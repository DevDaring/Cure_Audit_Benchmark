"""
deploy_final.py -- one Vast.ai GPU per final model for bootstrap_final.sh; status, logs, destroy.
Reuses Extended_Research_Codes/deploy_vast.py (key handling, offer filter, scrubbed CLI calls,
REST destroy, ssh) with its own state file and onstart script.

Usage
  python deploy_final.py check
  python deploy_final.py launch --models gemma-2-2b-it llama-3.1-8b-instruct [--hours 12] [--rate auto]
  python deploy_final.py status
  python deploy_final.py logs --model gemma-2-2b-it [--tail 200]
  python deploy_final.py destroy [--model M]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Extended_Research_Codes"))
import deploy_vast as DV   # noqa: E402

STATE = HERE / ".vast_final_state.json"
DV.STATE = STATE
FINAL_MODELS = ["gemma-2-2b-it", "llama-3.1-8b-instruct"]


def onstart(model: str) -> str:
    return "\n".join([
        "#!/bin/bash",
        "export DEBIAN_FRONTEND=noninteractive",
        "mkdir -p /workspace && cd /workspace",
        "apt-get update -y >/dev/null 2>&1; apt-get install -y --no-install-recommends git >/dev/null 2>&1",
        "if [ ! -d /workspace/Cure_Audit_Benchmark ]; then",
        "  git clone -q https://${Github_Classic_Token}@github.com/DevDaring/Cure_Audit_Benchmark.git /workspace/Cure_Audit_Benchmark",
        "fi",
        "cd /workspace/Cure_Audit_Benchmark && git pull -q --no-rebase origin main || true",
        "nohup bash /workspace/Cure_Audit_Benchmark/Code/CURE/Next_Run/bootstrap_final.sh > /workspace/nr_boot.log 2>&1 &",
        "",
    ])


def cmd_launch(args):
    st = DV.load_state()
    pool = DV.offers(limit=40)
    if not pool:
        print("no offers"); sys.exit(1)
    used = set()
    for model in args.models:
        if model in st and st[model].get("instance_id"):
            print(model, "already has instance", st[model]["instance_id"]); continue
        cand = [o for o in pool if o["id"] not in used and o.get("machine_id") not in {st[m].get("machine_id") for m in st}]
        if not cand:
            print("ran out of offers"); break
        o = cand[0]; used.add(o["id"])
        rate = o.get("dph_total", 0) if args.rate == "auto" else float(args.rate)
        script = HERE / (".onstart_final_%s.sh" % model)
        script.write_text(onstart(model), encoding="utf-8")
        env = ("-e NR_MODEL=%s -e NR_MODEL_HOURS=%s -e NR_GPU_RATE=%.3f -e RANDOM_SEED=%s -e HUGGINGFACE_TOKEN=%s -e Github_Classic_Token=%s%s"
               % (model, args.hours, rate, DV.E.get("RANDOM_SEED", "20260101"), DV.E["HUGGINGFACE_TOKEN"], DV.E["Github_Classic_Token"],
                  (" -e NR_TIER=%s" % args.tier) if args.tier else ""))
        out = DV.vast("create", "instance", str(o["id"]), "--image", DV.IMAGE, "--disk", str(DV.DISK_GB),
                      "--onstart", str(script), "--env", env, "--ssh", "--direct", "--label", "final-%s" % model)
        script.unlink(missing_ok=True)
        try:
            rec = json.loads(out)
        except json.JSONDecodeError:
            print("unexpected create output:", out[:300]); continue
        st[model] = {"instance_id": rec.get("new_contract"), "offer_id": o["id"], "machine_id": o.get("machine_id"),
                     "gpu": o.get("gpu_name"), "dph": o.get("dph_total"), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        DV.save_state(st)
        print("launched %s on %s ($%.3f/h) instance %s" % (model, o.get("gpu_name"), o.get("dph_total", 0), rec.get("new_contract")))


def github_status(model: str) -> str:
    import urllib.request
    url = "https://raw.githubusercontent.com/DevDaring/Cure_Audit_Benchmark/main/Code/CURE/results/NR_STATUS_%s.txt" % model
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.read().decode().strip()
    except Exception as ex:
        return "(no status yet: %s)" % str(ex)[:60]


def cmd_status(_):
    st = DV.load_state()
    inst = {str(i.get("id")): i for i in json.loads(DV.vast("show", "instances"))}
    for m, rec in st.items():
        i = inst.get(str(rec.get("instance_id")), {})
        print("%-24s inst %s %-10s %-20s $%.3f/h  gpu %s%%  | %s" % (m, rec.get("instance_id"), i.get("actual_status", "?"),
              rec.get("gpu"), float(i.get("dph_total") or rec.get("dph") or 0), i.get("gpu_util", "?"), github_status(m)))


def cmd_logs(args):
    st = DV.load_state(); iid = st[args.model]["instance_id"]
    print(DV.ssh_run(int(iid), "tail -n %s /workspace/nr_boot.log; echo ----; ls /workspace/Cure_Audit_Benchmark/Code/CURE/logs/ 2>/dev/null; "
                     "for f in /workspace/Cure_Audit_Benchmark/Code/CURE/logs/nr_*.log; do echo == $f; tail -n 15 $f; done" % args.tail, timeout=180))


def cmd_destroy(args):
    st = DV.load_state()
    for m in list(st):
        if args.model and m != args.model:
            continue
        code, txt = DV.api_destroy(int(st[m]["instance_id"]))
        print(m, "destroy ->", code, txt[:80])
        if code in (200, 204):
            del st[m]
    DV.save_state(st)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("check"); sub.add_parser("status"); sub.add_parser("offers")
    l = sub.add_parser("launch"); l.add_argument("--models", nargs="+", default=FINAL_MODELS); l.add_argument("--hours", default="12")
    l.add_argument("--rate", default="auto"); l.add_argument("--tier", default=None)
    g = sub.add_parser("logs"); g.add_argument("--model", required=True); g.add_argument("--tail", default="200")
    d = sub.add_parser("destroy"); d.add_argument("--model", default=None)
    a = ap.parse_args()
    {"check": DV.cmd_check, "status": cmd_status, "offers": DV.cmd_offers, "launch": cmd_launch, "logs": cmd_logs, "destroy": cmd_destroy}[a.cmd](a)


if __name__ == "__main__":
    main()
