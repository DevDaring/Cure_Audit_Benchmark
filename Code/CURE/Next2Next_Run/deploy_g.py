"""
deploy_g.py -- Vast.ai orchestration for bootstrap_g.sh (ICLR-feedback cycle): one VM per
model group. Reuses Extended_Research_Codes/deploy_vast.py (key VAST_AI_API_2009_KEY from
Code/CURE/.env, offer filter, scrubbed CLI calls, REST destroy, ssh).

Usage
  python deploy_g.py check
  python deploy_g.py launch --groups "gemma-2-2b-it llama-3.1-8b-instruct" "qwen2.5-7b-instruct phi-4-mini-instruct"
  python deploy_g.py status
  python deploy_g.py logs --group "gemma-2-2b-it llama-3.1-8b-instruct" [--tail 200]
  python deploy_g.py restart [--group G] [--skip-smoke]
  python deploy_g.py destroy [--group G]
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

STATE = HERE / ".vast_g_state.json"
DV.STATE = STATE
DEFAULT_GROUPS = ["gemma-2-2b-it llama-3.1-8b-instruct", "qwen2.5-7b-instruct phi-4-mini-instruct"]


def tag_of(group: str) -> str:
    return group.replace(" ", "+")


def onstart() -> str:
    return "\n".join([
        "#!/bin/bash",
        "export DEBIAN_FRONTEND=noninteractive",
        "mkdir -p /workspace && cd /workspace",
        "apt-get update -y >/dev/null 2>&1; apt-get install -y --no-install-recommends git >/dev/null 2>&1",
        "if [ ! -d /workspace/Cure_Audit_Benchmark ]; then",
        "  git clone -q https://${Github_Classic_Token}@github.com/DevDaring/Cure_Audit_Benchmark.git /workspace/Cure_Audit_Benchmark",
        "fi",
        "cd /workspace/Cure_Audit_Benchmark && git pull -q --no-rebase origin main || true",
        "nohup bash /workspace/Cure_Audit_Benchmark/Code/CURE/Next2Next_Run/bootstrap_g.sh > /workspace/nn_boot.log 2>&1 &",
        "",
    ])


def cmd_launch(args):
    st = DV.load_state()
    pool = DV.offers(limit=40)
    if not pool:
        print("no offers"); sys.exit(1)
    used = set()
    for group in args.groups:
        tag = tag_of(group)
        if tag in st and st[tag].get("instance_id"):
            print(tag, "already has instance", st[tag]["instance_id"]); continue
        cand = [o for o in pool if o["id"] not in used and o.get("machine_id") not in {st[g].get("machine_id") for g in st}]
        if not cand:
            print("ran out of offers"); break
        o = cand[0]; used.add(o["id"])
        rate = o.get("dph_total", 0)
        script = HERE / (".onstart_g_%s.sh" % tag)
        script.write_text(onstart(), encoding="utf-8")
        env = ("-e NN_MODELS='%s' -e NN_GPU_RATE=%.3f -e RANDOM_SEED=%s -e HUGGINGFACE_TOKEN=%s -e Github_Classic_Token=%s%s"
               % (group, rate, DV.E.get("RANDOM_SEED", "20260101"), DV.E["HUGGINGFACE_TOKEN"], DV.E["Github_Classic_Token"],
                  " -e NN_SKIP_SMOKE=1" if args.skip_smoke else ""))
        out = DV.vast("create", "instance", str(o["id"]), "--image", DV.IMAGE, "--disk", str(DV.DISK_GB),
                      "--onstart", str(script), "--env", env, "--ssh", "--direct", "--label", "feedback-%s" % tag)
        script.unlink(missing_ok=True)
        try:
            rec = json.loads(out)
        except json.JSONDecodeError:
            print("unexpected create output:", out[:300]); continue
        st[tag] = {"instance_id": rec.get("new_contract"), "offer_id": o["id"], "machine_id": o.get("machine_id"), "gpu": o.get("gpu_name"),
                   "dph": o.get("dph_total"), "models": group, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        DV.save_state(st)
        print("launched %s on %s ($%.3f/h) instance %s" % (tag, o.get("gpu_name"), o.get("dph_total", 0), rec.get("new_contract")))


def github_status(tag: str) -> str:
    import urllib.request
    url = "https://raw.githubusercontent.com/DevDaring/Cure_Audit_Benchmark/main/Code/CURE/results/NN_STATUS_%s.txt" % tag
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.read().decode().strip()
    except Exception as ex:
        return "(no status yet: %s)" % str(ex)[:60]


def cmd_status(_):
    st = DV.load_state()
    inst = {str(i.get("id")): i for i in json.loads(DV.vast("show", "instances"))}
    for tag, rec in st.items():
        i = inst.get(str(rec.get("instance_id")), {})
        print("%-42s inst %s %-10s %-18s $%.3f/h gpu %s%% | %s" % (tag, rec.get("instance_id"), i.get("actual_status", "?"), rec.get("gpu"),
              float(i.get("dph_total") or rec.get("dph") or 0), i.get("gpu_util", "?"), github_status(tag)))


def cmd_logs(args):
    st = DV.load_state(); iid = st[tag_of(args.group)]["instance_id"]
    print(DV.ssh_run(int(iid), "tail -n %s /workspace/nn_boot.log; echo ----; for f in /workspace/Cure_Audit_Benchmark/Code/CURE/logs/nn_*.log; do echo == $f; tail -n 12 $f; done" % args.tail, timeout=180))


RESTART = r'''
S=$(printf '%s%s' 'boot' 'strap_g.sh')
pkill -f "^bash .*${S}$" >/dev/null 2>&1; pkill -x sleep >/dev/null 2>&1
pkill -f "[r]un_all.py" >/dev/null 2>&1; pkill -f "[g][0-9]_.*\.py" >/dev/null 2>&1; sleep 3; pkill -9 -f "[g][0-9]_.*\.py" >/dev/null 2>&1; sleep 1
while IFS= read -r -d '' kv; do case "$kv" in NN_*=*|HUGGINGFACE_TOKEN=*|Github_Classic_Token=*|RANDOM_SEED=*) export "$kv";; esac; done < /proc/1/environ
cd /workspace/Cure_Audit_Benchmark && git add -u -- Code/CURE/results >/dev/null 2>&1; git commit -q -m "nn: local before pull" >/dev/null 2>&1
git pull --no-rebase --no-edit -q origin main >/dev/null 2>&1 || git merge --abort >/dev/null 2>&1
export NN_SKIP_SMOKE=__SKIP__
nohup bash "Code/CURE/Next2Next_Run/${S}" > /workspace/nn_boot.log 2>&1 &
sleep 5; echo "models=$NN_MODELS running=$(pgrep -fc "^bash .*${S}$") head=$(git log --oneline -1 | cut -c1-60)"
'''


def cmd_restart(args):
    st = DV.load_state()
    for tag, rec in st.items():
        if args.group and tag != tag_of(args.group):
            continue
        cmd = RESTART.replace("__SKIP__", "1" if args.skip_smoke else "0")
        print("%-42s %s" % (tag, DV.ssh_run(int(rec["instance_id"]), cmd, timeout=240).strip()[-300:]))


def cmd_destroy(args):
    st = DV.load_state()
    for tag in list(st):
        if args.group and tag != tag_of(args.group):
            continue
        code, txt = DV.api_destroy(int(st[tag]["instance_id"]))
        print(tag, "destroy ->", code, txt[:80])
        if code in (200, 204):
            del st[tag]
    DV.save_state(st)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("check"); sub.add_parser("status"); sub.add_parser("offers")
    l = sub.add_parser("launch"); l.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS); l.add_argument("--skip-smoke", action="store_true")
    g = sub.add_parser("logs"); g.add_argument("--group", required=True); g.add_argument("--tail", default="200")
    d = sub.add_parser("destroy"); d.add_argument("--group", default=None)
    rs = sub.add_parser("restart"); rs.add_argument("--group", default=None); rs.add_argument("--skip-smoke", action="store_true")
    a = ap.parse_args()
    {"check": DV.cmd_check, "status": cmd_status, "offers": DV.cmd_offers, "launch": cmd_launch, "logs": cmd_logs, "destroy": cmd_destroy, "restart": cmd_restart}[a.cmd](a)


if __name__ == "__main__":
    main()
