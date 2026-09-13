"""
deploy_vast.py -- provision one Vast.ai GPU per model for bootstrap_extended.sh, monitor,
and destroy. Reads Code/CURE/.env for VAST_AI_API_2009_KEY (the CLI key), HUGGINGFACE_TOKEN
and Github_Classic_Token (injected into the container as env vars; never printed, never
written to disk here). State is kept in .vast_ext_state.json (gitignored: *.json under this
directory is not, so the file name is listed in Code/CURE/.gitignore).

Image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel (torch 2.5.1 cu124 preinstalled, so the
bootstrap only adds the requirements and the precompiled flash-attention wheel).

Usage
  python deploy_vast.py check                       # key works, balance
  python deploy_vast.py offers                      # candidate machines, cheapest first
  python deploy_vast.py launch --models qwen2.5-7b-instruct llama-3.1-8b-instruct gemma-2-2b-it phi-4-mini-instruct
  python deploy_vast.py status                      # instance state + last GitHub status line per model
  python deploy_vast.py logs --model qwen2.5-7b-instruct
  python deploy_vast.py restart [--model M]         # pull + relaunch the bootstrap (resume-aware)
  python deploy_vast.py destroy [--model M]         # all, or one

Offer filter (Next_Plan.md Section 5: one suitable GPU per model, measured memory): a single
verified, reliable GPU with >= 40 GB so an 8B bf16 model plus generation never approaches the
limit, CUDA >= 12.4 for the cu124 wheels, fast inbound network for the model download.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = HERE.parent / ".env"
STATE = HERE / ".vast_ext_state.json"
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
DISK_GB = 90
MODELS = ["llama-3.1-8b-instruct", "qwen2.5-7b-instruct", "gemma-2-2b-it", "phi-4-mini-instruct"]
GPU_OK = ("A100", "H100", "L40S", "L40", "A6000", "RTX 6000 Ada", "A40", "H200", "RTX 5090", "RTX PRO 6000")
QUERY = ("num_gpus=1 gpu_ram>=40 cuda_vers>=12.4 reliability>0.97 inet_down>400 "
         "disk_space>=%d rentable=true verified=true dph<=%.2f" % (DISK_GB, 1.80))


def read_env() -> dict:
    out = {}
    for line in ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


E = read_env()
KEY = E.get("VAST_AI_API_2009_KEY", "")
SECRETS = [v for v in (KEY, E.get("HUGGINGFACE_TOKEN", ""), E.get("Github_Classic_Token", "")) if v]


def scrub(t: str) -> str:
    for s in SECRETS:
        t = t.replace(s, "***")
    return t


def vast(*args, raw=True, check=True) -> str:
    cmd = ["vastai", "--api-key", KEY] + (["--raw"] if raw else []) + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        print("vastai failed:", scrub(" ".join(cmd[3:])))
        print(scrub(r.stdout)[-600:]); print(scrub(r.stderr)[-600:])
        sys.exit(1)
    return scrub(r.stdout)


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=2))


# ---------------------------------------------------------------- commands

def cmd_check(_):
    if not KEY:
        print("VAST_AI_API_2009_KEY missing from Code/CURE/.env"); sys.exit(1)
    u = json.loads(vast("show", "user"))
    print("account:", u.get("email", "?"), "| credit: $%.2f" % float(u.get("credit", 0)))
    for k in ("HUGGINGFACE_TOKEN", "Github_Classic_Token"):
        print(k, "present" if E.get(k) else "MISSING")


def offers(limit=15) -> list[dict]:
    out = json.loads(vast("search", "offers", QUERY, "-o", "dph"))
    ok = [o for o in out if any(g.lower() in str(o.get("gpu_name", "")).lower() for g in GPU_OK)]
    return ok[:limit]


def cmd_offers(_):
    for o in offers():
        print("%-9s %-22s %5.0fGB  $%.3f/h  rel %.3f  down %5.0f Mb/s  disk %4.0fGB  cuda %s  %s"
              % (o["id"], o.get("gpu_name"), o.get("gpu_ram", 0) / 1024, o.get("dph_total", 0),
                 o.get("reliability2", 0), o.get("inet_down", 0), o.get("disk_space", 0),
                 o.get("cuda_max_good"), o.get("geolocation", "")))


def onstart_script(model: str) -> str:
    return "\n".join([
        "#!/bin/bash",
        "export DEBIAN_FRONTEND=noninteractive",
        "mkdir -p /workspace && cd /workspace",
        "apt-get update -y >/dev/null 2>&1; apt-get install -y --no-install-recommends git >/dev/null 2>&1",
        "if [ ! -d /workspace/Cure_Audit_Benchmark ]; then",
        "  git clone -q https://${Github_Classic_Token}@github.com/DevDaring/Cure_Audit_Benchmark.git /workspace/Cure_Audit_Benchmark",
        "fi",
        "cd /workspace/Cure_Audit_Benchmark && git pull -q --rebase origin main || true",
        "nohup bash /workspace/Cure_Audit_Benchmark/Code/CURE/Extended_Research_Codes/bootstrap_extended.sh "
        "> /workspace/ext_boot.log 2>&1 &",
        "",
    ])


def cmd_launch(args):
    st = load_state()
    pool = offers(limit=40)
    if not pool:
        print("no offers match", QUERY); sys.exit(1)
    used = set()
    for model in args.models:
        if model in st and st[model].get("instance_id"):
            print(model, "already has instance", st[model]["instance_id"]); continue
        cand = [o for o in pool if o["id"] not in used and o.get("machine_id") not in
                {st[m].get("machine_id") for m in st}]
        if not cand:
            print("ran out of distinct offers for", model); break
        o = cand[0]; used.add(o["id"])
        script = HERE / f".onstart_{model}.sh"
        script.write_text(onstart_script(model), encoding="utf-8")
        env = ("-e EXT_MODEL=%s -e RANDOM_SEED=%s -e HUGGINGFACE_TOKEN=%s -e Github_Classic_Token=%s "
               "-e EXT_P1_HOURS=%s -e EXT_P2_DEV_HOURS=%s -e EXT_P2_TEST_HOURS=%s -e EXT_P3_HOURS=%s "
               "-e EXT_EXTEND_RANKS=%s -e EXT_CAP_POLICY=%s -e EXT_SKIP_P3=%s"
               % (model, E.get("RANDOM_SEED", "20260101"), E["HUGGINGFACE_TOKEN"], E["Github_Classic_Token"],
                  args.p1_hours, args.p2_dev_hours, args.p2_test_hours, args.p3_hours,
                  1 if args.extend_ranks else 0, args.cap_policy, 1 if args.skip_p3 else 0))
        out = vast("create", "instance", str(o["id"]), "--image", IMAGE, "--disk", str(DISK_GB),
                   "--onstart", str(script), "--env", env, "--ssh", "--direct", "--label", f"ext-{model}")
        script.unlink(missing_ok=True)
        try:
            rec = json.loads(out)
        except json.JSONDecodeError:
            print("unexpected create output:", out[:300]); continue
        iid = rec.get("new_contract")
        st[model] = {"instance_id": iid, "offer_id": o["id"], "machine_id": o.get("machine_id"),
                     "gpu": o.get("gpu_name"), "dph": o.get("dph_total"), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        save_state(st)
        print("%-24s instance %s on %s at $%.3f/h (offer %s)" % (model, iid, o.get("gpu_name"), o.get("dph_total", 0), o["id"]))


def github_status(model: str) -> str:
    r = subprocess.run(["gh", "api", f"repos/DevDaring/Cure_Audit_Benchmark/contents/Code/CURE/results/EXT_STATUS_{model}.txt",
                        "-q", ".content"], capture_output=True, text=True)
    if r.returncode != 0:
        return "(no status file yet)"
    import base64
    try:
        return base64.b64decode(r.stdout.strip()).decode().strip()
    except Exception:
        return "(unreadable)"


def cmd_status(_):
    st = load_state()
    if not st:
        print("no instances in state"); return
    inst = {i["id"]: i for i in json.loads(vast("show", "instances"))}
    for model, rec in st.items():
        i = inst.get(rec.get("instance_id"), {})
        print("%-24s id=%s  %s  %s  $%.3f/h  up %s"
              % (model, rec.get("instance_id"), i.get("actual_status", "?"), i.get("gpu_name", rec.get("gpu")),
                 float(i.get("dph_total") or rec.get("dph") or 0),
                 ("%.1fh" % ((time.time() - float(i["start_date"])) / 3600)) if i.get("start_date") else "?"))
        print("      github: %s" % github_status(model))


def cmd_logs(args):
    st = load_state()
    iid = st.get(args.model, {}).get("instance_id")
    if not iid:
        print("no instance for", args.model); return
    print(vast("logs", str(iid), "--tail", str(args.tail), raw=False, check=False)[-6000:])


def api_destroy(iid: int) -> tuple[int, str]:
    """The CLI's `destroy instance` asks for interactive confirmation; the REST call does not."""
    import urllib.request, urllib.error
    req = urllib.request.Request(f"https://console.vast.ai/api/v0/instances/{iid}/", method="DELETE")
    req.add_header("Authorization", "Bearer " + KEY); req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode()[:120]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:160]


RESTART_CMD = r'''
S=$(printf '%s%s' 'boot' 'strap_extended.sh')
pkill -f "^bash .*${S}$" >/dev/null 2>&1; pkill -x sleep >/dev/null 2>&1
# the stage processes are children the bootstrap does not take down with it
pkill -f "[r]un_all.py" >/dev/null 2>&1; pkill -f "[p]1_pilot.py|[p]2_controlled_erasure.py|[p]3_explanation.py" >/dev/null 2>&1
sleep 3; pkill -9 -f "[p]1_pilot.py|[p]2_controlled_erasure.py|[p]3_explanation.py" >/dev/null 2>&1; sleep 1
echo "stage processes left: $(pgrep -fc '[p]1_pilot.py|[p]2_controlled_erasure.py|[p]3_explanation.py')"
while IFS= read -r -d '' kv; do case "$kv" in EXT_*=*|HUGGINGFACE_TOKEN=*|Github_Classic_Token=*|RANDOM_SEED=*) export "$kv";; esac; done < /proc/1/environ
cd /workspace/Cure_Audit_Benchmark && git checkout -q -- Code/CURE/results/reanalysis_v2 2>/dev/null; git clean -fdq Code/CURE/results/reanalysis_v2 2>/dev/null
git pull -q --rebase origin main >/dev/null 2>&1
nohup bash "Code/CURE/Extended_Research_Codes/${S}" > /workspace/ext_boot.log 2>&1 &
sleep 5; echo "model=$EXT_MODEL running=$(pgrep -fc "^bash .*${S}$") head=$(git log --oneline -1 | cut -c1-50)"
'''


def ssh_run(iid: int, cmd: str, timeout: int = 120) -> str:
    """Run a command on an instance over SSH with the account's registered RSA key."""
    out = vast("ssh-url", str(iid), raw=False, check=False).strip()
    m = re.search(r"ssh://root@([\w.\-]+):(\d+)", out)
    if not m:
        return "no ssh url"
    r = subprocess.run(["ssh", "-i", str(Path.home() / ".ssh" / "id_rsa"), "-o", "StrictHostKeyChecking=no",
                        "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", "-p", m.group(2), f"root@{m.group(1)}", cmd],
                       capture_output=True, text=True, timeout=timeout)
    lines = [l for l in scrub(r.stdout).strip().splitlines() if l and "vast.ai" not in l and "Have fun" not in l]
    return " | ".join(lines) if r.returncode == 0 else "ssh error: " + scrub(r.stderr)[-160:]


def cmd_restart(args):
    """Kill the bootstrap (and its idle sleep), pull the latest code, and start it again with the
    container's injected environment. The bootstrap is resume-aware, so finished stages are
    not repeated. The script name is assembled at runtime so pkill never matches this shell."""
    st = load_state()
    for model in ([args.model] if args.model else list(st)):
        iid = st.get(model, {}).get("instance_id")
        if iid:
            print("%-24s %s" % (model, ssh_run(int(iid), RESTART_CMD)))


def cmd_destroy(args):
    st = load_state()
    targets = [args.model] if args.model else list(st)
    for model in targets:
        iid = st.get(model, {}).get("instance_id")
        if iid:
            code, body = api_destroy(int(iid))
            print(model, "destroy", iid, "->", code, scrub(body))
            if code == 200:
                st.pop(model, None)
    save_state(st)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check"); sub.add_parser("offers"); sub.add_parser("status")
    l = sub.add_parser("launch")
    l.add_argument("--models", nargs="+", default=MODELS)
    l.add_argument("--p1-hours", default="2"); l.add_argument("--p2-dev-hours", default="4")
    l.add_argument("--p2-test-hours", default="4"); l.add_argument("--p3-hours", default="2")
    l.add_argument("--extend-ranks", action="store_true", default=False,
                   help="also run ranks 2 and 4 in P2 (roughly triples the control conditions)")
    l.add_argument("--skip-p3", action="store_true", default=False)
    l.add_argument("--cap-policy", default="all")
    g = sub.add_parser("logs"); g.add_argument("--model", required=True); g.add_argument("--tail", default="200")
    d = sub.add_parser("destroy"); d.add_argument("--model", default=None)
    rs = sub.add_parser("restart"); rs.add_argument("--model", default=None)
    args = ap.parse_args()
    {"check": cmd_check, "offers": cmd_offers, "launch": cmd_launch, "status": cmd_status,
     "logs": cmd_logs, "destroy": cmd_destroy, "restart": cmd_restart}[args.cmd](args)


if __name__ == "__main__":
    main()
