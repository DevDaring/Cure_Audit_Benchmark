"""
check_vms.py -- independent hourly health and correctness check of the four Extended
Research VMs, run from the author's machine. It does not trust the VMs' own status lines:

  liveness   (Vast API + SSH)  instance running; bootstrap process alive; a stage process
             running or a DONE marker present; GPU busy; free disk; no Traceback in the
             newest stage log; the last status line and its age
  results    (git pull of DevDaring/Cure_Audit_Benchmark into the scratch clone) for every
             results/v2_<model> and v2_smoke_<model> directory:
               pilot_per_item.parquet   conditions present, rows per condition, error rate,
                                        identity checks passed (pilot_protocol.json), seeds
                                        in the manifest DEV split
               p2_per_item.parquet      phases, conditions, cond_ids, rows, error rate,
                                        finite |C|, identity_alpha0 == unedited within 1e-3,
                                        dev/test seeds in the right manifest split, no
                                        duplicate keys, identity checks passed
               p2_basis_*               one file per fitted family, manifest status per family
               p2_capability.parquet    rows per condition, status
               p3_<account>_per_item    rows per condition, error rate
             cross-model: p2_test_seeds.csv / p2_dev_seeds.csv / pilot_seeds.csv agree
  budget     credit, burn rate, hours left, and (once P1 timings exist) the projected hours
             for P2 from pilot_protocol.json throughput

Output: a compact report to stdout and results/vm_check_<UTC>.json in the scratch clone's
parent (never pushed). Exit code 0 = no problems, 1 = warnings, 2 = a VM needs attention.

  push       (--push) over SSH, each VM commits and pushes its current outputs now (under the
             VM's own git lock); then the per-model directories are merged locally with
             merge_v2.merge() into results/v2 and that consolidated directory is pushed from
             here (no markdown, no secrets)

Usage
  python check_vms.py                      # full check
  python check_vms.py --push               # full check + pushes
  python check_vms.py --no-ssh             # results + budget only
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common as K
import deploy_vast as D

CLONE = Path(r"C:\Users\Debz\AppData\Local\Temp\claude\d--PhD-Audit-Benchmark\66e31894-16d3-477d-8371-6bd5e18be9e5\scratchpad\cure_repo")
SSH_KEY = Path.home() / ".ssh" / "id_rsa"
P1_CONDS = ["unedited", "identity_alpha0", "span_seq_r1", "span_frozen_r1", "last_token_r1"]
P2_FAMILIES = ["unedited", "identity_alpha0", "cure_centred_svd", "uncentred_svd", "mean_difference",
               "leace_faithful", "leace_sequential", "random_ortho_1", "random_ortho_2", "random_ortho_3",
               "neutral_contrast"]

problems: list[tuple[str, str, str]] = []   # (level, model, message)


def note(level: str, model: str, msg: str) -> None:
    problems.append((level, model, msg))


# ---------------------------------------------------------------- liveness

def ssh_url(iid: int) -> tuple[str, int] | None:
    out = D.vast("ssh-url", str(iid), raw=False, check=False).strip()
    m = re.search(r"ssh://root@([\w.\-]+):(\d+)", out)
    return (m.group(1), int(m.group(2))) if m else None


REMOTE = r'''
R=/workspace/Cure_Audit_Benchmark; C=$R/Code/CURE
echo "BOOT_PROCS=$(pgrep -fc '[b]ootstrap_extended.sh')"
echo "STAGE_PROCS=$(pgrep -fc '[p]1_pilot.py|[p]2_controlled_erasure.py|[p]3_explanation.py')"
echo "GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "DISK=$(df -h /workspace 2>/dev/null | awk 'NR==2{print $4" free of "$2}')"
echo "STATUS=$(cat $C/results/EXT_STATUS_$EXT_MODEL.txt 2>/dev/null | tail -1)"
echo "DONE=$(ls $C/results/v2_*/DONE_*.txt 2>/dev/null | wc -l)"
L=$(ls -t $C/logs/ext_*.log 2>/dev/null | head -1)
echo "LOG=$L"
if [ -n "$L" ]; then echo "TRACEBACKS=$(grep -c Traceback "$L")"; echo "LOG_AGE_S=$(( $(date +%s) - $(stat -c %Y "$L") ))"; echo "TAIL<<"; tail -4 "$L" | cut -c1-200; echo ">>"; fi
echo "BOOT_TAIL<<"; tail -3 /workspace/ext_boot.log 2>/dev/null | cut -c1-200; echo ">>"
'''


def check_liveness(model: str, rec: dict, inst: dict) -> dict:
    out = {"instance": rec.get("instance_id"), "state": inst.get("actual_status"),
           "gpu": inst.get("gpu_name"), "dph": inst.get("dph_total")}
    if inst.get("actual_status") != "running":
        note("ERROR", model, "instance state is %s" % inst.get("actual_status")); return out
    hp = ssh_url(int(rec["instance_id"]))
    if not hp:
        note("WARN", model, "no ssh url"); return out
    r = subprocess.run(["ssh", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                        "-o", "ConnectTimeout=25", "-p", str(hp[1]), f"root@{hp[0]}", "export EXT_MODEL=%s; %s" % (model, REMOTE)],
                       capture_output=True, text=True, timeout=90)
    if r.returncode != 0:
        note("WARN", model, "ssh failed: %s" % (r.stderr.strip().splitlines() or ["?"])[-1][:120]); return out
    txt = D.scrub(r.stdout)
    kv = dict(re.findall(r"^([A-Z_]+)=(.*)$", txt, flags=re.M))
    out.update(kv)
    tails = re.findall(r"TAIL<<\n(.*?)\n>>", txt, flags=re.S)
    out["log_tail"] = tails[0] if tails else ""
    out["boot_tail"] = tails[1] if len(tails) > 1 else ""
    done = int(kv.get("DONE", "0") or 0)
    if int(kv.get("BOOT_PROCS", "0") or 0) == 0 and not done:
        note("ERROR", model, "bootstrap process is not running and no DONE marker")
    if int(kv.get("STAGE_PROCS", "0") or 0) == 0 and not done and "starting" in kv.get("STATUS", ""):
        note("WARN", model, "status says a stage is starting but no stage process is running")
    if int(kv.get("TRACEBACKS", "0") or 0) > 0:
        note("WARN", model, "Traceback in %s (retry loop may recover)" % Path(kv.get("LOG", "")).name)
    gpu_pct = int(re.match(r"\s*(\d+)", kv.get("GPU", "0") or "0").group(1)) if re.match(r"\s*(\d+)", kv.get("GPU", "0") or "0") else 0
    if kv.get("LOG_AGE_S") and int(kv["LOG_AGE_S"]) > 2400 and not done and gpu_pct < 5:
        note("WARN", model, "newest stage log silent for %d min and GPU idle" % (int(kv["LOG_AGE_S"]) // 60))
    if "FATAL" in kv.get("STATUS", "") or "gave up" in kv.get("STATUS", ""):
        note("ERROR", model, "VM status: " + kv["STATUS"][:160])
    return out


# ---------------------------------------------------------------- results

def git_pull_clone() -> None:
    subprocess.run(["git", "-C", str(CLONE), "pull", "-q", "--rebase", "origin", "main"], capture_output=True, text=True)


def manifest_split() -> dict[str, str]:
    m = pd.read_csv(K.OUT_P0 / "split_manifest.csv")
    return dict(zip(m["seed_id"], m["split"]))


def _err_rate(df: pd.DataFrame) -> float:
    if "error" not in df:
        return 0.0
    return float(df["error"].notna().mean())


def check_pilot(model: str, d: Path, split: dict, smoke: bool) -> dict:
    p = d / "pilot_per_item.parquet"
    out = {}
    if not p.exists():
        return {"present": False}
    df = pd.read_parquet(p)
    out.update({"present": True, "rows": len(df), "conditions": df["condition"].value_counts().to_dict() if "condition" in df else {}})
    tag = "smoke " if smoke else ""
    missing = [c for c in P1_CONDS if c not in out["conditions"]]
    if missing and (smoke or len(df) > 50):
        note("WARN", model, tag + "P1 missing conditions %s" % missing)
    er = _err_rate(df); out["error_rate"] = er
    if er > 0.1:
        note("WARN", model, tag + "P1 error rate %.0f%%" % (100 * er))
    bad = sorted({s for s in df["seed_id"] if split.get(s) != "dev"})
    if bad:
        note("ERROR", model, tag + "P1 used %d seeds outside the DEV split: %s" % (len(bad), bad[:3]))
    pj = d / "pilot_protocol.json"
    if pj.exists():
        proto = json.loads(pj.read_text(encoding="utf-8"))
        mrec = (proto.get("models") or {}).get(model, {})
        ic = mrec.get("identity_checks") or {}
        out["identity_pass_all"] = ic.get("pass_all")
        if ic and not ic.get("pass_all", False):
            note("ERROR", model, tag + "P1 identity checks FAILED: %s" % json.dumps(ic)[:200])
        thr = mrec.get("throughput") or mrec.get("sec_per_item") or mrec.get("timing")
        out["throughput"] = thr
        out["attn"] = mrec.get("attn_implementation") or proto.get("attn_implementation")
    return out


def check_p2(model: str, d: Path, split: dict, smoke: bool) -> dict:
    p = d / "p2_per_item.parquet"
    out = {}
    if not p.exists():
        return {"present": False}
    df = pd.read_parquet(p)
    tag = "smoke " if smoke else ""
    if "status" in df:      # status rows (a condition that could not run) carry placeholder keys
        out["status_rows"] = df[df["status"] != "ok"]["cond_id"].value_counts().to_dict() if "cond_id" in df else int((df["status"] != "ok").sum())
        df = df[df["status"] == "ok"]
    out.update({"present": True, "rows": len(df),
                "phases": df["phase"].value_counts().to_dict() if "phase" in df else {},
                "conditions": df["condition"].value_counts().to_dict() if "condition" in df else {},
                "n_cond_ids": int(df["cond_id"].nunique()) if "cond_id" in df else None})
    key = [c for c in ("model_name", "phase", "cond_id", "seed_id", "pair_type") if c in df]
    dup = int(df.duplicated(subset=key).sum()) if key else 0
    if dup:
        note("ERROR", model, tag + "P2 has %d duplicate key rows" % dup)
    er = _err_rate(df); out["error_rate"] = er
    if er > 0.1:
        note("WARN", model, tag + "P2 error rate %.0f%% (%s)" % (100 * er, df.loc[df["error"].notna(), "error"].astype(str).str[:60].value_counts().head(2).to_dict()))
    if "absC" in df:
        fin = df["absC"].astype(float)
        out["absC_finite"] = float(np.isfinite(fin).mean())
        if out["absC_finite"] < 0.9:
            note("WARN", model, tag + "P2 only %.0f%% finite |C|" % (100 * out["absC_finite"]))
    # identity_alpha0 must equal unedited on the same item
    if {"condition", "seed_id", "pair_type", "phase", "absC"} <= set(df.columns):
        u = df[df["condition"] == "unedited"].set_index(["phase", "seed_id", "pair_type"])["absC"]
        i = df[df["condition"] == "identity_alpha0"].set_index(["phase", "seed_id", "pair_type"])["absC"]
        j = pd.concat([u.rename("u"), i.rename("i")], axis=1, join="inner").dropna()
        if len(j):
            gap = float((j["u"] - j["i"]).abs().max()); out["identity_vs_unedited_max_gap"] = gap
            if gap > 1e-3:
                note("ERROR", model, tag + "P2 identity_alpha0 differs from unedited by up to %.4g" % gap)
    for ph, want in (("dev", "dev"), ("test", "test")):
        seeds = set(df.loc[df["phase"] == ph, "seed_id"]) if "phase" in df else set()
        bad = sorted(s for s in seeds if split.get(s) != want)
        if bad:
            note("ERROR", model, tag + "P2 %s phase used %d seeds outside the %s split" % (ph, len(bad), want.upper()))
    stat = df["status"].value_counts().to_dict() if "status" in df else {}
    out["status"] = stat
    # bases + manifest
    man = list(d.glob(f"p2_basis_manifest_{model}_*.json"))
    if man:
        mj = json.loads(man[0].read_text(encoding="utf-8"))
        out["basis_status"] = mj.get("status")
        for fam, st in (mj.get("status") or {}).items():
            if st not in ("ok", "insufficient", "skipped"):
                note("WARN", model, tag + "P2 basis %s status %s" % (fam, st))
        files = mj.get("files") or {}
        for fam, fn in files.items():
            if not (d / fn).exists():
                note("ERROR", model, tag + "P2 basis file missing on GitHub: %s" % fn)
    pj = d / "p2_protocol.json"
    if pj.exists():
        proto = json.loads(pj.read_text(encoding="utf-8"))
        mrec = (proto.get("models") or {}).get(model, {})
        ic = mrec.get("identity_checks") or {}
        out["identity_pass_all"] = ic.get("pass_all")
        if ic and not ic.get("pass_all", False):
            note("ERROR", model, tag + "P2 identity checks FAILED")
        out["attn"] = mrec.get("attn_implementation")
        out["sec_model_total"] = mrec.get("sec_model_total")
    cap = d / "p2_capability.parquet"
    if cap.exists():
        c = pd.read_parquet(cap)
        out["capability_rows"] = len(c)
        if "status" in c and (c["status"] != "ok").any():
            out["capability_status"] = c["status"].value_counts().to_dict()
    return out


def check_p3(model: str, d: Path, smoke: bool) -> dict:
    out = {}
    tag = "smoke " if smoke else ""
    for acc in ("magnitude", "depth", "massive"):
        p = d / f"p3_{acc}_per_item.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        er = _err_rate(df)
        out[acc] = {"rows": len(df), "phases": df["phase"].value_counts().to_dict() if "phase" in df else {},
                    "n_conditions": int(df["condition"].nunique()) if "condition" in df else None, "error_rate": er}
        if er > 0.1:
            note("WARN", model, tag + "P3 %s error rate %.0f%%" % (acc, 100 * er))
    return out


def check_results(models: list[str]) -> dict:
    git_pull_clone()
    split = manifest_split()
    res = CLONE / "Code" / "CURE" / "results"
    out = {}
    for model in models:
        for smoke in (True, False):
            d = res / (("v2_smoke_" if smoke else "v2_") + model)
            key = ("smoke" if smoke else "full")
            if not d.exists():
                out.setdefault(model, {})[key] = {"present": False}; continue
            rec = {"present": True, "files": len(list(d.rglob("*"))),
                   "done": (d / f"DONE_{model}.txt").exists(),
                   "p1": check_pilot(model, d, split, smoke), "p2": check_p2(model, d, split, smoke),
                   "p3": check_p3(model, d, smoke)}
            out.setdefault(model, {})[key] = rec
    # cross-model agreement of the frozen seed lists
    for fn in ("p2_test_seeds.csv", "p2_dev_seeds.csv", "pilot_seeds.csv"):
        sets = {}
        for model in models:
            p = res / f"v2_{model}" / fn
            if p.exists():
                sets[model] = set(pd.read_csv(p)["seed_id"])
        if len(sets) > 1 and len({frozenset(s) for s in sets.values()}) > 1:
            note("ERROR", "all", "%s differs across VMs: %s" % (fn, {m: len(s) for m, s in sets.items()}))
    out["_seed_lists_checked"] = True
    return out


# ---------------------------------------------------------------- pushes

VM_PUSH = r"""
cd /workspace/Cure_Audit_Benchmark || exit 3
( flock 9
  git checkout -q -- Code/CURE/results/reanalysis_v2 >/dev/null 2>&1 || true
  find Code/CURE/results/v2_* -type f ! -name '*.md' ! -name '*.log' -print0 2>/dev/null | xargs -0 -r git add -f >/dev/null 2>&1
  git add -u -- Code/CURE/results >/dev/null 2>&1
  git add -f "Code/CURE/results/EXT_STATUS_$EXT_MODEL.txt" >/dev/null 2>&1
  git commit -q -m "ext-results[$EXT_MODEL]: hourly check push" >/dev/null 2>&1 || true
  git pull --rebase -q origin main >/dev/null 2>&1 || git rebase --abort >/dev/null 2>&1
  git push -q origin main >/dev/null 2>&1 && echo PUSHED || echo PUSH_FAILED
) 9>/tmp/ext_git.lock
"""


def vm_push(model: str, rec: dict, inst: dict) -> str:
    if inst.get("actual_status") != "running":
        return "not running"
    hp = ssh_url(int(rec["instance_id"]))
    if not hp:
        return "no ssh url"
    r = subprocess.run(["ssh", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
                        "-o", "ConnectTimeout=25", "-p", str(hp[1]), f"root@{hp[0]}",
                        "export EXT_MODEL=%s; %s" % (model, VM_PUSH)], capture_output=True, text=True, timeout=300)
    return (r.stdout.strip().splitlines() or ["?"])[-1] if r.returncode == 0 else "ssh error"


def push_merged(models: list[str]) -> str:
    """Merge results/v2_<model> from the clone into the local results/v2 and push it."""
    import shutil
    import merge_v2 as M
    src = CLONE / "Code" / "CURE" / "results"
    n = 0
    for model in models:
        d = src / f"v2_{model}"
        if d.exists():
            shutil.copytree(d, K.RESULTS / f"v2_{model}", dirs_exist_ok=True); n += 1
    if n == 0:
        return "nothing to merge yet"
    man = M.merge(M.per_model_dirs())
    dst = src / "v2"
    dst.mkdir(parents=True, exist_ok=True)
    for p in (K.RESULTS / "v2").rglob("*"):
        if p.is_file() and p.suffix != ".md":
            t = dst / p.relative_to(K.RESULTS / "v2"); t.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, t)
    tok = D.E.get("Github_Classic_Token", "")
    g = lambda *a: subprocess.run(["git", "-C", str(CLONE)] + list(a), capture_output=True, text=True)
    g("remote", "set-url", "origin", f"https://{tok}@github.com/DevDaring/Cure_Audit_Benchmark.git")
    files = [str(p.relative_to(CLONE)) for p in dst.rglob("*") if p.is_file() and p.suffix != ".md"]
    if files:
        g("add", "-f", *files)
    staged = g("diff", "--cached", "--name-only").stdout.splitlines()
    if any(x.endswith(".md") or ".env" in x for x in staged):
        g("reset", "-q"); g("remote", "set-url", "origin", D.CLEAN_URL if hasattr(D, "CLEAN_URL") else "https://github.com/DevDaring/Cure_Audit_Benchmark.git")
        return "REFUSED: markdown or env staged"
    msg = "merged results/v2 (hourly check): %d files from %d VMs" % (len(files), n)
    g("commit", "-q", "-m", msg)
    g("pull", "--rebase", "-q", "origin", "main")
    r = g("push", "-q", "origin", "main")
    g("remote", "set-url", "origin", "https://github.com/DevDaring/Cure_Audit_Benchmark.git")
    return ("pushed %d files (%d merged, %d warnings)" % (len(files), len(man["files"]), len(man["warnings"]))) if r.returncode == 0 else "push failed: " + D.scrub(r.stderr)[-200:]


# ---------------------------------------------------------------- budget

def check_budget(st: dict, inst_by_id: dict) -> dict:
    u = json.loads(D.vast("show", "user"))
    credit = float(u.get("credit", 0))
    burn = sum(float((inst_by_id.get(r.get("instance_id")) or {}).get("dph_total") or r.get("dph") or 0) for r in st.values()
               if (inst_by_id.get(r.get("instance_id")) or {}).get("actual_status") == "running")
    out = {"credit": credit, "burn_per_h": burn, "hours_left": (credit / burn) if burn else None}
    if credit < 3.0:
        note("ERROR", "all", "credit $%.2f: instances stop at zero" % credit)
    elif burn and credit / burn < 6:
        note("WARN", "all", "credit $%.2f covers only %.1f more hours at $%.2f/h" % (credit, credit / burn, burn))
    return out


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ssh", action="store_true")
    ap.add_argument("--push", action="store_true", help="VM checkpoint pushes over SSH, then merge + push results/v2 from here")
    args = ap.parse_args()
    st = D.load_state()
    inst_by_id = {i["id"]: i for i in json.loads(D.vast("show", "instances"))}
    report = {"utc": K.utc_now(), "liveness": {}, "results": {}, "budget": {}}
    if not args.no_ssh:
        for model, rec in st.items():
            report["liveness"][model] = check_liveness(model, rec, inst_by_id.get(rec.get("instance_id"), {}))
    if args.push and not args.no_ssh:
        report["vm_push"] = {m: vm_push(m, r, inst_by_id.get(r.get("instance_id"), {})) for m, r in st.items()}
    report["results"] = check_results(list(st))
    if args.push:
        report["merged_push"] = push_merged(list(st))
    report["budget"] = check_budget(st, inst_by_id)
    report["problems"] = [{"level": l, "model": m, "msg": s} for l, m, s in problems]

    # compact print
    print("=== VM check %s ===" % report["utc"])
    for model in st:
        lv = report["liveness"].get(model, {})
        r = report["results"].get(model, {})
        sm, fu = r.get("smoke", {}), r.get("full", {})
        def brief(x):
            if not x or not x.get("present"):
                return "-"
            p1 = x["p1"].get("rows", 0) if x["p1"].get("present") else 0
            p2 = x["p2"].get("rows", 0) if x["p2"].get("present") else 0
            p3 = sum(v["rows"] for v in x["p3"].values()) if x["p3"] else 0
            return "p1=%d p2=%d(%s) p3=%d%s" % (p1, p2, ",".join("%s:%d" % kv for kv in x["p2"].get("phases", {}).items()) or "-", p3, " DONE" if x.get("done") else "")
        print("%-22s %-8s gpu[%s] procs boot=%s stage=%s | smoke: %s | full: %s"
              % (model, lv.get("state", "?"), lv.get("GPU", "?")[:22], lv.get("BOOT_PROCS", "?"), lv.get("STAGE_PROCS", "?"), brief(sm), brief(fu)))
        print("    status: %s" % lv.get("STATUS", "(no ssh)")[:150])
        if lv.get("log_tail"):
            last = [x for x in lv["log_tail"].splitlines() if x.strip()]
            if last:
                print("    log: %s" % last[-1][:150])
    if report.get("vm_push"):
        print("vm pushes: %s" % report["vm_push"])
    if report.get("merged_push"):
        print("merged push: %s" % report["merged_push"])
    b = report["budget"]
    print("credit $%.2f | burn $%.2f/h | %s h left" % (b["credit"], b["burn_per_h"], ("%.1f" % b["hours_left"]) if b["hours_left"] else "?"))
    if problems:
        print("--- problems ---")
        for l, m, s in problems:
            print("  [%s] %s: %s" % (l, m, s))
    else:
        print("no problems detected")
    out = CLONE.parent / ("vm_check_%s.json" % report["utc"].replace(":", "").replace("-", ""))
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    levels = {l for l, _, _ in problems}
    return 2 if "ERROR" in levels else (1 if "WARN" in levels else 0)


if __name__ == "__main__":
    sys.exit(main())
