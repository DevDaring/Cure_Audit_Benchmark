"""
run_all.py -- single entry point for the Extended Research Codes (Submission2/Next_Plan.md).

Stages, in dependency order. Each stage is idempotent and resumable; rerunning a stage that
already has outputs is safe.

  p0     CPU. Evidence repair on the EXISTING artifacts: checksums and corrected dedup keys,
         metric audit (W vs D vs R vs X), degeneracy join and stage overlap, seed-group split
         manifest, prognosis rebuild, raw-archive search, TAU audit, evidence ledger and
         Submission2/Evidence_Contract.md. No model, no GPU, no mutation of results/.
  test   CPU. Functional tests of the edit (tiny model) and of the pilot selection logic.
  p1     GPU. 24-seed protocol-validation pilot on two models under a hard wall-clock cap.
         Gate: identity checks must pass; the report states which way the decision gate went.
  p2     GPU. Corrected, controlled erasure experiment with faithful LEACE, random and
         neutral controls, matched energy, canonical option scoring, external utility.
  p3     GPU. One explanation (--account depth|magnitude|massive), only after p2.
  p4     CPU. Optional independent risk forecast on p2 outputs, with its stop condition.

Usage
  python run_all.py --stage p0
  python run_all.py --stage test
  python run_all.py --stage p1 --models qwen2.5-7b-instruct llama-3.1-8b-instruct --gpu-hours-cap 8
  python run_all.py --stage p2 --phase dev --gpu-hours-cap 20
  python run_all.py --stage p2 --phase test
  python run_all.py --stage p3 --account magnitude
  python run_all.py --stage p4
  python run_all.py --stage all            (p0, test, p1; stops at the P1 gate for a human)

Everything after p1 is deliberately NOT chained by --stage all: Next_Plan.md P1 says "Choose
the final research claim from that pilot. Proceed with P2 only when it tests a viable,
distinct question." That choice is the author's.

Part of the CURE codebase (ICLR 2027, Submission2).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

P0_MODULES = [
    "p0_snapshot_integrity.py",
    "p0_metric_audit.py",
    "p0_data_integrity.py",
    "p0_split_manifest.py",
    "p0_prognosis_rebuild.py",
    "p0_raw_archive_search.py",
    "p0_tau_audit.py",
    "p0_claims_ledger.py",
]


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(HERE))
    print("[exit %d in %.0fs]" % (rc, time.time() - t0), flush=True)
    return rc


def stage_p0(args) -> int:
    for m in P0_MODULES:
        rc = run([sys.executable, m])
        if rc != 0:
            print("P0 stopped at %s" % m); return rc
    return 0


def stage_test(args) -> int:
    return run([sys.executable, "test_p1_cpu.py"])


def stage_p1(args) -> int:
    cmd = [sys.executable, "p1_pilot.py", "--fmt", args.fmt, "--gpu-hours-cap", str(args.gpu_hours_cap)]
    if args.models:
        cmd += ["--models"] + args.models
    if args.smoke:
        cmd.append("--smoke")
    if args.seeds_file:
        cmd += ["--seeds-file", args.seeds_file]
    return run(cmd + args.extra)


def stage_p2(args) -> int:
    cmd = [sys.executable, "p2_controlled_erasure.py", "--phase", args.phase,
           "--gpu-hours-cap", str(args.gpu_hours_cap)]
    if args.models:
        cmd += ["--models"] + args.models
    for flag in ("extend_ranks", "match_energy", "summarise_only"):
        if getattr(args, flag, False):
            cmd.append("--" + flag.replace("_", "-"))
    if args.capability_policy:
        cmd += ["--capability-policy", args.capability_policy]
    return run(cmd + args.extra)


def stage_p3(args) -> int:
    cmd = [sys.executable, "p3_explanation.py", "--account", args.account, "--phase", args.phase,
           "--gpu-hours-cap", str(args.gpu_hours_cap)]
    if args.models:
        cmd += ["--models"] + args.models
    return run(cmd + args.extra)


def stage_p4(args) -> int:
    cmd = [sys.executable, "p4_risk_forecast.py"]
    if args.models:
        cmd += ["--models"] + args.models
    return run(cmd + args.extra)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["p0", "test", "p1", "p2", "p3", "p4", "all"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--gpu-hours-cap", type=float, default=8.0)
    ap.add_argument("--fmt", choices=["chat", "raw"], default="chat")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds-file", default=None)
    ap.add_argument("--phase", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--extend-ranks", action="store_true")
    ap.add_argument("--match-energy", action="store_true")
    ap.add_argument("--summarise-only", action="store_true")
    ap.add_argument("--capability-policy", choices=["none", "all"], default=None)
    ap.add_argument("--account", choices=["depth", "magnitude", "massive"], default="magnitude")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="anything after '--' is forwarded verbatim to the stage script, "
                         "e.g. --stage p2 -- --leace-sequential --n-fit-pairs 300")
    args = ap.parse_args()
    args.extra = [a for a in args.extra if a != "--"]

    stages = {"p0": stage_p0, "test": stage_test, "p1": stage_p1, "p2": stage_p2,
              "p3": stage_p3, "p4": stage_p4}
    if args.stage == "all":
        for s in ("p0", "test", "p1"):
            rc = stages[s](args)
            if rc != 0:
                return rc
        print("\nP1 gate reached. Read Code/CURE/results/v2/pilot_report.md, decide the research "
              "claim, then run --stage p2 explicitly.")
        return 0
    return stages[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
