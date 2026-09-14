"""
run_all.py -- single entry point for Code/CURE/Next_Run (Submission2/Next_Plan.md, final cycle).

  python run_all.py --stage f0                       # CPU: provenance, energy audit, legacy reanalysis
  python run_all.py --stage f1                       # CPU: fresh BBQ manifests, exclusions, freshness
  python run_all.py --stage f2 --model gemma-2-2b-it [--smoke] [--f2-stage all|checks|calibrate|timing|final|control|baseline]
  python run_all.py --stage f4                       # CPU: confirmatory / preservation / secondary tables, claims, COMPLETION
  python run_all.py --stage all                      # f0, f1 (f2 needs a GPU and one model at a time)

Every stage is resumable and writes only under results/final_20260914/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(HERE / script), *args]
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit("%s failed (%d)" % (script, r.returncode))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["f0", "f1", "f2", "f4", "all"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--f2-stage", default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model-hours", default=None)
    a, rest = ap.parse_known_args()
    if a.stage in ("f0", "all"):
        run("f0_provenance.py")
    if a.stage in ("f1", "all"):
        run("f1_data.py")
    if a.stage == "f2":
        if not a.model:
            raise SystemExit("--model is required for f2")
        extra = ["--stage", a.f2_stage] + (["--smoke"] if a.smoke else []) + (["--model-hours", a.model_hours] if a.model_hours else []) + rest
        run("f2_runner.py", "--model", a.model, *extra)
    if a.stage == "f4":
        run("f4_analysis.py")


if __name__ == "__main__":
    main()
