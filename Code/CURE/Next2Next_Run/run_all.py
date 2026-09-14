"""
run_all.py -- single entry point of the ICLR-feedback cycle (Submission2/Next_Plan.md, groups
A and B). Stages:

  g4      CPU analyses from stored files (strata, length decomposition, correlations, drift)
  g3      LEACE refits (part a on every model; part b on the fresh models)         GPU
  g1      pooled-audit re-read with shift-invariant statistics (+ the g3 erasers)  GPU
  g2      span-alignment re-run on the fresh replication                            GPU
  g5      steering baselines on the fresh replication                               GPU
  g6      merged statistics (author's machine)
  all     g4 g3 g1 g2 g5 for one model (g2 / g5 / g3-b only on the fresh models)

Usage: python run_all.py --stage all --model gemma-2-2b-it [--smoke]
       python run_all.py --stage g4
       python run_all.py --stage g6
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nn_common as C

HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run([sys.executable] + cmd, check=True, cwd=HERE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["g1", "g2", "g3", "g4", "g5", "g6", "all"])
    ap.add_argument("--model", default=None, choices=C.ALL_MODELS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    smoke = ["--smoke"] if a.smoke else []
    stages = [a.stage] if a.stage != "all" else ["g4", "g3", "g1", "g2", "g5"]
    for st in stages:
        if st == "g4":
            run(["g4_analyses.py"])
        elif st == "g6":
            run(["g6_analysis.py"])
        else:
            if not a.model:
                raise SystemExit("--model is required for %s" % st)
            fresh = a.model in C.FRESH_MODELS
            if st == "g3":
                run(["g3_leace_conditioned.py", "--model", a.model, "--parts", "ab" if fresh else "a"] + smoke)
            elif st == "g1":
                run(["g1_invariant.py", "--model", a.model, "--extra-leace"] + smoke)
            elif st == "g2" and fresh:
                run(["g2_alignment.py", "--model", a.model] + smoke)
            elif st == "g5" and fresh:
                run(["g5_steering.py", "--model", a.model] + smoke)


if __name__ == "__main__":
    main()
