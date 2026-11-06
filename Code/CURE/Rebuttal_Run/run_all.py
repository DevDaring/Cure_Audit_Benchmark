"""
run_all.py -- single entry point of the rebuttal cycle (Submission2, ICLR 2027 author response).

Stages
  h5    spans, identity checks and steering vector for the two replication models GPU
  h1a   site x operator cells at total-energy fractions 10, 1, 0.1 and alpha 1       GPU
  h1b   the dose curve at fractions 0.3 and 0.03                                      GPU
  h2    LEACE / mean-difference baselines at equal energy on the gender-flip set     GPU
  h3    every demographic pair of the pooled test seeds, unselected                   GPU
  h4    prompt-rewrite baseline                                                       GPU
  h6    statistics and tables (author's machine, CPU)
  h7    per-token option scores (explains the length-normalised reversal), after h1a GPU
  all   the model's own sequence (below)

Per model
  fresh  (gemma-2-2b-it, llama-3.1-8b-instruct)   h1a h7 h4 h2 h3 h1b
  replication (qwen2.5-7b-instruct, phi-4-mini-instruct)   h5 h1a h7 h4 h3 h1b

Usage: python run_all.py --stage all --model gemma-2-2b-it [--smoke]
       python run_all.py --stage h6
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import hh_common as H

HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run([sys.executable] + cmd, check=True, cwd=HERE)


def sequence(model: str) -> list[str]:
    if model in H.FRESH_MODELS:
        return ["h1a", "h7", "h4", "h2", "h3", "h1b"]
    return ["h5", "h1a", "h7", "h4", "h3", "h1b"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["h1a", "h1b", "h2", "h3", "h4", "h5", "h6", "h7", "all"])
    ap.add_argument("--model", default=None, choices=H.ALL_MODELS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    smoke = ["--smoke"] if a.smoke else []
    if a.stage == "h6":
        run(["h6_analysis.py"]); return
    if not a.model:
        raise SystemExit("--model is required")
    stages = [a.stage] if a.stage != "all" else sequence(a.model)
    for st in stages:
        if st == "h5" and a.model in H.REPLICATION_MODELS:
            run(["h5_replication_prep.py", "--model", a.model] + smoke)
        elif st == "h1a":
            run(["h1_dose_site.py", "--model", a.model, "--pass", "a"] + smoke)
        elif st == "h1b":
            run(["h1_dose_site.py", "--model", a.model, "--pass", "b"] + smoke)
        elif st == "h2" and a.model in H.FRESH_MODELS:
            run(["h2_gender_dose.py", "--model", a.model] + smoke)
        elif st == "h3" and a.model in H.POOLED_MODELS:
            run(["h3_all_pairs.py", "--model", a.model] + smoke)
        elif st == "h7":
            run(["h7_token_scores.py", "--model", a.model] + smoke)
        elif st == "h4":
            run(["h4_rewrite.py", "--model", a.model] + smoke)
        else:
            print("stage %s does not apply to %s; skipped" % (st, a.model), flush=True)


if __name__ == "__main__":
    main()
