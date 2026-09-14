"""
nn_common.py -- shared paths and helpers for the ICLR-feedback cycle (Submission2/Next_Plan.md,
14 September 2026, groups A and B). Builds on Code/CURE/Next_Run (nr_common, f1_scoring,
f2_runner) and Code/CURE/Extended_Research_Codes (P1 hooks, P2 bases); never modifies anything
under results/v2 or results/final_20260914. Every output of this cycle goes to
results/feedback_20260915/ with model-suffixed file names (two VMs never write one file).

Outputs (results/feedback_20260915/)
  g1_invariant_<model>.parquet    A1: pooled-audit patching effect re-read with shift-invariant
                                  statistics (log-prob, option margin) next to the raw logit,
                                  unedited / span erasure / three random subspaces / LEACE rows
  g2_alignment_<model>.parquet    A2: fresh-replication bridge on unequal-length pairs under the
                                  truncate / mean-pad / last-pad alignment rules
  g3_leace_<model>.npz, g3_leace_<model>.json
                                  A3: man/woman LEACE refit on a large BBQ pool (N >> d), with
                                  effective ranks, probes, cosine to the contrast basis
  g3_rows_<model>.parquet         A3: LC2 (conditioned LEACE) on the fresh gender-eligible seeds
  g5_steering_<model>.parquet     B: debiasing-vector steering at span / last token / every
                                  prefill position, energy-calibrated, fresh set + control set
  g5_calibration_<model>.json     the steering vector provenance and strengths
  g4_*.csv / .json                CPU analyses (strata, length decomposition, correlations, drift)
  g_summary.csv, g_tests.csv      merged statistics (g6_analysis.py, author's machine)
  COMPLETION_G.txt                closing record

Part of the CURE codebase (ICLR 2027, Submission2).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                      # Code/CURE/Next2Next_Run
CURE = HERE.parent
NR = CURE / "Next_Run"
EXT = CURE / "Extended_Research_Codes"
for p in (str(NR), str(EXT), str(CURE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import nr_common as N          # noqa: E402  (Next_Run)
import common as K             # noqa: E402  (Extended_Research_Codes)

OUT = K.RESULTS / os.environ.get("NN_OUT_DIR", "feedback_20260915")
FINAL = N.FINAL                # results/final_20260914 (read only)
V2 = N.V2                      # results/v2 (read only)

ALL_MODELS = K.MODELS
FRESH_MODELS = N.FINAL_MODELS  # gemma-2-2b-it, llama-3.1-8b-instruct
RANDOM_SEED = 20260915
N_BOOT = N.N_BOOT_FINAL
MATCH_TOL = N.MATCH_TOL

log = N.setup_logging("cure.feedback")


def ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def utc_now() -> str:
    return N.utc_now()


def write_json(obj, path: Path) -> Path:
    return N.write_json(obj, path)


def read_json(path: Path, default=None):
    return N.read_json(path, default)


def append_rows(path: Path, rows: list[dict]) -> None:
    N.append_rows(path, rows)


def done_keys(path: Path, cols: list[str]) -> set:
    return N.done_keys(path, cols)


def load_model(model_name: str):
    """GPU load through P1 (bf16, flash attention when present); NR_CPU=1 -> float32 CPU."""
    import f2_runner as F2
    return F2.load_for_run(model_name)


def unload_model(model_name: str) -> None:
    try:
        from load_osm import unload_model as _u
        _u(model_name)
    except Exception:
        pass
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
