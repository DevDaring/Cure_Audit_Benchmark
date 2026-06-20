"""
config_scope.py -- central configuration for SCOPE (Surgical COncept-ablation via
Patchscope-guided Editing). SCOPE replaces CURE's all-layer erasure with a localized,
massive-activation-protected edit, and reuses the entire CURE/audit stack for everything
else (models, datasets, audit, baselines, metrics, judge, checkpointing).

Implements / builds on:
  - Patchscopes (Ghandeharioun et al. 2024, arXiv:2401.06102): the localisation and the
    post-edit verification use a Few-Shot Token-Identity Patchscope.
  - LEACE concept erasure (Belrose et al. 2023, arXiv:2306.03819): the projection eraser.
  - The causal discriminative-validity audit (Code/audit) and the CURE harness (Code/CURE).

All secrets come from Code/CURE/.env via config_cure (no secret literal lives here).
SCOPE writes its results INTO the CURE results dir, so the audit, baseline, SCOPE, and
held-out/behavioural artifacts are all taken from one folder (Code/CURE/results).
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # Code/SCOPE
CURE = HERE.parent / "CURE"                            # Code/CURE
AUDIT = HERE.parent / "audit"                          # Code/audit

# Make the CURE and audit packages importable (config_cure already adds the audit paths).
sys.path.insert(0, str(CURE))
sys.path.insert(0, str(AUDIT))
sys.path.insert(0, str(AUDIT / "GPU_CPU"))

# Reuse the whole CURE configuration: this loads Code/CURE/.env, the four open models,
# the dataset paths, tau, the judge tiers, and RESULTS=Code/CURE/results.
import config_cure as C  # noqa: E402

# Re-export the pieces the SCOPE code needs, so scope.py / run_scope.py import one module.
OSM_MODELS = C.OSM_MODELS
OSM_NAMES = C.OSM_NAMES
RESULTS = C.RESULTS                                    # Code/CURE/results (one place for all)
LOGS = C.LOGS
PENTAD_PATH = C.PENTAD_PATH
TAU = C.TAU
RANDOM_SEED = C.RANDOM_SEED
SUBSPACE_PAIRS = C.SUBSPACE_PAIRS
SWEEP_SUBSET = C.SWEEP_SUBSET
BASELINE_E4_LIMIT = C.BASELINE_E4_LIMIT
E4_MAX_TOKENS = C.E4_MAX_TOKENS
DRY_LIMIT = C.DRY_LIMIT
MAX_UTILITY_COST = C.MAX_UTILITY_COST
model_cfg = C.model_cfg

RESULTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


# ---------------------------------------------------------------------------
# SCOPE knobs (safe defaults; every one is read from the environment / .env).
#
#   LOCALIZE_PAIRS   counterfactual pairs used for the Patchscope decodability sweep.
#                    The bias direction is robust from a few hundred pairs, and the
#                    decodability map is a per-layer average, so a bounded subset suffices.
#   DECODE_PCTILE    a layer is a "localised site" when its mean decodability is at or
#                    above this percentile of the per-layer decodability map.
#   PCTILE_LADDER    if the edit at DECODE_PCTILE breaches the utility budget, SCOPE
#                    tightens the localisation by climbing this ladder (fewer layers).
#   MASSIVE_K        per layer, the number of top-magnitude ("massive activation")
#                    dimensions whose subspace the demographic direction is removed
#                    ORTHOGONAL to, so the load-bearing structure is protected.
#   PROTECT          1 = protect massive activations (SCOPE); 0 = the no-protect ablation.
#   VERIFY_DROP      success target for the post-edit Patchscope: the localised-layer
#                    decodability must fall by at least this fraction.
# ---------------------------------------------------------------------------
LOCALIZE_PAIRS = _env_int("SCOPE_LOCALIZE_PAIRS", 128)
DECODE_PCTILE = _env_float("SCOPE_DECODE_PCTILE", 75.0)
PCTILE_LADDER = [float(x) for x in os.environ.get("SCOPE_PCTILE_LADDER", "75,85,92").split(",")]
MASSIVE_K = _env_int("SCOPE_MASSIVE_K", 8)
PROTECT = os.environ.get("SCOPE_PROTECT", "1").strip() == "1"
VERIFY_DROP = _env_float("SCOPE_VERIFY_DROP", 0.30)

# The Few-Shot Token-Identity Patchscope target prompt (Patchscopes section 4.1): a short
# repetition prompt whose final token is patched with the source representation; the
# next-token distribution then reveals what that representation decodes to.
PATCHSCOPE_TARGET = os.environ.get(
    "SCOPE_TARGET_PROMPT", "cat cat\n135 135\nhello hello\nstop stop\nx")

# Train/test split fraction for the held-out evaluation (matches run_tacl_extra).
FRAC_TRAIN = _env_float("TACL_FRAC_TRAIN", 0.55)
TEST_PAIR_CAP = _env_int("TACL_TEST_PAIR_CAP", 600)
BEHAV_SEED_CAP = _env_int("TACL_BEHAV_SEED_CAP", 160)
