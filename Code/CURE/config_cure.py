"""
config_cure.py -- central configuration for CURE (Causal aUdit and REpair).

Loads CURE/.env into the process environment, then reuses the audit model list
and dataset paths from Code/audit. Every key is read from the environment; no
secret literal lives in this file or any other tracked file.

CURE extends the causal discriminative-validity audit (Code/audit) from diagnosis
to repair. It runs on the same four open models, the same three datasets, and the
same causal stack, so the comparison with prior debiasing methods stays fair.

Implements / builds on:
  - The causal discriminative-validity audit (Code/audit).
  - LEACE concept erasure (Belrose et al. 2023, arXiv:2306.03819).
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # Code/CURE
AUDIT = HERE.parent / "audit"                          # Code/audit (the audit base)
REPO = HERE.parent.parent                             # repo root
ENV_PATH = HERE / ".env"

# Make the audit package importable (config, GPU_CPU.*, dataset, results).
sys.path.insert(0, str(AUDIT))


def load_env(path: Path = ENV_PATH) -> int:
    """Load KEY=VALUE pairs from a .env into os.environ. No external dependency.

    Keys already present are not overwritten. Returns the count loaded. Whitespace
    around the key name is tolerated (a few keys in this .env have trailing spaces).
    """
    n = 0
    if not path.exists():
        return n
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


# Load CURE secrets BEFORE importing the audit config, so its import-time key
# validation passes from the environment we just populated.
load_env()

# Reuse the audited model list and the research system prompt from Code/audit.
from config import OSM_MODELS, RESEARCH_SYSTEM_PROMPT  # noqa: E402

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
GITHUB_TOKEN = os.environ.get("Github_Classic_Token", "") or os.environ.get("GITHUB_CLASSIC_TOKEN", "")
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "20260101") or "20260101")

# Dataset and audited-result paths (shipped in the audit folder).
PENTAD_PATH = AUDIT / "Dataset" / "seeds" / "pentad_dataset.parquet"
CDVA_PATH = AUDIT / "results" / "cdva_results.parquet"
BEHAVIORAL_PATH = AUDIT / "results" / "behavioral_results.parquet"

# CURE output locations.
RESULTS = HERE / "results"
LOGS = HERE / "logs"
RESULTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Judge / answer-extraction providers.
#
# One active tier is chosen by CURE_JUDGE_PROVIDER (default gemini). Keys are
# round-robined WITHIN the active tier. There is no automatic fallback between
# tiers: if the active tier fails, the item is recorded as a judge failure. All
# keys are read from the environment populated above; none are hardcoded.
# ---------------------------------------------------------------------------

def _keys(*names: str) -> list[str]:
    """Return the non-empty values for the given env names, in order."""
    out = []
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            out.append(v)
    return out


JUDGE_PROVIDERS = {
    # Primary: Google Gemini (the four Gemini / GCP keys in the .env).
    "gemini": {
        "model": os.environ.get("CURE_GEMINI_MODEL", "gemini-2.5-flash"),
        "keys": _keys("GEMINI_API_KEY_1", "GEMINI_API_KEY_2",
                      "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"),
        "kind": "gemini",
        "base_url": None,
    },
    # Secondary: DeepSeek (OpenAI-compatible).
    "deepseek": {
        "model": os.environ.get("DEEPSEEK_JUDGE_MODEL_NAME", "deepseek-chat"),
        "keys": _keys("DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2"),
        "kind": "openai",
        "base_url": os.environ.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com/v1"),
    },
    # Tertiary: Mistral small (OpenAI-compatible).
    "mistral": {
        "model": os.environ.get("CURE_MISTRAL_MODEL", "mistral-small-latest"),
        "keys": _keys("MISTRAL_API_KEY1", "MISTRAL_API_KEY2"),
        "kind": "openai",
        "base_url": os.environ.get("MISTRAL_API_BASE_URL", "https://api.mistral.ai/v1"),
    },
    # Alternative gateway: OpenRouter (OpenAI-compatible).
    "openrouter": {
        "model": os.environ.get("CURE_OPENROUTER_MODEL", "google/gemini-2.5-flash"),
        "keys": _keys("OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2"),
        "kind": "openai",
        "base_url": os.environ.get("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1"),
    },
}

ACTIVE_JUDGE = os.environ.get("CURE_JUDGE_PROVIDER", "gemini").strip().lower()

# The four open models receive the causal audit and the cure.
OSM_NAMES = [m["name"] for m in OSM_MODELS]

# Sweep settings.
DRY_LIMIT = int(os.environ.get("CURE_DRY_LIMIT", "2"))
ERASE_RANKS = [int(x) for x in os.environ.get("CURE_ERASE_RANKS", "1,2,4,8").split(",")]

# ---------------------------------------------------------------------------
# Safe expedite knobs (each preserves correctness, harmony, and statistical
# soundness; see README and the head-to-head design).
#
#   HEADLINE_RANK   the single operating rank at which the headline numbers
#                   (residual removed, the head-to-head) are reported on ALL pairs.
#   SUBSPACE_PAIRS  number of counterfactual pairs used to ESTIMATE the bias
#                   subspace once (a low-rank direction is robust from a few hundred
#                   pairs; the full set is not needed to estimate a direction).
#   SWEEP_SUBSET    stratified, fixed-seed subset on which the multi-rank sweep runs,
#                   feeding the prognosis (E6) and the fairness-utility curve (E4).
#                   The headline numbers still use ALL pairs, so no reported statistic
#                   loses power.
#   E4_MAX_TOKENS   short generations suffice for option-answer accuracy.
#   E4_LIMIT        prompts used for the utility measurement (None = all slot-a).
# ---------------------------------------------------------------------------
HEADLINE_RANK = int(os.environ.get("CURE_HEADLINE_RANK", "4"))
SUBSPACE_PAIRS = int(os.environ.get("CURE_SUBSPACE_PAIRS", "600"))
SWEEP_SUBSET = int(os.environ.get("CURE_SWEEP_SUBSET", "1000"))
E4_MAX_TOKENS = int(os.environ.get("CURE_E4_MAX_TOKENS", "64"))
_e4 = os.environ.get("CURE_E4_LIMIT", "").strip()
E4_LIMIT = int(_e4) if _e4 else None

# Causal threshold tau: reuse the audit default (75th percentile of |C|).
TAU = float(os.environ.get("CURE_TAU", "0.7644"))

# Minimum natural swap effect below which a recovery ratio is undefined.
MIN_EFFECT = float(os.environ.get("CURE_MIN_EFFECT", "0.5"))


def model_cfg(name: str) -> dict:
    for m in OSM_MODELS:
        if m["name"] == name:
            return m
    raise KeyError(f"unknown model {name!r}; known: {OSM_NAMES}")
