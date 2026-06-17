"""
File: Dataset/download_crows_pairs.py
Purpose: Download and cache the CrowS-Pairs dataset from HuggingFace.

Implements / builds on / cites:
  - Nangia et al. (2020). "CrowS-Pairs: A Challenge Dataset for Measuring
    Social Biases in Masked Language Models." EMNLP 2020.
    https://aclanthology.org/2020.emnlp-main.154

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_CACHE, HUGGINGFACE_TOKEN, ensure_dirs

logger = logging.getLogger(__name__)

CROWS_HF_DATASET = "nyu-mll/crows_pairs"
CROWS_BIAS_TYPES = [
    "race-color",
    "socioeconomic",
    "gender",
    "disability",
    "nationality",
    "sexual-orientation",
    "physical-appearance",
    "religion",
    "age",
]

_CACHE_PATH = DATASET_CACHE / "crows_pairs_raw.parquet"


def download_crows_pairs(force: bool = False) -> pd.DataFrame:
    """
    Download CrowS-Pairs from HuggingFace and persist to parquet cache.

    Parameters
    ----------
    force : bool
        If True, re-download even if cached copy exists.

    Returns
    -------
    pd.DataFrame
    """
    ensure_dirs()

    if _CACHE_PATH.exists() and not force:
        logger.info("CrowS-Pairs cache hit: %s", _CACHE_PATH)
        df = pd.read_parquet(_CACHE_PATH)
        logger.info("CrowS-Pairs loaded %d rows from cache.", len(df))
        return df

    # Strategy: nyu-mll/crows_pairs on HuggingFace only has a loading script
    # (crows_pairs.py) with no parquet conversion. The canonical raw data is
    # the anonymised CSV from the original GitHub repository (10 kB).
    # We download it directly with retry and no token required (public repo).
    import io
    import time
    import requests  # type: ignore

    _URL = (
        "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/"
        "data/crows_pairs_anonymized.csv"
    )
    _TIMEOUT = 60
    _RETRIES = 3

    logger.info("Downloading CrowS-Pairs CSV from GitHub (nyu-mll/crows-pairs).")
    last_exc: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            resp = requests.get(_URL, timeout=_TIMEOUT)
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content))
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("Attempt %d/%d failed: %s", attempt, _RETRIES, exc)
            if attempt < _RETRIES:
                time.sleep(4 * attempt)
    else:
        raise RuntimeError(
            f"CrowS-Pairs download failed after {_RETRIES} attempts: {last_exc}"
        ) from last_exc

    df.to_parquet(_CACHE_PATH, index=False)
    logger.info("CrowS-Pairs: %d rows cached to %s.", len(df), _CACHE_PATH)
    return df


def validate_crows_pairs(df: pd.DataFrame) -> None:
    """Assert required columns exist and report row counts per bias type."""
    required = {"sent_more", "sent_less", "stereo_antistereo", "bias_type"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"CrowS-Pairs missing columns: {missing_cols}")

    for bt in CROWS_BIAS_TYPES:
        n = (df["bias_type"] == bt).sum() if "bias_type" in df.columns else 0
        logger.info("CrowS-Pairs type %-25s : %d rows", bt, n)

    logger.info("CrowS-Pairs validation passed. Total rows: %d", len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = download_crows_pairs()
    validate_crows_pairs(data)
