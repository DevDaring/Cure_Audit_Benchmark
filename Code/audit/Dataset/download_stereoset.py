"""
File: Dataset/download_stereoset.py
Purpose: Download and cache the StereoSet dataset from HuggingFace.

Implements / builds on / cites:
  - Nadeem et al. (2021). "StereoSet: Measuring stereotypical bias in
    pretrained language models." ACL-IJCNLP 2021.
    https://aclanthology.org/2021.acl-long.416

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_CACHE, HUGGINGFACE_TOKEN, ensure_dirs

logger = logging.getLogger(__name__)

STEREOSET_HF_DATASET = "McGill-NLP/stereoset"
STEREOSET_DOMAINS = ["gender", "profession", "race", "religion"]

_CACHE_PATH = DATASET_CACHE / "stereoset_raw.parquet"


def download_stereoset(force: bool = False) -> pd.DataFrame:
    """
    Download StereoSet (intra-sentence split) from HuggingFace and cache.

    Parameters
    ----------
    force : bool
        Re-download if True.

    Returns
    -------
    pd.DataFrame
        Intra-sentence subset with normalised columns.
    """
    ensure_dirs()

    if _CACHE_PATH.exists() and not force:
        logger.info("StereoSet cache hit: %s", _CACHE_PATH)
        df = pd.read_parquet(_CACHE_PATH)
        logger.info("StereoSet loaded %d rows from cache.", len(df))
        return df

    logger.info("Downloading StereoSet from HuggingFace (%s).", STEREOSET_HF_DATASET)
    from datasets import load_dataset  # type: ignore

    # Load intra-sentence split
    ds = load_dataset(
        STEREOSET_HF_DATASET,
        "intrasentence",
        split="validation",
        token=HUGGINGFACE_TOKEN,
        cache_dir=str(DATASET_CACHE),
    )
    df = ds.to_pandas()

    # Keep only the 4 target domains
    if "domain" in df.columns:
        df = df[df["domain"].isin(STEREOSET_DOMAINS)].reset_index(drop=True)

    df.to_parquet(_CACHE_PATH, index=False)
    logger.info("StereoSet: %d rows (intra-sentence) cached to %s.", len(df), _CACHE_PATH)
    return df


def validate_stereoset(df: pd.DataFrame) -> None:
    """Validate required columns and per-domain counts."""
    required = {"id", "target", "bias_type", "sentences"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"StereoSet missing columns: {missing_cols}")

    for domain in STEREOSET_DOMAINS:
        n = (df["bias_type"] == domain).sum() if "bias_type" in df.columns else 0
        logger.info("StereoSet domain %-12s : %d rows", domain, n)

    logger.info("StereoSet validation passed. Total rows: %d", len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = download_stereoset()
    validate_stereoset(data)
