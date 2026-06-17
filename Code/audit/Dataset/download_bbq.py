"""
File: Dataset/download_bbq.py
Purpose: Download and cache the BBQ bias benchmark dataset from HuggingFace.

Implements / builds on / cites:
  - Parrish et al. (2022). "BBQ: A Hand-Built Bias Benchmark for Question
    Answering." Findings of ACL 2022.
    https://aclanthology.org/2022.findings-acl.165

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_CACHE, HUGGINGFACE_TOKEN, ensure_dirs

logger = logging.getLogger(__name__)

BBQ_HF_DATASET = "heegyu/bbq"
BBQ_CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",
    "Sexual_orientation",
]

_CACHE_PATH = DATASET_CACHE / "bbq_raw.parquet"


def download_bbq(force: bool = False) -> pd.DataFrame:
    """
    Download BBQ from HuggingFace and persist to parquet cache.

    Parameters
    ----------
    force : bool
        If True, re-download even if cached copy exists.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame across all BBQ categories.
    """
    ensure_dirs()

    if _CACHE_PATH.exists() and not force:
        logger.info("BBQ cache hit: %s", _CACHE_PATH)
        df = pd.read_parquet(_CACHE_PATH)
        logger.info("BBQ loaded %d rows from cache.", len(df))
        return df

    # Strategy: use the HuggingFace datasets parquet API.
    # heegyu/bbq has been auto-converted to parquet by HuggingFace.
    # API endpoint: GET /api/datasets/heegyu/bbq/parquet/{Category}/test
    # Returns a list of parquet file URLs. Authenticate with HUGGINGFACE_TOKEN.
    import io
    import time
    import requests  # type: ignore

    _HF_PARQUET_API = "https://huggingface.co/api/datasets/heegyu/bbq/parquet/{cat}/test"
    _TIMEOUT = 60
    _RETRIES = 3
    _AUTH = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

    def _download_bytes(url: str, headers: dict) -> bytes:
        """GET url with retry. Returns raw bytes on success."""
        for attempt in range(1, _RETRIES + 1):
            try:
                resp = requests.get(url, headers=headers, timeout=_TIMEOUT)
                resp.raise_for_status()
                return resp.content
            except Exception as exc:
                logger.warning(
                    "    Attempt %d/%d failed (%s): %s", attempt, _RETRIES, url, exc
                )
                if attempt < _RETRIES:
                    time.sleep(3 * attempt)
        raise RuntimeError(f"All {_RETRIES} attempts failed for {url}")

    logger.info("Downloading BBQ via HuggingFace parquet API (heegyu/bbq).")
    frames: list[pd.DataFrame] = []
    for cat in BBQ_CATEGORIES:
        logger.info("  Loading BBQ category: %s", cat)
        try:
            api_url = _HF_PARQUET_API.format(cat=cat)
            meta_bytes = _download_bytes(api_url, _AUTH)
            import json
            parquet_urls: list[str] = json.loads(meta_bytes)
            if not parquet_urls:
                raise RuntimeError("API returned empty parquet URL list.")
            cat_frames: list[pd.DataFrame] = []
            for purl in parquet_urls:
                raw = _download_bytes(purl, _AUTH)
                cat_frames.append(pd.read_parquet(io.BytesIO(raw)))
            df_cat = pd.concat(cat_frames, ignore_index=True)
            df_cat["bbq_category"] = cat
            frames.append(df_cat)
            logger.info("    %d rows.", len(df_cat))
        except Exception as exc:
            logger.warning("    Failed to load category %s: %s", cat, exc)

    if not frames:
        raise RuntimeError("BBQ download produced no data. Check HF token and network.")

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(_CACHE_PATH, index=False)
    logger.info("BBQ: %d total rows cached to %s.", len(df), _CACHE_PATH)
    return df


def validate_bbq(df: pd.DataFrame) -> None:
    """Assert required columns exist and report row counts per category."""
    required = {"question", "ans0", "ans1", "ans2", "label", "bbq_category"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"BBQ missing columns: {missing_cols}")

    for cat in BBQ_CATEGORIES:
        n = (df["bbq_category"] == cat).sum()
        logger.info("BBQ category %-30s : %d rows", cat, n)

    logger.info("BBQ validation passed. Total rows: %d", len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = download_bbq()
    validate_bbq(data)
