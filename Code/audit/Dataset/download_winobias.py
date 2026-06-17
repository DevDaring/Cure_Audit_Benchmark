"""
File: Dataset/download_winobias.py
Purpose: Download and cache WinoBias from the canonical GitHub source.
         WinoBias is HELD OUT for predictive validity -- never used for
         calibration, training, or tau tuning.

Implements / builds on / cites:
  - Zhao et al. (2018). "Gender Bias in Coreference Resolution: Evaluation
    and Debiasing Methods." NAACL 2018.
    https://aclanthology.org/N18-2003
  - Stanovsky et al. (2019). "Evaluating Gender Bias in Machine Translation."
    ACL 2019. https://aclanthology.org/P19-1164
  - Webster et al. (2020). "Measuring and Reducing Gendered Correlations in
    Pre-Trained Models." arXiv:2010.06032

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_CACHE, ensure_dirs

logger = logging.getLogger(__name__)

_WINOBIAS_BASE = (
    "https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/data/"
)
_FILES = {
    "type1_pro": "pro_stereotyped_type1.txt.test",
    "type1_anti": "anti_stereotyped_type1.txt.test",
    "type2_pro": "pro_stereotyped_type2.txt.test",
    "type2_anti": "anti_stereotyped_type2.txt.test",
}

_CACHE_PATH = DATASET_CACHE / "winobias_raw.parquet"
_TIMEOUT_SECONDS = 30


def _download_file(filename: str) -> list[str]:
    url = _WINOBIAS_BASE + filename
    logger.info("  Fetching %s", url)
    resp = requests.get(url, timeout=_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.text.strip().splitlines()


def download_winobias(force: bool = False) -> pd.DataFrame:
    """
    Download WinoBias Type-1 and Type-2 (pro/anti) from GitHub and cache.

    NOTE: This dataset is held out for predictive validity.
    It MUST NOT be used for any calibration step.

    Returns
    -------
    pd.DataFrame
    """
    ensure_dirs()

    if _CACHE_PATH.exists() and not force:
        logger.info("WinoBias cache hit: %s", _CACHE_PATH)
        df = pd.read_parquet(_CACHE_PATH)
        logger.info("WinoBias loaded %d rows from cache.", len(df))
        return df

    logger.info("Downloading WinoBias from %s", _WINOBIAS_BASE)
    rows: list[dict] = []
    for split_name, filename in _FILES.items():
        wino_type = "type1" if split_name.startswith("type1") else "type2"
        stereotyped = "pro" if "pro" in split_name else "anti"
        try:
            lines = _download_file(filename)
            for line in lines:
                line = line.strip()
                if line:
                    rows.append(
                        {
                            "sentence": line,
                            "wino_type": wino_type,
                            "stereo_direction": stereotyped,
                            "split": split_name,
                        }
                    )
        except Exception as exc:
            logger.warning("  Failed to download %s: %s", filename, exc)

    if not rows:
        raise RuntimeError("WinoBias download produced no data. Check network.")

    df = pd.DataFrame(rows)
    df.to_parquet(_CACHE_PATH, index=False)
    logger.info("WinoBias: %d rows cached to %s.", len(df), _CACHE_PATH)
    return df


def validate_winobias(df: pd.DataFrame) -> None:
    """Validate required columns and per-type counts."""
    required = {"sentence", "wino_type", "stereo_direction"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"WinoBias missing columns: {missing_cols}")

    for wtype in ["type1", "type2"]:
        for direction in ["pro", "anti"]:
            n = ((df["wino_type"] == wtype) & (df["stereo_direction"] == direction)).sum()
            logger.info("WinoBias %-6s %-4s : %d rows", wtype, direction, n)

    logger.info("WinoBias validation passed. Total rows: %d", len(df))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = download_winobias()
    validate_winobias(data)
