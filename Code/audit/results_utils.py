"""
Helpers for deduplicating and loading MIRAGE result parquets.
"""

from __future__ import annotations

import pandas as pd

_BEHAVIORAL_KEYS = ["prompt_id", "model_name", "sample_index"]
_CDVA_KEYS = ["seed_id", "model_name", "pair_A_subvariant", "pair_B_subvariant"]


def reparse_failed_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Re-evaluate parse_model_response on failed rows (in-place safe copy)."""
    if df.empty or "success_flag" not in df.columns:
        return df
    from parse_utils import parse_model_response

    out = df.copy()
    failed = ~out["success_flag"].fillna(False)
    for idx, row in out[failed].iterrows():
        ok, ans, conf, rat, method, _reason = parse_model_response(str(row.get("raw_response", "")))
        if ok:
            out.at[idx, "success_flag"] = True
            out.at[idx, "parsed_answer"] = ans
            out.at[idx, "parsed_confidence"] = conf
            out.at[idx, "parsed_rationale"] = rat
            out.at[idx, "parse_method"] = method
            out.at[idx, "failure_reason"] = ""
    return out


def dedup_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicate behavioral rows.

    Prefers successful rows; when multiple successes exist, keeps the latest
    timestamp.  This prevents restart loops from leaving failed rows that
    would win under naive keep='last'.
    """
    if df.empty:
        return df
    out = df.copy()
    if "timestamp_utc" not in out.columns:
        out["timestamp_utc"] = ""
    if "success_flag" not in out.columns:
        out["success_flag"] = False
    out["_rank"] = out["success_flag"].astype(int)
    out = out.sort_values(["_rank", "timestamp_utc"])
    out = (
        out.drop_duplicates(subset=_BEHAVIORAL_KEYS, keep="last")
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )
    return out


def dedup_cdva(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate CDVA pair rows, preferring successful runs."""
    if df.empty:
        return df
    out = df.copy()
    if "timestamp_utc" not in out.columns:
        out["timestamp_utc"] = ""
    if "success_flag" not in out.columns:
        out["success_flag"] = False
    out["_rank"] = out["success_flag"].astype(int)
    out = out.sort_values(["_rank", "timestamp_utc"])
    out = (
        out.drop_duplicates(subset=_CDVA_KEYS, keep="last")
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )
    return out


def load_behavioral_deduped(path) -> pd.DataFrame:
    """Load behavioral_results.parquet with deduplication applied."""
    df = pd.read_parquet(path)
    return dedup_behavioral(df)
