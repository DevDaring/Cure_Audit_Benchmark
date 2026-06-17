"""
Normalize seed_category labels across BBQ, CrowS-Pairs, and StereoSet.
"""

from __future__ import annotations

import pandas as pd

# Map raw source-specific labels to canonical buckets for aggregation.
_CATEGORY_MAP: dict[str, str] = {
    "age": "age",
    "Age": "age",
    "disability": "disability",
    "Disability_status": "disability",
    "gender": "gender",
    "Gender": "gender",
    "Gender_identity": "gender",
    "nationality": "nationality",
    "Nationality": "nationality",
    "physical-appearance": "physical_appearance",
    "physical_appearance": "physical_appearance",
    "Physical_appearance": "physical_appearance",
    "profession": "profession",
    "race": "race",
    "race-color": "race",
    "Race_ethnicity": "race",
    "religion": "religion",
    "Religion": "religion",
    "socioeconomic": "socioeconomic",
    "SES": "socioeconomic",
    "sexual-orientation": "sexual_orientation",
    "sexual_orientation": "sexual_orientation",
    "Sexual_orientation": "sexual_orientation",
}


def normalize_seed_category(raw: str) -> str:
    """Return a canonical category label for cross-benchmark aggregation."""
    key = str(raw).strip()
    if not key:
        return "unknown"
    return _CATEGORY_MAP.get(key, key.lower().replace("-", "_"))


def add_normalized_category(df: pd.DataFrame, col: str = "seed_category") -> pd.DataFrame:
    """Add seed_category_norm column without mutating the input frame."""
    out = df.copy()
    if col in out.columns:
        out["seed_category_norm"] = out[col].map(normalize_seed_category)
    else:
        out["seed_category_norm"] = "unknown"
    return out
