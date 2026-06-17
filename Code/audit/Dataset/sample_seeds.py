"""
File: Dataset/sample_seeds.py
Purpose: Stratified seed selection from three audited source benchmarks plus
         a held-out WinoBias set for predictive validity.

ACTUAL SEED COUNTS (what the code can obtain given available data):
  bbq        : up to 270 (30 per BBQ category × 9)
  crows_pairs: up to 200 (~22 per bias type)
  stereoset  : up to 200 (50 per domain × 4)
  winobias   : up to 200 (50 per wino_type × stereo_direction quadrant) -- HELD OUT
  dev set    : 50 seeds drawn from the non-WinoBias pool (disjoint from main)

WinoBias is NOT included in the main audited set (main_seeds).  It is saved
separately as winobias_heldout.parquet for predictive-validity evaluation only.
This fixes review finding B2 ("WinoBias is NOT held out").

seed_category is normalised to a single controlled 10-value vocabulary to fix
review finding B6 (21 inconsistent category names across sources).

Implements / builds on / cites:
  - Parrish et al. (2022). BBQ. Findings of ACL 2022.
  - Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
  - Nadeem et al. (2021). StereoSet. ACL-IJCNLP 2021.
  - Zhao et al. (2018). WinoBias. NAACL 2018.

RNG: numpy.random.default_rng(seed=20260101) -- fixed for reproducibility.
Part of the audit codebase (diagnosis half of CURE).
"""

import hashlib
import json
import logging
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RANDOM_SEED, SEEDS_DIR, ensure_dirs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical 10-value seed_category vocabulary (fixes B6)
# ---------------------------------------------------------------------------
SEED_CATEGORY_CANONICAL: dict[str, str] = {
    # BBQ
    "Age": "age",
    "Disability_status": "disability",
    "Gender_identity": "gender",
    "Nationality": "nationality",
    "Physical_appearance": "physical_appearance",
    "Race_ethnicity": "race",
    "Religion": "religion",
    "SES": "socioeconomic",
    "Sexual_orientation": "sexual_orientation",
    # CrowS-Pairs
    "race-color": "race",
    "socioeconomic": "socioeconomic",
    "gender": "gender",
    "disability": "disability",
    "nationality": "nationality",
    "sexual-orientation": "sexual_orientation",
    "physical-appearance": "physical_appearance",
    "religion": "religion",
    "age": "age",
    # StereoSet
    "profession": "profession",
    "race": "race",
    # WinoBias
    "Gender": "gender",
    # Already-canonical pass-throughs
    "sexual_orientation": "sexual_orientation",
    "physical_appearance": "physical_appearance",
}


def _normalise_category(raw: str) -> str:
    """Return the canonical category for a raw source-specific category string."""
    return SEED_CATEGORY_CANONICAL.get(raw, raw.lower().replace("-", "_").replace(" ", "_"))


# ---------------------------------------------------------------------------
# Target seed counts
# ---------------------------------------------------------------------------
SEED_COUNTS = {
    "bbq": 270,          # 30 per category × 9
    "crows_pairs": 200,
    "stereoset": 200,
    "winobias": 200,     # held out for predictive validity
}
DEV_SEED_COUNT = 50      # disjoint dev set for tau calibration (from non-WinoBias pool)

_SEEDS_PATH = SEEDS_DIR / "seeds.parquet"
_WINO_HELDOUT_PATH = SEEDS_DIR / "winobias_heldout.parquet"
_MANIFEST_PATH = SEEDS_DIR / "seeds_manifest.json"


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Per-source samplers
# ---------------------------------------------------------------------------

def _sample_bbq(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sample up to 30 seeds per BBQ category (target 270 total)."""
    per_cat = 30
    frames: list[pd.DataFrame] = []
    for cat in sorted(df["bbq_category"].unique()):
        cat_df = df[df["bbq_category"] == cat]
        n = min(per_cat, len(cat_df))
        if n == 0:
            logger.warning("BBQ category %s has no rows; skipping.", cat)
            continue
        if n < per_cat:
            logger.warning(
                "BBQ category %s has only %d rows (target %d); using all.",
                cat, n, per_cat,
            )
        sample = cat_df.sample(n=n, random_state=int(rng.integers(0, 2**31)), replace=False).copy()
        sample["seed_source"] = "bbq"
        sample["seed_category"] = _normalise_category(cat)
        frames.append(sample)

    if not frames:
        raise RuntimeError("BBQ sampling produced no data.")
    return pd.concat(frames, ignore_index=True)


def _sample_crows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sample ~22 seeds per bias type (target 200 total)."""
    per_type = 22
    bias_col = "bias_type" if "bias_type" in df.columns else df.columns[0]
    frames: list[pd.DataFrame] = []
    for bt in sorted(df[bias_col].unique()):
        bt_df = df[df[bias_col] == bt]
        n = min(per_type, len(bt_df))
        if n == 0:
            continue
        sample = bt_df.sample(n=n, random_state=int(rng.integers(0, 2**31)), replace=False).copy()
        sample["seed_source"] = "crows_pairs"
        sample["seed_category"] = _normalise_category(bt)
        frames.append(sample)

    combined = pd.concat(frames, ignore_index=True)
    if len(combined) > 200:
        combined = combined.sample(n=200, random_state=int(rng.integers(0, 2**31)), replace=False)
    return combined.reset_index(drop=True)


def _sample_stereoset(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sample balanced across 4 domains (target 200 total)."""
    per_domain = 50
    domain_col = "bias_type" if "bias_type" in df.columns else "domain"
    if domain_col not in df.columns:
        domain_col = df.columns[0]
    frames: list[pd.DataFrame] = []
    for domain in sorted(df[domain_col].unique()):
        dom_df = df[df[domain_col] == domain]
        n = min(per_domain, len(dom_df))
        if n == 0:
            continue
        sample = dom_df.sample(n=n, random_state=int(rng.integers(0, 2**31)), replace=False).copy()
        sample["seed_source"] = "stereoset"
        sample["seed_category"] = _normalise_category(domain)
        frames.append(sample)
    return pd.concat(frames, ignore_index=True)


def _sample_winobias(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sample up to 200 rows balanced across type1/type2 × pro/anti quadrants."""
    per_group = 50
    frames: list[pd.DataFrame] = []
    for wtype in ["type1", "type2"]:
        for direction in ["pro", "anti"]:
            sub = df[(df["wino_type"] == wtype) & (df["stereo_direction"] == direction)]
            n = min(per_group, len(sub))
            if n == 0:
                continue
            sample = sub.sample(n=n, random_state=int(rng.integers(0, 2**31)), replace=False).copy()
            sample["seed_source"] = "winobias"
            sample["seed_category"] = "gender"          # normalised (was "Gender")
            sample["seed_subcategory"] = f"{wtype}_{direction}"
            frames.append(sample)

    if not frames:
        raise RuntimeError("WinoBias sampling produced no data.")
    return pd.concat(frames, ignore_index=True)


def _assign_seed_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df["seed_id"] = [f"{prefix}_{uuid.uuid4().hex[:8]}" for _ in range(len(df))]
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_seeds(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the main audited seed set and dev set.

    Main set: BBQ + CrowS-Pairs + StereoSet (WinoBias is held out separately).
    Dev set: 50 seeds drawn from the main set, disjoint from evaluation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (main_seeds, dev_seeds)
    """
    ensure_dirs()

    if _SEEDS_PATH.exists() and not force:
        logger.info("Seeds cache hit: %s", _SEEDS_PATH)
        main_seeds = pd.read_parquet(_SEEDS_PATH)
        # Re-normalize categories for legacy caches built before B6 fix.
        main_seeds["seed_category"] = main_seeds["seed_category"].apply(_normalise_category)
        # Strip WinoBias if an older cache incorrectly included it in main seeds.
        n_wino = (main_seeds["seed_source"] == "winobias").sum()
        if n_wino > 0:
            logger.warning(
                "Removing %d WinoBias rows from cached main_seeds (held-out only).", n_wino
            )
            main_seeds = main_seeds[main_seeds["seed_source"] != "winobias"].reset_index(drop=True)
        dev_path = SEEDS_DIR / "dev_seeds.parquet"
        dev_seeds = pd.read_parquet(dev_path) if dev_path.exists() else pd.DataFrame()
        logger.info(
            "Loaded %d main seeds and %d dev seeds from cache.",
            len(main_seeds), len(dev_seeds),
        )
        return main_seeds, dev_seeds

    logger.info("Sampling seeds with RNG seed %d.", RANDOM_SEED)
    rng = np.random.default_rng(seed=RANDOM_SEED)

    from Dataset.download_bbq import download_bbq
    from Dataset.download_crows_pairs import download_crows_pairs
    from Dataset.download_stereoset import download_stereoset
    from Dataset.download_winobias import download_winobias

    bbq_df = download_bbq()
    crows_df = download_crows_pairs()
    stereo_df = download_stereoset()
    wino_df = download_winobias()

    bbq_seeds = _assign_seed_ids(_sample_bbq(bbq_df, rng), "bbq")
    crows_seeds = _assign_seed_ids(_sample_crows(crows_df, rng), "crows")
    stereo_seeds = _assign_seed_ids(_sample_stereoset(stereo_df, rng), "stereo")

    # WinoBias is HELD OUT -- not part of main_seeds (fixes B2).
    wino_seeds = _assign_seed_ids(_sample_winobias(wino_df, rng), "wino")
    wino_seeds.to_parquet(_WINO_HELDOUT_PATH, index=False)
    logger.info(
        "WinoBias held-out set: %d seeds saved to %s (NOT audited in main run).",
        len(wino_seeds), _WINO_HELDOUT_PATH,
    )

    # Main audited set: BBQ + CrowS + StereoSet only (WinoBias held out separately).
    main_seeds = pd.concat([bbq_seeds, crows_seeds, stereo_seeds], ignore_index=True)

    if (main_seeds["seed_source"] == "winobias").any():
        raise RuntimeError("WinoBias rows found in main_seeds — must be held out only.")

    # Log actual vs target counts; fail if any source is critically short.
    actual = {src: int((main_seeds["seed_source"] == src).sum())
              for src in ["bbq", "crows_pairs", "stereoset"]}
    _MIN_ACCEPTABLE = {"bbq": 200, "crows_pairs": 150, "stereoset": 150}
    for src, n in actual.items():
        target = SEED_COUNTS[src]
        if n < _MIN_ACCEPTABLE[src]:
            raise RuntimeError(
                f"Seed count for {src} is {n} (minimum {_MIN_ACCEPTABLE[src]}, "
                f"target {target}). Re-download source data before building pentad."
            )
        if n < target:
            logger.warning(
                "Seed count shortfall for %s: expected %d, got %d. "
                "Report actual count (%d) in all paper claims.",
                src, target, n, n,
            )

    # Dev set: 50 disjoint seeds from the main set
    dev_seeds = main_seeds.sample(
        n=min(DEV_SEED_COUNT, len(main_seeds)),
        random_state=int(rng.integers(0, 2**31)),
        replace=False,
    )
    main_seeds = main_seeds[~main_seeds["seed_id"].isin(dev_seeds["seed_id"])].reset_index(drop=True)
    dev_seeds = dev_seeds.reset_index(drop=True)

    # Integrity checks
    assert main_seeds["seed_id"].is_unique, "Duplicate seed_id detected."
    assert len(main_seeds) > 0, "No seeds produced."
    overlap = set(main_seeds["seed_id"]) & set(dev_seeds["seed_id"])
    assert not overlap, f"main/dev overlap: {overlap}"

    # Save
    main_seeds.to_parquet(_SEEDS_PATH, index=False)
    dev_seeds.to_parquet(SEEDS_DIR / "dev_seeds.parquet", index=False)

    # SHA-256 manifest
    sha = _sha256(_SEEDS_PATH)
    manifest = {
        "seeds_sha256": sha,
        "n_main": len(main_seeds),
        "n_dev": len(dev_seeds),
        "n_wino_heldout": len(wino_seeds),
        "sources_in_main": actual,
    }
    with open(_MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info(
        "Sampled %d main seeds + %d dev seeds + %d WinoBias held-out. SHA-256: %s",
        len(main_seeds), len(dev_seeds), len(wino_seeds), sha,
    )
    return main_seeds, dev_seeds


def verify_seeds_integrity() -> None:
    """Re-check SHA-256 of seeds file against stored manifest. Fail loudly if changed."""
    if not _SEEDS_PATH.exists():
        raise FileNotFoundError(f"Seeds file not found: {_SEEDS_PATH}")
    if not _MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {_MANIFEST_PATH}")

    with open(_MANIFEST_PATH) as fh:
        manifest = json.load(fh)

    current_sha = _sha256(_SEEDS_PATH)
    if current_sha != manifest["seeds_sha256"]:
        raise RuntimeError(
            f"Seeds file integrity check FAILED. "
            f"Expected SHA-256: {manifest['seeds_sha256']}, Got: {current_sha}"
        )
    logger.info("Seeds integrity check passed. SHA-256: %s", current_sha)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main_s, dev_s = sample_seeds()
    logger.info(
        "Main seeds: %d (bbq=%d crows=%d stereo=%d), Dev seeds: %d",
        len(main_s),
        (main_s["seed_source"] == "bbq").sum(),
        (main_s["seed_source"] == "crows_pairs").sum(),
        (main_s["seed_source"] == "stereoset").sum(),
        len(dev_s),
    )
