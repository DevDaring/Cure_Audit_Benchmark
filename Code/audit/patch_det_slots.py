"""
File: patch_det_slots.py
Purpose: After a full build where deterministic slots (a/b/c) had incorrect counts
         (e.g., < 5 slot-c variants due to small equivalence sets), patch only those
         slots without re-calling any DeepSeek API.

Approach:
  1. Load saved pentad_dataset.parquet (which contains valid API rows d + e)
  2. Re-generate all deterministic rows (a/b/c) from updated equivalence_sets.yaml
  3. Replace old det rows in parquet with new ones
  4. Save and validate

Usage:
    python patch_det_slots.py
    python patch_det_slots.py --dry-run   # show stats only, do not save

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RANDOM_SEED, SEEDS_DIR
from logger_setup import setup_logging

logger = logging.getLogger(__name__)

_PENTAD_PATH = SEEDS_DIR / "pentad_dataset.parquet"
_SEEDS_PATH = SEEDS_DIR / "seeds.parquet"
_AUDIT_SOURCES = frozenset({"bbq", "crows_pairs", "stereoset"})


def main() -> bool:
    parser = argparse.ArgumentParser(description="Patch deterministic slots in pentad_dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print stats without saving")
    args = parser.parse_args()

    setup_logging()

    if not _PENTAD_PATH.exists():
        logger.error("pentad_dataset.parquet not found at %s — run run_dataset.py first", _PENTAD_PATH)
        return False

    if not _SEEDS_PATH.exists():
        logger.error("seeds.parquet not found at %s", _SEEDS_PATH)
        return False

    # ---------------------------------------------------------------- load
    logger.info("Loading existing pentad dataset ...")
    old_df = pd.read_parquet(_PENTAD_PATH)
    logger.info("  Existing rows: %d", len(old_df))

    api_df = old_df[old_df["slot"].isin(["d", "e"])].copy()
    api_df = api_df[
        api_df["seed_source"].astype(str).str.lower().isin(_AUDIT_SOURCES)
    ]
    det_old = old_df[old_df["slot"].isin(["a", "b", "c"])]
    logger.info("  API rows (d/e): %d", len(api_df))
    logger.info("  Det rows (a/b/c) before patch: %d", len(det_old))

    logger.info("Loading main seeds ...")
    from Dataset.sample_seeds import sample_seeds
    main_seeds, _ = sample_seeds()
    main_seeds = main_seeds[
        main_seeds["seed_source"].astype(str).str.lower().isin(_AUDIT_SOURCES)
    ].reset_index(drop=True)
    logger.info("  %d audit seeds loaded (WinoBias excluded)", len(main_seeds))

    # ---------------------------------------------------------------- regen
    logger.info("Re-generating deterministic slots (a/b/c) with updated equivalence sets ...")
    from Dataset.pentad_generator import generate_pentad_deterministic
    rng = np.random.default_rng(seed=RANDOM_SEED)
    new_det_rows = generate_pentad_deterministic(main_seeds, rng)
    new_det_df = pd.DataFrame(new_det_rows)
    logger.info("  New det rows: %d", len(new_det_df))
    ok_seed_ids = set(new_det_df["seed_id"].unique())
    excluded = set(main_seeds["seed_id"]) - ok_seed_ids
    manifest = {
        "n_excluded": len(excluded),
        "n_included": len(ok_seed_ids),
        "excluded_seed_ids": sorted(excluded),
        "reason": "pentad deterministic generation failed (no valid swap token or gold)",
    }
    manifest_path = SEEDS_DIR / "excluded_seeds.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Seed manifest: %d included, %d excluded -> %s", len(ok_seed_ids), len(excluded), manifest_path)
    if excluded:
        logger.warning("Excluded seed sample: %s", sorted(excluded)[:10])

    # Verify new slot-c counts
    new_c_counts = new_det_df[new_det_df["slot"] == "c"].groupby("seed_id").size()
    bad_after = (new_c_counts != 5).sum()
    if bad_after > 0:
        logger.warning("  Still %d seeds with slot-c != 5 after regeneration!", bad_after)
    else:
        logger.info("  All seeds have exactly 5 slot-c variants.")

    # Det-only rows until API regen completes; drop partial d/e to avoid mixed state.
    combined = new_det_df.copy()
    combined = combined.sort_values(["seed_id", "slot", "subvariant"]).reset_index(drop=True)
    logger.info("  Det rows saved: %d (expected %d)", len(combined), len(ok_seed_ids) * 7)
    logger.info("  Next step: run regenerate_api_slots.py for d/e DeepSeek prompts.")

    if args.dry_run:
        logger.info("[dry-run] Not saving. Rows that would be written: %d", len(combined))
        return True

    # ---------------------------------------------------------------- save
    combined.to_parquet(_PENTAD_PATH, index=False)
    logger.info("Patched pentad_dataset.parquet saved.")

    # ---------------------------------------------------------------- validate
    logger.info("Validating patched dataset ...")
    from Dataset.validate_pentad import run_all_validations
    try:
        run_all_validations(combined, require_api_slots=False)
    except Exception as exc:
        logger.error("Validation FAILED after patch: %s", exc)
        return False

    from Dataset.validate_pentad import write_pentad_manifest

    write_pentad_manifest(combined)

    logger.info("Patch complete. Dataset valid.")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
