"""
File: run_dataset.py
Purpose: Single entry-point for full MIRAGE dataset preparation.
         Generates all 668×12 = ~8,016 pentad probe variants.
         Resumes automatically from checkpoints -- safe to Ctrl-C and restart.

All credentials loaded from .env via config.py -- no hardcoded keys.

Usage:
    python run_dataset.py               # full build
    python run_dataset.py --det-only    # skip DeepSeek, slots a/b/c only
    python run_dataset.py --force       # regenerate even if cached

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167

Part of the audit codebase (diagnosis half of CURE).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SEEDS_DIR, validate_all_keys
from logger_setup import setup_logging

logger = logging.getLogger(__name__)


def _check_env() -> bool:
    missing = validate_all_keys()
    if missing:
        logger.error("Missing required env vars: %s", missing)
        return False
    logger.info("All env keys present.")
    return True


def main() -> bool:
    parser = argparse.ArgumentParser(description="MIRAGE dataset builder")
    parser.add_argument("--det-only", action="store_true",
                        help="Generate only deterministic slots (a/b/c); skip DeepSeek API calls")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if pentad_dataset.parquet already exists")
    args = parser.parse_args()

    run_id = setup_logging()
    logger.info("=== MIRAGE Dataset Builder (run_id=%s) ===", run_id)

    if not _check_env():
        return False

    t0 = time.monotonic()

    # ------------------------------------------------------------------ seeds
    logger.info("Step 1/3: Sampling seeds ...")
    from Dataset.sample_seeds import sample_seeds
    main_seeds, dev_seeds = sample_seeds()
    logger.info(
        "  Seeds ready: %d main + %d dev (total %d)",
        len(main_seeds), len(dev_seeds), len(main_seeds) + len(dev_seeds),
    )

    # -------------------------------------------------------- pentad generation
    logger.info(
        "Step 2/3: Building pentad dataset "
        "(%s, %d seeds, %d expected rows) ...",
        "det-only" if args.det_only else "full with DeepSeek API",
        len(main_seeds),
        len(main_seeds) * (7 if args.det_only else 12),
    )
    from Dataset.pentad_generator import build_pentad_dataset
    pentad_df = build_pentad_dataset(
        main_seeds,
        include_api_slots=not args.det_only,
        force=args.force,
    )
    elapsed = time.monotonic() - t0
    logger.info(
        "  Pentad dataset ready: %d rows in %.1fs",
        len(pentad_df), elapsed,
    )

    # ---------------------------------------------------------------- validate
    logger.info("Step 3/3: Validating pentad dataset ...")
    from Dataset.validate_pentad import assert_production_ready, write_pentad_manifest
    try:
        if args.det_only:
            from Dataset.validate_pentad import run_all_validations
            run_all_validations(pentad_df, require_api_slots=False)
        else:
            assert_production_ready(pentad_df)
        write_pentad_manifest(pentad_df)
    except Exception as exc:
        logger.error("Validation FAILED: %s", exc)
        return False

    # ---------------------------------------------------------------- summary
    logger.info("\n=== Dataset Build Summary ===")
    logger.info("  Seeds parquet:    %s", SEEDS_DIR / "seeds.parquet")
    logger.info("  Pentad parquet:   %s", SEEDS_DIR / "pentad_dataset.parquet")
    logger.info("  Total rows:       %d", len(pentad_df))
    logger.info("  Unique seed_ids:  %d", pentad_df["seed_id"].nunique())
    logger.info("  Elapsed:          %.1fs", time.monotonic() - t0)
    logger.info("  Slot distribution:")
    for slot in ["a", "b", "c", "d", "e"]:
        n = (pentad_df["slot"] == slot).sum()
        logger.info("    slot %s: %d prompts", slot, n)
    logger.info("  Source distribution:")
    for src, grp in pentad_df.groupby("seed_source"):
        logger.info("    %-15s %d prompts", src, len(grp))

    logger.info("\n  BUILD COMPLETE")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
    ok = main()
    sys.exit(0 if ok else 1)
