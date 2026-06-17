"""
Full CPU_Only pipeline: API behavioral evaluation + post-processing.

Run dry run first:
    python Dry_Run/dry_run_cpu_only.py

Then (sequential API calls, checkpoints every 50 prompts):
    python run_cpu_full.py
"""

import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd

from CPU_Only.api_behavioral import run_api_behavioral
from config import RESULTS_DIR, SEEDS_DIR, ensure_dirs
from logger_setup import setup_logging

# Reuse post-processing from run_cpu_postprocess
from run_cpu_postprocess import main as run_postprocess

logger = logging.getLogger(__name__)


def main() -> None:
    run_id = setup_logging()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_dirs()

    pentad_path = SEEDS_DIR / "pentad_dataset.parquet"
    if not pentad_path.exists():
        logger.error("Missing pentad dataset: %s", pentad_path)
        sys.exit(1)

    pentad = pd.read_parquet(pentad_path)
    logger.info("=== CPU_Only Full Pipeline (run_id=%s) ===", run_id)
    logger.info("Pentad: %d rows", len(pentad))
    logger.info("Started: %s", datetime.now().isoformat())

    # Phase 1: API behavioral (appends to existing OSM behavioral results)
    logger.info("--- Phase 1: API Behavioral Evaluation (4 models) ---")
    run_api_behavioral(pentad, run_id=run_id)

    # Phase 2: Scoring, leaderboard, validity gap, figures
    logger.info("--- Phase 2: Post-processing ---")
    run_postprocess()

    logger.info("=== CPU_Only Full Pipeline COMPLETE ===")
    logger.info("Outputs: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
