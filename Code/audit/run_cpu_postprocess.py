"""
Run CPU-only post-processing on existing GPU results (no GPU, no API calls).

Steps: tau calibration (percentile fallback) -> scoring -> leaderboard ->
       validity gap table -> predictive validity -> results analysis.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_DIR, ensure_dirs
from CPU_Only.leaderboard import build_leaderboard
from CPU_Only.predictive_validity import run_predictive_validity
from CPU_Only.results_analysis import run_results_analysis
from CPU_Only.scoring import score_all
from CPU_Only.validity_gap_table import build_validity_gap_table, _write_markdown
from results_utils import dedup_behavioral, dedup_cdva

logger = logging.getLogger(__name__)


def calibrate_tau_percentile(cdva_df: pd.DataFrame) -> float:
    """75th percentile of |delta_logit| when dev seeds are not tagged."""
    valid = cdva_df[
        (cdva_df["success_flag"] == True)  # noqa: E712
        & (cdva_df["position_fallback_used"] == False)
    ]
    tau = float(np.percentile(valid["delta_logit"].abs().values, 75))
    out = {
        "global_tau": tau,
        "method": "75th_percentile_abs_delta_logit",
        "note": "Dev seeds not tagged; percentile fallback used.",
    }
    path = RESULTS_DIR / "tau_calibration.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Tau = %.4f saved to %s", tau, path)
    return tau


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_dirs()

    beh_path = RESULTS_DIR / "behavioral_results.parquet"
    cdva_path = RESULTS_DIR / "cdva_results.parquet"
    if not beh_path.exists():
        logger.error("Missing %s", beh_path)
        sys.exit(1)
    if not cdva_path.exists():
        logger.error("Missing %s", cdva_path)
        sys.exit(1)

    behavioral = dedup_behavioral(pd.read_parquet(beh_path))
    cdva = dedup_cdva(pd.read_parquet(cdva_path))
    cdva = cdva[cdva["position_fallback_used"] == False]

    logger.info("Loaded behavioral=%d rows, cdva=%d rows", len(behavioral), len(cdva))

    tau = calibrate_tau_percentile(cdva)

    scored = score_all(behavioral, cdva, tau)
    logger.info(
        "Scoring done: MIRAGE-B=%.3f MIRAGE-Full=%.3f",
        scored["mirage_b_pass"].mean(),
        scored["mirage_full_pass"].mean(),
    )

    leaderboard = build_leaderboard(behavioral, cdva)
    logger.info("Leaderboard:\n%s", leaderboard.to_string())

    gap_df = build_validity_gap_table(behavioral, scored)
    gap_df.to_parquet(RESULTS_DIR / "validity_gap_leaderboard.parquet", index=False)
    _write_markdown(gap_df, RESULTS_DIR / "validity_gap_leaderboard.md")
    logger.info("Validity gap table written")

    pred = run_predictive_validity(behavioral, cdva)
    if pred:
        (RESULTS_DIR / "predictive_validity.json").write_text(
            json.dumps(pred, indent=2), encoding="utf-8"
        )
        logger.info("Predictive validity: %s", pred)
    else:
        logger.warning("Predictive validity skipped (no WinoBias rows or single class)")

    run_results_analysis()
    logger.info("CPU post-processing complete. Outputs in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
