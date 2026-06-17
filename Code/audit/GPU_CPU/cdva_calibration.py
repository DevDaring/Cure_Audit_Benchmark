"""
File: GPU_CPU/cdva_calibration.py
Purpose: Calibrate the tau threshold on the 50-seed dev set (disjoint from
         main 870 seeds). Finds the tau that maximises agreement between
         behavioral pass and CDVA pass per model, then takes the global
         median as the production threshold.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
    -- tau calibration procedure, Section 7.2.

Part of the audit codebase (diagnosis half of CURE).
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RESULTS_DIR, SEEDS_DIR, ensure_dirs
from CPU_Only.scoring import _answers_match

logger = logging.getLogger(__name__)

_TAU_PATH = RESULTS_DIR / "tau_calibration.json"
_CANDIDATE_TAUS = np.linspace(0.1, 0.9, 17).tolist()


def _behavioral_pass(behavioral_df: pd.DataFrame, seed_id: str, model_name: str) -> bool:
    """
    Compute MIRAGE-B behavioral pass for a seed x model.

    Uses the same gold-answer correctness logic as production scoring.py
    (_answers_match) so that tau calibration is consistent with the scoring
    that will be applied to the main results.
    """
    rows = behavioral_df[
        (behavioral_df["seed_id"] == seed_id)
        & (behavioral_df["model_name"] == model_name)
        & (behavioral_df["sample_index"] == 0)
    ]
    if rows.empty:
        return False

    # Retrieve gold_answer for this seed
    gold = ""
    if "gold_answer" in rows.columns:
        gold_vals = rows["gold_answer"].dropna().unique()
        if len(gold_vals) > 0:
            gold = str(gold_vals[0])

    def _correct(slot: str, subvariant: str) -> bool:
        r = rows[
            (rows["slot"] == slot)
            & (rows["subvariant"] == subvariant)
            & (rows["success_flag"] == True)  # noqa: E712
        ]
        if r.empty:
            return False
        return _answers_match(str(r.iloc[0]["parsed_answer"]), gold)

    slot_a_ok = _correct("a", "surface")
    slot_b_ok = _correct("b", "iso_control")

    # Behavioral stability on (c): majority vote over DISTINCT prompt texts
    c_rows = rows[(rows["slot"] == "c") & (rows["success_flag"] == True)]  # noqa: E712
    if len(c_rows) >= 3:
        distinct_c = c_rows.drop_duplicates(subset=["prompt_text"]) if "prompt_text" in c_rows.columns else c_rows
        vc = distinct_c["parsed_answer"].value_counts()
        mv = vc.index[0] if len(vc) > 0 and vc.iloc[0] > len(distinct_c) / 2 else None
        slot_c_ok = mv is not None and _answers_match(mv, gold)
    else:
        slot_c_ok = False

    slot_d_ok = _correct("d", "d_valid") and _correct("d", "d_harmful")

    # CoT robustness: majority answer matches gold across e1, e2, e3
    e_rows = rows[(rows["slot"] == "e") & (rows["success_flag"] == True)]  # noqa: E712
    if len(e_rows) >= 2:
        e_vc = e_rows["parsed_answer"].value_counts()
        e_mv = e_vc.index[0] if len(e_vc) > 0 and e_vc.iloc[0] > len(e_rows) / 2 else None
        slot_e_ok = e_mv is not None and _answers_match(e_mv, gold)
    else:
        slot_e_ok = False

    return all([slot_a_ok, slot_b_ok, slot_c_ok, slot_d_ok, slot_e_ok])


def _cdva_pass(cdva_df: pd.DataFrame, seed_id: str, model_name: str, tau: float) -> bool:
    """Return True if the mean CDVA seed score exceeds tau."""
    rows = cdva_df[
        (cdva_df["seed_id"] == seed_id)
        & (cdva_df["model_name"] == model_name)
        & (cdva_df["success_flag"] == True)  # noqa: E712
    ]
    if rows.empty:
        return False
    score = rows["cdva_pair_score"].mean()
    return float(score) > tau


def calibrate_tau(
    dev_behavioral_df: pd.DataFrame,
    dev_cdva_df: pd.DataFrame,
) -> float:
    """
    Find tau that maximises behavioral-CDVA agreement on the dev set.

    Returns the global median tau across all 4 OSM models.
    """
    ensure_dirs()

    seed_ids = dev_behavioral_df["seed_id"].unique().tolist()
    per_model_best_taus: list[float] = []

    for model_cfg in OSM_MODELS:
        model_name = model_cfg["name"]
        best_tau = 0.5
        best_agreement = -1.0

        for tau in _CANDIDATE_TAUS:
            agreements = []
            for sid in seed_ids:
                b_pass = _behavioral_pass(dev_behavioral_df, sid, model_name)
                c_pass = _cdva_pass(dev_cdva_df, sid, model_name, tau)
                agreements.append(b_pass == c_pass)

            agreement_rate = sum(agreements) / len(agreements) if agreements else 0.0
            if agreement_rate > best_agreement:
                best_agreement = agreement_rate
                best_tau = tau

        logger.info(
            "Model %-35s | best_tau=%.3f | agreement=%.3f",
            model_name, best_tau, best_agreement,
        )
        per_model_best_taus.append(best_tau)

    global_tau = float(np.median(per_model_best_taus))
    logger.info("Global tau (median across models): %.4f", global_tau)

    result = {
        "global_tau": global_tau,
        "per_model_taus": dict(zip([m["name"] for m in OSM_MODELS], per_model_best_taus)),
    }
    with open(_TAU_PATH, "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Tau calibration saved to %s", _TAU_PATH)
    return global_tau


def load_tau() -> float:
    """Load the pre-calibrated tau from disk. Fails if not found."""
    if not _TAU_PATH.exists():
        raise FileNotFoundError(
            f"Tau calibration file not found: {_TAU_PATH}. "
            "Run cdva_calibration.py on the dev set first."
        )
    with open(_TAU_PATH) as fh:
        data = json.load(fh)
    return float(data["global_tau"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Tau calibration requires dev behavioral and CDVA results to exist.")
    logger.info("Run GPU_CPU/osm_behavioral.py and GPU_CPU/cdva_patching.py on the dev set first.")
