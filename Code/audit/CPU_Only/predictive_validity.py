"""
File: CPU_Only/predictive_validity.py
Purpose: Predictive validity -- trains on BBQ+CrowS+StereoSet MIRAGE-B pass
         patterns, tests whether they predict WinoBias coreference accuracy.

Labels are NOT derived from the same FM helper functions as features (fixes
review finding B3 circularity).  Training label = MIRAGE-B pass (gold-answer
based).  Test label = WinoBias coreference correctness (independent parse).

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - Zhao et al. (2018). "WinoBias." NAACL 2018.

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR, SEEDS_DIR, ensure_dirs
from CPU_Only.scoring import _answers_match, compute_mirage_b

logger = logging.getLogger(__name__)

_TRAIN_SOURCES = {"bbq", "crows_pairs", "stereoset"}


def _build_feature_matrix(behavioral_df: pd.DataFrame, cdva_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(seed_id, model_name) behavioral feature vector."""
    rows: list[dict] = []
    pairs = (
        behavioral_df[["seed_id", "model_name", "seed_source", "seed_category"]]
        .drop_duplicates(subset=["seed_id", "model_name"])
    )

    for _, pair in pairs.iterrows():
        seed_id = pair["seed_id"]
        model_name = pair["model_name"]
        b_rows = behavioral_df[
            (behavioral_df["seed_id"] == seed_id)
            & (behavioral_df["model_name"] == model_name)
            & (behavioral_df["sample_index"] == 0)
        ]

        def _correct_rate(slot: str) -> float:
            s = b_rows[(b_rows["slot"] == slot) & (b_rows["success_flag"] == True)]  # noqa: E712
            if s.empty or "gold_answer" not in s.columns:
                return 0.0
            gold = str(s["gold_answer"].iloc[0])
            return float(
                sum(_answers_match(str(r["parsed_answer"]), gold) for _, r in s.iterrows())
                / len(s)
            )

        e_rows = b_rows[(b_rows["slot"] == "e") & (b_rows["success_flag"] == True)]  # noqa: E712
        cot_robust = 0.0
        if len(e_rows) >= 2 and "gold_answer" in e_rows.columns:
            gold = str(e_rows["gold_answer"].iloc[0])
            vc = e_rows["parsed_answer"].value_counts()
            if len(vc) > 0 and vc.iloc[0] > len(e_rows) / 2:
                cot_robust = 1.0 if _answers_match(str(vc.index[0]), gold) else 0.0

        cdva_seed = cdva_df[
            (cdva_df["seed_id"] == seed_id)
            & (cdva_df["model_name"] == model_name)
            & (cdva_df["success_flag"] == True)  # noqa: E712
        ] if len(cdva_df) > 0 else pd.DataFrame()
        cdva_score = float(cdva_seed["cdva_pair_score"].mean()) if len(cdva_seed) > 0 else 0.0

        rows.append(
            {
                "seed_id": seed_id,
                "model_name": model_name,
                "seed_source": pair["seed_source"],
                "seed_category": pair["seed_category"],
                "feat_a_correct": _correct_rate("a"),
                "feat_b_correct": _correct_rate("b"),
                "feat_c_correct": _correct_rate("c"),
                "feat_d_correct": _correct_rate("d"),
                "feat_e_correct": _correct_rate("e"),
                "feat_cot_robust": cot_robust,
                "feat_cdva_score": cdva_score,
            }
        )

    return pd.DataFrame(rows)


def _compute_mirage_b_labels(behavioral_df: pd.DataFrame) -> pd.DataFrame:
    """Independent training label: MIRAGE-B pass from gold-answer scoring."""
    rows: list[dict] = []
    for seed_id in behavioral_df["seed_id"].unique():
        for model_name in behavioral_df["model_name"].unique():
            rows.append(
                {
                    "seed_id": seed_id,
                    "model_name": model_name,
                    "mirage_b_pass": int(
                        compute_mirage_b(behavioral_df, seed_id, model_name)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _compute_winobias_coreference_labels(behavioral_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test label for held-out WinoBias: slot-a coreference answer matches gold.
    """
    rows: list[dict] = []
    wino = behavioral_df[behavioral_df["seed_source"] == "winobias"]
    for seed_id in wino["seed_id"].unique():
        for model_name in wino["model_name"].unique():
            m_rows = wino[
                (wino["seed_id"] == seed_id)
                & (wino["model_name"] == model_name)
                & (wino["slot"] == "a")
                & (wino["sample_index"] == 0)
                & (wino["success_flag"] == True)  # noqa: E712
            ]
            if m_rows.empty:
                correct = 0
            else:
                row = m_rows.iloc[0]
                correct = int(
                    _answers_match(
                        str(row["parsed_answer"]),
                        str(row.get("gold_answer", "")),
                    )
                )
            rows.append(
                {
                    "seed_id": seed_id,
                    "model_name": model_name,
                    "coreference_correct": correct,
                }
            )
    return pd.DataFrame(rows)


def run_predictive_validity(
    behavioral_df: pd.DataFrame,
    cdva_df: pd.DataFrame,
) -> dict:
    """
    Train logistic regression on audit benchmarks to predict MIRAGE-B pass.
    Evaluate on WinoBias whether high-risk audit patterns predict coreference errors.
    """
    ensure_dirs()
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # type: ignore

    features_df = _build_feature_matrix(behavioral_df, cdva_df)
    train_labels = _compute_mirage_b_labels(behavioral_df)
    test_labels = _compute_winobias_coreference_labels(behavioral_df)
    if test_labels.empty or "seed_id" not in test_labels.columns:
        logger.warning(
            "No WinoBias behavioral rows found. Run WinoBias evaluation separately."
        )
        return {}

    merged = features_df.merge(train_labels, on=["seed_id", "model_name"])
    feat_cols = [c for c in merged.columns if c.startswith("feat_")]

    train = merged[merged["seed_source"].isin(_TRAIN_SOURCES)]
    test_feat = features_df[features_df["seed_source"] == "winobias"].merge(
        test_labels, on=["seed_id", "model_name"]
    )

    if len(test_feat) == 0:
        logger.warning("WinoBias feature rows empty after merge; skipping.")
        return {}

    X_train = train[feat_cols].fillna(0).values
    y_train = train["mirage_b_pass"].values
    X_test = test_feat[feat_cols].fillna(0).values
    y_test = test_feat["coreference_correct"].values

    if len(np.unique(y_train)) < 2:
        logger.warning("Training set has only one MIRAGE-B class; skipping.")
        return {}

    clf = LogisticRegression(max_iter=500, random_state=20260101)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else y_pred

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc = float("nan")

    results = {
        "winobias_coreference": {
            "accuracy": acc,
            "f1": f1,
            "roc_auc": auc,
            "n_test": len(y_test),
        }
    }
    logger.info(
        "Predictive validity (WinoBias coreference) | acc=%.3f f1=%.3f auc=%.3f n=%d",
        acc, f1, auc, len(y_test),
    )
    return results
