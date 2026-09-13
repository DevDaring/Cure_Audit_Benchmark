"""
p0_metric_audit.py -- recompute directional decrease, magnitude reduction, worsening and
threshold crossing SEPARATELY, with exact definitions and denominators.

Next_Plan.md, Section 2.1 and P0 row 2. The shipped "removed" quantity is

    W = mean( |C_post| < |C_pre| )                      (directional win rate)

which counts a decrease of 0.0001 and a decrease of 10 equally. It is neither the fraction of
signal magnitude removed nor proof of erased concept information. This module reports, on the
same rows:

    W        directional win rate, as shipped
    T        tie share,        mean( |C_post| == |C_pre| )
    L        worsened share,   mean( |C_post| >  |C_pre| )
    D        paired absolute change, mean( |C_post| - |C_pre| )   [primary effect]
    R        ratio of means,   1 - mean(|C_post|) / mean(|C_pre|) [secondary, normalised]
    X        threshold crossing among initially failing pairs:
             mean( |C_post| <= TAU  |  |C_pre| > TAU )
    B        share already below TAU before the edit, mean( |C_pre| <= TAU )

Per Section 4 of the plan: paired absolute change is primary; the ratio of means is secondary;
per-pair fractional reductions are NOT averaged (near-zero denominators); intervals are
seed-clustered percentile bootstraps (all pairs of a seed resampled together); pooled and
per-benchmark strata are both reported.

Inputs (read only)
  cure_recovery_sweep_<model>.parquet   1000 pairs x ranks {1,2,4,8}, per-pair |C| pre/post
  tacl_extra_<model>.parquet            held-out summary rows (means only, no per-pair data)
  cure_final_<model>.parquet            head-to-head summary rows (means only)

Outputs
  reanalysis_v2/metric_audit.csv                 every (model, rank, stratum) with all seven
  reanalysis_v2/metric_audit_heldout.csv         R recomputed from the held-out summary means
  reanalysis_v2/metric_audit_headtohead.csv      R recomputed from the head-to-head means
  reanalysis_v2/metric_definitions.md            the formulas above, verbatim

Usage:
  python p0_metric_audit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as K

log = K.setup_logging("cure.ext.p0.metric")

DEFINITIONS = """# Metric definitions (p0_metric_audit)

All quantities are computed on the same rows: one row = one counterfactual pair (seed_id,
subvariant_A, subvariant_B) at one erasure rank. `pre` = orig_commutator = |C| before the
edit; `post` = erased_commutator = |C| after the edit. Both are absolute values as stored.

| symbol | name | formula | denominator |
|---|---|---|---|
| W | directional win rate (as shipped) | mean(post < pre) | all pairs with finite pre and post |
| T | tie share | mean(post == pre) | same |
| L | worsened share | mean(post > pre) | same |
| D | paired absolute change (primary) | mean(post - pre) | same |
| R | ratio of means (secondary) | 1 - mean(post) / mean(pre) | same |
| X | threshold crossing | mean(post <= TAU) over pairs with pre > TAU | pairs with pre > TAU |
| B | already-below share | mean(pre <= TAU) | all pairs |

TAU = 0.7644 (config_cure.TAU), compared against raw |C| exactly as the CURE code does.
Confidence intervals: 95% percentile bootstrap resampling seeds (clusters), 2000 draws,
seed 20260912. Per-pair fractional reductions are deliberately not averaged.
Negative D or negative R means the mean audit magnitude increased after the edit.
"""


def _finite(d: pd.DataFrame) -> pd.DataFrame:
    return d[np.isfinite(d["orig_commutator"]) & np.isfinite(d["erased_commutator"])].copy()


def metrics(d: pd.DataFrame) -> dict:
    pre = d["orig_commutator"].to_numpy(float)
    post = d["erased_commutator"].to_numpy(float)
    fail = pre > K.TAU
    out = {
        "n_pairs": int(len(d)), "n_seeds": K.n_clusters(d),
        "W_win_rate": float(np.mean(post < pre)),
        "T_tie_share": float(np.mean(post == pre)),
        "L_worsened_share": float(np.mean(post > pre)),
        "D_paired_abs_change": float(np.mean(post - pre)),
        "mean_pre": float(pre.mean()), "mean_post": float(post.mean()),
        "R_ratio_of_means": float(1.0 - post.mean() / pre.mean()) if pre.mean() > 0 else float("nan"),
        "n_initially_failing": int(fail.sum()),
        "X_threshold_crossing": float(np.mean(post[fail] <= K.TAU)) if fail.any() else float("nan"),
        "B_already_below_share": float(np.mean(~fail)),
    }
    return out


def with_ci(d: pd.DataFrame) -> dict:
    """metrics() plus seed-clustered 95% intervals for W, L, D, R and X (vectorised)."""
    m = metrics(d)
    pre = d["orig_commutator"].to_numpy(float)
    post = d["erased_commutator"].to_numpy(float)
    cl = d["seed_id"].to_numpy()
    ones = np.ones_like(pre)
    specs = {
        "W_win_rate": ((post < pre).astype(float), ones, None),
        "L_worsened_share": ((post > pre).astype(float), ones, None),
        "D_paired_abs_change": (post - pre, ones, None),
        "R_ratio_of_means": (post, pre, lambda r: 1.0 - r),
    }
    for key, (num, den, tf) in specs.items():
        _, lo, hi = K.ratio_bootstrap(num, den, cl, transform=tf)
        m[key + "_ci_lo"], m[key + "_ci_hi"] = lo, hi
    fail = pre > K.TAU
    if fail.sum() >= 2:
        _, lo, hi = K.ratio_bootstrap((post[fail] <= K.TAU).astype(float), ones[fail], cl[fail])
        m["X_threshold_crossing_ci_lo"], m["X_threshold_crossing_ci_hi"] = lo, hi
    return m


def sweep_audit() -> pd.DataFrame:
    rows = []
    for model in K.MODELS:
        sw = _finite(K.read_sweep(model))
        op = K.read_rankcurve(model).get("operating_rank")
        for rank, dr in sw.groupby("erase_rank"):
            base = {"model_name": model, "source": f"cure_recovery_sweep_{model}.parquet",
                    "erase_rank": int(rank), "is_operating_rank": bool(op == rank)}
            rows.append({**base, "stratum": "pooled", **with_ci(dr)})
            for bm, db in dr.groupby("benchmark"):
                rows.append({**base, "stratum": bm, **metrics(db)})
            log.info("%s rank %d: W=%.3f D=%+.3f R=%+.3f X=%.3f (n=%d pairs, %d seeds)",
                     K.DISPLAY[model], rank, rows[-1 - dr["benchmark"].nunique()]["W_win_rate"],
                     rows[-1 - dr["benchmark"].nunique()]["D_paired_abs_change"],
                     rows[-1 - dr["benchmark"].nunique()]["R_ratio_of_means"],
                     rows[-1 - dr["benchmark"].nunique()]["X_threshold_crossing"],
                     len(dr), K.n_clusters(dr))
    return pd.DataFrame(rows)


def heldout_audit() -> pd.DataFrame:
    """R from the saved held-out means. Only means are stored, so no CI is possible here and
    no per-pair quantity (W is stored as heldout_residual_removed)."""
    rows = []
    for model in K.MODELS:
        h = K.read_heldout(model)
        for _, r in h.iterrows():
            pre, post = r.get("orig_commutator_test"), r.get("mean_erased_commutator")
            R = (1.0 - post / pre) if (pd.notna(pre) and pd.notna(post) and pre > 0) else np.nan
            rows.append({"model_name": model, "method": r["method"], "op_rank": r.get("op_rank"),
                         "W_win_rate_as_stored": r.get("heldout_residual_removed"),
                         "mean_pre_stored": pre, "mean_post_stored": post,
                         "R_ratio_of_means": R, "n_test_pairs": r.get("n_test_pairs"),
                         "n_acc_seeds": r.get("n_acc_seeds"), "n_flip_seeds": r.get("n_flip_seeds"),
                         "behav_accuracy": r.get("behav_accuracy"),
                         "behav_flip_rate": r.get("behav_flip_rate"),
                         "note": "means only in artifact; no per-pair rows, no CI possible; "
                                 "R from rounded stored means"})
    return pd.DataFrame(rows)


def headtohead_audit() -> pd.DataFrame:
    rows = []
    for model in K.MODELS:
        f = K.read_final(model)
        # pre-edit mean for the head-to-head eval pairs is not stored per row; reconstruct
        # it from the sweep (same seed pool) is NOT valid because eval_pairs used
        # RANDOM_SEED+7. Report only what the artifact contains.
        for _, r in f.iterrows():
            rows.append({"model_name": model, "method": r["method"], "status": r.get("status"),
                         "erase_rank": r.get("erase_rank"),
                         "W_win_rate_as_stored": r.get("causal_residual_removed"),
                         "mean_post_stored": r.get("mean_erased_commutator"),
                         "utility_cost": r.get("utility_cost"), "n_pairs": r.get("n_pairs"),
                         "baseline_acc": r.get("baseline_acc"), "erased_acc": r.get("erased_acc"),
                         "note": "no pre-edit mean stored for these eval pairs; R not computable "
                                 "from this artifact"})
    return pd.DataFrame(rows)


def main() -> None:
    K.OUT_P0.mkdir(parents=True, exist_ok=True)
    (K.OUT_P0 / "metric_definitions.md").write_text(DEFINITIONS, encoding="utf-8")

    sw = sweep_audit()
    K.write_csv(sw, K.OUT_P0 / "metric_audit.csv")
    ho = heldout_audit()
    K.write_csv(ho, K.OUT_P0 / "metric_audit_heldout.csv")
    hh = headtohead_audit()
    K.write_csv(hh, K.OUT_P0 / "metric_audit_headtohead.csv")

    pd.set_option("display.width", 200)
    print("\n=== SWEEP, pooled, all ranks: W (shipped) vs D (primary) vs R (secondary) vs X ===")
    v = sw[sw["stratum"] == "pooled"][["model_name", "erase_rank", "is_operating_rank", "n_pairs",
                                       "n_seeds", "W_win_rate", "L_worsened_share",
                                       "D_paired_abs_change", "D_paired_abs_change_ci_lo",
                                       "D_paired_abs_change_ci_hi", "R_ratio_of_means",
                                       "X_threshold_crossing", "B_already_below_share"]]
    print(v.round(4).to_string(index=False))

    print("\n=== HELD-OUT (tacl_extra) CURE rows: W as stored vs R from stored means ===")
    c = ho[ho["method"] == "cure"][["model_name", "op_rank", "W_win_rate_as_stored", "mean_pre_stored",
                                    "mean_post_stored", "R_ratio_of_means", "n_test_pairs",
                                    "n_acc_seeds", "n_flip_seeds", "behav_accuracy", "behav_flip_rate"]]
    print(c.round(4).to_string(index=False))
    print("\nNext_Plan.md Section 2.1 states R = +29.35%, +19.12%, +0.48%, -61.57% for "
          "Llama, Qwen, Gemma, Phi. Compare with the R column above.")


if __name__ == "__main__":
    main()
