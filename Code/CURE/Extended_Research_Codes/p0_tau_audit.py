"""
p0_tau_audit.py -- audit the scale of TAU.

Next_Plan.md, Section 4 (P0):

  The shared audit exposes both raw logit changes and a transformed pair score, while CURE
  compares raw |C| against 0.7644. Trace the calibration artifact rather than assuming the
  same numeric cutoff has the same meaning on both scales.

cdva_results.parquet stores, per pair, delta_logit (raw C) and cdva_pair_score (transformed).
CURE's e3_reaudit compares abs(delta_logit) against TAU = 0.7644 (config_cure.TAU, described
as "the audit default, 75th percentile of |C|"). This module answers, from the data:

  1. Where 0.7644 sits on the raw |C| distribution, pooled and per model (percentile rank).
  2. Where 0.7644 sits on the cdva_pair_score distribution, pooled and per model.
  3. The empirical relation between cdva_pair_score and |delta_logit| (is the score a
     monotone transform of |C|? fit score = a + b*|C| and score = a + b*log1p(|C|) and report
     which fits; report the |C| value at which the fitted score equals 0.7644).
  4. Per-model 75th percentiles of |C| on the successful, no-fallback rows the CURE loader
     uses, so a per-model TAU can be compared with the single pooled one.
  5. Sensitivity: the share of sweep pairs "initially failing" (|C_pre| > tau) and the win
     rate W at the operating rank for tau in {pooled 50th, 75th, 90th pct; per-model 75th}.

Outputs
  reanalysis_v2/tau_audit.csv           percentile placement and per-model quantiles
  reanalysis_v2/tau_relation.csv        fitted score-vs-|C| relations
  reanalysis_v2/tau_sensitivity.csv     failing share and W under alternative cutoffs

Usage:
  python p0_tau_audit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as K

log = K.setup_logging("cure.ext.p0.tau")


def pct_rank(x: np.ndarray, v: float) -> float:
    return float(np.mean(x <= v)) if len(x) else float("nan")


def main() -> None:
    K.OUT_P0.mkdir(parents=True, exist_ok=True)
    c = K.read_cdva()
    c = c[(c["success_flag"] == True) & (c["position_fallback_used"] == False)].copy()   # noqa: E712
    c["absC"] = c["delta_logit"].abs()
    c = c[np.isfinite(c["absC"]) & np.isfinite(c["cdva_pair_score"])]

    rows = []
    pooled_abs = c["absC"].to_numpy(float); pooled_sc = c["cdva_pair_score"].to_numpy(float)
    rows.append({"scope": "pooled", "n_pairs": len(c),
                 "tau": K.TAU,
                 "pct_rank_of_tau_on_absC": pct_rank(pooled_abs, K.TAU),
                 "pct_rank_of_tau_on_pair_score": pct_rank(pooled_sc, K.TAU),
                 "absC_p50": float(np.percentile(pooled_abs, 50)),
                 "absC_p75": float(np.percentile(pooled_abs, 75)),
                 "absC_p90": float(np.percentile(pooled_abs, 90)),
                 "score_p50": float(np.percentile(pooled_sc, 50)),
                 "score_p75": float(np.percentile(pooled_sc, 75)),
                 "score_min": float(pooled_sc.min()), "score_max": float(pooled_sc.max()),
                 "absC_min": float(pooled_abs.min()), "absC_max": float(pooled_abs.max())})
    for m in K.MODELS:
        d = c[c["model_name"] == m]
        a = d["absC"].to_numpy(float); s = d["cdva_pair_score"].to_numpy(float)
        rows.append({"scope": m, "n_pairs": len(d), "tau": K.TAU,
                     "pct_rank_of_tau_on_absC": pct_rank(a, K.TAU),
                     "pct_rank_of_tau_on_pair_score": pct_rank(s, K.TAU),
                     "absC_p50": float(np.percentile(a, 50)), "absC_p75": float(np.percentile(a, 75)),
                     "absC_p90": float(np.percentile(a, 90)),
                     "score_p50": float(np.percentile(s, 50)), "score_p75": float(np.percentile(s, 75)),
                     "score_min": float(s.min()), "score_max": float(s.max()),
                     "absC_min": float(a.min()), "absC_max": float(a.max())})
    tau = pd.DataFrame(rows)
    K.write_csv(tau, K.OUT_P0 / "tau_audit.csv")

    # relation between the transformed score and |C|
    rel = []
    x = pooled_abs; y = pooled_sc
    for name, fx in (("linear", x), ("log1p", np.log1p(x))):
        A = np.vstack([np.ones_like(fx), fx]).T
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ coef
        r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        absC_at_tau = ((K.TAU - coef[0]) / coef[1]) if coef[1] != 0 else np.nan
        if name == "log1p" and np.isfinite(absC_at_tau):
            absC_at_tau = float(np.expm1(absC_at_tau))
        rel.append({"fit": f"score = a + b*{name}(|C|)", "a": float(coef[0]), "b": float(coef[1]),
                    "r2": float(r2), "absC_where_fitted_score_equals_tau": float(absC_at_tau),
                    "spearman_score_vs_absC": float(pd.Series(x).corr(pd.Series(y), method="spearman"))})
    # monotone check: is score a deterministic function of |C|? group by rounded |C|
    g = c.groupby(c["absC"].round(4))["cdva_pair_score"].agg(["nunique", "size"])
    rel.append({"fit": "determinism check", "a": np.nan, "b": np.nan, "r2": np.nan,
                "absC_where_fitted_score_equals_tau": np.nan,
                "spearman_score_vs_absC": float((g["nunique"] == 1).mean()),
                })
    rel[-1]["note"] = ("share of distinct |C| values (4 dp) mapping to exactly one score; 1.0 means "
                       "the score is a deterministic function of |C| alone")
    K.write_csv(pd.DataFrame(rel), K.OUT_P0 / "tau_relation.csv")

    # sensitivity on the sweep at the operating rank
    sens = []
    cuts = {"pooled_p50": float(np.percentile(pooled_abs, 50)),
            "pooled_p75": float(np.percentile(pooled_abs, 75)),
            "pooled_p90": float(np.percentile(pooled_abs, 90)),
            "config_TAU": K.TAU}
    for m in K.MODELS:
        sw = K.read_sweep(m)
        op = K.read_rankcurve(m).get("operating_rank")
        d = sw[sw["erase_rank"] == op]
        d = d[np.isfinite(d["orig_commutator"]) & np.isfinite(d["erased_commutator"])]
        per_model_p75 = float(np.percentile(c[c["model_name"] == m]["absC"], 75))
        for label, t in {**cuts, "per_model_p75": per_model_p75}.items():
            fail = d["orig_commutator"] > t
            sens.append({"model_name": m, "operating_rank": op, "cutoff": label, "tau_value": t,
                         "share_initially_failing": float(fail.mean()),
                         "W_win_rate": float((d["erased_commutator"] < d["orig_commutator"]).mean()),
                         "X_threshold_crossing": float((d.loc[fail, "erased_commutator"] <= t).mean()) if fail.any() else np.nan,
                         "n_pairs": len(d)})
    K.write_csv(pd.DataFrame(sens), K.OUT_P0 / "tau_sensitivity.csv")

    pd.set_option("display.width", 200)
    print("\n=== WHERE TAU = 0.7644 SITS ===")
    print(tau[["scope", "n_pairs", "pct_rank_of_tau_on_absC", "pct_rank_of_tau_on_pair_score",
               "absC_p75", "score_p75", "score_min", "score_max"]].round(4).to_string(index=False))
    print("\n=== SCORE vs |C| RELATION ===")
    print(pd.DataFrame(rel).round(4).to_string(index=False))
    print("\n=== SENSITIVITY at operating rank ===")
    print(pd.DataFrame(sens).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
