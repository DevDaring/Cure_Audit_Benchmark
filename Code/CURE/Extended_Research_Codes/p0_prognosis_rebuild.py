"""
p0_prognosis_rebuild.py -- rebuild the prognosis target honestly.

Next_Plan.md, Section 2.6 and P0 row 5:

  The actual outcome in e6_prognosis is the first tested rank whose erased |C| crosses TAU;
  it takes the first pair for each seed. It does not require correct or invariant generated
  answers. It does not impose utility preservation. Already-passing items are assigned a
  positive rank because rank zero is absent. reanalyse.py defines the high-rank class as
  repair_rank >= 8, which includes both rank-8 successes and sentinel-9 failures. Repeated
  overlapping splits are not 200 independent replications.

  Required: rank 0 for initially passing seeds; all paired variants handled by a predeclared
  seed aggregation; separate audit repair, generation validity and task damage.

What this does, all from cure_recovery_sweep_<model>.parquet (every pair, every rank):

  1. REPRODUCE the shipped first-pair prognosis and the plan's restricted-AUC table, as a
     check that this module reads the same records the plan did.
  2. REBUILD with two predeclared seed aggregations over ALL pairs of a seed:
        worst_pair  score = max_pairs |C_pre|; repaired at rank r iff every pair <= TAU
        mean_pair   score = mean_pairs |C_pre|; repaired at rank r iff mean <= TAU
     rank 0 for seeds already at or below TAU before any edit; SENTINEL for never repaired.
  3. TARGETS, kept separate because they are different claims:
        never_repaired      repair_rank == SENTINEL            (strict failure)
        high_or_never       repair_rank >= 8                   (shipped, mixes both)
        worsened_at_oprank  aggregated |C_post| > |C_pre| at the operating rank
     generation validity and task damage: NOT COMPUTABLE from the artifacts (no raw
     generations; see p0_raw_archive_search) -> status REQUIRES_RERUN in the ledger.
  4. HELD-OUT evaluation with ONE fixed split from split_manifest.csv (fit+dev seeds choose
     the threshold, test seeds are scored once), instead of 200 overlapping random halves.
     Reports AUROC, average precision with prevalence, Brier score, and the restricted
     (initially failing only) versions of each.

  Circularity caveat, recorded in every row: the predictor is |C_pre| and the target is a
  function of |C_pre| crossing TAU after the edit. A seed just above TAU needs a small
  reduction to cross; a seed far above needs a large one. Part of any AUC here is mechanical.
  Only a behavioural target (P2/P4) can remove that.

Outputs
  reanalysis_v2/prognosis_reproduction.csv    shipped first-pair numbers and restricted AUC
  reanalysis_v2/prognosis_rebuilt_seed_level.csv  one row per (model, aggregation, seed)
  reanalysis_v2/prognosis_rebuilt.csv         per (model, aggregation, target, population)
  reanalysis_v2/prognosis_rebuilt_heldout.csv one fixed manifest split, test seeds scored once

Usage:
  python p0_prognosis_rebuild.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import common as K

log = K.setup_logging("cure.ext.p0.prog")


def _auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def reproduce_shipped() -> pd.DataFrame:
    """Shipped first-pair records: all-seed AUC for rank>=8 and the restricted AUC on seeds
    with initial score > TAU. Next_Plan.md Section 2.6 reports 0.9292/0.8593/0.9296/0.8506
    (all) and 0.7152/0.7723/0.6190/0.7060 (restricted) with 105/205/89/113 failing seeds."""
    rows = []
    for m in K.MODELS:
        d = K.read_prognosis(m)
        hard = (d["repair_rank"] >= 8).astype(int)
        fail = d["audit_score"] > K.TAU
        rows.append({"model_name": m, "n_seeds": len(d), "source": f"cure_prognosis_{m}.parquet",
                     "aggregation": "first_pair (shipped)",
                     "auc_all_seeds_rank_ge8": _auc(hard, d["audit_score"]),
                     "n_initially_failing": int(fail.sum()),
                     "auc_restricted_to_failing": _auc(hard[fail], d.loc[fail, "audit_score"]),
                     "n_sentinel": int((d["repair_rank"] == K.SENTINEL_RANK).sum()),
                     "n_rank8_success": int((d["repair_rank"] == 8).sum()),
                     "n_initially_passing_given_positive_rank": int((~fail).sum()),
                     "note": "already-passing seeds carry a positive rank in the shipped file"})
    return pd.DataFrame(rows)


def rebuild_seed_level(model: str) -> pd.DataFrame:
    sw = K.read_sweep(model)
    sw = sw[np.isfinite(sw["orig_commutator"]) & np.isfinite(sw["erased_commutator"])]
    op = K.read_rankcurve(model).get("operating_rank")
    ranks = sorted(sw["erase_rank"].unique())
    rows = []
    for agg in ("worst_pair", "mean_pair"):
        f = (lambda s: s.max()) if agg == "worst_pair" else (lambda s: s.mean())
        # pre-edit score is rank-independent; take it from rank 1 rows (identical across ranks)
        pre = sw[sw["erase_rank"] == ranks[0]].groupby("seed_id")["orig_commutator"].agg(f)
        post = {r: sw[sw["erase_rank"] == r].groupby("seed_id")["erased_commutator"].agg(f)
                for r in ranks}
        for sid, score in pre.items():
            if score <= K.TAU:
                rank = 0
            else:
                rank = K.SENTINEL_RANK
                for r in ranks:
                    v = post[r].get(sid, np.nan)
                    if np.isfinite(v) and v <= K.TAU:
                        rank = int(r); break
            post_op = post[op].get(sid, np.nan) if op in post else np.nan
            rows.append({"model_name": model, "aggregation": agg, "seed_id": sid,
                         "benchmark": K.benchmark_of(sid), "audit_score": float(score),
                         "initially_failing": bool(score > K.TAU), "repair_rank": rank,
                         "never_repaired": int(rank == K.SENTINEL_RANK),
                         "high_or_never": int(rank >= 8),
                         "worsened_at_oprank": int(np.isfinite(post_op) and post_op > score),
                         "post_at_oprank": float(post_op), "operating_rank": op,
                         "n_pairs": int((sw[sw["erase_rank"] == ranks[0]]["seed_id"] == sid).sum())})
    return pd.DataFrame(rows)


def summarise(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, agg), d in seed_df.groupby(["model_name", "aggregation"]):
        for target in ("never_repaired", "high_or_never", "worsened_at_oprank"):
            for pop_name, pop in (("all_seeds", d), ("initially_failing", d[d["initially_failing"]])):
                y = pop[target].to_numpy(int); s = pop["audit_score"].to_numpy(float)
                rows.append({
                    "model_name": m, "aggregation": agg, "target": target, "population": pop_name,
                    "n_seeds": len(pop), "prevalence": float(y.mean()) if len(pop) else np.nan,
                    "auroc": _auc(y, s),
                    "avg_precision": float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else np.nan,
                    "n_rank0": int((pop["repair_rank"] == 0).sum()),
                    "n_sentinel": int((pop["repair_rank"] == K.SENTINEL_RANK).sum()),
                    "caveat": "predictor and target both derive from |C_pre| vs TAU; part of AUC is "
                              "mechanical; behavioural target REQUIRES_RERUN",
                })
    return pd.DataFrame(rows)


def heldout_fixed_split(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Threshold chosen on manifest fit+dev seeds (Youden J on the raw score), evaluated once
    on manifest test seeds. No repeated overlapping halves."""
    man = pd.read_csv(K.OUT_P0 / "split_manifest.csv")[["seed_id", "split"]]
    d = seed_df.merge(man, on="seed_id", how="left")
    rows = []
    for (m, agg), g in d.groupby(["model_name", "aggregation"]):
        for target in ("never_repaired", "high_or_never"):
            for pop_name in ("all_seeds", "initially_failing"):
                gg = g if pop_name == "all_seeds" else g[g["initially_failing"]]
                tr = gg[gg["split"].isin(["fit", "dev"])]; te = gg[gg["split"] == "test"]
                if tr[target].nunique() < 2 or te[target].nunique() < 2:
                    rows.append({"model_name": m, "aggregation": agg, "target": target,
                                 "population": pop_name, "n_train": len(tr), "n_test": len(te),
                                 "note": "degenerate class in train or test"})
                    continue
                grid = np.quantile(tr["audit_score"], np.linspace(0.05, 0.95, 37))
                best_t, best_j = grid[0], -np.inf
                for t in grid:
                    pred = (tr["audit_score"] >= t).astype(int)
                    j = pred[tr[target] == 1].mean() - pred[tr[target] == 0].mean()
                    if j > best_j:
                        best_j, best_t = j, t
                y = te[target].to_numpy(int); s = te["audit_score"].to_numpy(float)
                pred = (s >= best_t).astype(int)
                # probability calibration for Brier: isotonic-free, use train prevalence within
                # score bins (5 quantile bins) -- simple and honest for n of this size
                bins = np.quantile(tr["audit_score"], [0, .2, .4, .6, .8, 1.0])
                bins[-1] = np.inf
                tr_bin = np.clip(np.searchsorted(bins, tr["audit_score"], side="right") - 1, 0, 4)
                p_bin = np.array([tr[target].to_numpy()[tr_bin == b].mean() if (tr_bin == b).any()
                                  else tr[target].mean() for b in range(5)])
                te_bin = np.clip(np.searchsorted(bins, s, side="right") - 1, 0, 4)
                prob = p_bin[te_bin]
                rows.append({
                    "model_name": m, "aggregation": agg, "target": target, "population": pop_name,
                    "n_train": len(tr), "n_test": len(te), "test_prevalence": float(y.mean()),
                    "threshold_from_train": float(best_t),
                    "test_auroc": _auc(y, s),
                    "test_avg_precision": float(average_precision_score(y, s)),
                    "test_accuracy_at_threshold": float((pred == y).mean()),
                    "test_tpr": float(pred[y == 1].mean()), "test_fpr": float(pred[y == 0].mean()),
                    "test_brier_binned": float(brier_score_loss(y, prob)),
                    "test_brier_constant_baseline": float(brier_score_loss(y, np.full_like(prob, tr[target].mean()))),
                    "note": "one fixed manifest split; threshold and bins fit on fit+dev only",
                })
    return pd.DataFrame(rows)


def main() -> None:
    K.OUT_P0.mkdir(parents=True, exist_ok=True)
    rep = reproduce_shipped()
    K.write_csv(rep, K.OUT_P0 / "prognosis_reproduction.csv")

    seed_df = pd.concat([rebuild_seed_level(m) for m in K.MODELS], ignore_index=True)
    K.write_csv(seed_df, K.OUT_P0 / "prognosis_rebuilt_seed_level.csv")
    summ = summarise(seed_df)
    K.write_csv(summ, K.OUT_P0 / "prognosis_rebuilt.csv")
    ho = heldout_fixed_split(seed_df)
    K.write_csv(ho, K.OUT_P0 / "prognosis_rebuilt_heldout.csv")

    pd.set_option("display.width", 220)
    print("\n=== REPRODUCTION of shipped first-pair prognosis (plan: all 0.9292/0.8593/0.9296/0.8506; "
          "restricted 0.7152/0.7723/0.6190/0.7060; failing 105/205/89/113) ===")
    print(rep[["model_name", "n_seeds", "auc_all_seeds_rank_ge8", "n_initially_failing",
               "auc_restricted_to_failing", "n_sentinel", "n_rank8_success",
               "n_initially_passing_given_positive_rank"]].round(4).to_string(index=False))
    print("\n=== REBUILT: rank 0 for already-passing seeds, all pairs aggregated ===")
    v = summ[summ["target"].isin(["never_repaired", "high_or_never"])]
    print(v[["model_name", "aggregation", "target", "population", "n_seeds", "prevalence",
             "auroc", "avg_precision", "n_rank0", "n_sentinel"]].round(3).to_string(index=False))
    print("\n=== HELD-OUT, one fixed manifest split (test seeds scored once) ===")
    w = ho[ho["target"] == "never_repaired"] if "test_auroc" in ho else ho
    cols = [c for c in ["model_name", "aggregation", "population", "n_train", "n_test", "test_prevalence",
                        "test_auroc", "test_avg_precision", "test_accuracy_at_threshold",
                        "test_brier_binned", "test_brier_constant_baseline"] if c in w.columns]
    print(w[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
