"""
g4_analyses.py -- the CPU-only analyses of the ICLR-feedback cycle (Submission2/Next_Plan.md
A2 part 1, A4, C7, C9), computed from the stored per-item files of the final cycle and the
pooled audit. No model is loaded.

  g4_strata.csv            A2.1  reduction in |C| (pooled audit, raw logit) and in |C_answer|
                                 (fresh replication) on equal-length vs unequal-length span
                                 pairs, with template- (fresh) or seed-cluster (pooled) intervals
  g4_length.csv            A4    per condition: T under raw-sum, per-token-mean and
                                 unknown-option-contrast scorings; the share of the per-option
                                 score change explained by option length (regression slope)
  g4_correlation.csv       C7    per model: correlation between the per-seed change in
                                 |C_answer| and the change in T under the span erasure
  g4_energy_drift.csv      C9    per model and calibrated cell: dev vs test energy, and the
                                 regression of test energy on span length and prompt length

Usage: python g4_analyses.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import nn_common as C

log = C.log


def tmean_ci(vals: np.ndarray, clusters: np.ndarray) -> tuple[float, float, float, int]:
    """Cluster-weighted mean with a cluster bootstrap interval."""
    df = pd.DataFrame({"v": np.asarray(vals, float), "c": np.asarray(clusters)})
    g = df.groupby("c")["v"].mean().to_numpy()
    est, lo, hi = C.N.paired_interval(g, n_boot=C.N_BOOT, seed=C.RANDOM_SEED)
    return est, lo, hi, len(g)


def strata() -> pd.DataFrame:
    rows = []
    # pooled audit (raw logit C) from results/v2
    p = pd.read_parquet(C.V2 / "p2_per_item.parquet",
                        columns=["model_name", "phase", "pair_type", "cond_id", "seed_id", "absC", "span_len_A", "span_len_B"])
    p = p[(p.phase == "test") & (p.pair_type == "demographic") & (p.cond_id.isin(["unedited_r0_a0", "cure_centred_svd_r1_a1"]))]
    for m, g in p.groupby("model_name"):
        w = g.pivot_table(index=["seed_id", "span_len_A", "span_len_B"], columns="cond_id", values="absC").reset_index().dropna()
        w["D"] = w["unedited_r0_a0"] - w["cure_centred_svd_r1_a1"]
        w["unequal"] = w.span_len_A != w.span_len_B
        for lab, sub in (("equal", w[~w.unequal]), ("unequal", w[w.unequal]), ("all", w)):
            if len(sub) < 3:
                continue
            est, lo, hi, n = tmean_ci(sub.D.to_numpy(), sub.seed_id.to_numpy())
            rows.append({"study": "pooled_audit", "model_name": m, "quantity": "D = |C|_B - |C|_S1 (raw logit)", "stratum": lab,
                         "n_pairs": int(len(sub)), "n_clusters": n, "estimate": est, "lo": lo, "hi": hi,
                         "R_pct": 100 * (1 - sub["cure_centred_svd_r1_a1"].mean() / sub["unedited_r0_a0"].mean())})
    # fresh replication (C_answer) from results/final_20260914 per-item bridge rows
    for m in C.FRESH_MODELS:
        q = pd.read_parquet(C.FINAL / ("per_item_%s.parquet" % m))
        b = q[(q.kind == "bridge") & (q.status == "ok") & (q.group == "final") & (q.cond.isin(["B", "S1"]))]
        w = b.pivot_table(index=["seed_id", "template", "span_len_a", "span_len_b"], columns="cond", values="C_answer", aggfunc="first").reset_index().dropna()
        w["D"] = w["B"].abs() - w["S1"].abs(); w["unequal"] = w.span_len_a != w.span_len_b
        for lab, sub in (("equal", w[~w.unequal]), ("unequal", w[w.unequal]), ("all", w)):
            if len(sub) < 3:
                continue
            est, lo, hi, n = tmean_ci(sub.D.to_numpy(), sub.template.to_numpy())
            rows.append({"study": "fresh_replication", "model_name": m, "quantity": "|C_answer|_B - |C_answer|_S1", "stratum": lab,
                         "n_pairs": int(len(sub)), "n_clusters": n, "estimate": est, "lo": lo, "hi": hi,
                         "R_pct": 100 * (1 - sub["S1"].abs().mean() / sub["B"].abs().mean())})
    return pd.DataFrame(rows)


def T_from_scores(sa: list[float], sb: list[float]) -> float:
    from f1_scoring import softmax, total_variation
    return total_variation(softmax(sa), softmax(sb))


def length_decomposition() -> pd.DataFrame:
    rows = []
    for m in C.FRESH_MODELS:
        q = pd.read_parquet(C.FINAL / ("per_item_%s.parquet" % m))
        s = q[(q.kind == "score") & (q.status == "ok") & (q.group == "final") & (q.variant == "std")]
        man = pd.read_csv(C.FINAL / "source_manifest.csv").set_index("seed_id")
        base = s[s.cond == "B"].set_index(["seed_id", "side"])
        for cond in sorted(s.cond.unique()):
            g = s[s.cond == cond].set_index(["seed_id", "side"])
            recs = []
            for sid in g.index.get_level_values(0).unique():
                if (sid, "A") not in g.index or (sid, "B") not in g.index:
                    continue
                a, b = g.loc[(sid, "A")], g.loc[(sid, "B")]
                sa, sb = json.loads(a.s), json.loads(b.s); ma, mb = json.loads(a.s_mean), json.loads(b.s_mean)
                na, nb = json.loads(a.n_cand_tokens), json.loads(b.n_cand_tokens)
                gold = int(a.gold_index)
                # unknown-option contrast: the gold (unknown) option against the best entity option
                ua = sa[gold] - max(x for i, x in enumerate(sa) if i != gold); ub = sb[gold] - max(x for i, x in enumerate(sb) if i != gold)
                rec = {"seed_id": sid, "template": a.template, "T_raw": T_from_scores(sa, sb), "T_mean": T_from_scores(ma, mb),
                       "unknown_contrast_A": ua, "unknown_contrast_B": ub}
                if (sid, "A") in base.index and (sid, "B") in base.index and cond != "B":
                    ba, bb = base.loc[(sid, "A")], base.loc[(sid, "B")]
                    ds = np.asarray(sa) - np.asarray(json.loads(ba.s)); n = np.asarray(na, float)
                    # slope of the per-option score change on option length (3 options; a slope of zero means no length component)
                    slope = float(np.polyfit(n, ds, 1)[0]) if len(set(n.tolist())) > 1 else float("nan")
                    rec.update({"delta_s_slope_on_length_A": slope,
                                "delta_s_mean_A": float(ds.mean()), "delta_s_spread_A": float(ds.max() - ds.min())})
                recs.append(rec)
            d = pd.DataFrame(recs)
            if d.empty:
                continue
            out = {"model_name": m, "cond": cond, "n_seeds": int(len(d))}
            for col in ("T_raw", "T_mean"):
                est, lo, hi, _ = tmean_ci(d[col].to_numpy(), d.template.to_numpy()); out[col] = est; out[col + "_lo"] = lo; out[col + "_hi"] = hi
            est, lo, hi, _ = tmean_ci((d.unknown_contrast_A - d.unknown_contrast_B).abs().to_numpy(), d.template.to_numpy())
            out.update({"T_unknown_contrast_absdiff": est, "T_unknown_contrast_absdiff_lo": lo, "T_unknown_contrast_absdiff_hi": hi})
            if "delta_s_slope_on_length_A" in d:
                v = d.delta_s_slope_on_length_A.dropna()
                if len(v) > 2:
                    est, lo, hi, _ = tmean_ci(v.to_numpy(), d.loc[v.index, "template"].to_numpy())
                    out.update({"length_slope": est, "length_slope_lo": lo, "length_slope_hi": hi,
                                "share_seeds_negative_slope": float((v < 0).mean())})
            rows.append(out)
    return pd.DataFrame(rows)


def correlation() -> pd.DataFrame:
    rows = []
    for m in C.FRESH_MODELS:
        q = pd.read_parquet(C.FINAL / ("per_item_%s.parquet" % m))
        pl = pd.read_csv(C.FINAL / "pair_level.csv")
        pl = pl[(pl.model_name == m) & (pl.group == "final") & (pl.variant == "std")]
        dT = pl.pivot_table(index="seed_id", columns="cond", values="T")
        dT = (dT["S1"] - dT["B"]).rename("dT")
        b = q[(q.kind == "bridge") & (q.status == "ok") & (q.group == "final")].pivot_table(index="seed_id", columns="cond", values="C_answer", aggfunc="first")
        dC = (b["S1"].abs() - b["B"].abs()).rename("dC")
        j = pd.concat([dT, dC], axis=1).dropna()
        if len(j) > 3:
            r = float(np.corrcoef(j.dT, j.dC)[0, 1])
            from scipy.stats import spearmanr
            rho = float(spearmanr(j.dT, j.dC).correlation)
            rows.append({"model_name": m, "n_seeds": int(len(j)), "pearson_dT_dCanswer": r, "spearman_dT_dCanswer": rho,
                         "share_seeds_both_fall": float(((j.dT < 0) & (j.dC < 0)).mean()),
                         "share_seeds_C_falls_T_rises": float(((j.dC < 0) & (j.dT > 0)).mean())})
    return pd.DataFrame(rows)


def energy_drift() -> pd.DataFrame:
    rows = []
    for m in C.FRESH_MODELS:
        cal = C.read_json(C.FINAL / ("calibration_%s.json" % m))
        pl = pd.read_csv(C.FINAL / "pair_level.csv")
        pl = pl[(pl.model_name == m) & (pl.group == "final") & (pl.variant == "std")]
        sp = C.read_json(C.FINAL / ("spans_%s.json" % m))
        D = float(cal["denominator"])
        for cond in ("SE", "NE", "RE", "RNE"):
            g = pl[pl.cond == cond].copy()
            g["energy_rel"] = g.energy_num / D
            g["span_len"] = [len(sp["seeds"][s]["A"]["demo"]) + len(sp["seeds"][s]["B"]["demo"]) for s in g.seed_id]
            g["prompt_len"] = [sp["seeds"][s]["A"]["n_prefill"] + sp["seeds"][s]["B"]["n_prefill"] for s in g.seed_id]
            dev_e = cal["cells"][cond]["achieved"]
            X = np.column_stack([np.ones(len(g)), g.span_len, g.prompt_len]); y = g.energy_rel.to_numpy()
            beta = np.linalg.lstsq(X, y, rcond=None)[0] if len(g) > 3 else [np.nan] * 3
            rows.append({"model_name": m, "cond": cond, "alpha": cal["alphas"][cond], "dev_energy": dev_e, "test_energy_mean": float(y.mean()),
                         "test_over_dev": float(y.mean() / dev_e) if dev_e else float("nan"), "test_energy_cv": float(y.std() / y.mean()) if y.mean() else float("nan"),
                         "corr_energy_span_len": float(np.corrcoef(g.span_len, y)[0, 1]) if len(g) > 3 else float("nan"),
                         "corr_energy_prompt_len": float(np.corrcoef(g.prompt_len, y)[0, 1]) if len(g) > 3 else float("nan"),
                         "beta_span_len": float(beta[1]), "beta_prompt_len": float(beta[2]), "n_seeds": int(len(g)),
                         "mean_span_len_test": float(g.span_len.mean())})
    return pd.DataFrame(rows)


def main() -> None:
    C.ensure_out()
    for name, fn in (("g4_strata.csv", strata), ("g4_length.csv", length_decomposition), ("g4_correlation.csv", correlation),
                     ("g4_energy_drift.csv", energy_drift)):
        df = fn(); df.to_csv(C.OUT / name, index=False); log.info("wrote %s (%d rows)", name, len(df))


if __name__ == "__main__":
    main()
