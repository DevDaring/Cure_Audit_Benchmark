"""
g6_analysis.py -- statistics for the ICLR-feedback cycle (author's machine, CPU): merges the
per-model outputs of g1, g2, g3 and g5 with the final-cycle per-item rows and writes

  g_summary.csv    levels per (study, model, condition): T, generation accuracy, validity,
                   both-correct, |C_answer|, energy (template-weighted means, cluster intervals)
  g_tests.csv      paired differences with studentised cluster-bootstrap tests:
                     A1  D on C_raw / C_logp / C_margin (S1 vs B; S1 vs mean of the random
                         subspaces; each LEACE eraser vs B), seed clusters, Holm within model
                         across the three statistics of each comparison
                     A2  |C_answer| reduction on unequal-length pairs under each alignment rule
                     A3  fresh F<->M set: each eraser vs B on T, accuracy, |C_answer|
                     B   steering cells vs B on T, accuracy, both-correct, control accuracy;
                         template clusters
  g_leace_fits.csv the eraser descriptions of g3 (rows, ranks, probes, cosines)
  COMPLETION_G.txt the closing record

Usage: python g6_analysis.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import nn_common as C

log = C.log
N = C.N


def paired(diff: np.ndarray, clusters: np.ndarray, alternative: str = "two-sided") -> dict:
    df = pd.DataFrame({"d": np.asarray(diff, float), "c": np.asarray(clusters)})
    g = df.groupby("c")["d"].mean().to_numpy()
    t = N.studentized_paired_test(g, n_boot=C.N_BOOT, seed=C.RANDOM_SEED, alternative=alternative)
    return {"n_pairs": int(len(df)), "n_clusters": int(len(g)), "estimate": t["mean"], "lo": t["lo"], "hi": t["hi"], "p": t["p"],
            "degenerate": t["degenerate"]}


def level(vals: np.ndarray, clusters: np.ndarray) -> dict:
    df = pd.DataFrame({"v": np.asarray(vals, float), "c": np.asarray(clusters)})
    g = df.groupby("c")["v"].mean().to_numpy()
    est, lo, hi = N.paired_interval(g, n_boot=C.N_BOOT, seed=C.RANDOM_SEED)
    return {"n": int(len(df)), "n_clusters": int(len(g)), "estimate": est, "lo": lo, "hi": hi}


# ---------------------------------------------------------------- A1
def a1(tests: list, summary: list) -> None:
    for m in C.ALL_MODELS:
        p = C.OUT / ("g1_invariant_%s.parquet" % m)
        if not p.exists():
            continue
        d = pd.read_parquet(p); d = d[d.status == "ok"]
        w = {stat: d.pivot_table(index="seed_id", columns="cond_id", values=stat) for stat in ("C_raw", "C_logp", "C_margin")}
        # reproduction of the pooled audit's stored C on the same pairs: bf16 logits are quantised at
        # 0.125-0.25 for logit magnitudes of 16-64, so a different GPU reproduces C only to that
        # resolution (qwen: exact; gemma/phi: differences are multiples of 0.125). Report the
        # correlation and the mean-level agreement, not an exact-match share.
        b = d[d.cond_id.isin(["unedited_r0_a0", "cure_centred_svd_r1_a1"]) & np.isfinite(d.C_stored)]
        if len(b) > 3:
            summary.append({"study": "A1", "model_name": m, "condition": "-", "quantity": "reproduction: corr(C_raw, stored C)", "estimate": float(np.corrcoef(b.C_raw, b.C_stored)[0, 1])})
            summary.append({"study": "A1", "model_name": m, "condition": "-", "quantity": "reproduction: median |C_raw - stored C|", "estimate": float((b.C_raw - b.C_stored).abs().median())})
            for c in ("unedited_r0_a0", "cure_centred_svd_r1_a1"):
                x = b[b.cond_id == c]
                summary.append({"study": "A1", "model_name": m, "condition": c, "quantity": "reproduction: mean|C_raw| / mean|stored C|", "estimate": float(x.C_raw.abs().mean() / x.C_stored.abs().mean()) if x.C_stored.abs().mean() else float("nan")})
        rnd = [c for c in w["C_raw"].columns if c.startswith("random_ortho")]
        comps = [("S1_vs_B", "cure_centred_svd_r1_a1", "unedited_r0_a0"), ("S1_vs_random_mean", "cure_centred_svd_r1_a1", "RANDOM_MEAN")]
        for c in w["C_raw"].columns:
            if c.startswith("leace"):
                comps.append(("%s_vs_B" % c.replace("_r1_a1", ""), c, "unedited_r0_a0"))
        for name, post, pre in comps:
            fam = []
            for stat in ("C_raw", "C_logp", "C_margin"):
                x = w[stat]
                if post not in x.columns:
                    continue
                pre_v = x[rnd].abs().mean(axis=1) if pre == "RANDOM_MEAN" else x[pre].abs()
                j = pd.concat([pre_v.rename("pre"), x[post].abs().rename("post")], axis=1).dropna()
                if pre == "RANDOM_MEAN":
                    # D = mean|C_random| - |C_S1| (positive = the erasure removes more than random)
                    diff = (j.pre - j.post).to_numpy()
                else:
                    diff = (j.pre - j.post).to_numpy()
                r = paired(diff, j.index.to_numpy())
                r.update({"study": "A1", "model_name": m, "comparison": name, "statistic": stat,
                          "R_pct": 100 * (1 - j.post.mean() / j.pre.mean()) if j.pre.mean() else float("nan")})
                fam.append(r)
            if fam:
                adj = N.holm([f["p"] for f in fam])
                for f, a in zip(fam, adj):
                    f["p_holm_within_comparison"] = a
                tests += fam
        # the confound itself
        for c in ("cure_centred_svd_r1_a1",):
            s = d[d.cond_id == c]
            summary.append({"study": "A1", "model_name": m, "condition": c, "quantity": "mean logit shift (patched - clean, mean over vocab)",
                            **level(s.mean_logit_shift.to_numpy(), s.seed_id.to_numpy())})
            summary.append({"study": "A1", "model_name": m, "condition": c, "quantity": "log-partition shift LSE(patched) - LSE(clean)",
                            **level(s.logZ_shift.to_numpy(), s.seed_id.to_numpy())})
        for c in sorted(d.cond_id.unique()):
            s = d[d.cond_id == c]
            for stat in ("C_raw", "C_logp", "C_margin"):
                summary.append({"study": "A1", "model_name": m, "condition": c, "quantity": "mean |%s|" % stat, **level(s[stat].abs().to_numpy(), s.seed_id.to_numpy())})


# ---------------------------------------------------------------- A2
def a2(tests: list, summary: list) -> None:
    for m in C.FRESH_MODELS:
        p = C.OUT / ("g2_alignment_%s.parquet" % m)
        if not p.exists():
            continue
        d = pd.read_parquet(p); d = d[d.status == "ok"]
        for rule in sorted(d.rule.unique()):
            x = d[d.rule == rule].pivot_table(index=["seed_id", "template"], columns="cond", values="C_answer").reset_index().dropna()
            if len(x) < 3:
                continue
            r = paired((x.B.abs() - x.S1.abs()).to_numpy(), x.template.to_numpy())
            r.update({"study": "A2", "model_name": m, "comparison": "|C_answer|_B - |C_answer|_S1 (unequal-length pairs)", "statistic": rule,
                      "R_pct": 100 * (1 - x.S1.abs().mean() / x.B.abs().mean())})
            tests.append(r)
            for cond in ("B", "S1"):
                summary.append({"study": "A2", "model_name": m, "condition": "%s/%s" % (cond, rule), "quantity": "mean |C_answer| on unequal-length pairs",
                                **level(x[cond].abs().to_numpy(), x.template.to_numpy())})


# ---------------------------------------------------------------- A3
def a3(tests: list, summary: list, fits: list) -> None:
    for m in C.ALL_MODELS:
        rec = C.read_json(C.OUT / ("g3_leace_%s.json" % m))
        if not rec:
            continue
        for part in ("part_a", "part_b"):
            for tag, v in (rec.get(part) or {}).items():
                if not isinstance(v, dict) or "per_layer" not in v:
                    continue
                for pl in v["per_layer"]:
                    fits.append({"model_name": m, "part": part, "eraser": tag, "n_rows_fit": v["n_rows_fit"], "n_rows_heldout": v["n_rows_heldout"],
                                 "n_pairs": v.get("n_pairs"), "shrinkage": v.get("shrinkage", True), "d": v["d"], **pl})
        p = C.OUT / ("g3_rows_%s.parquet" % m)
        if not p.exists():
            continue
        d = pd.read_parquet(p); d = d[d.status == "ok"]
        sc = d[d.kind == "score"]
        # T per seed per condition
        recs = []
        for (sid, cond), g in sc.groupby(["seed_id", "cond"]):
            if set(g.side) != {"A", "B"}:
                continue
            a, b = g[g.side == "A"].iloc[0], g[g.side == "B"].iloc[0]
            from f1_scoring import softmax, total_variation
            recs.append({"seed_id": sid, "template": a.template, "cond": cond, "T": total_variation(json.loads(a.q), json.loads(b.q)),
                         "acc": 0.5 * (float(a.gen_correct) + float(b.gen_correct)), "both": float(bool(a.gen_correct) and bool(b.gen_correct)),
                         "validity": 0.5 * (float(a.gen_valid) + float(b.gen_valid)), "energy": float(a.energy_num) + float(b.energy_num)})
        pl = pd.DataFrame(recs)
        if pl.empty:
            continue
        for cond in sorted(pl.cond.unique()):
            g = pl[pl.cond == cond]
            for q in ("T", "acc", "both", "validity", "energy"):
                summary.append({"study": "A3_fm_eval", "model_name": m, "condition": cond, "quantity": q, **level(g[q].to_numpy(), g.template.to_numpy())})
            if cond != "B":
                x = pl[pl.cond.isin(["B", cond])].pivot_table(index=["seed_id", "template"], columns="cond", values=["T", "acc"]).reset_index().dropna()
                if len(x) > 3:
                    for q in ("T", "acc"):
                        r = paired((x[(q, "B")] - x[(q, cond)]).to_numpy(), x["template"].to_numpy())
                        r.update({"study": "A3_fm_eval", "model_name": m, "comparison": "%s: B - %s" % (q, cond), "statistic": q}); tests.append(r)
        br = d[d.kind == "bridge"].pivot_table(index=["seed_id", "template"], columns="cond", values="C_answer", aggfunc="first").reset_index()
        for cond in [c for c in br.columns if c not in ("seed_id", "template", "B")]:
            x = br[["seed_id", "template", "B", cond]].dropna()
            if len(x) > 3:
                r = paired((x.B.abs() - x[cond].abs()).to_numpy(), x.template.to_numpy())
                r.update({"study": "A3_fm_eval", "model_name": m, "comparison": "|C_answer|: B - %s" % cond, "statistic": "C_answer",
                          "R_pct": 100 * (1 - x[cond].abs().mean() / x.B.abs().mean())}); tests.append(r)


# ---------------------------------------------------------------- B
def b_steering(tests: list, summary: list) -> None:
    from f1_scoring import softmax, total_variation
    for m in C.FRESH_MODELS:
        p = C.OUT / ("g5_steering_%s.parquet" % m)
        if not p.exists():
            continue
        d = pd.read_parquet(p); d = d[d.status == "ok"]
        base = pd.read_parquet(C.FINAL / ("per_item_%s.parquet" % m))
        base = base[(base.kind == "score") & (base.status == "ok") & (base.variant == "std") & (base.cond.isin(["B", "S1", "SE"]))]
        allsc = pd.concat([d[d.kind == "score"], base], ignore_index=True)
        recs = []
        for (grp, sid, cond), g in allsc.groupby(["group", "seed_id", "cond"]):
            if set(g.side) != {"A", "B"}:
                continue
            a, b = g[g.side == "A"].iloc[0], g[g.side == "B"].iloc[0]
            recs.append({"group": grp, "seed_id": sid, "template": a.template, "cond": cond, "T": total_variation(json.loads(a.q), json.loads(b.q)),
                         "acc": 0.5 * (float(a.gen_correct) + float(b.gen_correct)), "both": float(bool(a.gen_correct) and bool(b.gen_correct)),
                         "validity": 0.5 * (float(a.gen_valid) + float(b.gen_valid)), "energy": float(a.energy_num) + float(b.energy_num)})
        pl = pd.DataFrame(recs)
        for grp in ("final", "control"):
            q = pl[pl.group == grp]
            for cond in sorted(q.cond.unique()):
                g = q[q.cond == cond]
                for qty in ("T", "acc", "both", "validity", "energy"):
                    summary.append({"study": "B_%s" % grp, "model_name": m, "condition": cond, "quantity": qty, **level(g[qty].to_numpy(), g.template.to_numpy())})
                if cond not in ("B",):
                    x = q[q.cond.isin(["B", cond])].pivot_table(index=["seed_id", "template"], columns="cond", values=["T", "acc", "both"]).reset_index().dropna()
                    if len(x) > 3:
                        for qty in ("T", "acc", "both"):
                            r = paired((x[(qty, "B")] - x[(qty, cond)]).to_numpy(), x["template"].to_numpy())
                            r.update({"study": "B_%s" % grp, "model_name": m, "comparison": "%s: B - %s" % (qty, cond), "statistic": qty}); tests.append(r)
        # bridge: steering at the span vs B
        brs = pd.concat([d[d.kind == "bridge"], pd.read_parquet(C.FINAL / ("per_item_%s.parquet" % m)).query("kind == 'bridge' and status == 'ok' and group == 'final'")], ignore_index=True)
        br = brs.pivot_table(index=["seed_id", "template"], columns="cond", values="C_answer", aggfunc="first").reset_index()
        for cond in [c for c in br.columns if c not in ("seed_id", "template", "B")]:
            x = br[["seed_id", "template", "B", cond]].dropna()
            if len(x) > 3:
                r = paired((x.B.abs() - x[cond].abs()).to_numpy(), x.template.to_numpy())
                r.update({"study": "B_final", "model_name": m, "comparison": "|C_answer|: B - %s" % cond, "statistic": "C_answer",
                          "R_pct": 100 * (1 - x[cond].abs().mean() / x.B.abs().mean())}); tests.append(r)


def main() -> None:
    C.ensure_out()
    tests, summary, fits = [], [], []
    a1(tests, summary); a2(tests, summary); a3(tests, summary, fits); b_steering(tests, summary)
    pd.DataFrame(tests).to_csv(C.OUT / "g_tests.csv", index=False)
    pd.DataFrame(summary).to_csv(C.OUT / "g_summary.csv", index=False)
    pd.DataFrame(fits).to_csv(C.OUT / "g_leace_fits.csv", index=False)
    lines = ["COMPLETION record (feedback cycle) %s" % C.utc_now(), "tests: %d rows; summary: %d rows; leace fits: %d rows" % (len(tests), len(summary), len(fits)), ""]
    for t in tests:
        lines.append("%-12s %-24s %-58s %-9s est %+.4f [%+.4f, %+.4f] p %.4f%s" % (
            t["study"], t["model_name"], t["comparison"][:58], t.get("statistic", ""), t["estimate"], t["lo"], t["hi"], t["p"],
            ("  p_holm %.4f" % t["p_holm_within_comparison"]) if "p_holm_within_comparison" in t else ""))
    (C.OUT / "COMPLETION_G.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("g6: %d tests, %d summary rows", len(tests), len(summary))


if __name__ == "__main__":
    main()
