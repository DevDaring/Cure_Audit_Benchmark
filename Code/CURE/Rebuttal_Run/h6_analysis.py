"""
h6_analysis.py -- the declared tests of the rebuttal cycle (h0_protocol.py), author's machine, CPU.

Writes to results/rebuttal_20261107/
  h_levels.csv     per (model, group, cond): T, T_mean, option and generation accuracy, validity,
                   both-correct, energy fraction of E_S1, feasible share, |C_answer|; cluster intervals
  h_tests.csv      every declared test: estimate, interval, p, Holm p, verdict
  h_k5.csv         the selection analysis per model and statistic
  h_k6.csv         the rewrite comparison per model
  h_k8_tokens.csv  attribution of the change in T to the option's tokens
  COMPLETION_H.txt the closing record

Statistics (Efron and Tibshirani 1993; Holm 1979): studentised bootstrap over clusters (BBQ template
on the fresh and gender sets, seed on the pooled pairs), 10,000 resamples, Holm within each named
family and model. Paired tests use this cycle's B rows only.

Usage: python h6_analysis.py [--dir results/rebuttal_20261107]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import hh_common as H

log = H.log
M_T, M_ACC, M_C = 0.02, 0.02, 0.1       # the declared margins (h0_protocol)
READOUTS = ("T", "T_mean")


def verdict(est: float, p_adj: float, alternative: str = "greater") -> str:
    if not np.isfinite(p_adj):
        return "not testable"
    if p_adj < 0.05:
        return "supported"
    return "limited"


def tagged(tests: list, rows: list[dict]) -> None:
    """Holm over the family, then verdicts."""
    adj = H.N.holm([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = a; r["verdict"] = verdict(r["estimate"], a)
    tests.extend(rows)


def diff_test(wide: pd.DataFrame, plus: str, minus: str, shift: float = 0.0, alternative: str = "greater") -> dict | None:
    if plus not in wide.columns or minus not in wide.columns:
        return None
    j = wide[[plus, minus, "template"]].dropna()
    if len(j) < 3:
        return None
    r = H.paired((j[plus] - j[minus] + shift).to_numpy(), j["template"].to_numpy(), alternative=alternative)
    r.update({"plus": plus, "minus": minus, "shift": shift, "mean_plus": float(j[plus].mean()), "mean_minus": float(j[minus].mean())})
    return r


def wide_of(pt: pd.DataFrame, value: str) -> pd.DataFrame:
    w = pt.pivot_table(index="seed_id", columns="cond", values=value, aggfunc="first")
    w["template"] = pt.drop_duplicates("seed_id").set_index("seed_id").template
    return w


def bridge_wide(d: pd.DataFrame) -> pd.DataFrame:
    br = d[(d.kind == "bridge") & (d.status == "ok")].copy()
    if br.empty:
        return pd.DataFrame()
    br["absC"] = br.C_answer.abs()
    w = br.pivot_table(index="seed_id", columns="cond", values="absC", aggfunc="first")
    w["template"] = br.drop_duplicates("seed_id").set_index("seed_id").template
    return w


# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------

def levels_of(model: str, d: pd.DataFrame, pt: pd.DataFrame, study: str) -> list[dict]:
    out = []
    frac = d[(d.kind == "score") & (d.status == "ok")].assign(x=lambda v: v.energy_per_pass / v.E_S1_side)
    for (grp, cond), g in pt.groupby(["group", "cond"]):
        rec = {"study": study, "model_name": model, "group": grp, "cond": cond, "n_pairs": len(g), "feasible_share": float(g.feasible.mean())}
        fr = frac[(frac.group == grp) & (frac.cond == cond)].x.dropna()
        rec["energy_fraction_median"] = float(fr.median()) if len(fr) else np.nan
        for q in ("T", "T_mean", "score_acc", "acc", "both", "validity"):
            v = g[q].dropna()
            if len(v) >= 2:
                L = H.level(g[q].to_numpy(), g.template.to_numpy())
                rec.update({q: L["estimate"], q + "_lo": L["lo"], q + "_hi": L["hi"]})
        out.append(rec)
    br = d[(d.kind == "bridge") & (d.status == "ok")]
    for cond, g in br.groupby("cond"):
        L = H.level(g.C_answer.abs().to_numpy(), g.template.to_numpy())
        out.append({"study": study, "model_name": model, "group": "final", "cond": cond, "n_pairs": len(g), "abs_C_answer": L["estimate"], "abs_C_answer_lo": L["lo"], "abs_C_answer_hi": L["hi"]})
    return out


# ---------------------------------------------------------------------------
# K1 K2 K3 K7 K8 on the fresh replication
# ---------------------------------------------------------------------------

def fresh_tests(model: str, d: pd.DataFrame, pt: pd.DataFrame, tests: list) -> None:
    final = pt[pt.group == "final"]; ctl = pt[pt.group == "control"]
    base = {"model_name": model}
    for readout in READOUTS:
        w = wide_of(final, readout)
        # K1 at f = 1: the span erasure matched to the energy each site actually achieved
        fam = []
        for other, span in (("P_ne@1", "P_span@E(P_ne)"), ("P_last@1", "P_span@E(P_last)"), ("P_all@1", "P_span@1")):
            r = diff_test(w, other, span)
            if r:
                fam.append({**base, **r, "hypothesis": "K1_f1", "readout": readout, "statement": "%s - %s > 0" % (other, span)})
        tagged(tests, fam)
        fam = []
        for other, span in (("P_ne@0.1", "P_span@0.1"), ("P_last@0.1", "P_span@0.1"), ("P_all@0.1", "P_span@0.1")):
            r = diff_test(w, other, span)
            if r:
                fam.append({**base, **r, "hypothesis": "K1_f0.1", "readout": readout, "statement": "%s - %s > 0" % (other, span)})
        tagged(tests, fam)
        # K3
        r = diff_test(w, "P_span@1", "CV_span@1", shift=M_T)
        if r:
            tagged(tests, [{**base, **r, "hypothesis": "K3", "readout": readout, "statement": "CV_span@1 non-inferior to P_span@1 within %g" % M_T}])
        r = diff_test(w, "P_span@1", "CV_span@1", alternative="two-sided")
        if r:
            tagged(tests, [{**base, **r, "hypothesis": "K3_difference", "readout": readout, "statement": "P_span@1 - CV_span@1 (two-sided, descriptive)"}])
        # K7 (all four reported; out-of-sample only on the two replication models)
        r = diff_test(w, "B", "P_span@1")
        if r:
            hyp = "K7_out_of_sample" if model in H.REPLICATION_MODELS else "H1_follow_up"
            tests.append({**base, **r, "hypothesis": hyp, "readout": readout, "statement": "B - P_span@1 > 0", "p_holm": np.nan, "verdict": ""})
    # K1 on the patching effect (secondary) and K7's second endpoint
    bw = bridge_wide(d)
    if len(bw):
        fam = []
        for other, span in (("P_ne@1", "P_span@E(P_ne)"), ("P_last@1", "P_span@E(P_last)"), ("P_all@1", "P_span@1")):
            r = diff_test(bw, other, span)
            if r:
                fam.append({**base, **r, "hypothesis": "K1_f1", "readout": "abs_C_answer", "statement": "%s - %s > 0" % (other, span)})
        tagged(tests, fam)
        r = diff_test(bw, "B", "P_span@1")
        if r:
            hyp = "K7_out_of_sample" if model in H.REPLICATION_MODELS else "H1_follow_up"
            tests.append({**base, **r, "hypothesis": hyp, "readout": "abs_C_answer", "statement": "B - P_span@1 > 0", "p_holm": np.nan, "verdict": ""})
    # Holm over K7's two endpoints within model (T and C_answer), T_mean reported beside them
    fam = [t for t in tests if t.get("model_name") == model and t["hypothesis"] in ("K7_out_of_sample", "H1_follow_up") and t["readout"] in ("T", "abs_C_answer")]
    adj = H.N.holm([t["p"] for t in fam])
    for t, a in zip(fam, adj):
        t["p_holm"] = a; t["verdict"] = verdict(t["estimate"], a)
    # K2: against the UNEDITED model
    wa = wide_of(final, "acc")
    r = diff_test(wa, "P_all@1", "B", shift=M_ACC)
    if r:
        tagged(tests, [{**base, **r, "hypothesis": "K2", "readout": "acc", "statement": "P_all@1 non-inferior to B within %g" % M_ACC}])
    r = diff_test(wa, "B", "P_all@a1")
    if r:
        tagged(tests, [{**base, **r, "hypothesis": "K2", "readout": "acc", "statement": "B - P_all@a1 > 0 (equal strength damages)"}])
    for cond in ("P_all@0.1", "P_all@1", "P_all@10", "P_all@a1"):
        r = diff_test(wa, "B", cond, alternative="two-sided")
        if r:
            tagged(tests, [{**base, **r, "hypothesis": "K2_dose", "readout": "acc", "statement": "B - %s (descriptive)" % cond}])
    # control set, descriptive
    if len(ctl):
        wc = wide_of(ctl, "acc")
        for cond in ("P_span@1", "P_span@0.3", "P_span@0.1", "CV_span@1", "CV_span@0.3", "CV_span@0.1", "P_ne@1", "P_all@1", "R_all@1"):
            r = diff_test(wc, "B", cond, alternative="two-sided")
            if r:
                tagged(tests, [{**base, **r, "hypothesis": "control_accuracy", "readout": "acc", "statement": "B - %s (descriptive)" % cond}])


# ---------------------------------------------------------------------------
# K4 on the gender-flip set
# ---------------------------------------------------------------------------

def gender_tests(model: str, d: pd.DataFrame, tests: list) -> None:
    pt = H.pair_table(d[d.kind == "score"])
    base = {"model_name": model}
    baselines = ("LEACE_gl", "MDg_span", "MDf3_span", "LEACE_f3")
    for readout, margin in (("T", M_T), ("T_mean", M_T), ("abs_C_answer", M_C)):
        w = wide_of(pt, readout) if readout != "abs_C_answer" else bridge_wide(d)
        if not len(w):
            continue
        fam = []
        for b in baselines:
            native, matched = "%s@native" % b, "P_span@E(%s)" % b
            r = diff_test(w, matched, native, shift=margin)
            if r:
                r["erasure_minus_baseline"] = r["mean_plus"] - r["mean_minus"]
                fam.append({**base, **r, "hypothesis": "K4", "readout": readout, "statement": "the erasure at %s's energy is not lower than it by more than %g" % (b, margin)})
        tagged(tests, fam)


# ---------------------------------------------------------------------------
# K5 selection
# ---------------------------------------------------------------------------

def selection(model: str, d: pd.DataFrame, tests: list, k5: list) -> None:
    d = d[d.status == "ok"]
    base = {"model_name": model}
    fam = []
    for stat in ("C_raw", "C_logp", "C_margin"):
        w = d.pivot_table(index=["seed_id", "subvariant_A", "subvariant_B"], columns="cond_id", values=stat, aggfunc="first").dropna()
        if "cure_centred_svd_r1_a1" not in w or "unedited_r0_a0" not in w or w.empty:
            continue
        x = (w["unedited_r0_a0"].abs() - w["cure_centred_svd_r1_a1"].abs()).rename("D").reset_index()
        meta = d.drop_duplicates(["seed_id", "subvariant_A", "subvariant_B"])[["seed_id", "subvariant_A", "subvariant_B", "is_selected_pair", "rank_by_Cpre"]]
        x = x.merge(meta, on=["seed_id", "subvariant_A", "subvariant_B"])
        pre = w["unedited_r0_a0"].abs().mean()
        r = H.paired(x.D.to_numpy(), x.seed_id.to_numpy(), alternative="greater")
        r.update({"hypothesis": "K5", "readout": stat, "statement": "D > 0 on every eligible pair"})
        if stat in ("C_raw", "C_logp"):
            fam.append({**base, **r})
        else:
            tests.append({**base, **r, "p_holm": np.nan, "verdict": "descriptive"})
        sel = x[x.is_selected_pair]; rest = x[~x.is_selected_pair]
        k5.append({"model_name": model, "statistic": stat, "n_pairs": len(x), "n_seeds": x.seed_id.nunique(), "D_all": x.D.mean(), "R_all_pct": 100 * x.D.mean() / pre if pre else np.nan,
                   "D_selected": sel.D.mean(), "D_unselected": rest.D.mean(), "ratio_all_over_selected": x.D.mean() / sel.D.mean() if sel.D.mean() else np.nan,
                   "share_lower": float((x.D > 1e-9).mean()), "share_higher": float((x.D < -1e-9).mean()), "D_q25": x.D.quantile(.25), "D_q50": x.D.quantile(.5), "D_q75": x.D.quantile(.75),
                   "top10pct_share_of_net": x.D.sort_values(ascending=False).head(max(1, len(x) // 10)).sum() / x.D.sum() if x.D.sum() else np.nan})
    tagged(tests, fam)


# ---------------------------------------------------------------------------
# K6 rewrite
# ---------------------------------------------------------------------------

def rewrite(model: str, h4: pd.DataFrame, h1: pd.DataFrame, k6: list) -> None:
    ok = h4[(h4.kind == "score") & (h4.status == "ok")]
    pt = H.pair_table(ok)
    rec = {"model_name": model}
    q1 = h1[(h1.kind == "score") & (h1.status == "ok") & (h1.cond == "P_span@1")].set_index(["seed_id", "side"]).q
    qb = ok[ok.cond == "B"].set_index(["seed_id", "side"]).q
    qr = ok[ok.cond == "RW"]
    for grp in ("final", "control"):
        p = pt[pt.group == grp]
        for cond in ("B", "RW"):
            g = p[p.cond == cond]
            if len(g):
                rec.update({"%s_%s_acc" % (grp, cond): g.acc.mean(), "%s_%s_T" % (grp, cond): g["T"].mean(), "%s_%s_validity" % (grp, cond): g.validity.mean(), "%s_%s_n" % (grp, cond): len(g)})
        x = h1[(h1.kind == "score") & (h1.status == "ok")]
        y = H.pair_table(x[x.cond == "P_span@1"])
        y = y[y.group == grp]
        rec["%s_S1_acc" % grp] = y.acc.mean(); rec["%s_S1_T" % grp] = y["T"].mean()
        tv_e, tv_b = [], []
        for r in qr[qr.group == grp].itertuples(index=False):
            k = (r.seed_id, r.side)
            if k in q1.index and k in qb.index:
                qq = np.asarray(json.loads(r.q)); tv_e.append(0.5 * np.abs(np.asarray(json.loads(q1[k])) - qq).sum()); tv_b.append(0.5 * np.abs(np.asarray(json.loads(qb[k])) - qq).sum())
        rec["%s_TV_edit_to_rewrite" % grp] = float(np.mean(tv_e)) if tv_e else np.nan
        rec["%s_TV_unedited_to_rewrite" % grp] = float(np.mean(tv_b)) if tv_b else np.nan
    rec["share_pairs_identical_after_rewrite"] = float(qr.identical_after_rewrite.mean()) if len(qr) else np.nan
    k6.append(rec)


# ---------------------------------------------------------------------------
# K8 token attribution
# ---------------------------------------------------------------------------

def tv(q1, q2) -> float:
    return float(0.5 * np.abs(q1 - q2).sum())


def token_attribution(model: str, d: pd.DataFrame, k8: list) -> None:
    """For each pair, T under the unedited scores with only ONE component's change from the edit
    applied: the descriptor's tokens, the shared JSON wrapper, the option's other tokens. Reported
    for raw-sum and for per-token-mean scores."""
    import f1_scoring as S
    n_pre = None

    def parts(row):
        toks, lps, masks = json.loads(row.tokens), json.loads(row.lp), json.loads(row.desc_mask)
        out = []
        for t, l, m in zip(toks, lps, masks):
            text = "".join(t)
            a = len(S.PREFIX); b = len(text) - len(S.SUFFIX)
            pos = 0; comp = []
            for tk, mk in zip(t, m):
                mid = pos + len(tk) / 2.0
                comp.append("desc" if mk else ("json" if (mid < a or mid >= b) else "other"))
                pos += len(tk)
            out.append((np.asarray(l, float), comp))
        return out

    for cond in ("P_span@1", "CV_span@1", "P_ne@1"):
        base = d[d.cond == "B"].set_index(["seed_id", "side"]); ed = d[d.cond == cond].set_index(["seed_id", "side"])
        res = {k: [] for k in ("T_B", "T_edit", "T_desc", "T_json", "T_other", "Tm_B", "Tm_edit", "Tm_desc", "Tm_json", "Tm_other", "d_desc_ent1", "d_desc_all")}
        for sid in sorted(set(base.index.get_level_values(0)) & set(ed.index.get_level_values(0))):
            if not all(((sid, s) in base.index and (sid, s) in ed.index) for s in ("A", "B")):
                continue
            q = {}
            for s in ("A", "B"):
                pb, pe = parts(base.loc[(sid, s)]), parts(ed.loc[(sid, s)])
                sums, means = {}, {}
                for key in ("B", "edit", "desc", "json", "other"):
                    vs, vm = [], []
                    for (lb, comp), (le, _) in zip(pb, pe):
                        if key == "B":
                            v = lb
                        elif key == "edit":
                            v = le
                        else:
                            v = np.array([e if c == key else b for b, e, c in zip(lb, le, comp)])
                        vs.append(v.sum()); vm.append(v.mean())
                    sums[key], means[key] = H.softmax_np(vs), H.softmax_np(vm)
                q[s] = (sums, means)
            for key in ("B", "edit", "desc", "json", "other"):
                res["T_" + key].append(tv(q["A"][0][key], q["B"][0][key])); res["Tm_" + key].append(tv(q["A"][1][key], q["B"][1][key]))
        if res["T_B"]:
            k8.append({"model_name": model, "cond": cond, "n_pairs": len(res["T_B"]), **{k: float(np.mean(v)) for k, v in res.items() if v}})


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    a = ap.parse_args()
    out = Path(a.dir) if a.dir else H.OUT
    tests, levels, k5, k6, k8 = [], [], [], [], []
    for model in H.ALL_MODELS:
        p = out / ("h1_dose_%s.parquet" % model)
        if p.exists():
            d = H.read_rows(p)
            pt = H.pair_table(d[d.kind == "score"])
            levels += levels_of(model, d, pt, "h1")
            fresh_tests(model, d, pt, tests)
            q = out / ("h4_rewrite_%s.parquet" % model)
            if q.exists():
                rewrite(model, H.read_rows(q), d, k6)
        q = out / ("h2_gender_%s.parquet" % model)
        if q.exists():
            g = H.read_rows(q)
            levels += levels_of(model, g, H.pair_table(g[g.kind == "score"]), "h2")
            gender_tests(model, g, tests)
        q = out / ("h3_pairs_%s.parquet" % model)
        if q.exists():
            selection(model, H.read_rows(q), tests, k5)
        q = out / ("h7_tokens_%s.parquet" % model)
        if q.exists():
            token_attribution(model, H.read_rows(q), k8)
    pd.DataFrame(levels).to_csv(out / "h_levels.csv", index=False)
    td = pd.DataFrame(tests); td.to_csv(out / "h_tests.csv", index=False)
    pd.DataFrame(k5).to_csv(out / "h_k5.csv", index=False)
    pd.DataFrame(k6).to_csv(out / "h_k6.csv", index=False)
    pd.DataFrame(k8).to_csv(out / "h_k8_tokens.csv", index=False)
    lines = ["COMPLETION record (rebuttal cycle) %s; declared protocol hash %s" % (H.utc_now(), (H.read_json(H.OUT / "h_protocol.json", {}) or {}).get("protocol_hash")), ""]
    for t in tests:
        lines.append("%-14s %-22s %-13s %-60s est %+.4f [%+.4f, %+.4f] n=%d p %.4f p_holm %s %s" % (t["hypothesis"], t["model_name"], t["readout"], str(t["statement"])[:60], t["estimate"], t["lo"], t["hi"], t["n_pairs"], t["p"],
                     ("%.4f" % t["p_holm"]) if np.isfinite(t.get("p_holm", np.nan)) else "-", t.get("verdict", "")))
    (out / "COMPLETION_H.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
