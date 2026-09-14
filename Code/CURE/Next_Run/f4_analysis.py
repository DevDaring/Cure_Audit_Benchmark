"""
f4_analysis.py -- the analysis fixed before the final run (Next_Plan.md Sections 9, 10, 11).
CPU only; reads results/final_20260914/per_item_<model>.parquet and the calibration /
budget / baseline files, writes the confirmatory, preservation and secondary tables, the
claims record, number provenance and COMPLETION.txt.

Sampling unit: the source seed, kept whole in every resample; seeds of one BBQ template are
additionally grouped, and the primary tests resample TEMPLATE families (the more conservative
clustering, Section 9 "retain that grouping"); seed-level results are reported next to them.

Confirmatory family (eight tests, Holm at 0.05), per model:
  H1  mean(T_B  - T_S1)      operational sensitivity benefit
  H2  mean(T_NE - T_SE)      position specificity at calibrated energy
  H3  mean(T_RE - T_SE)      direction specificity at the demographic span
  H4  mean(acc_S1 - acc_B)   generated-answer improvement (attempted denominator, both sides)
Null-centred studentized paired bootstrap of the cluster-level differences, 10,000 resamples,
two-sided, finite-resampling correction. Degenerate samples are reported, never rejected.

Preservation family (six one-sided non-inferiority tests, Holm): for S1, SE and NE on each
model, H0: acc_cond - acc_B <= -margin (2 percentage points, frozen in final_protocol.json).

Secondary (estimates and intervals, no confirmatory status): direction-by-position
interaction (T_RNE - T_NE) - (T_RE - T_SE); relevant-information control accuracy under B, S1,
NE; both-correct rates; G1 effects on T and accuracy; C_answer and C_first for B and S1;
subgroup (category) results for H1 and H4; coherent-label LC / MC on the eligible subset;
the 32-seed scoring-sensitivity variants; achieved energies and the 10% caution flag;
validity and the parser-only view of every accuracy quantity.

Outcome rule (Section 10) is applied mechanically to write claims.json: each claim carries its
endpoint, model, population, estimate, interval, family, adjusted p, status in
{supported, limited, unsupported, not_measured} and permitted wording.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import nr_common as N
import f1_scoring as S

log = N.setup_logging("cure.final.f4")
OUT = N.FINAL


def load_rows() -> pd.DataFrame:
    parts = []
    for m in N.FINAL_MODELS:
        p = OUT / ("per_item_%s.parquet" % m)
        if p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        raise SystemExit("no per_item files under %s" % OUT)
    d = pd.concat(parts, ignore_index=True)
    return d


def pairwise_T(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, seed, cond, variant): T from the two sides' q vectors, plus the
    side-mean accuracy quantities and energies."""
    sc = d[(d.kind == "score") & (d.status == "ok")]
    out = []
    for (m, sid, cond, var), g in sc.groupby(["model_name", "seed_id", "cond", "variant"]):
        a = g[g.side == "A"]; b = g[g.side == "B"]
        if len(a) != 1 or len(b) != 1:
            continue
        a, b = a.iloc[0], b.iloc[0]
        qa, qb = json.loads(a.q), json.loads(b.q)
        rec = {"model_name": m, "seed_id": sid, "cond": cond, "variant": var, "group": a.group, "category": a.category,
               "template": a.template, "T": S.total_variation(qa, qb),
               "T_lengthnorm": S.total_variation(S.softmax(json.loads(a.s_mean)), S.softmax(json.loads(b.s_mean))),
               "score_acc": (float(a.score_correct) + float(b.score_correct)) / 2,
               "both_score_correct": float(bool(a.score_correct) and bool(b.score_correct)),
               "energy_num": (float(a.energy_num) + float(b.energy_num)) / 2, "n_edited": (int(a.n_edited_positions) + int(b.n_edited_positions)) / 2,
               "first_disambiguates": bool(a.first_disambiguates) and bool(b.first_disambiguates)}
        if var == "std" and "gen_valid" in g:
            rec.update({"gen_acc": (float(a.gen_correct) + float(b.gen_correct)) / 2,
                        "validity": (float(a.gen_valid) + float(b.gen_valid)) / 2,
                        "both_gen_correct": float(bool(a.gen_correct) and bool(b.gen_correct)),
                        "both_valid": float(bool(a.gen_valid) and bool(b.gen_valid)),
                        "flip_both_valid": float(bool(a.gen_valid) and bool(b.gen_valid) and int(a.gen_idx) != int(b.gen_idx)),
                        "fail_conservative": float((not bool(a.gen_valid)) or (not bool(b.gen_valid)) or int(a.gen_idx) != int(b.gen_idx))})
        out.append(rec)
    return pd.DataFrame(out)


def paired(pt: pd.DataFrame, m: str, c1: str, c2: str, col: str, group: str = "final", cluster: str = "template") -> tuple[np.ndarray, int, int]:
    """Cluster-level means of (col[c1] - col[c2]) over seeds present in both conditions."""
    a = pt[(pt.model_name == m) & (pt.group == group) & (pt.cond == c1) & (pt.variant == "std")].set_index("seed_id")
    b = pt[(pt.model_name == m) & (pt.group == group) & (pt.cond == c2) & (pt.variant == "std")].set_index("seed_id")
    common = a.index.intersection(b.index)
    if not len(common):
        return np.array([]), 0, 0
    diff = pd.DataFrame({"d": a.loc[common, col].to_numpy(float) - b.loc[common, col].to_numpy(float), "cl": a.loc[common, cluster].to_numpy()})
    return diff.groupby("cl")["d"].mean().to_numpy(), len(common), diff.cl.nunique()


def confirmatory(pt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    H = [("H1", "T", "B", "S1", "operational sensitivity benefit: mean(T_B - T_S1)"),
         ("H2", "T", "NE", "SE", "position specificity at calibrated energy: mean(T_NE - T_SE)"),
         ("H3", "T", "RE", "SE", "direction specificity at the span: mean(T_RE - T_SE)"),
         ("H4", "gen_acc", "S1", "B", "generated-answer improvement: mean(acc_S1 - acc_B)")]
    for m in N.FINAL_MODELS:
        for hid, col, c1, c2, desc in H:
            for cluster in ("template", "seed_id"):
                x, n_seeds, n_cl = paired(pt, m, c1, c2, col, cluster=cluster)
                t = N.studentized_paired_test(x) if len(x) else {"mean": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "n_seeds": 0, "degenerate": True, "se": np.nan, "t": np.nan}
                rows.append({"model_name": m, "id": hid, "description": desc, "cluster": cluster, "n_seeds": n_seeds, "n_clusters": n_cl,
                             "estimate": t["mean"], "lo": t["lo"], "hi": t["hi"], "se": t.get("se"), "t": t.get("t"), "p": t["p"],
                             "degenerate": t.get("degenerate", False), "n_boot": N.N_BOOT_FINAL})
    df = pd.DataFrame(rows)
    fam = df[df.cluster == "template"]
    df["p_holm"] = np.nan
    df.loc[fam.index, "p_holm"] = N.holm(fam.p.tolist())
    df["reject_holm_0.05"] = df.p_holm < 0.05
    return df


def preservation(pt: pd.DataFrame, margin_pp: float) -> pd.DataFrame:
    rows = []
    for m in N.FINAL_MODELS:
        for cond in ("S1", "SE", "NE"):
            x, n_seeds, n_cl = paired(pt, m, cond, "B", "gen_acc")
            t = N.studentized_paired_test(x, null_value=-margin_pp / 100.0, alternative="greater") if len(x) else {"mean": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "degenerate": True}
            rows.append({"model_name": m, "condition": cond, "quantity": "acc_%s - acc_B" % cond, "margin_pp": margin_pp, "n_seeds": n_seeds, "n_clusters": n_cl,
                         "estimate": t["mean"], "lo": t["lo"], "hi": t["hi"], "p_one_sided": t["p"], "degenerate": t.get("degenerate", False)})
    df = pd.DataFrame(rows)
    df["p_holm"] = N.holm(df.p_one_sided.tolist())
    df["non_inferior_holm_0.05"] = df.p_holm < 0.05
    return df


def secondary(d: pd.DataFrame, pt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(m, name, x, n, extra=None):
        p, lo, hi = N.paired_interval(x) if len(x) else (np.nan, np.nan, np.nan)
        rows.append({"model_name": m, "quantity": name, "n_clusters": n, "estimate": p, "lo": lo, "hi": hi, **(extra or {})})
    for m in N.FINAL_MODELS:
        # interaction
        a, n1, k1 = paired(pt, m, "RNE", "NE", "T"); b, n2, k2 = paired(pt, m, "RE", "SE", "T")
        if len(a) and len(b) and len(a) == len(b):
            add(m, "interaction (T_RNE-T_NE)-(T_RE-T_SE)", a - b, k1)
        for cond in N.CONDITIONS[1:]:
            x, n, k = paired(pt, m, cond, "B", "T"); add(m, "T_%s - T_B" % cond, x, k, {"n_seeds": n})
            x, n, k = paired(pt, m, cond, "B", "gen_acc"); add(m, "acc_%s - acc_B" % cond, x, k, {"n_seeds": n})
            x, n, k = paired(pt, m, cond, "B", "both_gen_correct"); add(m, "bothcorrect_%s - bothcorrect_B" % cond, x, k, {"n_seeds": n})
            x, n, k = paired(pt, m, cond, "B", "score_acc"); add(m, "scoreacc_%s - scoreacc_B" % cond, x, k, {"n_seeds": n})
            x, n, k = paired(pt, m, cond, "B", "flip_both_valid"); add(m, "flipvalid_%s - flipvalid_B" % cond, x, k, {"n_seeds": n})
            x, n, k = paired(pt, m, cond, "B", "fail_conservative"); add(m, "failcons_%s - failcons_B" % cond, x, k, {"n_seeds": n})
            x, n, k = paired(pt, m, cond, "B", "validity"); add(m, "validity_%s - validity_B" % cond, x, k, {"n_seeds": n})
        for cond in ("LC", "MC"):
            x, n, k = paired(pt, m, cond, "B", "T")
            if len(x):
                add(m, "T_%s - T_B (gender-eligible subset)" % cond, x, k, {"n_seeds": n})
                x, n, k = paired(pt, m, cond, "B", "gen_acc"); add(m, "acc_%s - acc_B (gender-eligible subset)" % cond, x, k, {"n_seeds": n})
                x, n, k = paired(pt, m, "S1", "B", "T", group="final");
        # control group
        for cond in ("S1", "NE"):
            x, n, k = paired(pt, m, cond, "B", "gen_acc", group="control"); add(m, "control acc_%s - acc_B" % cond, x, k, {"n_seeds": n})
            x, n, k = paired(pt, m, cond, "B", "T", group="control"); add(m, "control T_%s - T_B" % cond, x, k, {"n_seeds": n})
        for cond in ("B", "S1", "NE"):
            g = pt[(pt.model_name == m) & (pt.group == "control") & (pt.cond == cond) & (pt.variant == "std")]
            if len(g):
                add(m, "control acc_%s" % cond, g.groupby("template").gen_acc.mean().to_numpy(), g.template.nunique(), {"n_seeds": len(g)})
        # levels
        for cond in N.CONDITIONS:
            g = pt[(pt.model_name == m) & (pt.group == "final") & (pt.cond == cond) & (pt.variant == "std")]
            if len(g):
                add(m, "level T_%s" % cond, g.groupby("template").T.mean().to_numpy(), g.template.nunique(), {"n_seeds": len(g)})
                add(m, "level acc_%s" % cond, g.groupby("template").gen_acc.mean().to_numpy(), g.template.nunique(), {"n_seeds": len(g)})
                add(m, "level validity_%s" % cond, g.groupby("template").validity.mean().to_numpy(), g.template.nunique(), {"n_seeds": len(g)})
        # bridge
        br = d[(d.model_name == m) & (d.kind == "bridge") & (d.status == "ok")]
        for cond in ("B", "S1"):
            g = br[br.cond == cond]
            if len(g):
                add(m, "C_answer_%s (mean)" % cond, g.groupby("template").C_answer.mean().to_numpy(), g.template.nunique(), {"n_seeds": len(g)})
                add(m, "|C_answer|_%s (mean)" % cond, g.assign(a=g.C_answer.abs()).groupby("template").a.mean().to_numpy(), g.template.nunique(), {"n_seeds": len(g)})
                add(m, "|C_first|_%s (mean)" % cond, g.assign(a=g.C_first.abs()).groupby("template").a.mean().to_numpy(), g.template.nunique(),
                    {"n_seeds": len(g), "n_first_token_disambiguating": int(g.first_disambiguates.sum())})
        gb, gs = br[br.cond == "B"].set_index("seed_id"), br[br.cond == "S1"].set_index("seed_id")
        common = gb.index.intersection(gs.index)
        if len(common):
            dd = pd.DataFrame({"d": gb.loc[common, "C_answer"].abs().to_numpy() - gs.loc[common, "C_answer"].abs().to_numpy(), "cl": gb.loc[common, "template"].to_numpy()})
            add(m, "|C_answer|_B - |C_answer|_S1", dd.groupby("cl").d.mean().to_numpy(), dd.cl.nunique(), {"n_seeds": len(common)})
            dd = pd.DataFrame({"d": gb.loc[common, "C_first"].abs().to_numpy() - gs.loc[common, "C_first"].abs().to_numpy(), "cl": gb.loc[common, "template"].to_numpy()})
            add(m, "|C_first|_B - |C_first|_S1", dd.groupby("cl").d.mean().to_numpy(), dd.cl.nunique(), {"n_seeds": len(common)})
        # subgroups for H1 and H4
        for cat, g in pt[(pt.model_name == m) & (pt.group == "final")].groupby("category"):
            x, n, k = paired(g.assign(group="final"), m, "B", "S1", "T"); add(m, "H1 by category: %s" % cat, x, k, {"n_seeds": n})
            x, n, k = paired(g.assign(group="final"), m, "S1", "B", "gen_acc"); add(m, "H4 by category: %s" % cat, x, k, {"n_seeds": n})
        # scoring-sensitivity variants (32 seeds): T under permuted options and label-only candidates, B and S1
        for var in ("perm", "label"):
            a = pt[(pt.model_name == m) & (pt.cond == "B") & (pt.variant == var)].set_index("seed_id")
            b = pt[(pt.model_name == m) & (pt.cond == "S1") & (pt.variant == var)].set_index("seed_id")
            common = a.index.intersection(b.index)
            if len(common):
                dd = pd.DataFrame({"d": a.loc[common, "T"].to_numpy() - b.loc[common, "T"].to_numpy(), "cl": a.loc[common, "template"].to_numpy()})
                add(m, "diag32 %s: T_B - T_S1" % var, dd.groupby("cl").d.mean().to_numpy(), dd.cl.nunique(), {"n_seeds": len(common)})
        x, n, k = paired(pt, m, "B", "S1", "T_lengthnorm"); add(m, "length-normalised T_B - T_S1", x, k, {"n_seeds": n})
        # energies (test achieved, relative to the dev denominator)
        cal = N.read_json(OUT / ("calibration_%s.json" % m), {})
        D = cal.get("denominator")
        for cond in ("S1", "SE", "NE", "RE", "RNE", "G1"):
            g = pt[(pt.model_name == m) & (pt.group == "final") & (pt.cond == cond) & (pt.variant == "std")]
            if len(g) and D:
                e = g.energy_num.to_numpy() / D
                rows.append({"model_name": m, "quantity": "achieved energy %s (test, / dev denominator)" % cond, "n_clusters": g.template.nunique(),
                             "estimate": float(e.mean()), "lo": float(np.percentile(e, 2.5)), "hi": float(np.percentile(e, 97.5)),
                             "n_seeds": len(g), "dev_target": cal.get("target"), "alpha": float(g.alpha.iloc[0]), "n_edited_mean": float(g.n_edited.mean())})
    return pd.DataFrame(rows)


def energy_flags(sec: pd.DataFrame) -> dict:
    out = {}
    for m in N.FINAL_MODELS:
        e = sec[(sec.model_name == m) & sec.quantity.str.startswith("achieved energy")]
        cells = {r.quantity.split()[2]: r.estimate for r in e.itertuples(index=False)}
        cal = N.read_json(OUT / ("calibration_%s.json" % m), {})
        vals = [cells[c] for c in ("SE", "NE", "RE", "RNE") if c in cells]
        spread = (max(vals) - min(vals)) / np.mean(vals) if vals and np.mean(vals) > 0 else np.nan
        out[m] = {"dev_matching_status": cal.get("matching_status"), "dev_feasible": cal.get("feasible"), "test_cells": cells,
                  "test_relative_spread": spread, "caution_flag_gt_10pct": bool(np.isfinite(spread) and spread > N.CAUTION_MISMATCH)}
    return out


def outcome_claims(conf: pd.DataFrame, pres: pd.DataFrame, sec: pd.DataFrame, eflags: dict, proto: dict) -> dict:
    claims = []
    fam = conf[conf.cluster == "template"].set_index(["model_name", "id"])
    for m in N.FINAL_MODELS:
        def row(hid):
            return fam.loc[(m, hid)] if (m, hid) in fam.index else None
        h1, h2, h3, h4 = (row(h) for h in ("H1", "H2", "H3", "H4"))
        pr = pres[(pres.model_name == m)].set_index("condition")
        matched = eflags.get(m, {}).get("dev_feasible", False) and not eflags.get(m, {}).get("caution_flag_gt_10pct", True)
        def status(r):
            if r is None or not np.isfinite(r.p_holm):
                return "not_measured"
            return "supported" if (r["reject_holm_0.05"] and r.estimate > 0) else "limited"
        claims.append({"id": "H1_%s" % m, "model": m, "endpoint": "answer-score sensitivity T, S1 vs B", "population": "final BBQ set",
                       "estimate": None if h1 is None else float(h1.estimate), "interval": None if h1 is None else [float(h1.lo), float(h1.hi)],
                       "p_holm": None if h1 is None else float(h1.p_holm), "family": "confirmatory (8, Holm)", "status": status(h1),
                       "permitted_wording": "the span edit lowers answer-score sensitivity on this model" if status(h1) == "supported" else
                       "no confirmatory reduction in answer-score sensitivity on this model (estimate and interval reported)"})
        for hid, r, txt in (("H2", h2, "position specificity"), ("H3", h3, "direction specificity")):
            st = status(r)
            if st == "supported" and not matched:
                st = "limited"
            claims.append({"id": "%s_%s" % (hid, m), "model": m, "endpoint": txt, "population": "final BBQ set, paired location population",
                           "estimate": None if r is None else float(r.estimate), "interval": None if r is None else [float(r.lo), float(r.hi)],
                           "p_holm": None if r is None else float(r.p_holm), "family": "confirmatory (8, Holm)", "status": st,
                           "energy_matched": matched,
                           "permitted_wording": ("%s supported at calibrated energy" % txt) if st == "supported" else
                           ("%s: %s" % (txt, "unresolved (not significant after Holm)" if r is not None else "not measured")) +
                           ("" if matched else "; energy matching not verified, so no energy-controlled claim")})
        s4 = status(h4)
        pres_ok = bool(pr.loc["S1", "non_inferior_holm_0.05"]) if "S1" in pr.index else False
        claims.append({"id": "H4_%s" % m, "model": m, "endpoint": "generated-answer accuracy, S1 vs B", "population": "final BBQ set",
                       "estimate": None if h4 is None else float(h4.estimate), "interval": None if h4 is None else [float(h4.lo), float(h4.hi)],
                       "p_holm": None if h4 is None else float(h4.p_holm), "family": "confirmatory (8, Holm)",
                       "status": "supported" if s4 == "supported" else ("limited" if pres_ok else ("unsupported" if h4 is not None and h4.estimate < -proto.get("acc_margin_pp", 2) / 100 else "limited")),
                       "non_inferior_within_margin": pres_ok,
                       "permitted_wording": "generated-answer accuracy rises" if s4 == "supported" else
                       ("accuracy preserved within the %.0f-point margin" % proto.get("acc_margin_pp", 2) if pres_ok else "accuracy change uncertain; no preservation claim")})
    return {"generated_utc": N.utc_now(), "claims": claims, "energy_flags": eflags,
            "outcome_rule": "Section 10: two models must independently support a replication claim; one positive model supports a model-specific result"}


def main() -> None:
    proto = N.load_protocol()
    proto.setdefault("acc_margin_pp", N.ACC_MARGIN_PP)
    proto.setdefault("acc_margin_justification", "two percentage points of attempted-denominator accuracy on a three-option task is below "
                     "the half-width of the paired intervals on 160 seeds and is the largest loss the authors would accept for a bias edit "
                     "whose purpose is to leave the answer unchanged; fixed before the final run")
    proto.setdefault("n_boot", N.N_BOOT_FINAL); proto.setdefault("cluster", "template family (seed-level reported alongside)")
    N.save_protocol(proto)
    d = load_rows()
    pt = pairwise_T(d)
    N.write_csv(pt, OUT / "pair_level.csv")
    conf = confirmatory(pt); N.write_csv(conf, OUT / "confirmatory_tests.csv")
    pres = preservation(pt, proto["acc_margin_pp"]); N.write_csv(pres, OUT / "preservation_tests.csv")
    sec = secondary(d, pt); N.write_csv(sec, OUT / "secondary_results.csv")
    eflags = energy_flags(sec)
    claims = outcome_claims(conf, pres, sec, eflags, proto); N.write_json(claims, OUT / "claims.json")
    # merged per_item and provenance
    N.write_csv(pd.DataFrame([{"table": "confirmatory_tests.csv", "source": "per_item_<model>.parquet via pair_level.csv"},
                              {"table": "preservation_tests.csv", "source": "per_item_<model>.parquet via pair_level.csv"},
                              {"table": "secondary_results.csv", "source": "per_item_<model>.parquet, calibration_<model>.json"},
                              {"table": "legacy_behaviour.csv", "source": "results/v2/p2_per_item.parquet (F0)"},
                              {"table": "energy_audit.csv", "source": "results/v2/p3_magnitude_summary.csv (F0)"}]), OUT / "number_provenance.csv")
    fam = conf[conf.cluster == "template"]
    lines = ["COMPLETION record (%s)" % N.utc_now(), "",
             "Rows: %d per-item rows; models: %s" % (len(d), ", ".join(sorted(d.model_name.unique()))), "",
             "Confirmatory family (template clusters, Holm over 8):"]
    for r in fam.itertuples(index=False):
        lines.append("  %-24s %s  est %+.4f [%+.4f, %+.4f]  p %.4f  p_holm %.4f  %s" % (r.model_name, r.id, r.estimate, r.lo, r.hi, r.p, r.p_holm, "REJECT" if r._asdict()["reject_holm_0.05"] else ""))
    lines += ["", "Preservation family (one-sided, margin %.0f pp, Holm over 6):" % proto["acc_margin_pp"]]
    for r in pres.itertuples(index=False):
        lines.append("  %-24s %-3s est %+.4f [%+.4f, %+.4f]  p %.4f  p_holm %.4f  %s" % (r.model_name, r.condition, r.estimate, r.lo, r.hi, r.p_one_sided, r.p_holm, "NON-INFERIOR" if r._asdict()["non_inferior_holm_0.05"] else ""))
    lines += ["", "Energy: %s" % json.dumps(eflags, default=N.K._json_default)[:1500], "",
              "Budget: see runtime_budget_<model>.json; claims: claims.json; secondary: secondary_results.csv"]
    (OUT / "COMPLETION.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
