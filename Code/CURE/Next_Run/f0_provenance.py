"""
f0_provenance.py -- F0 (Next_Plan.md Section 5): repair provenance and analyse the existing
outputs before any GPU work. CPU only. Nothing under results/v2 or reanalysis_v2 is modified.

5.1  Freeze and verify: dated manifest with sha256 of the PDF, TeX, result files, fitted bases
     and code; row-key uniqueness and row counts of results/v2/p2_per_item.parquet; fit-manifest
     audit (how many fit pairs were neutral-type substitutions; the sequential P3 fit used
     blocks of four and 120 pairs); the estimator is named 'contrast-derived projection'.
5.2  Correct the energy claim: the P3 rows called 'matched' are relabelled with their achieved
     random-to-target energy ratio; every targeted_at_*_energy row is listed; feasibility is
     recomputed from the saved energies with a 5% tolerance instead of trusted.
5.3  Recover the behavioural evidence from the saved generations for all four models:
     attempted-denominator accuracy and validity with paired seed-cluster intervals; flips on
     both-valid pairs with that denominator; a conservative failure rate (either side invalid
     OR the answers differ); both-correct and both-wrong shares; by benchmark; parser-only
     next to parser-plus-judge; MMLU question-level intervals from the saved per-question
     outcomes; WikiText-2 sampling uncertainty disclosed as unavailable (aggregate only).

Outputs (results/final_20260914/)
  evidence_manifest.json   hashes and row counts
  fit_audit.json           fit provenance per model
  energy_audit.csv         P3 energy rows with achieved ratios and recomputed feasibility
  legacy_behaviour.csv     the reanalysis rows (model x condition x stratum x readout policy)
  capability_uncertainty.csv  MMLU intervals per condition; WikiText note
  provenance_audit.txt     narrative summary (plain text)
  claims_f0.json           what the existing evidence may still support, with statuses

Usage: python f0_provenance.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import nr_common as N
import p1_pilot as P1     # noqa: E402

log = N.setup_logging("cure.final.f0")
OUT = N.FINAL
K = N.K


def evidence_manifest() -> dict:
    files = []
    for pat in ("*.csv", "*.parquet", "*.json", "*.npz"):
        files += sorted(N.V2.glob(pat))
    files += sorted(N.P0.glob("*.csv"))
    sub = K.REPO / "Submission2" / "iclr2027"
    files += [sub / "iclr2027_CURE_Audit_Benchmark.tex", sub / "iclr2027_CURE_Audit_Benchmark.pdf"]
    files += sorted((N.EXT).glob("*.py")) + sorted((N.CURE / "Next_Run").glob("*.py")) + [N.CURE / "make_iclr_tables_v2.py"]
    rec = {"generated_utc": N.utc_now(), "files": []}
    for f in files:
        if f.exists() and f.name != "judge_cache.json":
            rec["files"].append({"path": K.rel(f), "sha256": N.sha256_file(f), "bytes": f.stat().st_size})
    per = pd.read_parquet(N.V2 / "p2_per_item.parquet")
    key = ["model_name", "phase", "cond_id", "seed_id", "pair_type", "subvariant_A", "subvariant_B"]
    ok = per[per.status == "ok"] if "status" in per else per
    dup = int(ok.duplicated(key).sum())
    rec["p2_per_item"] = {"rows": int(len(per)), "ok_rows": int(len(ok)), "duplicate_keys": dup, "key": key,
                          "rows_by_model_phase": {"%s|%s" % k: int(v) for k, v in ok.groupby(["model_name", "phase"]).size().items()},
                          "key_note": "the key lacks a tokenizer/prompt version and a metric version; final rows carry protocol_hash"}
    return rec


def fit_audit() -> dict:
    out = {}
    for m in N.ALL_MODELS:
        man = N.read_json(N.V2 / ("p2_basis_manifest_%s_chat.json" % m), {})
        info = (man.get("info") or {}).get("cure_centred_svd", {})
        seq = N.read_json(N.V2 / ("p3_basis_%s_sequential_b4_r1.json" % m), {})
        out[m] = {"estimator": "contrast-derived projection: leading right singular vectors of centred span-position "
                               "activation differences (not a direction selected by its effect on the output)",
                  "n_fit_pairs_used": info.get("n_fit_pairs_used"), "n_fit_pairs_neutral_type": info.get("n_fit_pairs_neutral_type"),
                  "fit_mixes_neutral_substitutions": bool(info.get("n_fit_pairs_neutral_type", 0)),
                  "n_span_tokens": info.get("n_span_tokens"), "fit_id": (man.get("fit_id") or {}).get("cure_centred_svd"),
                  "p3_sequential_fit": {"n_fit_pairs_used": seq.get("n_fit_pairs_used"), "block": seq.get("block"),
                                        "note": "blockwise refit on 120 pairs; not a controlled layer-by-layer comparison against the 300-pair frozen fit"},
                  "leace_v2_labels": ((man.get("info") or {}).get("leace_faithful") or {}).get("labels")}
    return out


def energy_audit() -> pd.DataFrame:
    m = pd.read_csv(N.V2 / "p3_magnitude_summary.csv")
    rows = []
    for (model, phase), g in m.groupby(["model_name", "phase"]):
        tgt = g[g.condition == "targeted"]
        if not len(tgt):
            continue
        e_t = float(tgt.energy_total_A_mean.iloc[0])
        for r in g.itertuples(index=False):
            if r.condition == "unedited":
                continue
            ratio = float(r.energy_total_A_mean) / e_t if e_t else np.nan
            if r.condition.endswith("_matched"):
                target_e = e_t; achieved = float(r.energy_total_A_mean)
            elif r.condition.startswith("targeted_at_"):
                target_e = float(r.energy_target) if np.isfinite(r.energy_target) else np.nan; achieved = float(r.energy_total_A_mean)
            else:
                target_e, achieved = np.nan, float(r.energy_total_A_mean)
            rel = (achieved - target_e) / target_e if (np.isfinite(target_e) and target_e > 0) else np.nan
            rows.append({"model_name": model, "phase": phase, "condition": r.condition, "alpha": r.alpha,
                         "energy_total_A_mean": r.energy_total_A_mean, "targeted_energy": e_t, "ratio_to_targeted": ratio,
                         "matching_target_energy": target_e, "relative_error": rel, "feasible_as_saved": bool(r.feasible),
                         "feasible_recomputed_5pct": bool(np.isfinite(rel) and abs(rel) <= N.MATCH_TOL),
                         "relabel": ("rank-matched random control, achieved energy %.1f%% of the targeted edit" % (100 * ratio))
                         if r.condition.endswith("_matched") else ("targeted edit scaled towards the %s energy; achieved %.0f%% of that target"
                                                                   % (r.condition.replace("targeted_at_", "").replace("_energy", ""), 100 * (1 + rel)) if np.isfinite(rel) else r.condition),
                         "absC_change_vs_unedited": r.absC_change_vs_unedited, "absC_change_lo": r.absC_change_vs_unedited_lo, "absC_change_hi": r.absC_change_vs_unedited_hi,
                         "acc_change_vs_unedited": -r.acc_loss_vs_unedited})
    return pd.DataFrame(rows)


def legacy_behaviour() -> pd.DataFrame:
    per = pd.read_parquet(N.V2 / "p2_per_item.parquet")
    per = per[(per.status == "ok") & (per.phase == "test") & (per.pair_type == "demographic")]
    pen = pd.read_parquet(K.PENTAD_CLEAN); pen = pen[pen.slot == "c"]
    prompts = {(r.seed_id, r.subvariant): str(r.prompt_text) for r in pen.itertuples(index=False)}
    conds = {"identity_alpha0_r1_a0": "B", "cure_centred_svd_r1_a1": "S1", "random_ortho_1_r1_a1": "R1", "leace_faithful_r1_a1": "LEACE_swapside"}
    rows = []
    for m in N.ALL_MODELS:
        for cid, lab in conds.items():
            d = per[(per.model_name == m) & (per.cond_id == cid)].copy()
            if not len(d):
                continue
            # parser-only readout (no judge): re-parse the raw generations deterministically
            for side in ("A", "B"):
                idx, val = [], []
                for r in d.itertuples(index=False):
                    up = prompts.get((r.seed_id, getattr(r, "subvariant_%s" % side)), "")
                    gi, _ = P1.parse_answer(str(getattr(r, "gen_raw_%s" % side)), up)
                    idx.append(-1 if gi is None else gi); val.append(gi is not None)
                d["po_idx_%s" % side] = idx; d["po_valid_%s" % side] = val
                d["po_correct_%s" % side] = [(i == g) if v else False for i, v, g in zip(idx, val, d["gold_idx_%s" % side])]
            for policy in ("parser_plus_judge", "parser_only"):
                pre = "gen" if policy == "parser_plus_judge" else "po"
                vA, vB = d["%s_valid_A" % pre].astype(bool), d["%s_valid_B" % pre].astype(bool)
                cA, cB = d["%s_correct_A" % pre].fillna(False).astype(bool), d["%s_correct_B" % pre].fillna(False).astype(bool)
                iA, iB = d["%s_idx_A" % pre].astype(float), d["%s_idx_B" % pre].astype(float)
                both_valid = vA & vB
                flip_bv = (iA != iB) & both_valid
                fail = (~both_valid) | (iA != iB)
                for stratum, mask in [("pooled", np.ones(len(d), bool))] + [(b, (d.benchmark == b).to_numpy()) for b in sorted(d.benchmark.unique())]:
                    dd = d[mask]
                    cl = dd.seed_id.to_numpy()
                    def ci(x):
                        p, lo, hi = K.ratio_bootstrap(np.asarray(x, float), np.ones(len(x)), cl, n_boot=2000, seed=N.RANDOM_SEED_FINAL)
                        return p, lo, hi
                    acc = ci(np.concatenate([cA[mask].to_numpy(), cB[mask].to_numpy()])) if False else K.ratio_bootstrap(
                        (cA[mask].astype(float) + cB[mask].astype(float)).to_numpy() / 2, np.ones(mask.sum()), cl, n_boot=2000, seed=N.RANDOM_SEED_FINAL)
                    validity = K.ratio_bootstrap((vA[mask].astype(float) + vB[mask].astype(float)).to_numpy() / 2, np.ones(mask.sum()), cl, n_boot=2000, seed=N.RANDOM_SEED_FINAL)
                    bv = both_valid[mask].to_numpy()
                    flip = K.ratio_bootstrap(flip_bv[mask].astype(float).to_numpy(), bv.astype(float), cl, n_boot=2000, seed=N.RANDOM_SEED_FINAL) if bv.sum() else (np.nan,) * 3
                    failr = ci(fail[mask].astype(float).to_numpy())
                    bothc = ci((cA & cB)[mask].astype(float).to_numpy())
                    bothw = ci((~cA & ~cB & both_valid)[mask].astype(float).to_numpy())
                    rows.append({"model_name": m, "condition": lab, "cond_id": cid, "readout_policy": policy, "stratum": stratum,
                                 "n_pairs": int(mask.sum()), "n_seeds": int(dd.seed_id.nunique()), "n_both_valid": int(bv.sum()),
                                 "accuracy_attempted": acc[0], "accuracy_lo": acc[1], "accuracy_hi": acc[2],
                                 "validity": validity[0], "validity_lo": validity[1], "validity_hi": validity[2],
                                 "flip_rate_both_valid": flip[0], "flip_lo": flip[1], "flip_hi": flip[2],
                                 "conservative_failure_rate": failr[0], "failure_lo": failr[1], "failure_hi": failr[2],
                                 "both_correct": bothc[0], "both_correct_lo": bothc[1], "both_correct_hi": bothc[2],
                                 "both_wrong_valid": bothw[0], "both_wrong_lo": bothw[1], "both_wrong_hi": bothw[2]})
    out = pd.DataFrame(rows)
    # paired differences S1 - B on the same seeds (pooled, both policies)
    diffs = []
    for m in N.ALL_MODELS:
        for policy in ("parser_plus_judge", "parser_only"):
            pre = "gen" if policy == "parser_plus_judge" else "po"
            b = per[(per.model_name == m) & (per.cond_id == "identity_alpha0_r1_a0")].set_index("seed_id")
            s = per[(per.model_name == m) & (per.cond_id == "cure_centred_svd_r1_a1")].set_index("seed_id")
            common = b.index.intersection(s.index)
            if not len(common):
                continue
            def side_vals(df, col):
                return (df.loc[common, col + "_A"].fillna(False).astype(float).to_numpy() + df.loc[common, col + "_B"].fillna(False).astype(float).to_numpy()) / 2
            if policy == "parser_only":
                def po(df):
                    outv = []
                    for sid in common:
                        r = df.loc[sid]; vals = []
                        for side in ("A", "B"):
                            gi, _ = P1.parse_answer(str(r["gen_raw_%s" % side]), prompts.get((sid, r["subvariant_%s" % side]), ""))
                            vals.append(float(gi == r["gold_idx_%s" % side]) if gi is not None else 0.0)
                        outv.append(np.mean(vals))
                    return np.asarray(outv)
                dacc = po(s) - po(b)
            else:
                dacc = side_vals(s, "gen_correct") - side_vals(b, "gen_correct")
            t = N.studentized_paired_test(dacc, n_boot=N.N_BOOT_FINAL)
            diffs.append({"model_name": m, "readout_policy": policy, "quantity": "accuracy_S1_minus_B", "n_seeds": t["n_seeds"],
                          "estimate": t["mean"], "lo": t["lo"], "hi": t["hi"], "p_two_sided": t["p"]})
    N.write_csv(pd.DataFrame(diffs), OUT / "legacy_behaviour_paired.csv")
    return out


def capability_uncertainty() -> pd.DataFrame:
    cap = pd.read_parquet(N.V2 / "p2_capability.parquet")
    cap = cap[(cap.status == "ok") & (cap.phase == "test")]
    rows = []
    rng = np.random.default_rng(N.RANDOM_SEED_FINAL)
    for m in N.ALL_MODELS:
        base = cap[(cap.model_name == m) & (cap.probe == "mmlu_200") & (cap.cond_id == "unedited_r0_a0")]
        b_ok = np.array([d["correct"] for d in json.loads(base.detail.iloc[0])], float) if len(base) else None
        for r in cap[(cap.model_name == m) & (cap.probe == "mmlu_200")].itertuples(index=False):
            ok = np.array([d["correct"] for d in json.loads(r.detail)], float)
            take = rng.integers(0, len(ok), size=(5000, len(ok)))
            acc = ok.mean(); lo, hi = np.percentile(ok[take].mean(1), [2.5, 97.5])
            rec = {"model_name": m, "probe": "mmlu_200", "cond_id": r.cond_id, "n_questions": len(ok), "accuracy": acc, "lo": lo, "hi": hi}
            if b_ok is not None and len(b_ok) == len(ok):
                d = ok - b_ok
                rec.update({"paired_diff_vs_unedited": d.mean(), "diff_lo": np.percentile(d[take].mean(1), 2.5), "diff_hi": np.percentile(d[take].mean(1), 97.5)})
            rows.append(rec)
        for r in cap[(cap.model_name == m) & (cap.probe == "wikitext2_20k")].itertuples(index=False):
            rows.append({"model_name": m, "probe": "wikitext2_20k", "cond_id": r.cond_id, "n_questions": int(float(r.n)), "accuracy": np.nan,
                         "perplexity": float(r.perplexity), "note": "aggregate only; per-window losses were not saved, so no sampling interval"})
    return pd.DataFrame(rows)


def claims_f0(energy: pd.DataFrame, legacy: pd.DataFrame, fit: dict) -> dict:
    conf = pd.read_csv(N.V2 / "p2_confirmatory.csv"); conf = conf[conf.phase == "test"]
    claims = []
    for m in N.ALL_MODELS:
        c = conf[conf.model_name == m].set_index("id")
        claims.append({"id": "L1_%s" % m, "endpoint": "paired reduction D in the first-gold-token patching effect, rank-1 span erasure",
                       "model": m, "population": "160 test seeds of the existing pool", "estimate": float(c.loc["C1", "point"]),
                       "interval": [float(c.loc["C1", "lo"]), float(c.loc["C1", "hi"])], "p_holm": float(c.loc["C1", "p_holm"]),
                       "status": "supported" if c.loc["C1", "reject_0.05_holm"] else "limited",
                       "permitted_wording": "reduces the legacy first-token patching statistic (Holm-corrected within model)" if c.loc["C1", "reject_0.05_holm"]
                       else "point estimate positive; not significant after Holm correction",
                       "validity_caveat": "the first gold token identifies a unique option on a minority of seeds; answer-level measures are added in F2"})
        claims.append({"id": "L2_%s" % m, "endpoint": "generation accuracy change (attempted denominator), S1 - B", "model": m,
                       "population": "160 test seeds of the existing pool", "estimate": float(c.loc["C2", "point"]),
                       "interval": [float(c.loc["C2", "lo"]), float(c.loc["C2", "hi"])], "p_holm": float(c.loc["C2", "p_holm"]),
                       "status": "supported" if c.loc["C2", "reject_0.05_holm"] else "limited",
                       "permitted_wording": "generation accuracy rises (Holm-corrected)" if c.loc["C2", "reject_0.05_holm"]
                       else "interval includes zero; no preservation claim without the prespecified non-inferiority test"})
    e = energy[(energy.phase == "test") & (energy.condition == "random_1_matched")]
    for r in e.itertuples(index=False):
        claims.append({"id": "E1_%s" % r.model_name, "endpoint": "energy-matched random control", "model": r.model_name,
                       "status": "unsupported", "achieved_ratio": r.ratio_to_targeted,
                       "permitted_wording": "rank-matched random control at %.1f%% of the targeted edit's activation energy; "
                                            "perturbation magnitude has NOT been ruled out" % (100 * r.ratio_to_targeted)})
    claims.append({"id": "M1", "endpoint": "massive-activation restoration", "status": "limited",
                   "permitted_wording": "restoring massive coordinates leaves the span effect unchanged; says nothing about the global damage"})
    claims.append({"id": "F1", "endpoint": "risk forecast", "status": "unsupported",
                   "permitted_wording": "secondary negative result; the audit score adds no held-out AUROC over the baseline"})
    claims.append({"id": "S1", "endpoint": "sequential fitting explanation", "status": "limited",
                   "permitted_wording": "blockwise refit (blocks of four layers, 120 pairs) behaved like the frozen 300-pair fit; not a controlled comparison"})
    claims.append({"id": "B1", "endpoint": "LEACE baseline", "status": "limited",
                   "permitted_wording": "LEACE fitted on swap-side labels (not a coherent demographic concept) left the patching statistic unchanged; "
                                        "coherent-label baseline is F3"})
    claims.append({"id": "D1", "endpoint": "direction selection", "status": "corrected",
                   "permitted_wording": "contrast-derived projection evaluated with patching; not 'identified by activation patching'"})
    return {"generated_utc": N.utc_now(), "claims": claims, "fit_audit": fit}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = evidence_manifest(); N.write_json(man, OUT / "evidence_manifest.json")
    fit = fit_audit(); N.write_json(fit, OUT / "fit_audit.json")
    energy = energy_audit(); N.write_csv(energy, OUT / "energy_audit.csv")
    legacy = legacy_behaviour(); N.write_csv(legacy, OUT / "legacy_behaviour.csv")
    capu = capability_uncertainty(); N.write_csv(capu, OUT / "capability_uncertainty.csv")
    claims = claims_f0(energy, legacy, fit); N.write_json(claims, OUT / "claims_f0.json")
    e = energy[(energy.phase == "test") & (energy.condition.str.endswith("_matched"))]
    lines = ["F0 provenance audit  (%s)" % N.utc_now(), "",
             "1. Evidence frozen: %d files hashed in evidence_manifest.json; p2_per_item ok rows %d, duplicate keys %d."
             % (len(man["files"]), man["p2_per_item"]["ok_rows"], man["p2_per_item"]["duplicate_keys"]), "",
             "2. Fit provenance: the rank-one basis is a centred-SVD contrast-derived projection; the 300 fit pairs include "
             "neutral-type substitutions on every model (%s); the P3 'sequential' fit used blocks of four layers on 120 pairs."
             % ", ".join("%s %s" % (N.DISPLAY[m], fit[m]["n_fit_pairs_neutral_type"]) for m in N.ALL_MODELS), "",
             "3. Energy: the P3 rows called matched achieved these fractions of the targeted edit's energy on the test seeds:",
             ] + ["   %s %s: %.1f%% (saved feasible=%s, recomputed feasible at 5%%=%s)" % (N.DISPLAY[r.model_name], r.condition, 100 * r.ratio_to_targeted, r.feasible_as_saved, r.feasible_recomputed_5pct) for r in e.itertuples(index=False)] + [
             "   -> relabelled as rank-matched controls with achieved energy; the perturbation-size exclusion is withdrawn.", "",
             "4. Legacy behaviour (legacy_behaviour.csv): accuracy, validity, valid-pair flips, conservative failure, both-correct "
             "and both-wrong shares, by benchmark, parser-only next to parser-plus-judge; paired S1-B accuracy tests in "
             "legacy_behaviour_paired.csv.", "",
             "5. Capability: MMLU question-level intervals in capability_uncertainty.csv; WikiText-2 has aggregate perplexity only.", "",
             "6. Permissible claims with statuses: claims_f0.json (the contract generator reads it)."]
    (OUT / "provenance_audit.txt").write_text("\n".join(lines), encoding="utf-8")
    log.info("F0 done")


if __name__ == "__main__":
    main()
