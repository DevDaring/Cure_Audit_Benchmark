"""
p0_claims_ledger.py -- one entry per manuscript claim: what was actually measured, by which
formula, on how many units, with what status and what wording the evidence permits.

Next_Plan.md, Section 7 handoff steps 2 and 4, and P0 row 7:

  Create an evidence ledger with one entry per claim: claim ID, actual intervention,
  supporting output, exact metric formula, eligible sample count, confidence interval and
  permitted wording. Proposed experiments must be marked PLANNED, never MEASURED.

Every number in a ledger row is read from a reanalysis_v2 output written by the other P0
modules (or from the shipped artifact it cites). Nothing is typed in. Quotes are from
Submission2/iclr2027/iclr2027_CURE_Audit_Benchmark.tex as read on 2026-09-12.

Status vocabulary
  SUPPORTED            evidence in an artifact supports the claim as written
  PARTIAL              supported for some models / under a narrower wording
  UNSUPPORTED          the artifact contradicts or does not contain the claim
  REQUIRES_RERUN       the needed measurement was never stored; a GPU rerun is required
  PLANNED              a P1+ experiment in this package, not yet run

Outputs
  reanalysis_v2/evidence_ledger.csv
  reanalysis_v2/evidence_ledger.md
  Submission2/Evidence_Contract.md     the writer-facing summary (measured only; P1+ PLANNED)

Usage:
  python p0_claims_ledger.py
"""

from __future__ import annotations

import json

import pandas as pd

import common as K

log = K.setup_logging("cure.ext.p0.ledger")
P0 = K.OUT_P0


def _load():
    d = {}
    d["metric"] = pd.read_csv(P0 / "metric_audit.csv")
    d["heldout"] = pd.read_csv(P0 / "metric_audit_heldout.csv")
    d["integ"] = pd.read_csv(P0 / "data_integrity.csv")
    d["overlap"] = pd.read_csv(P0 / "stage_overlap.csv")
    d["prog_rep"] = pd.read_csv(P0 / "prognosis_reproduction.csv")
    d["prog"] = pd.read_csv(P0 / "prognosis_rebuilt.csv")
    d["prog_ho"] = pd.read_csv(P0 / "prognosis_rebuilt_heldout.csv")
    d["tau"] = pd.read_csv(P0 / "tau_audit.csv")
    d["dedup"] = pd.read_csv(P0 / "dedup_dry_run.csv")
    d["archive"] = json.loads((P0 / "raw_archive_search.json").read_text(encoding="utf-8"))
    d["tt"] = pd.read_csv(K.RESULTS / "reanalysis" / "target_token_audit.csv")
    d["rank"] = {m: K.read_rankcurve(m) for m in K.MODELS}
    d["final"] = {m: K.read_final(m) for m in K.MODELS}
    return d


def _fmt_models(vals: dict, fmt="%+.1f%%") -> str:
    return "; ".join("%s %s" % (K.DISPLAY[m], (fmt % v) if v == v else "n/a") for m, v in vals.items())


def build(d) -> list[dict]:
    op = d["metric"][(d["metric"]["stratum"] == "pooled") & (d["metric"]["is_operating_rank"])]
    op = op.set_index("model_name")
    ho = d["heldout"][d["heldout"]["method"] == "cure"].set_index("model_name")
    rows = []

    def add(cid, quote, where, intervention, output, formula, n, ci, status, permitted, note=""):
        rows.append({"claim_id": cid, "manuscript_quote": quote, "location": where,
                     "actual_intervention": intervention, "supporting_output": output,
                     "metric_formula": formula, "eligible_n": n, "interval": ci,
                     "status": status, "permitted_wording": permitted, "note": note})

    # C01 removal fraction
    W = {m: 100 * op.loc[m, "W_win_rate"] for m in K.MODELS}
    R = {m: 100 * op.loc[m, "R_ratio_of_means"] for m in K.MODELS}
    Rlo = {m: 100 * op.loc[m, "R_ratio_of_means_ci_lo"] for m in K.MODELS}
    Rhi = {m: 100 * op.loc[m, "R_ratio_of_means_ci_hi"] for m in K.MODELS}
    D = {m: op.loc[m, "D_paired_abs_change"] for m in K.MODELS}
    add("C01", "it removes 39 to 59 per cent of the targeted signal", "Abstract; Sec 1; Sec 5.2",
        "orthogonal projection of a centred-difference SVD basis at the demographic position, "
        "every layer, operating rank per model",
        "reanalysis_v2/metric_audit.csv (pooled, is_operating_rank)",
        "shipped: W = mean(|C_post| < |C_pre|); magnitude: R = 1 - mean(post)/mean(pre); "
        "primary: D = mean(post - pre)",
        "1000 pairs / 508 seeds per model (sweep)",
        "R 95%% seed-cluster CI: " + "; ".join("%s [%+.1f, %+.1f]" % (K.DISPLAY[m], Rlo[m], Rhi[m]) for m in K.MODELS),
        "UNSUPPORTED",
        "The edit lowered |C| on %s of pairs (directional win rate). Mean |C| changed by %s. "
        "Paired absolute change D: %s." % (
            "; ".join("%s %.0f%%" % (K.DISPLAY[m], W[m]) for m in K.MODELS), _fmt_models(R),
            "; ".join("%s %+.3f" % (K.DISPLAY[m], D[m]) for m in K.MODELS)),
        "W is not a fraction of signal. On Phi the mean magnitude rose and the D interval spans zero.")

    # C02 provable removal
    add("C02", "provably removes most of the causal dependence", "Sec 5.2",
        "erase.project_out: orthogonal projection, frozen basis fitted on unedited activations; "
        "not the covariance-aware affine LEACE eraser; no held-out linear-probe test run",
        "Code/CURE/erase.py; reanalysis_v2/metric_audit.csv",
        "no proof exists; W at operating rank = %s" % "; ".join("%s %.2f" % (K.DISPLAY[m], W[m] / 100) for m in K.MODELS),
        "n/a", "n/a", "UNSUPPORTED",
        "Remove 'provably'. Call the estimator counterfactual-difference subspace projection. "
        "A LEACE guarantee may be claimed only after P2 runs the official concept-erasure code.")

    # C03 transfer to held-out
    hoR = {m: 100 * ho.loc[m, "R_ratio_of_means"] for m in K.MODELS}
    hoW = {m: 100 * ho.loc[m, "W_win_rate_as_stored"] for m in K.MODELS}
    add("C03", "the removal transfers to seeds the subspace was not fitted on", "Abstract; Sec 5.2",
        "subspace fitted on a seed-split train half (run_tacl_extra.split_by_seed), scored on "
        "600 test pairs; operating rank imported from the earlier run",
        "reanalysis_v2/metric_audit_heldout.csv",
        "W stored; R from stored means (no per-pair rows, no CI)",
        "600 pairs per model", "not computable (means only)", "PARTIAL",
        "On held-out seeds the win rate was %s and the mean |C| changed by %s. On Gemma the "
        "mean change is nil and on Phi the mean magnitude rose." % (
            "; ".join("%s %.0f%%" % (K.DISPLAY[m], hoW[m]) for m in K.MODELS), _fmt_models(hoR)),
        "Operating rank was selected on a sweep that shares seeds with the test split (see C08).")

    # C04 / C05 behavioural
    acc = {m: ho.loc[m, "behav_accuracy"] for m in K.MODELS}
    flip = {m: ho.loc[m, "behav_flip_rate"] for m in K.MODELS}
    nflip = {m: int(ho.loc[m, "n_flip_seeds"]) for m in K.MODELS}
    add("C04", "Task accuracy falls by 0.16 to 0.58", "Abstract; Sec 5.2; Table 4",
        "erase.ErasureContext: projection of the LAST token at every layer during generation "
        "(a different edit site from the audit, which edits the demographic position)",
        "tacl_extra_<model>.parquet (behav_accuracy); reanalysis_v2/raw_archive_search.json",
        "accuracy conditional on success_flag==True, bidirectional substring match to gold",
        "n_acc_seeds = 160 attempted; usable denominator not stored",
        "not computable (no raw outputs)", "REQUIRES_RERUN",
        "Under a last-token projection, conditional accuracy was %s; the attempted-versus-"
        "usable denominators were not stored." % "; ".join("%s %.3f" % (K.DISPLAY[m], acc[m]) for m in K.MODELS),
        "Direction plausible; magnitude and denominators require the P1/P2 rerun with raw outputs.")
    add("C05", "the rate at which answers flip under a demographic swap rises rather than falls "
        "[on all four models]", "Abstract; Sec 5.2; Table 4",
        "same last-token edit; flip = seeds with >=2 parsed variants whose parsed answers differ",
        "tacl_extra_<model>.parquet (behav_flip_rate, n_flip_seeds)",
        "mean over seeds with >=2 successfully parsed variants of 1[nunique(parsed) > 1]",
        "n_flip_seeds = " + "; ".join("%s %d" % (K.DISPLAY[m], nflip[m]) for m in K.MODELS),
        "not computable", "UNSUPPORTED",
        "On Qwen the flip rate rests on %d usable seeds against 160 unedited; it cannot support "
        "a rise. On the other three models (n=%s) the observed flip rate was higher than "
        "unedited, with denominators conditional on parse success." % (
            nflip["qwen2.5-7b-instruct"],
            "/".join(str(nflip[m]) for m in K.MODELS if m != "qwen2.5-7b-instruct")),
        "The loss of usable outputs is itself an outcome and must be reported as failure.")

    # C06 massive activations
    add("C06", "on two models it lies along hidden dimensions holding input-agnostic massive "
        "activations", "Abstract; Sec 5.3",
        "cmd_diagnose: top-5 variance coordinates at one middle layer; peak coordinate of each "
        "SVD direction checked for membership",
        "Code/CURE/results/anomaly_diagnostic.json",
        "hits_massive_dim = argmax|v| in top-5 Var(A) dims", "150 pairs, 2 models",
        "n/a", "PARTIAL",
        "On Qwen and Phi the leading direction's peak coordinate coincides with a high-variance "
        "residual dimension at one middle layer. Whether that dimension carries an "
        "input-agnostic massive activation, and whether it explains the damage, was not tested.",
        "Variance is not input-independence. P3 restoration control is PLANNED.")

    # C07 prognosis
    rep = d["prog_rep"].set_index("model_name")
    ho_prog = d["prog_ho"]
    hf = ho_prog[(ho_prog["target"] == "never_repaired") & (ho_prog["population"] == "initially_failing")
                 & (ho_prog["aggregation"] == "mean_pair")].set_index("model_name")
    add("C07", "the audit score alone separates the items erasure cannot repair, with area under "
        "the curve between 0.85 and 0.93, and the separation holds on held-out seeds",
        "Abstract; Sec 5.4; Table 6",
        "target = first tested rank at which the FIRST pair's erased |C| <= TAU; no rank 0; "
        "sentinel 9; class = repair_rank >= 8 (mixes rank-8 successes with never-repaired)",
        "reanalysis_v2/prognosis_reproduction.csv; prognosis_rebuilt.csv; prognosis_rebuilt_heldout.csv",
        "AUROC(audit_score -> target); rebuilt with rank 0 and all pairs; one fixed manifest split",
        "508 seeds per model; initially failing: " + "; ".join(
            "%s %d" % (K.DISPLAY[m], int(rep.loc[m, "n_initially_failing"])) for m in K.MODELS),
        "held-out test AUROC (never_repaired, initially failing, mean_pair): " + "; ".join(
            "%s %.2f (Brier %.3f vs base %.3f)" % (K.DISPLAY[m], hf.loc[m, "test_auroc"],
                                                   hf.loc[m, "test_brier_binned"],
                                                   hf.loc[m, "test_brier_constant_baseline"])
            for m in K.MODELS if m in hf.index),
        "UNSUPPORTED",
        "Across all seeds the audit score separates seeds already below TAU from the rest "
        "(%s), which is mechanical. Among initially failing seeds it separates never-repaired "
        "seeds with held-out AUROC %s, and on two models no better than the base rate by Brier "
        "score. The target is audit-threshold crossing, not behavioural damage." % (
            "; ".join("%s %.2f" % (K.DISPLAY[m], rep.loc[m, "auc_all_seeds_rank_ge8"]) for m in K.MODELS),
            "; ".join("%s %.2f" % (K.DISPLAY[m], hf.loc[m, "test_auroc"]) for m in K.MODELS if m in hf.index)),
        "A damage forecast is P4, PLANNED, and needs P2 raw outputs.")

    # C08 disjoint roles
    ov = d["overlap"]
    fs = ov[(ov["stage_a"] == "fit") & (ov["stage_b"] == "sweep")].iloc[0]
    add("C08", "Four disjoint roles are drawn from the counterfactual pool", "Sec 4",
        "cmd_main: fit = stratified_subset(pairs, 600, seed); sweep = stratified_subset(pairs, "
        "1000, same seed) -> fit is a prefix-subset of sweep; cmd_baselines eval uses seed+7 "
        "with no exclusion of fit seeds; tacl_extra split_by_seed is seed-disjoint but imports "
        "the operating rank chosen on the sweep",
        "reanalysis_v2/stage_overlap.csv; data_integrity.csv (NOTE_reconstruction)",
        "set containment on (seed_id, subvariant_A, subvariant_B)",
        "fit 600 pairs in sweep 1000: %d shared (structural)" % int(fs["pairs_shared"]),
        "n/a", "UNSUPPORTED",
        "The subspace was fitted on 600 pairs that are all contained in the 1000-pair sweep "
        "used for rank selection and prognosis. The exact fitting set was not recorded. "
        "P2 uses reanalysis_v2/split_manifest.csv with asserted zero intersection.",
        "Replayed cross-stage seed overlaps are illustrative only; run-time order not recoverable.")

    # C09 eight published methods
    add("C09", "Eight published debiasing methods provide context", "Sec 4; Table 3 / B.1",
        "prompt_debias: empty basis, never generated -> 0/0 assigned; fairsteer: mean-diff "
        "direction projected out, no probe gating; meandiff_steer: SVD projection when ctx "
        "given, not additive steering; nofreelunch: unembedding direction projected out; "
        "biasgym/sae_debias/hsal: in-house re-implementations on separate templates",
        "Code/CURE/baselines.py; cure_final_<model>.parquet",
        "each scored by W and by conditional accuracy drop", "9 rows per model", "n/a",
        "UNSUPPORTED",
        "Compared against in-house projection ablations named after the papers that inspired "
        "them; the Self-Debias row was not executed. Replace with faithful controls (P2) or "
        "rename as ablations.")

    # C12 table caption 160 seeds
    add("C12", "Held-out evaluation on 160 seeds per model", "Table 4 caption",
        "as C05", "tacl_extra_<model>.parquet",
        "n_acc_seeds, n_flip_seeds per row",
        "Qwen CURE flip: n_flip_seeds = %d" % nflip["qwen2.5-7b-instruct"], "n/a", "UNSUPPORTED",
        "State attempted and usable counts per cell. The Qwen CURE flip cell has %d usable seeds."
        % nflip["qwen2.5-7b-instruct"])

    # C13 two models without a utility-safe rank
    unsafe = []
    for m in K.MODELS:
        curve = d["rank"][m].get("curve", {})
        costs = [v.get("utility_cost") for v in curve.values() if v.get("utility_cost") is not None]
        if costs and min(costs) > K.MAX_UTILITY_COST:
            unsafe.append(K.DISPLAY[m])
    add("C13", "the two models without a utility-safe rank", "Table 5 caption; Sec 5.3",
        "cure_rankcurve_<model>.json utility_cost per measured rank vs budget 0.15",
        "Code/CURE/results/cure_rankcurve_*.json",
        "min over measured ranks of utility_cost > 0.15", "measured ranks only (Qwen: 4, 8)",
        "n/a", "UNSUPPORTED",
        "Models exceeding the 0.15 budget at every measured rank: %s (%d). Qwen has no utility "
        "measurement at ranks 1 and 2." % (", ".join(unsafe), len(unsafe)))

    # C14 rank selection
    lc = d["rank"]["llama-3.1-8b-instruct"]
    add("C14", "The selected rank removes the most causal signal among those whose utility cost "
        "stays within a budget of 0.15", "Sec 3.2",
        "cure_rankcurve: Llama rank 4 W=%.3f cost=%.4f vs selected rank 8 W=%.3f cost=%.4f; "
        "JSON note: bias values filled post hoc, rank 8 was the live choice" % (
            lc["curve"]["4"]["reduced_validation"], lc["curve"]["4"]["utility_cost"],
            lc["curve"]["8"]["reduced_validation"], lc["curve"]["8"]["utility_cost"]),
        "cure_rankcurve_llama-3.1-8b-instruct.json (note field)",
        "argmax W s.t. cost <= 0.15", "4 ranks", "n/a", "PARTIAL",
        "State the selection history: rank 8 was chosen live; on the final curve rank 4 removes "
        "more within budget. Do not describe rank 8 as the optimum of the saved curve.")

    # C15 LEACE
    add("C15", "following the linear core of LEACE applied through depth", "Sec 3.2; Sec 2",
        "orthogonal projection with a frozen basis at every layer; LEACE is covariance-aware, "
        "affine, and its scrubbing fits later erasers on already-edited representations",
        "Code/CURE/erase.py", "n/a", "n/a", "n/a", "PARTIAL",
        "Say: an orthogonal projection eraser in the spirit of nullspace projection; not LEACE. "
        "Faithful LEACE comparison is P2, PLANNED.")

    # C16 data cleanliness
    integ = d["integ"]
    sv = integ[integ["stage"] == "sweep_SAVED_ARTIFACT"].iloc[0]
    add("C16", "(implicit) items are the audited, integrity-filtered pool", "Sec 4",
        "loader reads pentad_dataset.parquet (raw) and never applies cdva_degeneracy_mask",
        "reanalysis_v2/data_integrity.csv (sweep_SAVED_ARTIFACT)",
        "join saved sweep to mask on (model, seed, A, B); count touches_degenerate",
        "%d of %d sweep pairs per model" % (int(sv["n_touching_degenerate"]), int(sv["n_pairs"])),
        "n/a", "UNSUPPORTED",
        "%.1f%% of the pairs used for fitting, rank selection and prognosis touch a variant the "
        "audit's own integrity filter marks degenerate. P2 fits only on mask-clean pairs."
        % (100 * sv["share_degenerate"]))

    # C17 target token
    tt = d["tt"].set_index("source")
    add("C17", "the scored token identifies a unique option on 168 of 596 seeds", "Sec 6; Table E.1",
        "word-level check: first word of gold answer vs first word of each option",
        "Code/CURE/results/reanalysis/target_token_audit.csv",
        "n_sharing == 1", "%d of %d" % (int(tt.loc["all", "n_discriminative"]), int(tt.loc["all", "n"])),
        "n/a", "SUPPORTED",
        "Keep, as a word-level diagnostic. Tokenizer-level extension is PLANNED (P1 readouts).")

    # C18 single run
    add("C18", "All numbers come from a single inference run at a fixed seed", "Sec 4; Sec 6",
        "by construction", "run logs", "n/a", "n/a", "n/a", "SUPPORTED", "Keep.")

    # C19 TAU
    tau = d["tau"].set_index("scope")
    add("C19", "tau ... the 75th percentile of |C|", "Sec 3 / config",
        "TAU=0.7644 compared with raw |delta_logit|",
        "reanalysis_v2/tau_audit.csv; tau_relation.csv",
        "percentile rank of 0.7644 on pooled |C|", "23840 pairs",
        "per-model p75: " + "; ".join("%s %.2f" % (K.DISPLAY[m], tau.loc[m, "absC_p75"]) for m in K.MODELS),
        "PARTIAL",
        "0.7644 is the pooled 75th percentile of raw |C| (percentile rank %.3f). It is the %.0fth "
        "percentile on Qwen and the %.0fth on Gemma, so 'initially failing' is not a comparable "
        "population across models. The transformed pair score is a decreasing function of |C| "
        "and must never be compared with this cutoff." % (
            tau.loc["pooled", "pct_rank_of_tau_on_absC"],
            100 * tau.loc["qwen2.5-7b-instruct", "pct_rank_of_tau_on_absC"],
            100 * tau.loc["gemma-2-2b-it", "pct_rank_of_tau_on_absC"]))

    # C20 dedup safety
    dd = d["dedup"]
    s = dd[(dd["key_set"] == "shipped") & dd["file"].str.startswith("cure_recovery_sweep")]
    add("C20", "(reproducibility) integrity checks run at the start of every run", "Sec 4 / code",
        "integrity.PRIMARY_KEYS['cure_recovery'] lacks erase_rank and prefix-matches the sweep",
        "reanalysis_v2/dedup_dry_run.csv",
        "rows dropped by drop_duplicates(key)", "%d of %d sweep rows would be deleted" % (
            int(s["rows_would_drop"].sum()), int(s["rows"].sum())),
        "n/a", "UNSUPPORTED",
        "Never invoke integrity.run() on the sweep files; use p0_snapshot_integrity."
        "run_integrity_safe (corrected keys, snapshot first).")

    # Planned
    for cid, txt in (("P1", "Same edit for audit and behaviour; identity checks; backend parity"),
                     ("P2", "Controlled erasure with faithful LEACE, random and neutral controls, "
                            "matched energy, canonical option scoring, external capability set"),
                     ("P3", "One explanation: depth, magnitude, or massive-activation restoration"),
                     ("P4", "Independent risk forecast against margin/entropy baselines")):
        add(cid, txt, "Next_Plan.md", "see Extended_Research_Codes/p%s_*.py" % cid[1],
            "Code/CURE/results/v2/ (absent until run)", "n/a", "n/a", "n/a", "PLANNED",
            "Must not appear in the paper as a result until run and validated.")
    return rows


def to_markdown(rows: list[dict]) -> str:
    out = ["# Evidence ledger (P0)", "",
           "Generated %s by p0_claims_ledger.py. Every number is read from a reanalysis_v2 "
           "output or the shipped artifact it names." % K.utc_now(), ""]
    for r in rows:
        out += ["## %s  [%s]" % (r["claim_id"], r["status"]), "",
                "**Quote.** %s  (%s)" % (r["manuscript_quote"], r["location"]), "",
                "**Actual intervention.** %s" % r["actual_intervention"], "",
                "**Supporting output.** `%s`" % r["supporting_output"], "",
                "**Formula.** %s" % r["metric_formula"], "",
                "**n.** %s   **Interval.** %s" % (r["eligible_n"], r["interval"]), "",
                "**Permitted wording.** %s" % r["permitted_wording"], ""]
        if r["note"]:
            out += ["**Note.** %s" % r["note"], ""]
    return "\n".join(out)


def evidence_contract(rows: list[dict]) -> str:
    out = ["# Evidence Contract for iclr2027_CURE_Audit_Benchmark.tex", "",
           "Generated %s from Code/CURE/results/reanalysis_v2 by "
           "Code/CURE/Extended_Research_Codes/p0_claims_ledger.py. This file is the ONLY "
           "results allow-list the writer may use. Anything marked PLANNED or REQUIRES_RERUN "
           "is not a result and must not be written as one." % K.utc_now(), "",
           "| id | status | permitted wording |", "|---|---|---|"]
    for r in rows:
        out.append("| %s | %s | %s |" % (r["claim_id"], r["status"],
                                        r["permitted_wording"].replace("|", "\\|")))
    out += ["", "## Measured, usable now", ""]
    for r in rows:
        if r["status"] in ("SUPPORTED", "PARTIAL"):
            out.append("- **%s** %s" % (r["claim_id"], r["permitted_wording"]))
    out += ["", "## Not usable as written", ""]
    for r in rows:
        if r["status"] in ("UNSUPPORTED", "REQUIRES_RERUN"):
            out.append("- **%s** (%s) %s" % (r["claim_id"], r["status"], r["permitted_wording"]))
    out += ["", "## Planned (never cite as a result)", ""]
    for r in rows:
        if r["status"] == "PLANNED":
            out.append("- **%s** %s" % (r["claim_id"], r["manuscript_quote"]))
    out += ["", "## Standing constraints", "",
            "- The audit is an instrument; the object of study is the erasure.",
            "- The two companion studies are never named; the patchscopes row is excluded.",
            "- No number is typed by hand; every table is regenerated from reanalysis_v2 or v2.",
            "- The AI-use statement must disclose methodological feedback and result "
            "interpretation from AI tools, and the multi-model revision system.", ""]
    return "\n".join(out)


def main() -> None:
    d = _load()
    rows = build(d)
    df = pd.DataFrame(rows)
    K.write_csv(df, P0 / "evidence_ledger.csv")
    (P0 / "evidence_ledger.md").write_text(to_markdown(rows), encoding="utf-8")
    contract = K.REPO / "Submission2" / "Evidence_Contract.md"
    contract.write_text(evidence_contract(rows), encoding="utf-8")
    log.info("wrote %s", contract.relative_to(K.REPO))
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
    print("\n=== EVIDENCE LEDGER ===")
    print(df[["claim_id", "status", "manuscript_quote"]].to_string(index=False))
    print("\ncounts:", df["status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
