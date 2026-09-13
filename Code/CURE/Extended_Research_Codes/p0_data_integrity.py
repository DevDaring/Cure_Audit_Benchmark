"""
p0_data_integrity.py -- join the degeneracy mask and quantify contamination by stage.

Next_Plan.md, Section 2.5 and P0 row 3:

  CURE loads pentad_dataset.parquet, not pentad_dataset_clean.parquet. The pair loader does
  not apply the existing degeneracy mask. Joining the saved rank-1 sweep to
  cdva_degeneracy_mask.parquet gives 191 malformed pairs out of 1,000 in every model.
  Filtering evaluation rows after the fact cannot remove the influence of malformed fitting
  pairs on the learned basis.

  cmd_main samples subspace and sweep pairs from the same pool with the same random seed.
  cmd_baselines uses a different random seed for the comparison, but does not exclude
  fitting or sweep seeds. Different random draws are not disjoint datasets.

What this does
  1. Reconstruct on CPU, with the shipped code's own logic (common.load_pairs_cpu and
     common.stratified_subset_cpu), the three pair sets each shipped run used:
        fit    = stratified_subset(pairs, 600,  RANDOM_SEED)      cmd_main and cmd_baselines
        sweep  = stratified_subset(pairs, 1000, RANDOM_SEED)      cmd_main (ranks, prognosis)
        eval   = stratified_subset(pairs, 1000, RANDOM_SEED + 7)  cmd_baselines head-to-head
     and VERIFY the reconstruction: the sweep set must equal the (seed_id, subvariant_A,
     subvariant_B) keys in cure_recovery_sweep_<model>.parquet exactly. If it does not, every
     downstream contamination count is marked unverified.
  2. Join each set to the degeneracy mask and count pairs that touch a degenerate variant.
  3. Measure overlap between stages at the PAIR level and at the SEED level: fit vs sweep,
     fit vs eval, sweep vs eval. Seed-level overlap is what matters for leakage, because a
     basis fitted on one pair of a seed has seen that seed's template.
  4. Report what the clean pentad would have done: pair counts with the clean file and the
     mask applied before fitting.

Outputs
  reanalysis_v2/data_integrity.csv           per model x stage: n, n_degenerate, share, verified
  reanalysis_v2/stage_overlap.csv            pair- and seed-level overlaps between stages
  reanalysis_v2/pair_sets/<model>_<stage>.csv the reconstructed keys, for later manifests

Usage:
  python p0_data_integrity.py
"""

from __future__ import annotations

import pandas as pd

import common as K

log = K.setup_logging("cure.ext.p0.data")

STAGES = {
    "fit": (K.SUBSPACE_PAIRS, K.RANDOM_SEED_CURE),
    "sweep": (K.SWEEP_SUBSET, K.RANDOM_SEED_CURE),
    "eval_headtohead": (K.SWEEP_SUBSET, K.RANDOM_SEED_CURE + 7),
}


def reconstruct(model: str) -> dict[str, list[dict]]:
    pairs = K.load_pairs_cpu(model, K.PENTAD_RAW)
    out = {"all": pairs}
    for stage, (n, seed) in STAGES.items():
        out[stage] = K.stratified_subset_cpu(pairs, n, seed)
    return out


def verify_against_sweep(model: str, sweep_pairs: list[dict]) -> tuple[bool, int, int]:
    saved = K.read_sweep(model)
    saved = saved[saved["erase_rank"] == 1]
    saved_keys = set(zip(saved["seed_id"], saved["subvariant_A"], saved["subvariant_B"]))
    recon_keys = {K.pair_key(p) for p in sweep_pairs}
    return saved_keys == recon_keys, len(saved_keys & recon_keys), len(saved_keys ^ recon_keys)


def degenerate_join(model: str, pairs: list[dict], mask: pd.DataFrame) -> tuple[int, int, int]:
    """Returns (n_pairs, n_touching_degenerate, n_unmatched_in_mask)."""
    if not pairs:
        return 0, 0, 0
    df = pd.DataFrame(pairs)[["seed_id", "subvariant_A", "subvariant_B"]]
    m = mask[mask["model_name"] == model].rename(
        columns={"pair_A_subvariant": "subvariant_A", "pair_B_subvariant": "subvariant_B"})
    j = df.merge(m[["seed_id", "subvariant_A", "subvariant_B", "touches_degenerate"]],
                 on=["seed_id", "subvariant_A", "subvariant_B"], how="left")
    unmatched = int(j["touches_degenerate"].isna().sum())
    return len(df), int(j["touches_degenerate"].fillna(False).astype(bool).sum()), unmatched


def overlap(a: list[dict], b: list[dict]) -> dict:
    pa, pb = {K.pair_key(p) for p in a}, {K.pair_key(p) for p in b}
    sa, sb = {p["seed_id"] for p in a}, {p["seed_id"] for p in b}
    return {"pairs_a": len(pa), "pairs_b": len(pb), "pairs_shared": len(pa & pb),
            "seeds_a": len(sa), "seeds_b": len(sb), "seeds_shared": len(sa & sb),
            "seed_share_of_b_seen_in_a": (len(sa & sb) / len(sb)) if sb else float("nan")}


def saved_sweep_join(model: str, mask: pd.DataFrame) -> dict:
    """The plan's own computation: join the SAVED rank-1 sweep rows to the mask. This does
    not depend on any reconstruction, so it is the authoritative contamination count."""
    sw = K.read_sweep(model)
    sw1 = sw[sw["erase_rank"] == 1]
    pairs = sw1[["seed_id", "subvariant_A", "subvariant_B"]].to_dict("records")
    n, nd, unmatched = degenerate_join(model, pairs, mask)
    keys = {(r["seed_id"], r["subvariant_A"], r["subvariant_B"]) for r in pairs}
    return {"model_name": model, "stage": "sweep_SAVED_ARTIFACT", "n_pairs": n,
            "n_seeds": int(sw1["seed_id"].nunique()), "n_touching_degenerate": nd,
            "share_degenerate": (nd / n) if n else float("nan"), "n_unmatched_in_mask": unmatched,
            "reconstruction_verified_against_saved_sweep": True,
            "pentad_used_by_shipped_loader": "authoritative: joined from cure_recovery_sweep parquet",
            "_keys": keys}


def main() -> None:
    K.OUT_P0.mkdir(parents=True, exist_ok=True)
    (K.OUT_P0 / "pair_sets").mkdir(exist_ok=True)
    mask = K.read_mask()
    integ_rows, ov_rows = [], []

    for model in K.MODELS:
        # 0. authoritative count from the saved artifact (no reconstruction involved)
        saved = saved_sweep_join(model, mask)
        saved_keys = saved.pop("_keys")
        integ_rows.append(saved)
        pd.DataFrame(sorted(saved_keys), columns=["seed_id", "subvariant_A", "subvariant_B"]).to_csv(
            K.OUT_P0 / "pair_sets" / f"{model}_sweep_SAVED.csv", index=False)

        sets = reconstruct(model)
        ok, n_common, n_diff = verify_against_sweep(model, sets["sweep"])
        log.info("%s: sweep reconstruction %s (common=%d, symmetric diff=%d)",
                 K.DISPLAY[model], "VERIFIED" if ok else "MISMATCH", n_common, n_diff)
        # Structural fact independent of the draw: stratified_subset(pairs, 600, s) is a
        # per-benchmark prefix of the same shuffled groups as stratified_subset(pairs, 1000, s),
        # so the fit set is ALWAYS contained in the sweep set. Assert it on the replay.
        fit_in_sweep = {K.pair_key(p) for p in sets["fit"]} <= {K.pair_key(p) for p in sets["sweep"]}
        assert fit_in_sweep, "fit subset is not inside sweep subset; stratified_subset changed"
        all_in_pool = saved_keys <= {K.pair_key(p) for p in sets["all"]}
        integ_rows.append({
            "model_name": model, "stage": "NOTE_reconstruction", "n_pairs": n_common,
            "n_seeds": None, "n_touching_degenerate": None, "share_degenerate": None,
            "n_unmatched_in_mask": n_diff,
            "reconstruction_verified_against_saved_sweep": ok,
            "pentad_used_by_shipped_loader": (
                "n_pairs = replayed sweep keys that match the saved sweep; n_unmatched = symmetric "
                "difference. The saved sweep keys are %s inside the replayed pool, so the pool is "
                "right and the DRAW differs: the run-time row order of cdva_results.parquet is not "
                "recoverable, hence the exact fitting set was never recorded (no manifest). The "
                "containment fit SUBSET-OF sweep holds by construction and on the replay (%s)."
                % ("all" if all_in_pool else "NOT all", "confirmed" if fit_in_sweep else "FAILED")),
        })

        for stage, pairs in sets.items():
            n, nd, unmatched = degenerate_join(model, pairs, mask)
            integ_rows.append({
                "model_name": model, "stage": stage, "n_pairs": n,
                "n_seeds": len({p["seed_id"] for p in pairs}),
                "n_touching_degenerate": nd,
                "share_degenerate": (nd / n) if n else float("nan"),
                "n_unmatched_in_mask": unmatched,
                "reconstruction_verified_against_saved_sweep": ok,
                "pentad_used_by_shipped_loader": "pentad_dataset.parquet (raw, no mask)",
            })
            pd.DataFrame(pairs)[["seed_id", "subvariant_A", "subvariant_B", "benchmark"]].to_csv(
                K.OUT_P0 / "pair_sets" / f"{model}_{stage}.csv", index=False)

        # what the clean file gives, before any mask on pairs
        clean_pairs = K.load_pairs_cpu(model, K.PENTAD_CLEAN)
        n, nd, unmatched = degenerate_join(model, clean_pairs, mask)
        integ_rows.append({
            "model_name": model, "stage": "all_with_clean_pentad", "n_pairs": n,
            "n_seeds": len({p["seed_id"] for p in clean_pairs}),
            "n_touching_degenerate": nd, "share_degenerate": (nd / n) if n else float("nan"),
            "n_unmatched_in_mask": unmatched,
            "reconstruction_verified_against_saved_sweep": ok,
            "pentad_used_by_shipped_loader": "pentad_dataset_clean.parquet (counterfactual)",
        })
        # eligible pool: clean pentad AND mask-clean pairs (what P2 should fit and test on)
        m = mask[(mask["model_name"] == model) & (~mask["touches_degenerate"].astype(bool))]
        clean_keys = set(zip(m["seed_id"], m["pair_A_subvariant"], m["pair_B_subvariant"]))
        eligible = [p for p in clean_pairs if K.pair_key(p) in clean_keys]
        integ_rows.append({
            "model_name": model, "stage": "eligible_pool_clean_and_masked", "n_pairs": len(eligible),
            "n_seeds": len({p["seed_id"] for p in eligible}),
            "n_touching_degenerate": 0, "share_degenerate": 0.0, "n_unmatched_in_mask": 0,
            "reconstruction_verified_against_saved_sweep": ok,
            "pentad_used_by_shipped_loader": "clean pentad + mask applied before any stage",
        })
        pd.DataFrame(eligible)[["seed_id", "subvariant_A", "subvariant_B", "benchmark"]].to_csv(
            K.OUT_P0 / "pair_sets" / f"{model}_eligible.csv", index=False)

        for a, b in (("fit", "sweep"), ("fit", "eval_headtohead"), ("sweep", "eval_headtohead")):
            ov_rows.append({"model_name": model, "stage_a": a, "stage_b": b, **overlap(sets[a], sets[b]),
                            "basis": ("structural: fit is a prefix-subset of sweep by construction"
                                      if (a, b) == ("fit", "sweep") else
                                      "replay only; exact draw unverified (see NOTE_reconstruction)")})

    integ = pd.DataFrame(integ_rows)
    K.write_csv(integ, K.OUT_P0 / "data_integrity.csv")
    ov = pd.DataFrame(ov_rows)
    K.write_csv(ov, K.OUT_P0 / "stage_overlap.csv")

    pd.set_option("display.width", 220)
    print("\n=== CONTAMINATION BY STAGE (shipped loader = raw pentad, no mask) ===")
    print(integ[["model_name", "stage", "n_pairs", "n_seeds", "n_touching_degenerate",
                 "share_degenerate", "n_unmatched_in_mask",
                 "reconstruction_verified_against_saved_sweep"]].round(3).to_string(index=False))
    print("\nNext_Plan.md Section 2.5 states 191 malformed pairs out of 1,000 in every model's "
          "sweep, with no unmatched keys.")
    print("\n=== OVERLAP BETWEEN STAGES (a basis fitted on `a` has seen these seeds of `b`) ===")
    print(ov.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
