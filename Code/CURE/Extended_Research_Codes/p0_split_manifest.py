"""
p0_split_manifest.py -- seed-group fit / dev / test manifests with zero intersection.

Next_Plan.md, P0 row 4 and P2 "Suggested split":

  Build seed-group fit/dev/test manifests. Replace independent pair draws. Zero intersection
  of source-item groups; benchmark/category/variant counts; documented exclusions.

  After integrity filtering, allocate approximately 50% of eligible seed groups to fitting,
  20% to development and 30% to final testing, stratified by benchmark and feasible
  demographic categories. Keep related source templates/items together. All models and
  methods use the same final-test manifest where eligible.

Unit of allocation: the SEED (source item). Every counterfactual pair of a seed goes with it,
so no pair of a test seed can enter fitting. The split is model-agnostic (one manifest for all
four models); per-model eligibility is then the intersection with that model's mask-clean
pairs, which p0_data_integrity wrote to pair_sets/<model>_eligible.csv.

Eligibility (documented exclusions)
  1. prompt rows come from pentad_dataset_clean.parquet (the audit's integrity-filtered file)
  2. a pair is eligible for a model only if cdva_degeneracy_mask.touches_degenerate is False
  3. a seed is eligible if it has at least MIN_PAIRS eligible pairs on EVERY model, so the
     same test manifest serves all models
  4. seeds whose slot-c variants all share one gold answer string of "unknown" are kept but
     flagged, because the scored token then falls back to the swap token (Section 2.8)

Stratification: seed_source x seed_category. Within each stratum seeds are shuffled with a
fixed seed and cut 50/20/30; remainders go to test, then dev, then fit, so small strata never
lose their test share.

Outputs
  reanalysis_v2/split_manifest.csv        seed_id, seed_source, seed_category, split, n_pairs
                                          per model, flags
  reanalysis_v2/split_manifest_summary.csv counts by split x source x category
  reanalysis_v2/split_manifest.json       the assertion record (zero intersection) and the
                                          exclusions with counts

Usage:
  python p0_split_manifest.py
"""

from __future__ import annotations

import random

import pandas as pd

import common as K

log = K.setup_logging("cure.ext.p0.split")

FRACS = {"fit": 0.50, "dev": 0.20, "test": 0.30}
MIN_PAIRS = 2


def eligible_pairs_per_model() -> dict[str, pd.DataFrame]:
    out = {}
    for m in K.MODELS:
        p = K.OUT_P0 / "pair_sets" / f"{m}_eligible.csv"
        if not p.exists():
            raise SystemExit("run p0_data_integrity.py first (missing %s)" % p)
        out[m] = pd.read_csv(p)
    return out


def main() -> None:
    K.OUT_P0.mkdir(parents=True, exist_ok=True)
    pen = pd.read_parquet(K.PENTAD_CLEAN)
    seeds = (pen[pen["slot"] == "a"][["seed_id", "seed_source", "seed_category", "seed_subcategory"]]
             .drop_duplicates("seed_id").set_index("seed_id"))
    n_all = len(seeds)

    per_model = eligible_pairs_per_model()
    counts = pd.DataFrame({m: d.groupby("seed_id").size() for m, d in per_model.items()}).fillna(0).astype(int)
    counts = counts.reindex(seeds.index).fillna(0).astype(int)
    ok = (counts >= MIN_PAIRS).all(axis=1)
    excluded_min_pairs = int((~ok).sum())

    # unknown-gold flag (token fallback to swap token; Section 2.8 target-token specificity)
    c = pen[pen["slot"] == "c"]
    gold_unknown = c.groupby("seed_id")["gold_answer"].apply(
        lambda s: all(str(x).strip().lower() in ("unknown", "nan", "") for x in s))
    elig = seeds[ok].copy()
    elig["gold_unknown_fallback"] = elig.index.map(lambda s: bool(gold_unknown.get(s, False)))

    # stratified 50/20/30 by source x category, seeds shuffled with a fixed seed
    rng = random.Random(K.RANDOM_SEED_V2)
    split = {}
    for (src, cat), g in elig.groupby(["seed_source", "seed_category"], sort=True):
        ids = sorted(g.index.tolist())
        rng.shuffle(ids)
        n = len(ids)
        n_fit = int(n * FRACS["fit"]); n_dev = int(n * FRACS["dev"])
        n_test = n - n_fit - n_dev                     # remainder to test
        for s in ids[:n_fit]:
            split[s] = "fit"
        for s in ids[n_fit:n_fit + n_dev]:
            split[s] = "dev"
        for s in ids[n_fit + n_dev:]:
            split[s] = "test"
        if n_test == 0 and n >= 2:                     # never let a stratum lose its test share
            split[ids[-1]] = "test"
    elig["split"] = elig.index.map(split)
    for m in K.MODELS:
        elig[f"n_pairs_{m}"] = counts.loc[elig.index, m].values

    # assertions
    sets = {k: set(elig.index[elig["split"] == k]) for k in FRACS}
    assert not (sets["fit"] & sets["dev"]), "fit/dev intersect"
    assert not (sets["fit"] & sets["test"]), "fit/test intersect"
    assert not (sets["dev"] & sets["test"]), "dev/test intersect"
    assert sum(len(v) for v in sets.values()) == len(elig)
    # pair-level: no pair of a test seed appears in any model's fit pairs, by construction
    for m, d in per_model.items():
        d_test = d[d["seed_id"].isin(sets["test"])]
        d_fit = d[d["seed_id"].isin(sets["fit"])]
        assert not (set(d_test["seed_id"]) & set(d_fit["seed_id"])), m

    out = elig.reset_index()
    K.write_csv(out, K.OUT_P0 / "split_manifest.csv")
    summ = (out.groupby(["split", "seed_source", "seed_category"]).size()
            .rename("n_seeds").reset_index())
    K.write_csv(summ, K.OUT_P0 / "split_manifest_summary.csv")
    rec = {
        "generated_utc": K.utc_now(), "random_seed": K.RANDOM_SEED_V2, "fractions": FRACS,
        "unit": "seed_id (source item); all counterfactual pairs of a seed share its split",
        "n_seeds_clean_pentad": n_all, "n_seeds_eligible": int(len(elig)),
        "exclusions": {"fewer_than_%d_mask_clean_pairs_on_some_model" % MIN_PAIRS: excluded_min_pairs},
        "split_sizes": {k: len(v) for k, v in sets.items()},
        "pairs_per_split_per_model": {m: {k: int(d[d["seed_id"].isin(sets[k])].shape[0]) for k in FRACS}
                                      for m, d in per_model.items()},
        "n_gold_unknown_fallback": int(elig["gold_unknown_fallback"].sum()),
        "assertions": {"fit_dev_disjoint": True, "fit_test_disjoint": True, "dev_test_disjoint": True,
                       "no_test_seed_pair_in_fit_any_model": True},
        "note": ("The existing pool has already informed the project (Next_Plan.md P2): call the "
                 "test split a new controlled held-out evaluation, not a historically untouched "
                 "test. Cap dev/test evaluation by stratified sampling from these splits, never "
                 "by file-order truncation."),
    }
    K.write_json(rec, K.OUT_P0 / "split_manifest.json")

    print("\n=== SPLIT MANIFEST ===")
    print("eligible seeds: %d of %d (excluded %d with <%d clean pairs on some model)"
          % (len(elig), n_all, excluded_min_pairs, MIN_PAIRS))
    print("split sizes:", rec["split_sizes"])
    print("pairs per split per model:")
    for m, v in rec["pairs_per_split_per_model"].items():
        print("  %-24s %s" % (K.DISPLAY[m], v))
    print(out.groupby(["split", "seed_source"]).size().unstack(fill_value=0).to_string())
    print("all intersections empty: True")


if __name__ == "__main__":
    main()
