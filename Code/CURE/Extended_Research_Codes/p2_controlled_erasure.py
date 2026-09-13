"""
p2_controlled_erasure.py -- the corrected, controlled erasure experiment (Next_Plan.md, P2). GPU.

Purpose: test the central claim after the pilot, with faithful controls and independent
utility measurement, under ONE edit definition (p1_intervention: site = "span", source state
= "sequential", prefill only) applied identically to the audit, the option readout, the
generation readout and, under an explicit policy, the external capability probes.

Design (Next_Plan.md P2, item by item)
  Split       reanalysis_v2/split_manifest.csv (p0_split_manifest). Fit pairs are mask-clean
              pairs of FIT seeds, stratified by benchmark (p1_pilot.fit_pairs). Development
              evaluation is a stratified sample (by seed_source, never file order) of up to
              100 DEV seeds; final evaluation up to 160 TEST seeds. Every model and method
              uses the SAME test seed list, saved once to results/v2/p2_test_seeds.csv and
              re-read on every later run. The test split is a new controlled held-out
              evaluation, not a historically untouched test (P2, second paragraph).
  Per seed    up to three pairs from mask-clean slot-c variants (p1_pilot.pairs_for_seed):
              demographic (largest |C_pre| among clean pairs), identity (a variant against
              itself) and neutral (a 'person'-style substitution) when the seed has one.
  Conditions  (family; all edited conditions use site="span", source_state="sequential")
    unedited            no hooks                                   rank 0, alpha 0
    identity_alpha0     cure basis, alpha = 0; the alpha = 0 point of the strength screen and
                        the implementation-validity check (asserted equal to unedited)
    cure_centred_svd    the shipped estimator: leading right singular vectors of CENTRED
                        span-position difference vectors (erase._svd_components), correctly
                        applied at the span during prefill
    uncentred_svd       same differences, no centring
    mean_difference     rank-1 normalised mean difference (the contrast centring discards)
    leace_faithful      LEACE (Belrose et al. 2023) via the official concept_erasure package,
                        fitted on the same span-position activations with label = swap side,
                        applied as its exact affine map (p2_leace_faithful.LeaceAffineEdit);
                        frozen fitting, always.
    leace_sequential    the same package fitted by concept scrubbing (layer l on activations
                        already edited by layers < l), only with --leace-sequential.
                        status = "unavailable" if the package is not installed
    random_ortho_{1,2,3} random orthonormal subspaces of matched rank at the same layers,
                        seeds RANDOM_SEED_V2 + {1, 2, 3}
    neutral_contrast    centred SVD fitted on NEUTRAL fit pairs only (exactly one side's swap
                        token in p1_pilot.NEUTRAL_TOKENS); status = "insufficient" if fewer
                        than 30 such pairs exist for the model
  Screen      dev: cure at rank 1 with alpha in {0, 0.25, 0.5, 1}; controls at rank 1 and
              alpha 1; ranks 2 and 4 only with --extend-ranks; rank 8 (or any) via --ranks.
  Energy      every edited row records E[||h-h'||^2]/E[||h||^2] by layer (HookedEdit
              .energy_by_layer). --match-energy grid-searches, on DEV prompts, an alpha in
              linspace(0.05, 1, 20) per control so its mean removed energy matches cure's at
              alpha = 1, freezes the result in results/v2/p2_matched_alphas.json, and the
              test phase uses those alphas unchanged (conditions tagged "_matched").
  Readouts    per item: raw C and |C| (p1_intervention.commutator on the gold-answer token);
              option log-likelihoods on both prompts (argmax, margin, entropy, probs, per-
              option loglik_sum, correctness against the gold index mapped through the swap);
              greedy generation (raw text, parsed option index, validity, correctness); per-
              layer energy for the audit and the generation edits; basis fit id (sha256 of
              the basis npz); actual rank; alpha; site; seconds; model and tokenizer ids.
              Malformed output is recorded as invalid, never dropped.
  Sentence    for crows_pairs / stereoset items the per-option loglik_sum over the FULL option
  preference  strings is the benchmark-appropriate sentence preference; it is stored for every
              item and flagged by `sentence_pref_applicable`. The question-answer recast that
              the pentad prompts implement is NOT the native CrowS-Pairs / StereoSet protocol
              and is never described as such.
  Utility     capability_eval: (a) a fixed stratified 200-question MMLU test subset scored by
              option_loglik over the four answer letters; (b) about 20,000 WikiText-2 test
              tokens scored by mean negative log-likelihood per token in 1024-token windows.
              These prompts have no demographic span, so the edit's application site is an
              explicit policy: --capability-policy none (default) evaluates only the unedited
              model and records every edited condition as not evaluated; --capability-policy
              all applies the edit at EVERY position (site="all"). The policy is written into
              every capability row and the report. The site is never switched silently.
  Statistics  summarise(): per model x phase x condition x pair_type, pooled and by benchmark:
              W (share |C| decreased), L (worsened), U (unchanged), D = mean(|C_pre| -
              |C_post|) paired absolute change (primary), R = 1 - mean|C_post|/mean|C_pre|
              (secondary), X = share of pairs with |C_pre| >= TAU that fall below TAU (both
              denominators shown); option and generation accuracy with attempted AND usable
              denominators; validity; flip rate with both denominators; seed-cluster
              percentile bootstrap intervals (common.ratio_bootstrap). Four predeclared
              confirmatory comparisons (cure vs unedited on D and on generation accuracy;
              cure vs the mean of the three rank-matched random controls on D and on
              accuracy), Holm-corrected within model; everything else is exploratory and
              labelled so.
  Gate        Next_Plan.md P2 "Decision gate", written into p2_report.md per model.
  Budget      --gpu-hours-cap is a wall-clock cap including loading and fitting; the runner
              stops cleanly and everything done is on disk and resumable.

Outputs (Code/CURE/results/v2/)
  p2_dev_seeds.csv, p2_test_seeds.csv     the frozen evaluation seed lists
  p2_basis_<model>_<fmt>_<name>.npz       every fitted basis / eraser; sha256 = basis fit id
  p2_basis_manifest_<model>_<fmt>.json    fit records, statuses, fit ids
  p2_matched_alphas.json                  energy-matched control strengths chosen on DEV
  p2_per_item.parquet                     one row per model x phase x condition x seed x
                                          pair_type (plus one status row per unavailable or
                                          insufficient condition); resumable
  p2_capability.parquet                   one row per model x phase x condition x probe
  p2_protocol.json                        definitions, identity checks, timings
  p2_summary.csv, p2_confirmatory.csv, p2_report.md

Implements
  - Belrose et al. (2023) LEACE, arXiv:2306.03819, NeurIPS 2023 (faithful control, through
    p2_leace_faithful; the shipped eraser is the projection form only)
  - Efron and Tibshirani (1993) An Introduction to the Bootstrap (seed-cluster percentile
    intervals and bootstrap p-values)
  - Holm (1979) A simple sequentially rejective multiple test procedure, Scand. J. Statist.
    6(2):65-70 (multiplicity control over the predeclared set)

Usage
  python p2_controlled_erasure.py --phase dev  --models qwen2.5-7b-instruct --match-energy
  python p2_controlled_erasure.py --phase test --models qwen2.5-7b-instruct llama-3.1-8b-instruct \
                                  --match-energy --capability-policy all
  python p2_controlled_erasure.py --phase all --smoke
  python p2_controlled_erasure.py --summarise-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import common as K
import p1_intervention as I
import p1_pilot as P1

log = K.setup_logging("cure.ext.p2.erasure")

OUT = K.OUT_V2
SITE, SOURCE_STATE = "span", "sequential"
N_DEV_CAP, N_TEST_CAP = 100, 160
N_FIT_PAIRS = 300
MIN_NEUTRAL_PAIRS = 30
ALPHA_SCREEN = [0.25, 0.5, 1.0]          # identity_alpha0 is the alpha = 0 point
EXTENDED_RANKS = [2, 4]
ENERGY_GRID = [float(x) for x in np.linspace(0.05, 1.0, 20)]
MAX_NEW_TOKENS = 48
N_MMLU, N_WIKI_TOKENS, WINDOW = 200, 20000, 1024
PAIR_TYPES = ["demographic", "identity", "neutral"]
SENTENCE_PREF_BENCHMARKS = ("crows_pairs", "stereoset")

FAMILIES = ["unedited", "identity_alpha0", "cure_centred_svd", "uncentred_svd", "mean_difference",
            "leace_faithful", "leace_sequential", "random_ortho_1", "random_ortho_2", "random_ortho_3",
            "neutral_contrast"]
RANDOM_FAMILIES = ["random_ortho_1", "random_ortho_2", "random_ortho_3"]
LEACE_FAMILIES = ["leace_faithful", "leace_sequential"]   # frozen fit; concept-scrubbing fit
CONTROL_FAMILIES = ["uncentred_svd", "mean_difference"] + LEACE_FAMILIES + RANDOM_FAMILIES + ["neutral_contrast"]
RANKED_FAMILIES = ["cure_centred_svd", "uncentred_svd", "neutral_contrast"] + RANDOM_FAMILIES
BASIS_OF = {f: f for f in FAMILIES}
BASIS_OF["identity_alpha0"] = "cure_centred_svd"
KEY_COLS = ["model_name", "phase", "cond_id", "seed_id", "pair_type"]
UNEDITED_ID = "unedited_r0_a0"
CURE_ID = "cure_centred_svd_r1_a1"
RANDOM_IDS = [f"{f}_r1_a1" for f in RANDOM_FAMILIES]
CONFIRMATORY = [("C1", "cure vs unedited on D (paired absolute |C| change), demographic pairs"),
                ("C2", "cure vs unedited on generation accuracy (attempted denominator)"),
                ("C3", "cure vs mean of three rank-matched random controls on D"),
                ("C4", "cure vs mean of three rank-matched random controls on generation accuracy")]
SENTENCE_PREF_NOTE = ("loglik_sum over full option strings; the QA recast is not the native "
                      "CrowS-Pairs / StereoSet protocol")


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cond:
    family: str
    rank: int
    alpha: float
    tag: str = ""
    status: str = "ok"

    @property
    def cond_id(self) -> str:
        return f"{self.family}_r{self.rank}_a{self.alpha:g}" + (f"_{self.tag}" if self.tag else "")

    @property
    def kind(self) -> str:
        if self.family == "unedited":
            return "none"
        return "affine" if self.family in LEACE_FAMILIES else "proj"


def build_conditions(phase: str, ranks: list[int], store: "BasisStore", matched: dict | None,
                     families: list[str]) -> list[Cond]:
    """The condition list for one phase. Statuses come from the basis store; matched alphas
    (frozen on dev) add a tagged copy of each control."""
    conds = [Cond("unedited", 0, 0.0), Cond("identity_alpha0", 1, 0.0)]
    alphas = ALPHA_SCREEN if phase == "dev" else [1.0]
    for a in alphas:
        conds.append(Cond("cure_centred_svd", 1, a))
    for r in [x for x in ranks if x != 1]:
        conds.append(Cond("cure_centred_svd", r, 1.0))
    for fam in CONTROL_FAMILIES:
        st = store.status.get(BASIS_OF[fam], "ok")
        if st != "ok":
            conds.append(Cond(fam, 0, 0.0, status=st)); continue
        fam_ranks = ranks if fam in RANKED_FAMILIES else [store.natural_rank(fam)]
        for r in fam_ranks:
            conds.append(Cond(fam, r, 1.0))
        if matched and fam in matched.get("controls", {}):
            a = float(matched["controls"][fam]["alpha"])
            if abs(a - 1.0) > 1e-9:
                conds.append(Cond(fam, fam_ranks[0], a, tag="matched"))
    conds = [c for c in conds if c.family in set(families) | {"unedited"}]      # unedited is the pairing reference
    seen, out = set(), []
    for c in conds:
        if c.cond_id not in seen:
            seen.add(c.cond_id); out.append(c)
    return out


# ---------------------------------------------------------------------------
# data selection (CPU)
# ---------------------------------------------------------------------------

def stratified_seed_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional allocation by seed_source with largest-remainder rounding, seeds shuffled
    within stratum by a fixed generator. Never file-order truncation."""
    df = df.sort_values(["seed_source", "seed_id"]).reset_index(drop=True)
    if n >= len(df):
        return df
    rng = random.Random(seed)
    groups = {src: sorted(g["seed_id"]) for src, g in df.groupby("seed_source")}
    quota = {s: n * len(ids) / len(df) for s, ids in groups.items()}
    base = {s: int(q) for s, q in quota.items()}
    for s in sorted(quota, key=lambda k: -(quota[k] - base[k]))[:n - sum(base.values())]:
        base[s] += 1
    take = []
    for s in sorted(groups):
        ids = list(groups[s]); rng.shuffle(ids); take += ids[:base[s]]
    return df[df["seed_id"].isin(take)].sort_values(["seed_source", "seed_id"]).reset_index(drop=True)


def select_eval_seeds(split: str, n_cap: int, seed: int, path: Path) -> pd.DataFrame:
    """Frozen once: if `path` exists it is re-read so every model and method sees the same
    list; otherwise a stratified sample of the manifest split is drawn and written."""
    man = pd.read_csv(K.OUT_P0 / "split_manifest.csv")
    pool = man[man["split"] == split]
    if path.exists():
        df = pd.read_csv(path)
        missing = set(df["seed_id"]) - set(pool["seed_id"])
        if missing:
            raise SystemExit("%s holds seeds outside the manifest %s split: %s" % (path.name, split, sorted(missing)[:5]))
        return df
    df = stratified_seed_sample(pool, n_cap, seed)
    df = df[["seed_id", "seed_source", "seed_category", "split"]].copy()
    df["selected_utc"] = K.utc_now(); df["random_seed"] = seed
    K.write_csv(df, path)
    return df


def is_neutral_token(t) -> bool:
    return str(t).strip().lower().replace("_", " ") in P1.NEUTRAL_TOKENS


def neutral_fit_pairs(model: str, n: int, seed: int, c: pd.DataFrame) -> list[dict]:
    """Mask-clean FIT-seed pairs where exactly one side's swap token is neutral, stratified by
    benchmark. The neutral-contrast control is fitted on these only."""
    man = pd.read_csv(K.OUT_P0 / "split_manifest.csv")
    fit_seeds = set(man[man["split"] == "fit"]["seed_id"])
    elig = pd.read_csv(K.OUT_P0 / "pair_sets" / f"{model}_eligible.csv")
    elig = elig[elig["seed_id"].isin(fit_seeds)]
    swap = {(r["seed_id"], r["subvariant"]): r.get("swap_token", "") for _, r in c.iterrows()}
    ok = [is_neutral_token(swap.get((s, a), "")) != is_neutral_token(swap.get((s, b), ""))
          for s, a, b in zip(elig["seed_id"], elig["subvariant_A"], elig["subvariant_B"])]
    elig = elig[ok]
    if elig.empty:
        return []
    rng = random.Random(seed)
    take = []
    for _, g in elig.groupby("benchmark"):
        recs = g.to_dict("records"); rng.shuffle(recs)
        take += recs[:max(1, round(n * len(g) / len(elig)))]
    rng.shuffle(take)
    return take[:n]


# ---------------------------------------------------------------------------
# bases (GPU fit, CPU store)
# ---------------------------------------------------------------------------

class BasisStore:
    """Every basis of one model x fmt at its maximum fitted rank, sliced per condition.
    proj[name] = {layer: rows (max_rank x d)}; leace = {layer: eraser dict} or None."""

    def __init__(self, model: str, fmt: str):
        self.model, self.fmt = model, fmt
        self.proj: dict[str, dict[int, np.ndarray]] = {}
        self.leace: dict[int, dict] | None = None          # leace_faithful  (frozen fit)
        self.leace_seq: dict[int, dict] | None = None      # leace_sequential (concept scrubbing)
        self.status: dict[str, str] = {}
        self.info: dict[str, dict] = {}
        self.fit_id: dict[str, str] = {}
        self.max_rank = 1

    def path(self, name: str) -> Path:
        return OUT / f"p2_basis_{self.model}_{self.fmt}_{name}.npz"

    @property
    def manifest_path(self) -> Path:
        return OUT / f"p2_basis_manifest_{self.model}_{self.fmt}.json"

    def rows(self, name: str, rank: int) -> dict[int, np.ndarray]:
        return {l: r[:rank] for l, r in self.proj[name].items()}

    def erasers(self, family: str) -> dict[int, dict] | None:
        return {"leace_faithful": self.leace, "leace_sequential": self.leace_seq}.get(family)

    def natural_rank(self, family: str) -> int:
        er = self.erasers(family) if family in LEACE_FAMILIES else None
        if er:
            return int(max(e.get("actual_rank", 0) for e in er.values()))
        return 1

    def add_proj(self, name: str, basis: dict[int, np.ndarray], info: dict) -> None:
        self.proj[name] = {int(l): np.asarray(r, np.float32) for l, r in basis.items()}
        self.status[name] = "ok"; self.info[name] = info

    def mark(self, name: str, status: str, info: dict) -> None:
        self.status[name] = status; self.info[name] = info

    def save(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, basis in self.proj.items():
            np.savez_compressed(self.path(name), **{str(l): b for l, b in basis.items()})
            self.fit_id[name] = K.sha256(self.path(name))
        import p2_leace_faithful as LF
        leace_files = []
        for fam in LEACE_FAMILIES:
            er = self.erasers(fam)
            if er:
                np.savez_compressed(self.path(fam), **LF.erasers_to_npz_dict(er))
                self.fit_id[fam] = K.sha256(self.path(fam)); leace_files.append(fam)
        rec = {"model": self.model, "fmt": self.fmt, "max_rank": self.max_rank, "status": self.status,
               "info": self.info, "fit_id": self.fit_id, "saved_utc": K.utc_now(),
               "files": {n: str(self.path(n).name) for n in list(self.proj) + leace_files}}
        K.write_json(rec, self.manifest_path)

    @classmethod
    def load(cls, model: str, fmt: str) -> "BasisStore | None":
        st = cls(model, fmt)
        if not st.manifest_path.exists():
            return None
        rec = json.loads(st.manifest_path.read_text(encoding="utf-8"))
        st.status, st.info, st.fit_id = rec["status"], rec["info"], rec["fit_id"]
        st.max_rank = int(rec.get("max_rank", 1))
        for name in rec["files"]:
            if not st.path(name).exists():
                return None
            z = np.load(st.path(name))
            if name in LEACE_FAMILIES:
                import p2_leace_faithful as LF
                er = LF.erasers_from_npz(z)
                for l, e in er.items():
                    e["actual_rank"] = int(st.info.get(name, {}).get("actual_rank_by_layer", {}).get(str(l), LF._effective_rank(e["proj_left"])))
                if name == "leace_faithful":
                    st.leace = er
                else:
                    st.leace_seq = er
            else:
                st.proj[name] = {int(k): z[k] for k in z.files}
        return st


def collect_fit_activations(model, tok, fmt: str, pairs: list[dict], c: pd.DataFrame) -> tuple[dict, dict, np.ndarray, dict]:
    """One clean forward pass per fit prompt. Returns
      diffs {layer: [a - b per span token]}      for the SVD / mean-difference estimators
      acts  {layer: n x d}  and labels (n,)       for LEACE (label 0 = side A, 1 = side B)
    at exactly the same span positions (truncated to the shorter span)."""
    import p2_leace_faithful as LF
    sysm = P1.system_prompt() if fmt == "chat" else None
    items = LF.resolve_fit_items(tok, fmt, pairs, c, sysm)
    diffs: dict[int, list] = {}; acts: dict[int, list] = {}; labels = []
    hidden = []
    for it in items:
        _, hs = I.forward_hidden(model, tok, it["text"], None, None)
        hidden.append({l: hs[l][it["pos"], :].numpy() for l in hs})
        for l in hs:
            acts.setdefault(l, []).append(hidden[-1][l])
        labels += [it["label"]] * len(it["pos"])
    for ha, hb in zip(hidden[0::2], hidden[1::2]):                     # (A, B) in order
        for l in ha:
            for i in range(ha[l].shape[0]):
                diffs.setdefault(l, []).append(ha[l][i] - hb[l][i])
    acts = {l: np.concatenate(v, 0) for l, v in acts.items()}
    info = {"n_fit_pairs_requested": len(pairs), "n_fit_pairs_used": len(items) // 2,
            "n_span_tokens": int(len(labels) // 2), "site": "span (all tokens, truncated to shorter span)"}
    return diffs, acts, np.asarray(labels), info


def fit_bases(model, tok, args, mname: str, c: pd.DataFrame, budget) -> BasisStore:
    """Every basis from ONE activation pass over the fit pairs (plus a second pass over the
    neutral pairs and, with --leace-sequential, the scrubbing passes)."""
    import p2_leace_faithful as LF
    store = BasisStore(mname, args.fmt)
    store.max_rank = max(args.rank_list)
    t0 = time.time()
    fp = P1.fit_pairs(mname, args.n_fit_pairs, args.seed)
    diffs, acts, labels, info = collect_fit_activations(model, tok, args.fmt, fp, c)
    info.update({"sec_collect": time.time() - t0, "n_fit_pairs_neutral_type": int(sum(
        is_neutral_token(a) != is_neutral_token(b) for a, b in
        ((_swap(c, p["seed_id"], p["subvariant_A"]), _swap(c, p["seed_id"], p["subvariant_B"])) for p in fp)))})
    layers = sorted(diffs); d = int(next(iter(acts.values())).shape[1])
    store.add_proj("cure_centred_svd", I.basis_centred_svd(diffs, store.max_rank), {**info, "estimator": "centred_svd"})
    store.add_proj("uncentred_svd", I.basis_uncentred_svd(diffs, store.max_rank), {**info, "estimator": "uncentred_svd"})
    store.add_proj("mean_difference", I.basis_mean_difference(diffs), {**info, "estimator": "mean_difference", "rank": 1})
    for k, fam in enumerate(RANDOM_FAMILIES, start=1):
        store.add_proj(fam, I.basis_random(d, layers, store.max_rank, args.seed + k),
                       {"estimator": "random_orthonormal", "seed": args.seed + k, "layers": len(layers), "d": d})
    _fit_neutral(model, tok, args, mname, c, store)
    _fit_leace(model, tok, args, fp, c, acts, labels, store, budget)
    del acts, diffs
    store.save()
    return store


def _swap(c: pd.DataFrame, seed_id: str, sv: str) -> str:
    r = c[(c["seed_id"] == seed_id) & (c["subvariant"] == sv)]
    return str(r.iloc[0].get("swap_token", "")) if len(r) else ""


def _fit_neutral(model, tok, args, mname: str, c: pd.DataFrame, store: BasisStore) -> None:
    t0 = time.time()
    npairs = neutral_fit_pairs(mname, args.n_fit_pairs, args.seed, c)
    if len(npairs) < MIN_NEUTRAL_PAIRS:
        store.mark("neutral_contrast", "insufficient",
                   {"n_neutral_pairs": len(npairs), "min_required": MIN_NEUTRAL_PAIRS,
                    "note": "fewer neutral fit pairs than the predeclared minimum; condition skipped"})
        log.warning("neutral_contrast: only %d neutral fit pairs (< %d), skipping", len(npairs), MIN_NEUTRAL_PAIRS)
        return
    diffs, _, _, info = collect_fit_activations(model, tok, args.fmt, npairs, c)
    info.update({"estimator": "centred_svd on neutral pairs", "sec_collect": time.time() - t0,
                 "benchmarks": pd.Series([p["benchmark"] for p in npairs]).value_counts().to_dict(),
                 "note": ("token count is recorded (n_span_tokens); lexical-frequency matching to the "
                          "demographic fit pairs is NOT performed, see report; neutral swaps exist only "
                          "where the benchmark has a 'person'-style variant (BBQ in the current pentad)")})
    store.add_proj("neutral_contrast", I.basis_centred_svd(diffs, store.max_rank), info)


def _fit_leace(model, tok, args, fp, c, acts, labels, store: BasisStore, budget) -> None:
    """leace_faithful: fitted on the unedited activations (frozen protocol), always.
    leace_sequential: concept scrubbing, layer l fitted on activations already edited by the
    erasers of layers < l, only with --leace-sequential (L forward passes per prompt)."""
    import p2_leace_faithful as LF
    if not LF.leace_available():
        for fam in LEACE_FAMILIES:
            store.mark(fam, "unavailable", {"note": LF.PIP_HINT})
        log.warning("LEACE unavailable: %s", LF.PIP_HINT); return
    dev = str(next(model.parameters()).device)
    sysm = P1.system_prompt() if args.fmt == "chat" else None
    common = {"labels": "0 = side A, 1 = side B of the swap",
              "applied_as": "exact affine map eraser(x) at the span positions (LeaceAffineEdit)"}

    t0 = time.time()
    erasers = LF.fit_leace_erasers(acts, labels, device=dev)
    info = {"protocol": "frozen (fitted on unedited activations)", "n": int(len(labels)),
            "concept_erasure_version": LF.leace_version(), "sec_fit": time.time() - t0,
            "actual_rank_by_layer": {str(l): int(e["actual_rank"]) for l, e in erasers.items()}, **common}
    store.leace = erasers; store.status["leace_faithful"] = "ok"; store.info["leace_faithful"] = info

    if not args.leace_sequential:
        store.mark("leace_sequential", "skipped", {"note": "run with --leace-sequential to fit by concept scrubbing"})
        return
    t0 = time.time()
    seq, sinfo = LF.fit_leace_sequential(model, tok, args.fmt, fp, c, system=sysm,
                                         time_budget_s=max(0.0, budget.remaining()))
    sinfo.update({"sec_fit": time.time() - t0,
                  "actual_rank_by_layer": {str(l): int(e["actual_rank"]) for l, e in seq.items()}, **common})
    if not sinfo["complete"]:
        store.mark("leace_sequential", "partial", sinfo); return
    store.leace_seq = seq; store.status["leace_sequential"] = "ok"; store.info["leace_sequential"] = sinfo


# ---------------------------------------------------------------------------
# edit factories
# ---------------------------------------------------------------------------

def make_factory(model, store: BasisStore, cond: Cond, site: str = SITE):
    """A callable returning a FRESH edit for `cond` (None for unedited), so every pass logs
    its own energy. site is always "span" for the demographic items and "all" only for the
    capability probes under --capability-policy all."""
    if cond.kind == "none" or cond.status != "ok":
        return None
    if cond.kind == "affine":
        import p2_leace_faithful as LF
        er = store.erasers(cond.family)
        if not er:
            return None
        return lambda: LF.LeaceAffineEdit(model, er, alpha=cond.alpha, site=site)
    rows = store.rows(BASIS_OF[cond.family], cond.rank)
    return lambda: I.HookedEdit(model, rows, alpha=cond.alpha, site=site)


# ---------------------------------------------------------------------------
# energy matching (dev only)
# ---------------------------------------------------------------------------

def _dev_prompts(tok, fmt: str, seeds_df, mname: str, cdva, c, elig_keys, n: int) -> list[tuple[str, list[int]]]:
    sysm = P1.system_prompt() if fmt == "chat" else None
    out = []
    for _, s in seeds_df.iterrows():
        pr = [p for p in P1.pairs_for_seed(mname, s["seed_id"], cdva, c, elig_keys) if p["pair_type"] == "demographic"]
        if not pr:
            continue
        t = I.build_input(tok, pr[0]["prompt_A"], fmt, sysm)
        pos = I.resolve_span_positions(tok, t, pr[0]["swap_A"])
        if pos:
            out.append((t, pos))
        if len(out) >= n:
            break
    return out


def measure_energy(model, tok, factory, prompts: list[tuple[str, list[int]]]) -> tuple[float, dict]:
    """Pooled E[||h-h'||^2]/E[||h||^2] per layer over all prompts under one edit object, and
    its mean over layers."""
    e = factory()
    for t, pos in prompts:
        I.forward_hidden(model, tok, t, e, pos)
    s = e.summary()
    return float(s["energy_mean_over_layers"]), s["energy_by_layer"]


def match_energies(model, tok, store: BasisStore, prompts, mname: str, fmt: str) -> dict:
    """Grid search alpha in ENERGY_GRID per control so its mean removed energy on the dev
    prompts matches cure_centred_svd at rank 1, alpha 1. Frozen to p2_matched_alphas.json."""
    target, target_by_layer = measure_energy(model, tok, make_factory(model, store, Cond("cure_centred_svd", 1, 1.0)), prompts)
    rec = {"fmt": fmt, "target_condition": CURE_ID, "target_energy": target, "target_energy_by_layer": target_by_layer,
           "n_prompts": len(prompts), "grid": ENERGY_GRID, "controls": {}, "generated_utc": K.utc_now(),
           "definition": "mean over layers of pooled E[||h-h'||^2]/E[||h||^2] at the span positions of the A prompt"}
    for fam in CONTROL_FAMILIES:
        if store.status.get(BASIS_OF[fam], "ok") != "ok":
            rec["controls"][fam] = {"status": store.status[BASIS_OF[fam]]}; continue
        r = 1 if fam in RANKED_FAMILIES else store.natural_rank(fam)
        energies = [measure_energy(model, tok, make_factory(model, store, Cond(fam, r, a)), prompts)[0] for a in ENERGY_GRID]
        j = int(np.nanargmin([abs(e - target) for e in energies]))
        e1 = energies[-1]
        rec["controls"][fam] = {"rank": r, "alpha": ENERGY_GRID[j], "energy_achieved": energies[j],
                               "energy_at_alpha1": e1, "energies_by_alpha": energies,
                               "matchable": bool(e1 >= target),      # alpha <= 1 cannot raise a weaker control to the target
                               "quadratic_prediction_alpha": float(min(1.0, np.sqrt(target / e1))) if e1 > 0 else None}
        log.info("energy match %s %s: target %.4g, alpha %.2f -> %.4g", mname, fam, target, ENERGY_GRID[j], energies[j])
    return rec


def load_matched(mname: str) -> dict | None:
    p = OUT / "p2_matched_alphas.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get(mname)


def save_matched(mname: str, rec: dict) -> None:
    p = OUT / "p2_matched_alphas.json"
    allrec = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    allrec[mname] = rec
    K.write_json(allrec, p)


# ---------------------------------------------------------------------------
# per-item readouts (GPU)
# ---------------------------------------------------------------------------

def target_token_id(tok, pair: dict) -> int | None:
    """First token of the gold answer of B (else of the swap token), as in p1_pilot."""
    g = pair["gold_B"]
    tgt = g if g.strip().lower() not in ("unknown", "", "nan") else pair["swap_B"].replace("_", " ")
    ids = tok(" " + tgt.split()[0] if tgt.split() else tgt, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = tok(tgt, add_special_tokens=False)["input_ids"]
    return ids[0] if ids else None


def _audit(model, tok, ta, tb, pa, pb, tid, factory) -> dict:
    row = {}
    if pa and pb and tid is not None:
        C, info = I.commutator(model, tok, ta, tb, pa, pb, tid, factory, SOURCE_STATE)
        te = info.get("target_edit") or {}
        row.update({"C": C, "absC": abs(C) if C == C else np.nan, "patched_len": info.get("patched_len"),
                    "logit_patched": info.get("logit_patched"), "logit_clean": info.get("logit_clean"),
                    "energy_audit": te.get("energy_mean_over_layers"),
                    "energy_by_layer_audit": json.dumps(te.get("energy_by_layer")) if te else None})
    else:
        row.update({"C": np.nan, "absC": np.nan, "audit_error": "span or target unresolved"})
    return row


def _options(model, tok, side, text, pos, prompt, gold, factory) -> dict:
    opts = [f"({l}) {t}" for l, t in P1.options_of(prompt)]
    if not opts:
        return {f"n_options_{side}": 0}
    r = I.option_loglik(model, tok, text, opts, factory() if factory else None, pos)
    gi = P1.gold_index(prompt, gold)
    return {f"n_options_{side}": len(opts), f"opt_argmax_{side}": r["argmax"], f"opt_margin_{side}": r["margin_top1_top2"],
            f"opt_entropy_{side}": r["entropy"], f"gold_idx_{side}": gi,
            f"opt_correct_{side}": (r["argmax"] == gi) if gi is not None else None,
            f"opt_probs_{side}": json.dumps([round(x, 6) for x in r["probs"]]),
            f"opt_loglik_sum_{side}": json.dumps([round(x, 4) for x in r["loglik_sum"]])}


def _generation(model, tok, ta, tb, pa, pb, pair, factory, max_new_tokens: int) -> dict:
    e = factory() if factory else None
    texts = I.generate_under_edit(model, tok, [ta, tb], e, [pa, pb], max_new_tokens)
    row = {}
    for side, txt, prompt in (("A", texts[0], pair["prompt_A"]), ("B", texts[1], pair["prompt_B"])):
        idx, why = P1.parse_answer(txt, prompt)
        gi = P1.gold_index(prompt, pair[f"gold_{side}"])
        row.update({f"gen_raw_{side}": txt, f"gen_idx_{side}": idx, f"gen_parse_{side}": why,
                    f"gen_valid_{side}": idx is not None,
                    f"gen_correct_{side}": (idx == gi) if (idx is not None and gi is not None) else None})
    row["gen_flip_AB"] = (row["gen_idx_A"] != row["gen_idx_B"]) if (row["gen_valid_A"] and row["gen_valid_B"]) else None
    if e is not None:
        s = e.summary()
        row.update({"energy_gen": s["energy_mean_over_layers"], "energy_by_layer_gen": json.dumps(s["energy_by_layer"]),
                    "decode_step_edits": s["n_decode_step_edits"], "n_edits_gen": s["n_edits"]})
    return row


def run_item(model, tok, fmt: str, cond: Cond, factory, pair: dict, max_new_tokens: int) -> dict:
    """All readouts for one pair under one condition; same edit for audit, options, generation."""
    sysm = P1.system_prompt() if fmt == "chat" else None
    ta = I.build_input(tok, pair["prompt_A"], fmt, sysm)
    tb = I.build_input(tok, pair["prompt_B"], fmt, sysm)
    pa = I.resolve_span_positions(tok, ta, pair["swap_A"])
    pb = I.resolve_span_positions(tok, tb, pair["swap_B"])
    row = {"span_len_A": len(pa), "span_len_B": len(pb), "fmt": fmt}
    t0 = time.time()
    row.update(_audit(model, tok, ta, tb, pa, pb, target_token_id(tok, pair), factory))
    row["sec_audit"] = time.time() - t0
    t0 = time.time()
    row.update(_options(model, tok, "A", ta, pa, pair["prompt_A"], pair["gold_A"], factory))
    row.update(_options(model, tok, "B", tb, pb, pair["prompt_B"], pair["gold_B"], factory))
    row["opt_pref_changed_AB"] = (row.get("opt_argmax_A") != row.get("opt_argmax_B")) \
        if ("opt_argmax_A" in row and "opt_argmax_B" in row) else None
    row["sec_options"] = time.time() - t0
    t0 = time.time()
    row.update(_generation(model, tok, ta, tb, pa, pb, pair, factory, max_new_tokens))
    row["sec_generation"] = time.time() - t0
    return row


def item_meta(mname: str, phase: str, cond: Cond, store: BasisStore, s, pair: dict, model, tok, cfg) -> dict:
    fam_basis = BASIS_OF.get(cond.family)
    return {"model_name": mname, "hf_id": cfg.get("hf_id"), "model_name_or_path": getattr(model.config, "_name_or_path", None),
            "model_commit": getattr(model.config, "_commit_hash", None), "tokenizer_name_or_path": getattr(tok, "name_or_path", None),
            "phase": phase, "condition": cond.family, "cond_id": cond.cond_id, "basis_name": fam_basis if cond.kind != "none" else None,
            "basis_fit_id": store.fit_id.get(fam_basis) if cond.kind != "none" else None,
            "rank": cond.rank, "alpha": cond.alpha, "site": SITE if cond.kind != "none" else None,
            "source_state": SOURCE_STATE, "edit_kind": cond.kind, "status": cond.status,
            "seed_id": s["seed_id"], "benchmark": s["seed_source"], "pair_type": pair["pair_type"],
            "subvariant_A": pair["A"], "subvariant_B": pair["B"], "C_pre_shipped": pair["C_pre_shipped"],
            "sentence_pref_applicable": s["seed_source"] in SENTENCE_PREF_BENCHMARKS,
            "sentence_pref_protocol": SENTENCE_PREF_NOTE, "ts": K.utc_now()}


# ---------------------------------------------------------------------------
# independent utility (external capability probes)
# ---------------------------------------------------------------------------

_CAP_CACHE: dict = {}


def _datasets_or_none():
    try:
        import datasets
        return datasets
    except ImportError:
        return None


def _ds_revision(ds, datasets) -> dict:
    return {"datasets_version": datasets.__version__, "dataset_version": str(getattr(ds.info, "version", "")),
            "fingerprint": getattr(ds, "_fingerprint", None), "config_name": getattr(ds.info, "config_name", None)}


def mmlu_subset(n: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Deterministic stratified-by-subject subset of the MMLU test split (largest-remainder
    allocation, per-subject shuffle with one generator). Cached for the process."""
    if "mmlu" in _CAP_CACHE:
        return _CAP_CACHE["mmlu"]
    datasets = _datasets_or_none()
    ds = datasets.load_dataset("cais/mmlu", "all", split="test")
    df = pd.DataFrame({"question": ds["question"], "choices": ds["choices"], "answer": ds["answer"], "subject": ds["subject"]})
    df["row_index"] = np.arange(len(df))
    rng = random.Random(seed)
    groups = {s: g["row_index"].tolist() for s, g in df.groupby("subject")}
    quota = {s: n * len(v) / len(df) for s, v in groups.items()}
    base = {s: int(q) for s, q in quota.items()}
    for s in sorted(quota, key=lambda k: -(quota[k] - base[k]))[:n - sum(base.values())]:
        base[s] += 1
    take = []
    for s in sorted(groups):
        ids = sorted(groups[s]); rng.shuffle(ids); take += ids[:base[s]]
    sub = df[df["row_index"].isin(take)].sort_values("row_index").reset_index(drop=True)
    rec = {**_ds_revision(ds, datasets), "n": int(len(sub)), "subjects": sub["subject"].value_counts().to_dict(),
           "row_indices": sub["row_index"].tolist(), "random_seed": seed}
    _CAP_CACHE["mmlu"] = (sub, rec)
    return sub, rec


def wikitext_tokens(tok, n_tokens: int) -> tuple[list[int], dict]:
    """First non-empty WikiText-2 test documents until n_tokens tokens are accumulated,
    joined by blank lines and tokenised once; truncated to n_tokens."""
    key = ("wiki", getattr(tok, "name_or_path", id(tok)), n_tokens)
    if key in _CAP_CACHE:
        return _CAP_CACHE[key]
    datasets = _datasets_or_none()
    ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")   # namespaced id (datasets>=4)
    pieces, doc_ids, count = [], [], 0
    for i, t in enumerate(ds["text"]):
        if not t.strip():
            continue
        pieces.append(t); doc_ids.append(i); count += len(tok(t, add_special_tokens=False)["input_ids"])
        if count >= n_tokens:
            break
    ids = tok("\n\n".join(pieces), add_special_tokens=False)["input_ids"][:n_tokens]
    rec = {**_ds_revision(ds, datasets), "document_indices": doc_ids, "n_documents": len(doc_ids), "n_tokens": len(ids),
           "window": WINDOW, "note": "windows are scored independently (no cross-window context)"}
    _CAP_CACHE[key] = (ids, rec)
    return ids, rec


def _mmlu_prompt(r) -> tuple[str, list[str]]:
    body = "\n".join("%s. %s" % (L, c) for L, c in zip("ABCD", r["choices"]))
    return ("The following is a multiple choice question.\n\n%s\n%s\nAnswer:" % (r["question"], body),
            [" A", " B", " C", " D"])


def _mmlu_probe(model, tok, factory, n: int, seed: int) -> dict:
    sub, rec = mmlu_subset(n, seed)
    detail, energies = [], []
    for _, r in sub.iterrows():
        prompt, opts = _mmlu_prompt(r)
        e = factory() if factory else None
        o = I.option_loglik(model, tok, prompt, opts, e, None)
        detail.append({"row_index": int(r["row_index"]), "subject": r["subject"], "gold": int(r["answer"]),
                       "pred": o["argmax"], "correct": o["argmax"] == int(r["answer"]),
                       "loglik_sum": [round(x, 4) for x in o["loglik_sum"]]})
        if e is not None:
            energies.append(e.summary()["energy_mean_over_layers"])
    acc = float(np.mean([d["correct"] for d in detail])) if detail else float("nan")
    return {"probe": "mmlu_200", "metric": "accuracy_option_loglik_letters", "value": acc, "n": len(detail),
            "energy_mean": float(np.nanmean(energies)) if energies else None, "detail": json.dumps(detail),
            "dataset": json.dumps({k: v for k, v in rec.items() if k != "row_indices"}), "row_indices": json.dumps(rec["row_indices"]),
            "prompt_format": "raw MMLU header, options A-D, scored as ' A'..' D' by option_loglik"}


def _wiki_probe(model, tok, factory, n_tokens: int) -> dict:
    import torch
    ids, rec = wikitext_tokens(tok, n_tokens)
    dev = next(model.parameters()).device
    nll_sum, n_scored, energies = 0.0, 0, []
    for i in range(0, len(ids), WINDOW):
        chunk = ids[i:i + WINDOW]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk], device=dev)
        e = factory() if factory else None
        if e is not None:
            e.set_targets([[]], [0])
        with torch.no_grad(), (e if e is not None else I.no_edit()):
            lp = torch.log_softmax(model(input_ids=x).logits.float(), dim=-1)
        g = lp[0, :-1, :].gather(-1, x[0, 1:, None]).squeeze(-1)
        nll_sum += float(-g.sum().item()); n_scored += int(g.numel())
        if e is not None:
            energies.append(e.summary()["energy_mean_over_layers"])
    mean_nll = nll_sum / max(1, n_scored)
    return {"probe": "wikitext2_20k", "metric": "mean_nll_per_token", "value": mean_nll, "n": n_scored,
            "perplexity": float(np.exp(mean_nll)), "energy_mean": float(np.nanmean(energies)) if energies else None,
            "dataset": json.dumps(rec), "prompt_format": "raw text, %d-token windows" % WINDOW}


PROBES = ("mmlu_200", "wikitext2_20k")


def capability_eval(model, tok, edit_factory, policy: str, n_mmlu: int = N_MMLU,
                    n_wiki: int = N_WIKI_TOKENS, seed: int = K.RANDOM_SEED_V2,
                    probes: tuple = PROBES) -> list[dict]:
    """Two bounded probes under an explicit application policy. edit_factory must already
    build edits with site="all" when policy == "all"; with policy == "none" an edited
    condition is NOT evaluated (the prompts have no demographic span) and a status row says
    so. The unedited reference (edit_factory None) is evaluated under either policy."""
    if policy not in ("none", "all"):
        raise ValueError(policy)
    base = {"capability_policy": policy, "site": ("all" if edit_factory else None)}
    if _datasets_or_none() is None:
        return [{**base, "probe": p, "status": "unavailable", "note": "pip install datasets"} for p in ("mmlu_200", "wikitext2_20k")]
    if edit_factory is not None and policy == "none":
        return [{**base, "probe": p, "status": "not_evaluated",
                 "note": "no demographic span in external prompts; policy=none applies no edit and records nothing"}
                for p in ("mmlu_200", "wikitext2_20k")]
    rows = []
    for probe, fn in (("mmlu_200", lambda: _mmlu_probe(model, tok, edit_factory, n_mmlu, seed)),
                      ("wikitext2_20k", lambda: _wiki_probe(model, tok, edit_factory, n_wiki))):
        if probe not in probes:
            continue
        t0 = time.time()
        try:                                   # one probe failing must not discard the other
            rows.append({**base, **fn(), "status": "ok", "sec": time.time() - t0})
        except Exception as exc:
            rows.append({**base, "probe": probe, "status": "error", "note": str(exc)[:300], "sec": time.time() - t0})
            log.error("capability probe %s failed: %s", probe, str(exc)[:200])
    return rows


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

class Budget:
    def __init__(self, hours: float):
        self.t0, self.cap = time.time(), hours * 3600.0

    def remaining(self) -> float:
        return self.cap - (time.time() - self.t0)

    def exceeded(self) -> bool:
        return self.remaining() <= 0


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    d = pd.read_parquet(path, columns=KEY_COLS)
    return set(map(tuple, d.to_numpy().tolist()))


def cap_done_keys(path: Path) -> set:
    """Capability rows already produced, EXCLUDING error/unavailable rows so a rerun retries
    them (e.g. after a dataset-id fix)."""
    if not path.exists():
        return set()
    d = pd.read_parquet(path, columns=["model_name", "phase", "cond_id", "probe", "status"])
    d = d[~d["status"].isin(["error", "unavailable"])]
    return set(zip(d["model_name"], d["phase"], d["cond_id"], d["probe"]))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Design")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--models", nargs="*", default=["qwen2.5-7b-instruct", "llama-3.1-8b-instruct"])
    ap.add_argument("--fmt", choices=["chat", "raw"], default="chat")
    ap.add_argument("--gpu-hours-cap", type=float, default=8.0)
    ap.add_argument("--n-fit-pairs", type=int, default=N_FIT_PAIRS)
    ap.add_argument("--n-dev", type=int, default=N_DEV_CAP)
    ap.add_argument("--n-test", type=int, default=N_TEST_CAP)
    ap.add_argument("--conditions", nargs="*", default=FAMILIES, choices=FAMILIES, help="condition families to run")
    ap.add_argument("--pair-types", nargs="*", default=PAIR_TYPES, choices=PAIR_TYPES)
    ap.add_argument("--extend-ranks", action="store_true", help="add ranks 2 and 4 at alpha = 1")
    ap.add_argument("--ranks", nargs="*", type=int, default=[], help="extra ranks, e.g. 8")
    ap.add_argument("--match-energy", action="store_true")
    ap.add_argument("--match-energy-prompts", type=int, default=24)
    ap.add_argument("--leace-sequential", action="store_true", help="concept-scrubbing fit (L x cost)")
    ap.add_argument("--capability-policy", choices=["none", "all"], default="none")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--seed", type=int, default=K.RANDOM_SEED_V2)
    ap.add_argument("--refit", action="store_true", help="ignore cached bases")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarise-only", action="store_true")
    ap.add_argument("--n-mmlu", type=int, default=None, help="MMLU questions per capability probe (default %d)" % N_MMLU)
    ap.add_argument("--n-wiki-tokens", type=int, default=None, help="WikiText-2 tokens per probe (default %d)" % N_WIKI_TOKENS)
    args = ap.parse_args(argv)
    args.rank_list = sorted(set([1] + (EXTENDED_RANKS if args.extend_ranks else []) + list(args.ranks)))
    if args.smoke:
        args.models = ["gemma-2-2b-it"]; args.n_fit_pairs = 8; args.n_dev = 4; args.n_test = 4
        args.max_new_tokens = 16; args.match_energy_prompts = 2
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    per_item, cap_path, proto_path = OUT / "p2_per_item.parquet", OUT / "p2_capability.parquet", OUT / "p2_protocol.json"
    if args.summarise_only:
        summarise(); return
    budget = Budget(args.gpu_hours_cap)
    phases = ["dev", "test"] if args.phase == "all" else [args.phase]
    seeds = {"dev": select_eval_seeds("dev", args.n_dev, args.seed, OUT / "p2_dev_seeds.csv"),
             "test": select_eval_seeds("test", args.n_test, args.seed, OUT / "p2_test_seeds.csv")}
    c, _ = P1._pentad_c()
    cdva = K.read_cdva()
    proto = json.loads(proto_path.read_text(encoding="utf-8")) if proto_path.exists() else {}
    proto.update({"generated_utc": K.utc_now(), "fmt": args.fmt, "site": SITE, "source_state": SOURCE_STATE,
                  "ranks": args.rank_list, "alpha_screen_dev": [0.0] + ALPHA_SCREEN, "capability_policy": args.capability_policy,
                  "n_fit_pairs": args.n_fit_pairs, "n_dev_seeds": int(len(seeds["dev"])), "n_test_seeds": int(len(seeds["test"])),
                  "max_new_tokens": args.max_new_tokens, "gpu_hours_cap": args.gpu_hours_cap,
                  "edit_definition": I.__doc__.split("The edit")[1].split("Prompt format")[0].strip(),
                  "sentence_pref_protocol": SENTENCE_PREF_NOTE, "confirmatory": CONFIRMATORY})
    proto.setdefault("models", {})
    done, cap_done = done_keys(per_item), cap_done_keys(cap_path)

    for mname in args.models:
        if budget.exceeded():
            log.warning("cap reached before %s", mname); break
        log.info("=== P2: %s ===", mname)
        t_load = time.time()
        cfg, model, tok = P1.load(mname)
        mrec = proto["models"].setdefault(mname, {})
        mrec.update({"sec_load": time.time() - t_load, "torch": _ver("torch"), "transformers": _ver("transformers"),
                     **{k: cfg.get(k) for k in ("hf_id", "attn_implementation", "model_class", "dtype")}})
        try:
            elig = pd.read_csv(K.OUT_P0 / "pair_sets" / f"{mname}_eligible.csv")
            elig_keys = set(zip(elig["seed_id"], elig["subvariant_A"], elig["subvariant_B"]))
            store = None if args.refit else BasisStore.load(mname, args.fmt)
            if store is None:
                store = fit_bases(model, tok, args, mname, c, budget)
            mrec["bases"] = {"status": store.status, "info": store.info, "fit_id": store.fit_id, "max_rank": store.max_rank}
            mrec["identity_checks"] = _identity_gate(model, tok, args.fmt, seeds["dev"], mname, cdva, c, elig_keys, store)
            if not mrec["identity_checks"].get("pass_all", False):
                log.error("IDENTITY CHECKS FAILED on %s: stop and fix the implementation", mname)
                _save_proto(proto, proto_path); continue
            for phase in phases:
                matched = _matched_for(model, tok, args, store, seeds["dev"], mname, cdva, c, elig_keys, phase, budget)
                conds = build_conditions(phase, args.rank_list, store, matched, args.conditions)
                mrec.setdefault("conditions", {})[phase] = [c_.cond_id for c_ in conds]
                _run_phase(model, tok, cfg, args, mname, phase, conds, store, seeds[phase], cdva, c, elig_keys,
                           per_item, done, budget, mrec)
                _run_capability(model, tok, args, mname, phase, conds, store, cap_path, cap_done, budget)
            mrec["sec_model_total"] = time.time() - t_load
        finally:
            from load_osm import unload_model
            unload_model(mname)
        _save_proto(proto, proto_path)
    proto["sec_wall_total"] = time.time() - budget.t0
    _save_proto(proto, proto_path)
    summarise()


def _ver(pkg: str) -> str | None:
    try:
        return __import__(pkg).__version__
    except Exception:
        return None


def _save_proto(proto: dict, path: Path) -> None:
    path.write_text(json.dumps(proto, indent=2, default=K._json_default), encoding="utf-8")


def _identity_gate(model, tok, fmt, seeds_df, mname, cdva, c, elig_keys, store: BasisStore) -> dict:
    prompts = _dev_prompts(tok, fmt, seeds_df, mname, cdva, c, elig_keys, 2)
    if not prompts:
        return {"pass_all": False, "error": "no dev prompt with a resolved span"}
    chk = I.identity_checks(model, tok, [t for t, _ in prompts], store.rows("cure_centred_svd", 1), [p for _, p in prompts])
    log.info("identity checks %s: pass_all=%s", mname, chk.get("pass_all"))
    return chk


def _matched_for(model, tok, args, store, dev_seeds, mname, cdva, c, elig_keys, phase, budget) -> dict | None:
    """dev: run the grid search if asked and not yet frozen; test: only read the frozen file."""
    if not args.match_energy:
        return None
    rec = load_matched(mname)
    if rec is not None:
        return rec
    if phase == "test":
        raise SystemExit("--match-energy on test needs p2_matched_alphas.json for %s; run the dev phase first" % mname)
    if budget.exceeded():
        return None
    prompts = _dev_prompts(tok, args.fmt, dev_seeds, mname, cdva, c, elig_keys, args.match_energy_prompts)
    rec = match_energies(model, tok, store, prompts, mname, args.fmt)
    save_matched(mname, rec)
    return rec


def _status_row(mname, phase, cond: Cond, store: BasisStore) -> dict:
    return {"model_name": mname, "phase": phase, "condition": cond.family, "cond_id": cond.cond_id, "seed_id": "-",
            "pair_type": "-", "rank": cond.rank, "alpha": cond.alpha, "status": cond.status,
            "note": json.dumps(store.info.get(BASIS_OF[cond.family], {}), default=K._json_default), "ts": K.utc_now()}


def _run_phase(model, tok, cfg, args, mname, phase, conds, store, seeds_df, cdva, c, elig_keys, per_item, done, budget, mrec):
    rows, n_here = [], 0
    for cond in [x for x in conds if x.status != "ok"]:
        key = (mname, phase, cond.cond_id, "-", "-")
        if key not in done:
            rows.append(_status_row(mname, phase, cond, store)); done.add(key)
    live = [x for x in conds if x.status == "ok"]
    factories = {x.cond_id: make_factory(model, store, x) for x in live}
    for _, s in seeds_df.iterrows():
        if budget.exceeded():
            log.warning("cap reached at %s seed %s", phase, s["seed_id"]); break
        pairs = [p for p in P1.pairs_for_seed(mname, s["seed_id"], cdva, c, elig_keys) if p["pair_type"] in args.pair_types]
        if not pairs:
            log.warning("no clean pairs for %s on %s", s["seed_id"], mname); continue
        for pair in pairs:
            for cond in live:
                key = (mname, phase, cond.cond_id, s["seed_id"], pair["pair_type"])
                if key in done or budget.exceeded():
                    continue
                t0 = time.time()
                try:
                    r = run_item(model, tok, args.fmt, cond, factories[cond.cond_id], pair, args.max_new_tokens)
                except Exception as exc:
                    r = {"error": str(exc)[:300]}
                    log.error("item failed %s: %s", key, str(exc)[:200])
                r.update(item_meta(mname, phase, cond, store, s, pair, model, tok, cfg))
                r["sec_total"] = time.time() - t0
                rows.append(r); done.add(key); n_here += 1
                if len(rows) >= 10:
                    P1.append_rows(per_item, rows); rows = []
    P1.append_rows(per_item, rows)
    mrec.setdefault("n_items_this_run", {})[phase] = n_here


def _run_capability(model, tok, args, mname, phase, conds, store, cap_path, cap_done, budget):
    """Capability rows per condition; edits for policy=all are rebuilt with site='all'."""
    rows = []
    n_mmlu, n_wiki = (8, 2048) if args.smoke else (N_MMLU, N_WIKI_TOKENS)
    n_mmlu = args.n_mmlu if args.n_mmlu else n_mmlu
    n_wiki = args.n_wiki_tokens if args.n_wiki_tokens else n_wiki
    for cond in [x for x in conds if x.status == "ok"]:
        if budget.exceeded():
            log.warning("cap reached before capability %s %s", phase, cond.cond_id); break
        todo = tuple(p for p in PROBES if (mname, phase, cond.cond_id, p) not in cap_done)
        if not todo:
            continue
        factory = make_factory(model, store, cond, site="all") if args.capability_policy == "all" else make_factory(model, store, cond)
        try:
            out = capability_eval(model, tok, factory, args.capability_policy, n_mmlu, n_wiki, args.seed, probes=todo)
        except Exception as exc:
            out = [{"probe": p, "status": "error", "note": str(exc)[:300], "capability_policy": args.capability_policy}
                   for p in todo]
            log.error("capability failed %s %s: %s", mname, cond.cond_id, str(exc)[:200])
        for o in out:
            o.update({"model_name": mname, "phase": phase, "cond_id": cond.cond_id, "condition": cond.family,
                      "rank": cond.rank, "alpha": cond.alpha, "basis_fit_id": store.fit_id.get(BASIS_OF.get(cond.family, ""), None),
                      "ts": K.utc_now()})
            cap_done.add((mname, phase, cond.cond_id, o["probe"]))
        rows += out
    if rows and cap_path.exists():
        # a retried probe replaces its earlier error/unavailable row
        prev = pd.read_parquet(cap_path)
        new_keys = {(r["model_name"], r["phase"], r["cond_id"], r["probe"]) for r in rows}
        stale = prev.apply(lambda r: (r["model_name"], r["phase"], r["cond_id"], r["probe"]) in new_keys
                           and r["status"] in ("error", "unavailable"), axis=1)
        if stale.any():
            prev[~stale].to_parquet(cap_path, index=False)
    P1.append_rows(cap_path, rows)


# ---------------------------------------------------------------------------
# statistics (CPU)
# ---------------------------------------------------------------------------

def _cluster_draws(num, den, clusters, transform=None, n_boot=K.N_BOOT, seed=K.RANDOM_SEED_V2) -> np.ndarray:
    """The bootstrap draws behind common.ratio_bootstrap (same scheme and seed), returned so a
    two-sided percentile p-value can be read off (Efron and Tibshirani 1993, Section 15.4)."""
    num = np.asarray(num, float); den = np.asarray(den, float)
    transform = transform or (lambda r: r)
    codes, uniq = pd.factorize(np.asarray(clusters), sort=False)
    k = len(uniq)
    cnum = np.bincount(codes, weights=num, minlength=k); cden = np.bincount(codes, weights=den, minlength=k)
    if k < 2:
        return np.array([])
    rng = np.random.default_rng(seed)
    take = rng.integers(0, k, size=(n_boot, k))
    with np.errstate(divide="ignore", invalid="ignore"):
        return transform(cnum[take].sum(axis=1) / cden[take].sum(axis=1))


def boot_p(draws: np.ndarray) -> float:
    d = draws[np.isfinite(draws)]
    if d.size == 0:
        return float("nan")
    return float(min(1.0, max(1.0 / d.size, 2 * min((d <= 0).mean(), (d >= 0).mean()))))


def holm(pvals: list[float]) -> list[float]:
    """Holm (1979) step-down adjusted p-values."""
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (m - i) * p[j])
        adj[j] = min(1.0, running)
    return adj.tolist()


def _num01(s: pd.Series) -> pd.Series:
    """bool / None / NaN (object dtype after a parquet round trip) -> 1.0 / 0.0 / NaN."""
    def f(v):
        if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        return float(bool(v))
    return s.map(f).astype(float)


def _item_scores(d: pd.DataFrame) -> pd.DataFrame:
    """Per-item behavioural scores pooled over the two prompts of the pair."""
    d = d.copy()
    for col in ("opt_correct_A", "opt_correct_B", "gen_valid_A", "gen_valid_B", "gen_correct_A", "gen_correct_B", "gen_flip_AB"):
        if col not in d:
            d[col] = np.nan
    oc = pd.concat([_num01(d["opt_correct_A"]), _num01(d["opt_correct_B"])], axis=1)
    d["opt_acc"] = oc.mean(axis=1)
    gv = pd.concat([_num01(d["gen_valid_A"]), _num01(d["gen_valid_B"])], axis=1).fillna(0.0)
    gc = pd.concat([_num01(d["gen_correct_A"]), _num01(d["gen_correct_B"])], axis=1)
    d["gen_acc_attempted"] = gc.fillna(0.0).mean(axis=1)        # attempted denominator: invalid = wrong
    d["gen_acc_usable"] = gc.mean(axis=1)                        # usable denominator: NaN if no valid side
    d["validity"] = gv.mean(axis=1)
    d["both_valid"] = (gv.sum(axis=1) == 2)
    d["flip"] = _num01(d["gen_flip_AB"])
    return d


def _paired(post: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    cols = ["seed_id", "pair_type", "benchmark", "absC", "opt_acc", "gen_acc_attempted", "gen_acc_usable", "validity", "both_valid", "flip"]
    return post.merge(pre[cols], on=["seed_id", "pair_type", "benchmark"], suffixes=("", "_pre"))


def _ci(num, den, clusters, transform=None) -> tuple[float, float, float]:
    return K.ratio_bootstrap(np.asarray(num, float), np.asarray(den, float), np.asarray(clusters), transform)


def paired_metrics(m: pd.DataFrame) -> dict:
    """All P2 statistics for one (condition, pair_type, stratum) paired frame."""
    out = {"n_seeds": int(m["seed_id"].nunique()), "n_items": int(len(m))}
    a = m[np.isfinite(m["absC"]) & np.isfinite(m["absC_pre"])]
    pre, post, cl = a["absC_pre"].to_numpy(), a["absC"].to_numpy(), a["seed_id"].to_numpy()
    ones = np.ones(len(a))
    out.update({"n_audit": int(len(a)), "absC_pre_mean": float(pre.mean()) if len(a) else np.nan,
                "absC_post_mean": float(post.mean()) if len(a) else np.nan,
                "W": float((post < pre).mean()) if len(a) else np.nan, "L": float((post > pre).mean()) if len(a) else np.nan,
                "U": float((post == pre).mean()) if len(a) else np.nan})
    if len(a):
        out["D"], out["D_lo"], out["D_hi"] = _ci(pre - post, ones, cl)
        out["R"], out["R_lo"], out["R_hi"] = _ci(post, pre, cl, lambda r: 1 - r) if pre.sum() > 0 else (np.nan,) * 3
        above = (pre >= K.TAU).astype(float)
        out["X_den"] = int(above.sum()); out["X_num"] = int(((pre >= K.TAU) & (post < K.TAU)).sum())
        out["X_reverse_num"] = int(((pre < K.TAU) & (post >= K.TAU)).sum()); out["X_reverse_den"] = int((pre < K.TAU).sum())
        out["X"], out["X_lo"], out["X_hi"] = _ci(((pre >= K.TAU) & (post < K.TAU)).astype(float), above, cl) if above.sum() else (np.nan,) * 3
    for col in ("energy_audit", "energy_gen"):
        out[col + "_mean"] = float(pd.to_numeric(m[col], errors="coerce").mean()) if col in m else np.nan
    cl = m["seed_id"].to_numpy(); ones = np.ones(len(m))
    for col in ("opt_acc", "gen_acc_attempted", "validity"):
        x, x0 = m[col].to_numpy(float), m[col + "_pre"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(x0)
        out[col] = float(np.nanmean(x)) if ok.any() else np.nan; out[col + "_pre"] = float(np.nanmean(x0)) if ok.any() else np.nan
        out[f"d_{col}"], out[f"d_{col}_lo"], out[f"d_{col}_hi"] = _ci(x[ok] - x0[ok], ones[ok], cl[ok]) if ok.sum() else (np.nan,) * 3
    u, u0 = m["gen_acc_usable"].to_numpy(float), m["gen_acc_usable_pre"].to_numpy(float)
    out["gen_acc_usable"] = float(np.nanmean(u)) if np.isfinite(u).any() else np.nan
    out["gen_acc_usable_n"] = int(np.isfinite(u).sum())
    out["gen_acc_usable_pre"] = float(np.nanmean(u0)) if np.isfinite(u0).any() else np.nan
    f = m["flip"].to_numpy(float)
    out["flip_attempted_num"] = int(np.nansum(f)); out["flip_attempted_den"] = int(len(m))
    out["flip_attempted"] = out["flip_attempted_num"] / max(1, out["flip_attempted_den"])
    out["flip_usable_den"] = int(np.isfinite(f).sum()); out["flip_usable"] = out["flip_attempted_num"] / max(1, out["flip_usable_den"])
    return out


def _summary_rows(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mname, phase), dm in d.groupby(["model_name", "phase"]):
        pre = dm[dm["cond_id"] == UNEDITED_ID]
        if pre.empty:
            log.warning("no unedited rows for %s %s; nothing to pair", mname, phase); continue
        for cond_id, dc in dm[dm["cond_id"] != UNEDITED_ID].groupby("cond_id"):
            m_all = _paired(dc, pre)
            for pt, mp in m_all.groupby("pair_type"):
                strata = [("pooled", mp)] + [(b, g) for b, g in mp.groupby("benchmark")]
                for stratum, ms in strata:
                    rec = {"model_name": mname, "phase": phase, "cond_id": cond_id, "condition": dc["condition"].iloc[0],
                           "rank": dc["rank"].iloc[0], "alpha": dc["alpha"].iloc[0], "pair_type": pt, "stratum": stratum,
                           "label": "confirmatory-input" if (phase == "test" and cond_id in [CURE_ID] + RANDOM_IDS and pt == "demographic" and stratum == "pooled") else "exploratory"}
                    rec.update(paired_metrics(ms)); rows.append(rec)
    return pd.DataFrame(rows)


def _confirmatory(d: pd.DataFrame, phase: str = "test") -> pd.DataFrame:
    """The four predeclared comparisons per model on the given phase, demographic pairs,
    pooled; Holm within model. Random mean = the three rank-matched (alpha = 1) random
    controls averaged per item."""
    out = []
    for mname, dm in d[(d["phase"] == phase) & (d["pair_type"] == "demographic")].groupby("model_name"):
        pre = dm[dm["cond_id"] == UNEDITED_ID]; cure = dm[dm["cond_id"] == CURE_ID]
        if pre.empty or cure.empty:
            continue
        m = _paired(cure, pre)
        rnd = dm[dm["cond_id"].isin(RANDOM_IDS)]
        rmean = (rnd.groupby(["seed_id", "pair_type", "benchmark"])[["absC", "gen_acc_attempted"]].mean()
                 .rename(columns={"absC": "absC_rnd", "gen_acc_attempted": "gen_rnd"}).reset_index()
                 if rnd["cond_id"].nunique() == 3 else None)
        if rmean is not None:
            m = m.merge(rmean, on=["seed_id", "pair_type", "benchmark"], how="left")
        tests = {"C1": (m["absC_pre"] - m["absC"], "D"),
                 "C2": (m["gen_acc_attempted"] - m["gen_acc_attempted_pre"], "delta generation accuracy (attempted)"),
                 "C3": ((m["absC_rnd"] - m["absC"]) if rmean is not None else None, "D_cure - D_random_mean"),
                 "C4": ((m["gen_acc_attempted"] - m["gen_rnd"]) if rmean is not None else None, "acc_cure - acc_random_mean")}
        recs = []
        for (cid, desc), (x, stat) in zip(CONFIRMATORY, tests.values()):
            rec = {"model_name": mname, "phase": phase, "id": cid, "comparison": desc, "statistic": stat, "label": "confirmatory"}
            if x is None or not np.isfinite(x.to_numpy(float)).any():
                rec.update({"point": np.nan, "lo": np.nan, "hi": np.nan, "p_boot": np.nan, "n_seeds": 0, "note": "inputs missing"})
            else:
                ok = np.isfinite(x.to_numpy(float)); xv = x.to_numpy(float)[ok]; cl = m["seed_id"].to_numpy()[ok]
                rec["point"], rec["lo"], rec["hi"] = _ci(xv, np.ones(len(xv)), cl)
                rec["p_boot"] = boot_p(_cluster_draws(xv, np.ones(len(xv)), cl)); rec["n_seeds"] = int(len(set(cl)))
            recs.append(rec)
        ps = [r["p_boot"] for r in recs]
        adj = holm([p if np.isfinite(p) else 1.0 for p in ps])
        for r, a in zip(recs, adj):
            r["p_holm"] = a if np.isfinite(r["p_boot"]) else np.nan; r["reject_0.05_holm"] = bool(np.isfinite(r["p_boot"]) and a < 0.05)
        out += recs
    return pd.DataFrame(out)


def summarise() -> None:
    """CPU-only: p2_summary.csv, p2_confirmatory.csv, p2_report.md from the parquet files."""
    per_item = OUT / "p2_per_item.parquet"
    if not per_item.exists():
        log.warning("no per-item results yet"); return
    raw = pd.read_parquet(per_item)
    status_rows = raw[raw["status"] != "ok"] if "status" in raw else raw.iloc[0:0]
    d = raw[(raw["status"] == "ok")] if "status" in raw else raw
    if "error" in d:
        d = d[d["error"].isna()]
    d = _item_scores(d)
    summ = _summary_rows(d)
    K.write_csv(summ, OUT / "p2_summary.csv")
    conf = pd.concat([_confirmatory(d, ph) for ph in ("test", "dev")], ignore_index=True) if len(d) else pd.DataFrame()
    if len(conf):
        conf.loc[conf["phase"] == "dev", "label"] = "exploratory (dev; same four comparisons, not confirmatory)"
        K.write_csv(conf, OUT / "p2_confirmatory.csv")
    cap = pd.read_parquet(OUT / "p2_capability.parquet") if (OUT / "p2_capability.parquet").exists() else pd.DataFrame()
    write_report(summ, conf, cap, status_rows, raw)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _fmt_ci(r, key) -> str:
    v, lo, hi = r.get(key, np.nan), r.get(f"{key}_lo", np.nan), r.get(f"{key}_hi", np.nan)
    return "%.3f [%.3f, %.3f]" % (v, lo, hi) if np.isfinite(v) else "n/a"


def _excludes_zero(r, key) -> bool | None:
    lo, hi = r.get(f"{key}_lo", np.nan), r.get(f"{key}_hi", np.nan)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    return bool(lo > 0 or hi < 0)


def _gate_for_model(summ: pd.DataFrame, mname: str) -> list[str]:
    """Next_Plan.md P2 decision gate, applied to demographic pairs pooled on TEST (dev with a
    warning when no test rows exist)."""
    phase = "test" if ((summ["model_name"] == mname) & (summ["phase"] == "test")).any() else "dev"
    s = summ[(summ["model_name"] == mname) & (summ["phase"] == phase) & (summ["pair_type"] == "demographic") & (summ["stratum"] == "pooled")]
    lines = ["### %s (phase = %s%s)" % (K.DISPLAY.get(mname, mname), phase, "" if phase == "test" else "; DEVELOPMENT ONLY, not final evidence")]
    cure = s[s["cond_id"] == CURE_ID]
    if cure.empty:
        return lines + ["- no cure_centred_svd rank-1 alpha-1 rows; gate not evaluated"]
    r = cure.iloc[0].to_dict()
    supp, dmg = _excludes_zero(r, "D") and r["D"] > 0, _excludes_zero(r, "d_gen_acc_attempted") and r["d_gen_acc_attempted"] < 0
    lines += ["- cure: D = %s; delta generation accuracy = %s; delta option accuracy = %s; validity %.2f -> %.2f (n_seeds = %d)" % (
        _fmt_ci(r, "D"), _fmt_ci(r, "d_gen_acc_attempted"), _fmt_ci(r, "d_opt_acc"), r["validity_pre"], r["validity"], r["n_seeds"])]
    if not supp:
        lines.append("- suppression NOT established (D interval includes 0 or D <= 0); 'suppression with damage' is not retained.")
    if not dmg:
        lines.append("- damage NOT established (generation-accuracy change interval includes 0); 'suppression with damage' is not retained.")
    if not (supp and dmg):
        return lines
    verdict = "retain 'suppression with damage' for this model (both intervals exclude 0 under the same edit)"
    for fam_ids, label in ((RANDOM_IDS + [c for c in s["cond_id"] if c.startswith("random_ortho") and c.endswith("_matched")], "random controls"),
                           ([c for c in s["cond_id"] if c.startswith("neutral_contrast")], "neutral-contrast control")):
        ctrl = s[s["cond_id"].isin(fam_ids)]
        if ctrl.empty:
            lines.append("- %s: no rows" % label); continue
        similar = []
        for _, cr in ctrl.iterrows():
            cd = cr.to_dict()
            hurt = _excludes_zero(cd, "d_gen_acc_attempted") and cd["d_gen_acc_attempted"] < 0
            within = np.isfinite(cd["d_gen_acc_attempted"]) and r["d_gen_acc_attempted_lo"] <= cd["d_gen_acc_attempted"] <= r["d_gen_acc_attempted_hi"]
            lines.append("  - %s: D = %s; delta gen accuracy = %s; energy_gen %.3g vs cure %.3g%s" % (
                cd["cond_id"], _fmt_ci(cd, "D"), _fmt_ci(cd, "d_gen_acc_attempted"), cd.get("energy_gen_mean", np.nan),
                r.get("energy_gen_mean", np.nan), " (damage similar to cure)" if (hurt and within) else ""))
            similar.append(bool(hurt and within))
        if any(similar):
            verdict = ("report NON-SPECIFIC PERTURBATION, not demographic-computation entanglement: a %s shows damage "
                       "whose point estimate lies inside the cure interval" % label)
    for fam in LEACE_FAMILIES:
        le = s[s["cond_id"].str.startswith(fam)]
        if le.empty:
            continue
        ld = le.iloc[0].to_dict()
        lines.append("  - %s: D = %s; delta gen accuracy = %s" % (ld["cond_id"], _fmt_ci(ld, "D"), _fmt_ci(ld, "d_gen_acc_attempted")))
        if _excludes_zero(ld, "d_gen_acc_attempted") is False and (_excludes_zero(ld, "D") and ld["D"] > 0):
            verdict += ("; actual LEACE (%s) suppresses without an established accuracy loss, so NARROW the "
                        "conclusion to the studied projection protocol" % ("frozen" if fam == "leace_faithful" else "sequential"))
    lines.append("- **Verdict:** %s." % verdict)
    return lines


def _sample_size_note(summ: pd.DataFrame) -> list[str]:
    lines = ["## Development uncertainty and final sample size", ""]
    dev = summ[(summ["phase"] == "dev") & (summ["cond_id"] == CURE_ID) & (summ["pair_type"] == "demographic") & (summ["stratum"] == "pooled")]
    for _, r in dev.iterrows():
        hw = (r["D_hi"] - r["D_lo"]) / 2 if np.isfinite(r["D_hi"]) else np.nan
        proj = hw * np.sqrt(r["n_seeds"] / N_TEST_CAP) if np.isfinite(hw) else np.nan
        lines.append("- %s: dev D half-width %.3f on %d seeds; projected half-width at %d test seeds about %.3f (sqrt scaling); "
                     "if this is wider than the practically relevant effect, report inconclusive evidence rather than a null."
                     % (K.DISPLAY.get(r["model_name"], r["model_name"]), hw, r["n_seeds"], N_TEST_CAP, proj))
    return lines + [""]


def write_report(summ: pd.DataFrame, conf: pd.DataFrame, cap: pd.DataFrame, status_rows: pd.DataFrame, raw: pd.DataFrame) -> None:
    proto = json.loads((OUT / "p2_protocol.json").read_text(encoding="utf-8")) if (OUT / "p2_protocol.json").exists() else {}
    L = ["# P2 controlled erasure report", "", "Generated %s." % K.utc_now(), "",
         "Edit: site = span (every token of the demographic span, prefill only), source state = sequential, applied "
         "identically to the audit, the option readout and the generation readout. Capability policy: **%s**. "
         "Statistics: seed-cluster percentile bootstrap (%d draws); D = paired absolute |C| change is primary; "
         "R = ratio of means is secondary; X uses TAU = %.4f with its denominator shown. The test split is a new "
         "controlled held-out evaluation, not a historically untouched test." % (proto.get("capability_policy", "?"), K.N_BOOT, K.TAU), ""]
    if len(status_rows):
        L += ["## Conditions not run", ""] + ["- %s / %s / %s: %s" % (r["model_name"], r["phase"], r["cond_id"], r["status"]) for _, r in status_rows.iterrows()] + [""]
    for m, mrec in proto.get("models", {}).items():
        ic = mrec.get("identity_checks", {})
        L += ["## %s" % K.DISPLAY.get(m, m), "", "- identity checks pass_all = **%s**; bases: %s" % (ic.get("pass_all"), json.dumps(mrec.get("bases", {}).get("status", {}))),
              "- basis fit ids: %s" % json.dumps(mrec.get("bases", {}).get("fit_id", {})), ""]
    if len(summ):
        cols = ["model_name", "phase", "cond_id", "pair_type", "n_seeds", "absC_pre_mean", "absC_post_mean", "W", "L", "D", "D_lo", "D_hi",
                "R", "X_num", "X_den", "opt_acc_pre", "opt_acc", "d_opt_acc", "gen_acc_attempted_pre", "gen_acc_attempted", "d_gen_acc_attempted",
                "d_gen_acc_attempted_lo", "d_gen_acc_attempted_hi", "validity", "flip_attempted", "flip_usable", "energy_gen_mean", "label"]
        pooled = summ[summ["stratum"] == "pooled"][[c for c in cols if c in summ]]
        L += ["## Pooled per-condition statistics (every row exploratory unless labelled)", "", pooled.round(3).to_markdown(index=False), ""]
        L += ["## Benchmark strata (D and delta generation accuracy)", "",
              summ[summ["stratum"] != "pooled"][["model_name", "phase", "cond_id", "pair_type", "stratum", "n_seeds", "D", "D_lo", "D_hi",
                                                  "d_gen_acc_attempted", "d_gen_acc_attempted_lo", "d_gen_acc_attempted_hi"]].round(3).to_markdown(index=False), ""]
        dev_e = summ[(summ["phase"] == "dev") & (summ["stratum"] == "pooled") & (summ["pair_type"] == "demographic")]
        if len(dev_e):
            L += ["## Development screen: removed activation energy (mean over layers)", "",
                  dev_e[["model_name", "cond_id", "rank", "alpha", "energy_audit_mean", "energy_gen_mean", "D", "d_gen_acc_attempted"]].round(4).to_markdown(index=False), ""]
    if (OUT / "p2_matched_alphas.json").exists():
        ma = json.loads((OUT / "p2_matched_alphas.json").read_text(encoding="utf-8"))
        L += ["## Energy-matched control strengths (chosen on dev, frozen for test)", ""]
        for m, rec in ma.items():
            L.append("- %s: target %.4g (%s); " % (K.DISPLAY.get(m, m), rec["target_energy"], rec["target_condition"]) +
                     "; ".join("%s alpha=%.2f -> %.4g%s" % (f, v["alpha"], v["energy_achieved"],
                                                           "" if v.get("matchable", True) else " (NOT matchable: weaker than cure at alpha = 1)")
                               for f, v in rec["controls"].items() if "alpha" in v))
        L.append("")
    if len(conf):
        L += ["## Predeclared confirmatory comparisons (Holm within model over the four)", "",
              conf[["model_name", "phase", "id", "comparison", "point", "lo", "hi", "p_boot", "p_holm", "reject_0.05_holm", "n_seeds", "label"]].round(4).to_markdown(index=False), ""]
    if len(cap):
        keep = [c for c in ["model_name", "phase", "cond_id", "probe", "capability_policy", "site", "status", "value", "perplexity", "n", "energy_mean"] if c in cap]
        L += ["## Independent utility (external probes; narrow, not capability certification)", "",
              "Policy = %s. Under policy none only the unedited model is scored and every edited condition is recorded as not evaluated, "
              "because these prompts carry no demographic span. Under policy all the edit is applied at every position." % proto.get("capability_policy", "?"), "",
              cap[keep].round(4).to_markdown(index=False), ""]
    L += _sample_size_note(summ) if len(summ) else []
    L += ["## Decision gate (Next_Plan.md P2)", ""]
    for m in (summ["model_name"].unique() if len(summ) else []):
        L += _gate_for_model(summ, m) + [""]
    L += ["## Not implemented in this run", "",
          "- Lexical-frequency matching of the neutral-contrast fit pairs to the demographic fit pairs; only span token counts are recorded.",
          "- A label-order check of the option readout; option order is the frozen pentad order.",
          "- Sequential LEACE is opt-in (--leace-sequential) because it costs one forward pass per prompt per layer.", ""]
    (OUT / "p2_report.md").write_text("\n".join(L), encoding="utf-8")
    log.info("wrote %s", K.rel(OUT / "p2_report.md"))


if __name__ == "__main__":
    main()
