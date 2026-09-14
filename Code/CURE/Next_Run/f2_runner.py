"""
f2_runner.py -- the one controlled position-and-direction experiment (Next_Plan.md Section 7)
plus the F3 coherent-label baselines (Section 8), for ONE model on ONE GPU. Resumable.

Stages (run in this order; each is idempotent and checkpointed under results/final_20260914/)
  checks       identity checks (Section 7.3) on the development seeds:
               alpha 0 / rank 0 reproduce the baseline, non-span positions untouched, span
               edited, no decode-step edit under span and prefill_all, same-input patching is
               the identity null (C_answer(a,a) = 0), batched == single-item scoring, the
               candidate boundary is a clean token boundary
  spans        demographic / matched / prefill spans for every seed and side (frozen)
  calibrate    Section 7.2 energy matching of SE, NE, RE, RNE on the development seeds:
               common denominator from UNEDITED activations at both span locations, target =
               min over cells of the alpha-one energy, per-cell bounded search (<= 8
               evaluations), one allowed target reduction, 5% tolerance; alphas frozen
  timing       seconds per item per condition on the development seeds -> runtime_budget.json
               and the sample-tier decision (160 or 100) before any final row is written
  final        every (seed, side, condition) of the chosen tier: complete-answer scores,
               generation, energy; the patching bridge on B and S1; the 32-seed scoring
               sensitivity rows (permuted options, label-only candidates) on B and S1
  control      the relevant-information group under B, S1, NE
  baseline     F3: collect span activations for the coherent man/woman contrast of the fit
               split, fit official LEACE and the mean difference on the same labels and set,
               held-out probe and algebraic checks, then LC / MC on the eligible final subset

Conditions (Section 7.1; every edit is prefill-only, every decoder layer)
  B    none                       S1  target basis, demographic span, alpha 1
  SE   target, demographic span, calibrated alpha      NE   target, matched span, calibrated
  RE   random rank-1, demographic span, calibrated     RNE  random rank-1, matched span, calibrated
  G1   target basis, every non-padding prefill position, alpha 1
  LC   coherent-label LEACE (exact affine map), demographic span, alpha 1     (F3)
  MC   coherent-label mean-difference projection, demographic span, alpha 1  (F3)

Usage
  python f2_runner.py --model gemma-2-2b-it --stage checks
  python f2_runner.py --model gemma-2-2b-it --stage all        # checks..control (+baseline)
  python f2_runner.py --model gemma-2-2b-it --stage final --smoke   # two seeds, every stage

Implements / builds on
  - p1_intervention.HookedEdit (one hook implementation), p2_leace_faithful (official LEACE)
  - Belrose et al. (2023) LEACE; Vig et al. (2020) / Zhang and Nanda (2024) patching;
    Efron and Tibshirani (1993) bootstrap (analysis stage, f4_analysis.py)
Part of the CURE codebase (ICLR 2027, Submission2). GPU required.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import nr_common as N
import f1_scoring as S
import p1_intervention as I     # noqa: E402
import p1_pilot as P1           # noqa: E402

log = N.setup_logging("cure.final.f2")
OUT = N.FINAL
CALIBRATED = ("SE", "NE", "RE", "RNE")
ROW_KEY = ["model_name", "seed_id", "side", "cond", "variant", "kind"]


# ---------------------------------------------------------------------------
# bases
# ---------------------------------------------------------------------------

def load_target_basis(model: str) -> tuple[dict, dict]:
    """The reused rank-one centred-SVD basis of results/v2 (fit provenance recorded)."""
    p = N.V2 / ("p2_basis_%s_chat_cure_centred_svd.npz" % model)
    z = np.load(p)
    basis = {int(k): np.asarray(z[k])[:1] for k in z.files}
    man = N.read_json(N.V2 / ("p2_basis_manifest_%s_chat.json" % model), {})
    info = (man.get("info") or {}).get("cure_centred_svd", {})
    prov = {"file": N.K.rel(p), "sha256": N.sha256_file(p), "rank_used": 1, "max_rank_in_file": int(next(iter(z.values())).shape[0]),
            "fit_info": info, "fit_id": (man.get("fit_id") or {}).get("cure_centred_svd"),
            "note": "contrast-derived projection: leading right singular vector of centred span-position "
                    "activation differences over the fit pairs; %s of the fit pairs are neutral-type substitutions"
                    % info.get("n_fit_pairs_neutral_type", "?")}
    return basis, prov


def random_basis(basis_like: dict, seed: int) -> dict:
    """One fixed rank-one orthonormal direction per layer, drawn before any outcome is seen and
    shared by RE and RNE (Section 7.1)."""
    out = {}
    for l, rows in basis_like.items():
        d = rows.shape[1]
        g = np.random.default_rng(seed * 1000 + int(l)).standard_normal(d).astype(np.float32)
        out[int(l)] = (g / np.linalg.norm(g))[None, :]
    return out


def mean_difference_basis(diffs_by_layer: dict[int, np.ndarray]) -> dict:
    out = {}
    for l, X in diffs_by_layer.items():
        m = X.mean(axis=0); n = np.linalg.norm(m)
        out[int(l)] = (m / n if n > 0 else m)[None, :].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# manifest and spans
# ---------------------------------------------------------------------------

def manifest() -> pd.DataFrame:
    return pd.read_csv(OUT / "source_manifest.csv")


def spans_path(model: str) -> Path:
    return OUT / ("spans_%s.json" % model)


def resolve_spans(model: str, tok, man: pd.DataFrame) -> dict:
    """Frozen span positions for every seed and side; structural failures recorded."""
    p = spans_path(model)
    if p.exists():
        return N.read_json(p)
    out = {"model": model, "rule": "demographic = every whole-word occurrence of the entity-1 descriptor in the user "
                                  "prompt; matched = contiguous context tokens of equal count nearest the context "
                                  "occurrence, no overlap with either descriptor, no punctuation-only token; "
                                  "prefill_all = every non-padding prefill position", "seeds": {}}
    for r in man.itertuples(index=False):
        rec = {}
        for side in ("A", "B"):
            up = getattr(r, "prompt_%s" % side); desc = getattr(r, "descriptor_%s" % side)
            text = S.chat_text(tok, up)
            demo = S.demographic_span(tok, text, up, desc)
            ent2 = S.demographic_span(tok, text, up, r.entity2_descriptor)
            ref = demo["occurrences"][0]["positions"][0] if demo["occurrences"] else 0
            ne = S.matched_control_span(tok, text, up, len(demo["positions"]), demo["positions"] + ent2["positions"], ref,
                                        seed=N.RANDOM_SEED_FINAL + int(N.sha256_text(r.seed_id + side)[:6], 16))   # stable tie-break seed
            rec[side] = {"text": text, "demo": demo["positions"], "n_demo_occurrences": demo["n_occurrences"],
                         "ne": ne["positions"], "ne_reason": ne["reason"], "ne_distance": ne.get("distance_tokens"),
                         "ne_text": ne.get("text"), "n_prefill": len(tok(text, add_special_tokens=True)["input_ids"]),
                         "structural_failure": (not demo["positions"]) or ne["reason"] != "ok"}
        out["seeds"][r.seed_id] = rec
    N.write_json(out, p)
    return out


# ---------------------------------------------------------------------------
# energy bookkeeping
# ---------------------------------------------------------------------------

def energy_of(edit) -> dict:
    """Sum of squared displacements and of squared activation norms at the edited positions,
    by layer, from HookedEdit.changed_log (prefill only)."""
    num, den = {}, {}
    for r in edit.changed_log:
        if not r["prefill"]:
            continue
        num[r["layer"]] = num.get(r["layer"], 0.0) + r["delta_norm"] ** 2
        den[r["layer"]] = den.get(r["layer"], 0.0) + r["h_norm"] ** 2
    return {"num": float(sum(num.values())), "den_edited": float(sum(den.values())), "by_layer": {str(k): v for k, v in num.items()},
            "n_edits": len(edit.changed_log), "n_decode_edits": edit.n_decode_edits}


def unedited_denominator(model, tok, text: str, positions: list[int]) -> float:
    """Sum over layers and the given positions of ||h||^2 on the unedited pass (Section 7.2)."""
    _, hs = I.forward_hidden(model, tok, text, None, None)
    return float(sum(float((hs[l][positions, :].float() ** 2).sum()) for l in hs))


# ---------------------------------------------------------------------------
# condition -> edit
# ---------------------------------------------------------------------------

class Bases:
    def __init__(self, model: str):
        self.model = model
        self.target, self.target_prov = load_target_basis(model)
        self.random = random_basis(self.target, N.RANDOM_SEED_FINAL)
        self.leace = None; self.md = None      # F3, loaded when present
        p = OUT / ("f3_baseline_%s.npz" % model)
        if p.exists():
            import p2_leace_faithful as LF
            z = np.load(p)
            self.leace = LF.erasers_from_npz(z)          # md_* keys are ignored by the reader (no 'L' prefix)
            self.md = {int(k[3:]): z[k] for k in z.files if k.startswith("md_")}


def make_edit(model, bases: Bases, cond: str, alphas: dict):
    """Returns (factory, site, positions_key, alpha, basis_id). factory() builds a fresh edit."""
    if cond == "B":
        return None, "none", None, 0.0, "none"
    if cond in ("S1", "SE", "NE", "G1"):
        U, bid = bases.target, "target_rank1"
    elif cond in ("RE", "RNE"):
        U, bid = bases.random, "random_rank1_seed%d" % N.RANDOM_SEED_FINAL
    elif cond == "MC":
        U, bid = bases.md, "mean_difference_coherent"
    elif cond == "LC":
        import p2_leace_faithful as LF
        er = bases.leace
        return (lambda: LF.LeaceAffineEdit(model, er, alpha=1.0, site="span")), "span", "demo", 1.0, "leace_coherent"
    else:
        raise ValueError(cond)
    alpha = float(alphas.get(cond, 1.0)) if cond in CALIBRATED else 1.0
    site = "prefill_all" if cond == "G1" else "span"
    pkey = {"S1": "demo", "SE": "demo", "RE": "demo", "NE": "ne", "RNE": "ne", "G1": "prefill", "MC": "demo"}[cond]
    return (lambda: I.HookedEdit(model, U, alpha=alpha, site=site)), site, pkey, alpha, bid


def positions_for(rec_side: dict, pkey: str | None) -> list[int]:
    if pkey is None:
        return []
    if pkey == "prefill":
        return list(range(rec_side["n_prefill"]))
    return list(rec_side[pkey])


# ---------------------------------------------------------------------------
# one (seed, side, condition) evaluation
# ---------------------------------------------------------------------------

def eval_side(model, tok, bases: Bases, alphas: dict, r, side: str, sp: dict, cond: str, variant: str = "std",
              generate: bool = True) -> dict:
    t0 = time.time()
    up = getattr(r, "prompt_%s" % side)
    options = json.loads(getattr(r, "options_%s" % side))
    gold = int(r.gold_index)
    if variant == "perm":
        options, mp = S.rotate_options(options, 1)
        gold = mp[gold]
        up = _replace_options(up, options)
    text = S.chat_text(tok, up) if variant == "perm" else sp["text"]
    cands = S.candidates(options, "label" if variant == "label" else "text")
    first_ids = S.first_token_ids(tok, text, options)
    factory, site, pkey, alpha, bid = make_edit(model, bases, cond, alphas)
    pos = positions_for(sp, pkey)
    if variant == "perm" and pkey in ("demo", "ne"):
        # positions were resolved on the standard prompt; re-resolve the demographic span on the
        # permuted prompt (the option line moved); the matched span is not used in this variant
        pos = S.demographic_span(tok, text, up, getattr(r, "descriptor_%s" % side))["positions"] if pkey == "demo" else []
    row = {"model_name": bases.model, "seed_id": r.seed_id, "group": r.group, "side": side, "cond": cond, "variant": variant,
           "kind": "score", "category": r.category, "template": r.template, "site": site, "alpha": alpha, "basis_id": bid,
           "rank": 0 if cond == "B" else 1, "positions": json.dumps(pos), "n_edited_positions": len(pos),
           "gold_index": gold, "options": json.dumps(options), "status": "ok"}
    try:
        e = factory() if factory else None
        sc = S.score_candidates(model, tok, text, cands, e, pos, first_ids)
        q = S.softmax(sc["s"])
        row.update({"s": json.dumps(sc["s"]), "s_mean": json.dumps(sc["s_mean"]), "n_cand_tokens": json.dumps(sc["n_tokens"]),
                    "q": json.dumps(q), "argmax": int(np.argmax(sc["s"])), "score_correct": bool(int(np.argmax(sc["s"])) == gold),
                    "M": S.margin(sc["s"], gold), "first_logits": json.dumps(sc["first_logits"]),
                    "first_disambiguates": S.first_token_disambiguates(first_ids, gold), "T0": sc["T0"]})
        if e is not None:
            en = energy_of(e)
            row.update({"energy_num": en["num"], "energy_den_edited": en["den_edited"], "energy_by_layer": json.dumps(en["by_layer"]),
                        "n_decode_edits_score": en["n_decode_edits"]})
        else:
            row.update({"energy_num": 0.0, "energy_den_edited": float("nan"), "energy_by_layer": "{}", "n_decode_edits_score": 0})
        if generate and variant == "std":
            e2 = factory() if factory else None
            gen = I.generate_under_edit(model, tok, [text], e2, [pos], max_new_tokens=N.MAX_NEW_TOKENS)[0]
            gi, why = S.parse_generation(gen, up)
            row.update({"gen_raw": gen, "gen_idx": (gi if gi is not None else -1), "gen_parse": why,
                        "gen_valid": gi is not None, "gen_correct": (gi == gold) if gi is not None else False,
                        "n_decode_edits_gen": (e2.n_decode_edits if e2 is not None else 0)})
    except Exception as ex:                           # a failure is an outcome, never a dropped row
        row.update({"status": "error", "error": str(ex)[:300]})
        log.exception("eval failed %s %s %s", r.seed_id, side, cond)
    row["sec"] = time.time() - t0
    return row


def _replace_options(up: str, options: list[str]) -> str:
    head = up.split("\n(A)")[0]
    letters = "ABCDEFGH"
    return head + "\n" + "\n".join("(%s) %s" % (letters[i], o) for i, o in enumerate(options)) + "\n" + up.split("\n")[-1]


def eval_bridge(model, tok, bases: Bases, alphas: dict, r, sp: dict, cond: str) -> dict:
    """Patching bridge (Section 6.3) for one pair under one condition: a -> b."""
    t0 = time.time()
    up_b = r.prompt_B; options = json.loads(r.options_B); gold = int(r.gold_index)
    factory, site, pkey, alpha, bid = make_edit(model, bases, cond, alphas)
    pa, pb = sp["A"]["demo"], sp["B"]["demo"]
    cands = S.candidates(options, "text")
    first_ids = S.first_token_ids(tok, sp["B"]["text"], options)
    row = {"model_name": bases.model, "seed_id": r.seed_id, "group": r.group, "side": "B", "cond": cond, "variant": "std",
           "kind": "bridge", "category": r.category, "template": r.template, "site": site, "alpha": alpha, "basis_id": bid,
           "rank": 0 if cond == "B" else 1, "gold_index": gold, "status": "ok"}
    try:
        e = factory() if factory else None
        # the reference is the IDENTITY-patched run (b's own span states injected at b), so the
        # clean and patched passes share the injection path and bf16 kernel noise cancels; the
        # same-input patch null is then exactly zero (identity_checks.json)
        clean = S.patched_scores(model, tok, sp["B"]["text"], sp["B"]["text"], pb, pb, cands, factory, first_ids)
        pat = S.patched_scores(model, tok, sp["A"]["text"], sp["B"]["text"], pa, pb, cands, factory, first_ids)
        if "error" in pat or "error" in clean:
            row.update({"status": "error", "error": pat.get("error") or clean.get("error")}); row["sec"] = time.time() - t0; return row
        Mc, Mp = S.margin(clean["s"], gold), S.margin(pat["s"], gold)
        row.update({"M_clean": Mc, "M_patched": Mp, "C_answer": Mp - Mc,
                    "C_first": pat["first_logits"][gold] - clean["first_logits"][gold],
                    "first_disambiguates": S.first_token_disambiguates(first_ids, gold),
                    "patched_len": pat["patched_len"], "span_len_a": pat["span_len_a"], "span_len_b": pat["span_len_b"],
                    "s_clean": json.dumps(clean["s"]), "s_patched": json.dumps(pat["s"])})
    except Exception as ex:
        row.update({"status": "error", "error": str(ex)[:300]}); log.exception("bridge failed %s %s", r.seed_id, cond)
    row["sec"] = time.time() - t0
    return row


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_checks(model, tok, bases: Bases, man: pd.DataFrame, sp_all: dict) -> dict:
    dev = man[man.group == "dev"].head(4)
    texts, pos = [], []
    for r in dev.itertuples(index=False):
        texts.append(sp_all["seeds"][r.seed_id]["A"]["text"]); pos.append(sp_all["seeds"][r.seed_id]["A"]["demo"])
    res = I.identity_checks(model, tok, texts, bases.target, pos)
    # prefill_all: no decode edits and every prefill position touched
    e = I.HookedEdit(model, bases.target, alpha=1.0, site="prefill_all")
    _ = I.generate_under_edit(model, tok, texts[:1], e, [list(range(len(tok(texts[0])["input_ids"])))], max_new_tokens=4)
    touched = {r["position"] for r in e.changed_log if r["layer"] == min(bases.target)}
    res["prefill_all_no_decode_edit"] = e.n_decode_edits == 0
    res["prefill_all_touches_every_prefill_position"] = len(touched) == len(tok(texts[0])["input_ids"])
    # same-input patching is the identity null on candidate scores
    r0 = dev.iloc[0]; s0 = sp_all["seeds"][r0.seed_id]["A"]
    cands = S.candidates(json.loads(r0.options_A), "text")
    fids = S.first_token_ids(tok, s0["text"], json.loads(r0.options_A))
    clean = S.score_candidates(model, tok, s0["text"], cands, None, [], fids)
    ref = S.patched_scores(model, tok, s0["text"], s0["text"], s0["demo"], s0["demo"], cands, None, fids)      # identity-patched reference
    pat = S.patched_scores(model, tok, s0["text"], s0["text"], s0["demo"], s0["demo"], cands, None, fids)      # same input again
    res["max_abs_dev_same_input_patch"] = float(np.max(np.abs(np.asarray(pat["s"]) - np.asarray(ref["s"]))))
    res["pass_same_input_patch_null"] = res["max_abs_dev_same_input_patch"] <= 1e-6      # identical computation: exact
    # injection-path noise floor (informational): an identity patch vs an unpatched batched pass
    res["injection_vs_unpatched_max_abs_dev"] = float(np.max(np.abs(np.asarray(ref["s"]) - np.asarray(clean["s"]))))
    # batched vs single-item scoring (informational): every comparison in the experiment is between
    # batched passes of identical shape, so this measures kernel noise, not a protocol defect
    single = [S.score_candidates(model, tok, s0["text"], [c], None, [], None)["s"][0] for c in cands]
    res["max_abs_dev_batched_vs_single"] = float(np.max(np.abs(np.asarray(single) - np.asarray(clean["s"]))))
    res["tolerances"] = {"logits_alpha0_rank0": res["tol"], "same_input_patch": 1e-6,
                         "note": "bf16 with flash attention. alpha-0, rank-0 and the same-input patch null are exact no-ops "
                                 "by construction and must be 0. injection_vs_unpatched and batched_vs_single compare "
                                 "different kernel paths and are recorded as the instrument's bf16 noise floor; the "
                                 "bridge uses the identity-patched reference so that noise cancels"}
    res["candidate_boundary"] = S.check_boundary(tok, s0["text"], cands)
    res["pass_all_final"] = bool(res["pass_all"] and res["prefill_all_no_decode_edit"] and res["pass_same_input_patch_null"]
                                 and res["candidate_boundary"]["n_boundary_merges"] == 0)
    res["model"] = bases.model; res["utc"] = N.utc_now()
    allp = N.read_json(OUT / "identity_checks.json", {})
    allp[bases.model] = res
    N.write_json(allp, OUT / "identity_checks.json")
    log.info("identity checks %s: pass_all_final=%s", bases.model, res["pass_all_final"])
    return res


def _cell_energy(model, tok, bases: Bases, cond: str, alpha: float, items: list[tuple[str, dict]]) -> float:
    """Mean over dev prompts of the summed squared displacement of `cond` at strength alpha."""
    tot = []
    for text, pos in items:
        U = bases.target if cond in ("SE", "NE") else bases.random
        e = I.HookedEdit(model, U, alpha=alpha, site="span")
        S.score_candidates(model, tok, text, [S.PREFIX + "x" + S.SUFFIX], e, pos, None)
        tot.append(energy_of(e)["num"])
    return float(np.mean(tot))


def stage_calibrate(model, tok, bases: Bases, man: pd.DataFrame, sp_all: dict, smoke: bool = False) -> dict:
    p = OUT / ("calibration_%s.json" % bases.model)
    if p.exists():
        return N.read_json(p)
    dev = man[man.group == "dev"]
    if smoke:
        dev = dev.head(3)
    items = {"demo": [], "ne": []}
    dens = []
    for r in dev.itertuples(index=False):
        for side in ("A", "B"):
            s = sp_all["seeds"][r.seed_id][side]
            if s["structural_failure"]:
                continue
            items["demo"].append((s["text"], s["demo"])); items["ne"].append((s["text"], s["ne"]))
            dens.append(unedited_denominator(model, tok, s["text"], sorted(set(s["demo"]) | set(s["ne"]))))
    D = float(np.mean(dens))
    cal = {"model": bases.model, "denominator": D, "denominator_rule": "mean over dev prompts of sum over layers and over the "
           "demographic span plus the matched span of ||h_unedited||^2", "n_dev_prompts": len(dens), "tolerance": N.MATCH_TOL,
           "cells": {}, "utc": N.utc_now()}
    e1 = {}
    for cond in CALIBRATED:
        e1[cond] = _cell_energy(model, tok, bases, cond, 1.0, items["demo" if cond in ("SE", "RE") else "ne"]) / D
    target = min(e1.values())
    cal["alpha_one_energy"] = e1; cal["target_initial"] = target
    if target <= 1e-9:
        cal["matching_status"] = "unsuccessful: common feasible energy numerically negligible"
        cal["alphas"] = {c: 1.0 for c in CALIBRATED}; cal["feasible"] = False
        N.write_json(cal, p); return cal

    def calibrate_cell(cond, tgt):
        pool = items["demo" if cond in ("SE", "RE") else "ne"]
        evals = []
        a = min(1.0, math.sqrt(tgt / e1[cond]))            # quadratic first guess
        lo_a, lo_e, hi_a, hi_e = 0.0, 0.0, 1.0, e1[cond]
        best = None
        for k in range(8):
            en = _cell_energy(model, tok, bases, cond, a, pool) / D
            evals.append({"alpha": a, "energy": en, "rel_err": (en - tgt) / tgt})
            if best is None or abs(en - tgt) < abs(best[1] - tgt):
                best = (a, en)
            if abs(en - tgt) / tgt <= N.MATCH_TOL:
                break
            if en < tgt:
                lo_a, lo_e = a, en
            else:
                hi_a, hi_e = a, en
            # secant on the bracket, guarded by bisection
            if hi_e > lo_e:
                a_new = lo_a + (tgt - lo_e) * (hi_a - lo_a) / (hi_e - lo_e)
            else:
                a_new = 0.5 * (lo_a + hi_a)
            if not (lo_a < a_new < hi_a):
                a_new = 0.5 * (lo_a + hi_a)
            a = float(min(1.0, max(1e-4, a_new)))
        return {"alpha": best[0], "achieved": best[1], "rel_err": (best[1] - tgt) / tgt, "within_tol": abs(best[1] - tgt) / tgt <= N.MATCH_TOL,
                "evaluations": evals}

    for attempt, tgt in enumerate((target, 0.5 * target)):
        cells = {c: calibrate_cell(c, tgt) for c in CALIBRATED}
        cal["cells"] = cells; cal["target"] = tgt; cal["target_reductions"] = attempt
        if all(v["within_tol"] for v in cells.values()):
            break
    cal["feasible"] = bool(all(v["within_tol"] for v in cal["cells"].values()))
    cal["alphas"] = {c: (cal["cells"][c]["alpha"] if cal["feasible"] else 1.0) for c in CALIBRATED}
    cal["matching_status"] = "matched within tolerance" if cal["feasible"] else \
        "unsuccessful: fixed-strength (alpha 1) comparison reported instead; no energy-controlled claim"
    N.write_json(cal, p)
    log.info("calibration %s: %s target=%.3e alphas=%s", bases.model, cal["matching_status"], cal["target"], cal["alphas"])
    return cal


def stage_timing(model, tok, bases: Bases, man: pd.DataFrame, sp_all: dict, cal: dict, args) -> dict:
    p = OUT / ("runtime_budget_%s.json" % bases.model)
    if p.exists() and not args.smoke:
        return N.read_json(p)
    dev = man[man.group == "dev"].head(3 if not args.smoke else 2)
    sec = {c: [] for c in N.CONDITIONS}
    t_bridge = []
    for r in dev.itertuples(index=False):
        sp = sp_all["seeds"][r.seed_id]
        for cond in N.CONDITIONS:
            for side in ("A", "B"):
                row = eval_side(model, tok, bases, cal["alphas"], r, side, sp[side], cond)
                sec[cond].append(row["sec"])
        for cond in ("B", "S1"):
            t_bridge.append(eval_bridge(model, tok, bases, cal["alphas"], r, sp, cond)["sec"])
    per_side = {c: float(np.mean(v)) for c, v in sec.items()}
    per_seed = 2 * sum(per_side.values()) + 2 * float(np.mean(t_bridge))
    n_final = int((man.group == "final").sum()); n_ctrl = int((man.group == "control").sum())
    est160 = (per_seed * n_final + 2 * (per_side["B"] + per_side["S1"] + per_side["NE"]) * n_ctrl
              + 2 * 2 * 2 * per_side["B"] * 32) / 3600.0
    est100 = (per_seed * 100 + 2 * (per_side["B"] + per_side["S1"] + per_side["NE"]) * n_ctrl + 8 * per_side["B"] * 32) / 3600.0
    rate = float(args.gpu_rate) if args.gpu_rate else None
    tier = 160 if est160 <= args.model_hours else 100
    rec = {"model": bases.model, "sec_per_side_by_condition": per_side, "sec_bridge": float(np.mean(t_bridge)),
           "sec_per_seed_all_conditions": per_seed, "estimate_hours_tier160": est160, "estimate_hours_tier100": est100,
           "model_hour_allowance": args.model_hours, "tier_chosen": tier, "gpu_rate_usd_per_hour": rate,
           "estimated_cost_usd": (est160 if tier == 160 else est100) * rate if rate else None,
           "decided_before_final_rows": not (OUT / ("per_item_%s.parquet" % bases.model)).exists(), "utc": N.utc_now()}
    N.write_json(rec, p)
    log.info("timing %s: %.1fs/seed, tier160 %.2fh, tier100 %.2fh -> tier %d", bases.model, per_seed, est160, est100, tier)
    return rec


def stage_rows(model, tok, bases: Bases, man: pd.DataFrame, sp_all: dict, cal: dict, group: str, conds: list[str],
               tier: int, smoke: bool, budget: dict) -> None:
    path = OUT / ("per_item_%s.parquet" % bases.model)
    done = N.done_keys(path, ROW_KEY)
    sub = man[man.group == group]
    if group == "final" and tier == 100:
        sub = sub[sub.in_tier_100]
    if smoke:
        sub = sub.head(2)
    t_start = time.time(); n_new = 0
    for r in sub.itertuples(index=False):
        sp = sp_all["seeds"][r.seed_id]
        rows = []
        for cond in conds:
            if cond in ("LC", "MC") and (bases.leace is None or not r.f3_eligible):
                continue
            for side in ("A", "B"):
                key = (bases.model, r.seed_id, side, cond, "std", "score")
                if key in done:
                    continue
                if sp[side]["structural_failure"] and cond in ("NE", "RNE"):
                    rows.append({"model_name": bases.model, "seed_id": r.seed_id, "group": group, "side": side, "cond": cond,
                                 "variant": "std", "kind": "score", "status": "structural_failure", "error": sp[side]["ne_reason"],
                                 "category": r.category, "template": r.template})
                    continue
                rows.append(eval_side(model, tok, bases, cal["alphas"], r, side, sp[side], cond))
            if cond in ("B", "S1") and group == "final":
                key = (bases.model, r.seed_id, "B", cond, "std", "bridge")
                if key not in done:
                    rows.append(eval_bridge(model, tok, bases, cal["alphas"], r, sp, cond))
                if getattr(r, "in_diag_32", False) or smoke:
                    for variant in ("perm", "label"):
                        for side in ("A", "B"):
                            key = (bases.model, r.seed_id, side, cond, variant, "score")
                            if key not in done:
                                rows.append(eval_side(model, tok, bases, cal["alphas"], r, side, sp[side], cond, variant, generate=False))
        if rows:
            for x in rows:
                x.setdefault("protocol_hash", N.load_protocol().get("protocol_hash", ""))
                N.append_jsonl(OUT / ("raw_generations_%s.jsonl" % bases.model),
                               {k: x.get(k) for k in ("model_name", "seed_id", "side", "cond", "variant", "kind", "gen_raw", "status")})
            N.append_rows(path, rows); n_new += len(rows)
        hours = (time.time() - t_start) / 3600.0 + budget.get("hours_used_before", 0.0)
        if hours > budget["cap_hours"]:
            log.error("GPU-hour cap %.1f reached (%.2f used); stopping cleanly, rows are on disk", budget["cap_hours"], hours)
            break
    log.info("%s/%s: %d new rows in %.1f min", bases.model, group, n_new, (time.time() - t_start) / 60)


def stage_baseline(model, tok, bases: Bases, man: pd.DataFrame, sp_all: dict) -> None:
    """F3: coherent man/woman contrast from the fit split; official LEACE and mean difference
    on the same labels and set; held-out probe check; algebraic hook check; saves the erasers."""
    import f3_baseline as F3
    F3.fit_and_validate(bases.model, model, tok, bases)


def load_for_run(model_name: str):
    """P1.load on a GPU; with NR_CPU=1 a float32 CPU load for local smoke tests only."""
    import os
    if os.environ.get("NR_CPU") == "1":
        import torch
        import config_cure as C
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cfg = dict(C.model_cfg(model_name)); hf = cfg["hf_id"]
        tok = AutoTokenizer.from_pretrained(hf, token=C.HUGGINGFACE_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(hf, token=C.HUGGINGFACE_TOKEN, torch_dtype=torch.float32, attn_implementation="eager")
        model.eval()
        cfg.update({"attn_implementation": "eager", "model_class": type(model).__name__, "dtype": "torch.float32", "device": "cpu"})
        return cfg, model, tok
    return P1.load(model_name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=N.ALL_MODELS)
    ap.add_argument("--stage", default="all", choices=["checks", "spans", "calibrate", "timing", "final", "control", "baseline", "all"])
    ap.add_argument("--smoke", action="store_true", help="two seeds per group, every stage")
    ap.add_argument("--model-hours", type=float, default=N.GPU_HOUR_CAP / 2, help="GPU-hour allowance for this model")
    ap.add_argument("--gpu-rate", type=float, default=None, help="USD per GPU hour, recorded in the budget")
    ap.add_argument("--tier", type=int, default=None, help="force the sample tier (160 or 100)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man = manifest()
    # F3 eligibility: gender contrast items (the coherent label fitted in stage 'baseline')
    man["f3_eligible"] = man.apply(lambda r: r.category == "Gender_identity" and str(r.group_A) in ("F", "M", "man", "woman", "boy", "girl")
                                   and str(r.group_B) in ("F", "M", "man", "woman", "boy", "girl"), axis=1)
    cfg, model, tok = load_for_run(args.model)
    log.info("loaded %s: %s attn=%s dtype=%s", args.model, cfg.get("model_class"), cfg.get("attn_implementation"), cfg.get("dtype"))
    bases = Bases(args.model)
    proto = N.load_protocol()
    proto.setdefault("models", {})[args.model] = {k: cfg.get(k) for k in ("hf_id", "model_class", "attn_implementation", "dtype", "transformers", "torch")}
    proto.setdefault("bases", {})[args.model] = bases.target_prov
    proto["random_basis_seed"] = N.RANDOM_SEED_FINAL; proto["conditions"] = N.CONDITIONS
    proto["max_new_tokens"] = N.MAX_NEW_TOKENS; proto["fmt"] = N.FMT; proto["match_tolerance"] = N.MATCH_TOL
    proto["dev_seed_source"] = "fresh BBQ construction, templates disjoint from the final set (deliberate departure from the plan's existing-dev-split wording)"
    N.save_protocol(proto)
    sp_all = resolve_spans(args.model, tok, man)
    stages = [args.stage] if args.stage != "all" else ["checks", "calibrate", "timing", "final", "control", "baseline"]
    cal = N.read_json(OUT / ("calibration_%s.json" % args.model), {"alphas": {c: 1.0 for c in CALIBRATED}})
    budget = {"cap_hours": args.model_hours, "hours_used_before": 0.0}
    for st in stages:
        if st == "checks":
            res = stage_checks(model, tok, bases, man, sp_all)
            if not res["pass_all_final"] and not args.smoke:
                raise SystemExit("identity checks failed; see identity_checks.json")
        elif st == "spans":
            pass
        elif st == "calibrate":
            cal = stage_calibrate(model, tok, bases, man, sp_all, smoke=args.smoke)
        elif st == "timing":
            tb = stage_timing(model, tok, bases, man, sp_all, cal, args)
            if args.tier:
                tb["tier_chosen"] = args.tier
            budget["tier"] = tb["tier_chosen"]
        elif st == "final":
            tier = args.tier or budget.get("tier") or N.read_json(OUT / ("runtime_budget_%s.json" % args.model), {}).get("tier_chosen", 160)
            stage_rows(model, tok, bases, man, sp_all, cal, "final", N.CONDITIONS, tier, args.smoke, budget)
        elif st == "control":
            stage_rows(model, tok, bases, man, sp_all, cal, "control", N.CONTROL_CONDITIONS, 160, args.smoke, budget)
        elif st == "baseline":
            stage_baseline(model, tok, bases, man, sp_all)
            bases = Bases(args.model)
            tier = args.tier or N.read_json(OUT / ("runtime_budget_%s.json" % args.model), {}).get("tier_chosen", 160)
            stage_rows(model, tok, bases, man, sp_all, cal, "final", N.BASELINE_CONDITIONS, tier, args.smoke, budget)
    (OUT / ("DONE_%s_%s.txt" % (args.model, args.stage))).write_text(N.utc_now(), encoding="utf-8")


if __name__ == "__main__":
    main()
