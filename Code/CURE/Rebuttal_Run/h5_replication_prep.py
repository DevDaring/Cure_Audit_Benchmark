"""
h5_replication_prep.py -- what Qwen2.5-7B and Phi-4-mini need before the fresh items can run on them
(rebuttal cycle; answers 'the behavioural benefit rests on one of two models': JwYi, dfbY W2, uG1a W1).
The pooled audit fitted their bases; the fresh items were never scored on them, so their results
are out-of-sample. GPU.

  1. resolve the demographic, control and prefill spans of every fresh / control / dev seed, under
     the final cycle's frozen rule
  2. the contrast steering vector, under the feedback cycle's rule (g5_steering): the pooled audit's
     mean-difference direction, with the sign that raises the unknown option on the development seeds
     at 5 per cent of the span erasure's energy (about the energy the calibration of the other two
     models used)
  3. identity checks: alpha 0 and rank 0 no-ops, non-span positions untouched at the first edited layer,
     no decode-step edit, the same-input patch null under a partial edit

Outputs: h5_spans_<model>.json, h5_vectors_<model>.npz, h5_vector_<model>.json, h5_identity_<model>.json.
Usage: python h5_replication_prep.py --model qwen2.5-7b-instruct [--smoke]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import hh_common as H

log = H.log


def resolve_spans(mname: str, tok) -> dict:
    import f1_scoring as S
    p = H.OUT / ("h5_spans_%s.json" % mname)
    if p.exists():
        return H.read_json(p)
    man = H.fresh_manifest()
    out = {"model": mname, "rule": "as final_20260914/spans_*.json: demographic = every whole-word occurrence of the entity-1 descriptor in the user prompt; "
                                  "matched = contiguous context tokens of equal count nearest the context occurrence, no overlap with either descriptor, "
                                  "no punctuation-only token; prefill_all = every non-padding prefill position", "seeds": {}}
    for r in man.itertuples(index=False):
        rec = {}
        for side in ("A", "B"):
            up = getattr(r, "prompt_%s" % side); desc = getattr(r, "descriptor_%s" % side)
            text = S.chat_text(tok, up)
            demo = S.demographic_span(tok, text, up, desc)
            ent2 = S.demographic_span(tok, text, up, r.entity2_descriptor)
            ref = demo["occurrences"][0]["positions"][0] if demo["occurrences"] else 0
            ne = S.matched_control_span(tok, text, up, len(demo["positions"]), demo["positions"] + ent2["positions"], ref,
                                        seed=H.N.RANDOM_SEED_FINAL + int(H.N.sha256_text(r.seed_id + side)[:6], 16))
            rec[side] = {"text": text, "demo": demo["positions"], "n_demo_occurrences": demo["n_occurrences"], "ne": ne["positions"], "ne_reason": ne["reason"],
                         "ne_distance": ne.get("distance_tokens"), "ne_text": ne.get("text"), "n_prefill": len(tok(text, add_special_tokens=True)["input_ids"]),
                         "structural_failure": (not demo["positions"]) or ne["reason"] != "ok"}
        out["seeds"][r.seed_id] = rec
    H.write_json(out, p)
    return out


def steering_vector(mname: str, model, tok, sp_all: dict, smoke: bool) -> None:
    import g5_steering as G5
    if (H.OUT / ("h5_vectors_%s.npz" % mname)).exists():
        return
    man = H.fresh_manifest()
    target, _ = H.target_basis(mname)
    items = G5.dev_items(man, sp_all, smoke)
    cv0, info = G5.contrast_vector(mname)
    alphas = []
    for text, pos, gold, cands, n_prefill in items:
        E = H.measure_energy(model, tok, text, H.make_factory(model, "proj", target, 1.0, "span"), pos)
        e1 = H.measure_energy(model, tok, text, H.make_factory(model, "steer", cv0, 1.0, "span"), pos)
        alphas.append(float(np.sqrt(0.05 * E / e1)) if e1 > 0 else 0.0)
    alpha = float(np.mean(alphas))
    cv, sign = G5.orient_contrast(model, tok, cv0, alpha, [(t, p, g, c) for t, p, g, c, _ in items])
    np.savez(H.OUT / ("h5_vectors_%s.npz" % mname), **{"CV_%d" % l: v for l, v in cv.items()})
    H.write_json({"model": mname, **info, **sign, "rule_energy": "alpha at 5 per cent of the span erasure's energy on the dev seeds", "utc": H.utc_now()}, H.OUT / ("h5_vector_%s.json" % mname))
    log.info("h5 %s: steering vector oriented (sign %+.0f, mean margin gain %.3f)", mname, sign["sign"], sign["mean_margin_gain_at_alpha"])


def identity(mname: str, model, tok, sp_all: dict) -> dict:
    import f1_scoring as S
    import p1_intervention as I
    man = H.fresh_manifest()
    dev = man[man.group == "dev"].head(4)
    target, _ = H.target_basis(mname)
    texts = [sp_all["seeds"][r.seed_id]["A"]["text"] for r in dev.itertuples(index=False)]
    pos = [sp_all["seeds"][r.seed_id]["A"]["demo"] for r in dev.itertuples(index=False)]
    res = I.identity_checks(model, tok, texts, target, pos)
    e = H.make_factory(model, "proj", target, 1.0, "all")()
    _ = I.generate_under_edit(model, tok, texts[:1], e, [list(range(len(tok(texts[0])["input_ids"])))], max_new_tokens=4)
    res["prefill_all_no_decode_edit"] = e.n_decode_edits == 0
    res["prefill_all_touches_every_prefill_position"] = len({r["position"] for r in e.changed_log if r["layer"] == min(target)}) == len(tok(texts[0])["input_ids"])
    r0 = dev.iloc[0]; s0 = sp_all["seeds"][r0.seed_id]["A"]
    cands = S.candidates(json.loads(r0.options_A), "text"); fids = S.first_token_ids(tok, s0["text"], json.loads(r0.options_A))
    fac = H.make_factory(model, "proj", target, 0.37, "span")
    ref = H.score_under_patch(model, tok, s0["text"], s0["text"], s0["demo"], s0["demo"], cands, fac, fac, s0["demo"], s0["demo"], fids)
    pat = H.score_under_patch(model, tok, s0["text"], s0["text"], s0["demo"], s0["demo"], cands, fac, fac, s0["demo"], s0["demo"], fids)
    res["max_abs_dev_same_input_patch_under_partial_edit"] = float(np.max(np.abs(np.asarray(pat["s"]) - np.asarray(ref["s"]))))
    res["pass_same_input_patch_null"] = res["max_abs_dev_same_input_patch_under_partial_edit"] <= 1e-6
    res["candidate_boundary"] = S.check_boundary(tok, s0["text"], cands)
    res["pass_all_h5"] = bool(res["pass_all"] and res["prefill_all_no_decode_edit"] and res["prefill_all_touches_every_prefill_position"]
                              and res["pass_same_input_patch_null"] and res["candidate_boundary"]["n_boundary_merges"] == 0)
    res["model"] = mname; res["utc"] = H.utc_now()
    H.write_json(res, H.OUT / ("h5_identity_%s.json" % mname))
    log.info("h5 identity %s: pass_all_h5=%s", mname, res["pass_all_h5"])
    return res


def run(mname: str, smoke: bool) -> None:
    H.ensure_out()
    cfg, model, tok = H.load_model(mname)
    H.write_json({"model": mname, "model_cfg": {k: cfg.get(k) for k in ("hf_id", "model_class", "attn_implementation", "dtype")}, "utc": H.utc_now()}, H.OUT / ("h5_model_%s.json" % mname))
    sp_all = resolve_spans(mname, tok)
    res = identity(mname, model, tok, sp_all)
    if not res["pass_all_h5"]:
        raise SystemExit("identity checks failed for %s; see h5_identity_%s.json" % (mname, mname))
    steering_vector(mname, model, tok, sp_all, smoke)
    (H.OUT / ("DONE_h5_%s.txt" % mname)).write_text(H.utc_now(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=H.REPLICATION_MODELS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.model, a.smoke)


if __name__ == "__main__":
    main()
