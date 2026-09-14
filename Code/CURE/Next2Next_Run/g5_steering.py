"""
g5_steering.py -- Group B of Submission2/Next_Plan.md (ICLR-feedback cycle): inference-time
activation-steering baselines next to the matched-position erasure, on the fresh replication.
GPU, one model at a time.

Steering adds a fixed vector instead of projecting a component out. Two vectors, three sites:

  DV  debiasing vector (FairSteer-style, Li et al. 2025): at every layer, the mean of the
      unedited span-position activations of fit prompts on which the unedited model's
      option-scoring argmax is the 'unknown' option (unbiased state) minus the mean on prompts
      where it is a stereotyped entity (biased state), unit-normalised. h <- h + alpha v.
  CV  contrast vector (RepE / ActAdd-style, Zou et al. 2023; Turner et al. 2023): the
      normalised mean of the span-position differences of the fit pairs, i.e. the pooled
      audit's mean-difference direction, applied with the sign that raises the unknown option
      on the development seeds. h <- h + alpha v.

  sites  span        the demographic span positions of the prompt (matched-position steering)
         last_token  the final prefill position and every generated token (the deployment
                     recipe of FairSteer / ITI)
         prefill_all every non-padding prefill position

The strength alpha is calibrated on the development seeds so that the summed squared
displacement over layers and edited positions equals the final cycle's calibrated target
(calibration_<model>.json: the random direction's alpha-one energy at the span) for the span
site, and the corresponding all-position target for prefill_all (the span erasure's G1
energy is the reference there); the last-token site uses the span alpha (one position).
Every (seed, side, condition) of the fresh final set and the control set is evaluated with
f2_runner.eval_side (T, generation accuracy, energy) and the B / DV_span bridge (C_answer).

Fit prompts for DV: the pooled audit's fit split BBQ seeds (pentad, slot c variants, chat
format), unedited option scoring by p1_intervention.option_loglik; labels from the argmax.

Outputs: g5_calibration_<model>.json, g5_steering_<model>.parquet (resumable).
Usage: python g5_steering.py --model gemma-2-2b-it [--smoke]
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

import nn_common as C
import f1_scoring as S          # noqa: E402
import f2_runner as F2          # noqa: E402
import p1_intervention as I     # noqa: E402
import p1_pilot as P1           # noqa: E402

log = C.log
ROW_KEY = ["model_name", "seed_id", "side", "cond", "variant", "kind"]
VECTORS = ("DV", "CV")
SITES = {"span": "span", "last": "last_token", "all": "prefill_all"}


class SteerEdit(I.HookedEdit):
    """h' = h + alpha * v at the target positions (additive steering); logs the same energy
    bookkeeping as HookedEdit so energy_of() applies unchanged."""

    def __init__(self, model, vec_by_layer: dict[int, np.ndarray], alpha: float, site: str):
        import torch
        super().__init__(model, {l: np.asarray(v, np.float32)[None, :] for l, v in vec_by_layer.items()}, alpha=alpha, site=site)
        dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
        self.V = {int(l): torch.as_tensor(np.asarray(v, np.float32), device=dev).to(dt) for l, v in vec_by_layer.items()}

    def _make_hook(self, li: int):
        v = self.V[li]

        def hook(module, inputs, output):
            import torch
            hs = output[0] if isinstance(output, tuple) else output
            B, T, _ = hs.shape
            prefill = T > 1
            if not prefill:
                self._step += 1
            for b in range(B):
                pos = self._positions_for(b, T, prefill)
                if not pos:
                    continue
                if not prefill:
                    self.n_decode_edits += 1
                cur = hs[b, pos, :]
                delta = (-self.alpha) * v.unsqueeze(0).expand(len(pos), -1)      # h' = h - delta = h + alpha v
                if self.record:
                    with torch.no_grad():
                        dn = delta.float().norm(dim=-1); hn = cur.float().norm(dim=-1)
                        for p, d_, h_ in zip(pos, dn.tolist(), hn.tolist()):
                            self.changed_log.append({"layer": li, "row": b, "position": int(p), "prefill": prefill,
                                                     "delta_norm": d_, "h_norm": h_})
                hs[b, pos, :] = cur - delta
            return output
        return hook


# ---------------------------------------------------------------------------
# vectors
# ---------------------------------------------------------------------------

UNKNOWN_PATTERNS = ("unknown", "can't answer", "cannot answer", "undetermined", "can't be determined", "cannot be determined",
                    "not enough info", "not answerable", "not known", "not enough information", "unclear")


def is_unknown_gold(gold: str) -> bool:
    g = gold.strip().lower()
    return any(g == u or g.startswith(u) for u in UNKNOWN_PATTERNS)


def fit_prompts_bbq(mname: str) -> list[dict]:
    """Ambiguous BBQ prompts of the pooled fit split (slot-c variants) with their span."""
    man = pd.read_csv(C.K.OUT_P0 / "split_manifest.csv")
    fit = set(man[(man.split == "fit") & (man.seed_source == "bbq")].seed_id)
    pen = pd.read_parquet(C.K.PENTAD_CLEAN); c = pen[(pen.slot == "c") & (pen.seed_id.isin(fit))]
    out = []
    for r in c.itertuples(index=False):
        gold = str(getattr(r, "gold_answer", "")).strip()
        if not is_unknown_gold(gold):                     # ambiguous items only: the unbiased answer is 'unknown'
            continue
        opts = P1.options_of(str(r.prompt_text))
        gi = P1.gold_index(str(r.prompt_text), gold)
        if gi is None or not opts:
            continue
        out.append({"seed_id": r.seed_id, "subvariant": r.subvariant, "prompt": str(r.prompt_text), "swap": str(r.swap_token), "gold_index": gi,
                    "options": [t for _, t in opts]})
    return out


def debias_vector(model, tok, items: list[dict], max_items: int) -> tuple[dict, dict]:
    """DV: mean span activation on unbiased-state prompts minus biased-state prompts."""
    sysm = P1.system_prompt()
    sums = {"unb": {}, "bias": {}}; cnt = {"unb": 0, "bias": 0}; n_used = 0
    for it in items[:max_items]:
        text = I.build_input(tok, it["prompt"], "chat", sysm)
        pos = I.resolve_span_positions(tok, text, it["swap"])
        if not pos:
            continue
        opts = ["(%s) %s" % (l, t) for l, t in P1.options_of(it["prompt"])]
        r = I.option_loglik(model, tok, text, opts, None, None)
        state = "unb" if int(r["argmax"]) == int(it["gold_index"]) else "bias"     # gold of an ambiguous item is 'unknown'
        _, hs = I.forward_hidden(model, tok, text, None, None)
        for l in hs:
            v = hs[l][pos, :].mean(0).numpy()
            sums[state][l] = sums[state].get(l, 0.0) + v
        cnt[state] += 1; n_used += 1
    if min(cnt.values()) == 0:
        raise RuntimeError("debias vector: one state has no prompts (%s)" % cnt)
    vec = {}
    for l in sums["unb"]:
        d = sums["unb"][l] / cnt["unb"] - sums["bias"][l] / cnt["bias"]
        vec[int(l)] = (d / np.linalg.norm(d)).astype(np.float32)
    return vec, {"n_prompts_used": n_used, "n_unbiased_state": cnt["unb"], "n_biased_state": cnt["bias"],
                 "rule": "state = option-scoring argmax equals the unknown gold (unbiased) or not (biased); "
                         "vector = mean(unbiased) - mean(biased) of span-position activations, unit norm per layer"}


def contrast_vector(mname: str) -> tuple[dict, dict]:
    p = C.V2 / ("p2_basis_%s_chat_mean_difference.npz" % mname)
    z = np.load(p)
    vec = {int(k): np.asarray(z[k])[0].astype(np.float32) for k in z.files}
    return vec, {"file": C.K.rel(p), "sha256": C.N.sha256_file(p), "rule": "pooled-audit mean-difference direction (unit norm), sign fixed on the dev seeds"}


def orient_contrast(model, tok, vec: dict, alpha: float, dev_items: list[tuple[str, list[int], int, list[str]]]) -> tuple[dict, dict]:
    """Choose the sign of CV that raises the unknown option's margin on the development seeds
    at the calibrated span strength (the energy is sign-invariant, so calibration precedes)."""
    gains = []
    for text, pos, gold, cands in dev_items:
        base = S.score_candidates(model, tok, text, cands, None, [], None)["s"]
        plus = S.score_candidates(model, tok, text, cands, SteerEdit(model, vec, alpha, "span"), pos, None)["s"]
        gains.append(S.margin(plus, gold) - S.margin(base, gold))
    sign = 1.0 if float(np.mean(gains)) >= 0 else -1.0
    return {l: sign * v for l, v in vec.items()}, {"sign": sign, "mean_margin_gain_at_alpha": float(np.mean(gains)), "alpha_used": alpha, "n_dev": len(gains)}


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def dev_items(man: pd.DataFrame, sp_all: dict, smoke: bool) -> list[tuple[str, list[int], int, list[str], int]]:
    out = []
    dev = man[man.group == "dev"]
    if smoke:
        dev = dev.head(3)
    for r in dev.itertuples(index=False):
        for side in ("A", "B"):
            s = sp_all["seeds"][r.seed_id][side]
            if s["structural_failure"]:
                continue
            out.append((s["text"], s["demo"], int(r.gold_index), S.candidates(json.loads(getattr(r, "options_%s" % side)), "text"), s["n_prefill"]))
    return out


def energy_at(model, tok, vec: dict, alpha: float, site: str, items, D: float) -> float:
    tot = []
    for text, pos, gold, cands, n_prefill in items:
        e = SteerEdit(model, vec, alpha, site)
        S.score_candidates(model, tok, text, [S.PREFIX + "x" + S.SUFFIX], e, (list(range(n_prefill)) if site == "prefill_all" else pos), None)
        tot.append(F2.energy_of(e)["num"])
    return float(np.mean(tot)) / D


def calibrate(model, tok, vec: dict, site: str, items, D: float, target: float) -> dict:
    """alpha with mean energy = target; energy is exactly quadratic in alpha for an additive
    vector (delta = alpha v at every edited position), so one evaluation fixes it and a
    second verifies it."""
    e1 = energy_at(model, tok, vec, 1.0, site, items, D)
    a = math.sqrt(target / e1) if e1 > 0 else 1.0
    e_a = energy_at(model, tok, vec, a, site, items, D)
    return {"alpha": a, "energy_alpha1": e1, "achieved": e_a, "rel_err": (e_a - target) / target if target else float("nan"),
            "within_tol": abs(e_a - target) / target <= C.MATCH_TOL if target else False}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(mname: str, smoke: bool) -> None:
    C.ensure_out()
    cal_path = C.OUT / ("g5_calibration_%s.json" % mname)
    path = C.OUT / ("g5_steering_%s.parquet" % mname)
    man = F2.manifest()
    sp_all = C.read_json(C.FINAL / ("spans_%s.json" % mname))
    cal_final = C.read_json(C.FINAL / ("calibration_%s.json" % mname))
    if sp_all is None or cal_final is None:
        raise SystemExit("final-cycle spans/calibration missing for %s" % mname)
    cfg, model, tok = C.load_model(mname)
    bases = F2.Bases(mname)
    rec = C.read_json(cal_path, None)
    if rec is None:
        items = dev_items(man, sp_all, smoke)
        D = float(cal_final["denominator"]); target_span = float(cal_final["target"])
        # all-position reference: the span basis at every prefill position (G1) on the dev seeds
        g1_energy = []
        for text, pos, gold, cands, n_prefill in items:
            e = I.HookedEdit(model, bases.target, alpha=1.0, site="prefill_all")
            S.score_candidates(model, tok, text, [S.PREFIX + "x" + S.SUFFIX], e, list(range(n_prefill)), None)
            g1_energy.append(F2.energy_of(e)["num"])
        target_all = float(np.mean(g1_energy)) / D
        dv, dv_info = debias_vector(model, tok, fit_prompts_bbq(mname), max_items=(12 if smoke else 400))
        cv0, cv_info = contrast_vector(mname)
        rec = {"model": mname, "denominator": D, "target_span": target_span, "target_all": target_all,
               "target_all_rule": "mean dev energy of the span basis applied at every prefill position at alpha 1 (G1)",
               "vectors": {"DV": dv_info, "CV": cv_info}, "alphas": {}, "utc": C.utc_now()}
        for vname, vec in (("DV", dv), ("CV", cv0)):
            rec["alphas"][vname] = {"span": calibrate(model, tok, vec, "span", items, D, target_span),
                                    "all": calibrate(model, tok, vec, "prefill_all", items, D, target_all)}
            rec["alphas"][vname]["last"] = {"alpha": rec["alphas"][vname]["span"]["alpha"], "rule": "span alpha (one position)"}
        cv, sign_info = orient_contrast(model, tok, cv0, float(rec["alphas"]["CV"]["span"]["alpha"]), [(t, p, g, c) for t, p, g, c, _ in items])
        rec["vectors"]["CV"].update(sign_info)
        np.savez(C.OUT / ("g5_vectors_%s.npz" % mname), **{"DV_%d" % l: v for l, v in dv.items()}, **{"CV_%d" % l: v for l, v in cv.items()})
        C.write_json(rec, cal_path)
    z = np.load(C.OUT / ("g5_vectors_%s.npz" % mname))
    vecs = {"DV": {int(k[3:]): z[k] for k in z.files if k.startswith("DV_")}, "CV": {int(k[3:]): z[k] for k in z.files if k.startswith("CV_")}}
    conds = [("%s_%s" % (v, s), v, s) for v in VECTORS for s in ("span", "last", "all")]
    done = C.done_keys(path, ROW_KEY)
    t0 = time.time(); n = 0
    for group in ("final", "control"):
        sub = man[man.group == group]
        if smoke:
            sub = sub.head(2)
        for r in sub.itertuples(index=False):
            sp = sp_all["seeds"][r.seed_id]
            rows = []
            for cname, vname, site_key in conds:
                if group == "control" and site_key != "span":
                    continue
                alpha = float(rec["alphas"][vname][site_key]["alpha"]); site = SITES[site_key]
                for side in ("A", "B"):
                    if (mname, r.seed_id, side, cname, "std", "score") in done:
                        continue
                    if sp[side]["structural_failure"] and not sp[side]["demo"]:
                        continue
                    row = eval_steer(model, tok, mname, vecs[vname], alpha, site, r, side, sp[side], cname)
                    rows.append(row)
                if group == "final" and site_key == "span" and (mname, r.seed_id, "B", cname, "std", "bridge") not in done:
                    rows.append(bridge_steer(model, tok, mname, vecs[vname], alpha, r, sp, cname))
            if rows:
                C.append_rows(path, rows); n += len(rows)
    log.info("g5 %s: %d new rows in %.1f min", mname, n, (time.time() - t0) / 60)
    (C.OUT / ("DONE_g5_%s.txt" % mname)).write_text(C.utc_now(), encoding="utf-8")


def eval_steer(model, tok, mname: str, vec: dict, alpha: float, site: str, r, side: str, sp: dict, cname: str) -> dict:
    """eval_side of the final cycle with a steering edit instead of a projection."""
    t0 = time.time()
    up = getattr(r, "prompt_%s" % side); options = json.loads(getattr(r, "options_%s" % side)); gold = int(r.gold_index)
    text = sp["text"]; cands = S.candidates(options, "text"); first_ids = S.first_token_ids(tok, text, options)
    pos = list(range(sp["n_prefill"])) if site == "prefill_all" else (sp["demo"] if site == "span" else [sp["n_prefill"] - 1])
    row = {"model_name": mname, "seed_id": r.seed_id, "group": r.group, "side": side, "cond": cname, "variant": "std", "kind": "score",
           "category": r.category, "template": r.template, "site": site, "alpha": alpha, "basis_id": cname.split("_")[0] + "_vector",
           "rank": 1, "positions": json.dumps(pos), "n_edited_positions": len(pos), "gold_index": gold, "options": json.dumps(options), "status": "ok"}
    try:
        e = SteerEdit(model, vec, alpha, site)
        sc = S.score_candidates(model, tok, text, cands, e, pos, first_ids)
        q = S.softmax(sc["s"]); en = F2.energy_of(e)
        row.update({"s": json.dumps(sc["s"]), "s_mean": json.dumps(sc["s_mean"]), "n_cand_tokens": json.dumps(sc["n_tokens"]), "q": json.dumps(q),
                    "argmax": int(np.argmax(sc["s"])), "score_correct": bool(int(np.argmax(sc["s"])) == gold), "M": S.margin(sc["s"], gold),
                    "first_logits": json.dumps(sc["first_logits"]), "first_disambiguates": S.first_token_disambiguates(first_ids, gold), "T0": sc["T0"],
                    "energy_num": en["num"], "energy_den_edited": en["den_edited"], "energy_by_layer": json.dumps(en["by_layer"]),
                    "n_decode_edits_score": en["n_decode_edits"]})
        e2 = SteerEdit(model, vec, alpha, site)
        gen = I.generate_under_edit(model, tok, [text], e2, [pos], max_new_tokens=C.N.MAX_NEW_TOKENS)[0]
        gi, why = S.parse_generation(gen, up)
        row.update({"gen_raw": gen, "gen_idx": (gi if gi is not None else -1), "gen_parse": why, "gen_valid": gi is not None,
                    "gen_correct": (gi == gold) if gi is not None else False, "n_decode_edits_gen": e2.n_decode_edits})
    except Exception as ex:
        row.update({"status": "error", "error": str(ex)[:300]}); log.exception("g5 %s %s %s", r.seed_id, side, cname)
    row["sec"] = time.time() - t0
    return row


def bridge_steer(model, tok, mname: str, vec: dict, alpha: float, r, sp: dict, cname: str) -> dict:
    t0 = time.time()
    options = json.loads(r.options_B); gold = int(r.gold_index)
    pa, pb = sp["A"]["demo"], sp["B"]["demo"]
    cands = S.candidates(options, "text"); first_ids = S.first_token_ids(tok, sp["B"]["text"], options)
    factory = (lambda: SteerEdit(model, vec, alpha, "span"))
    row = {"model_name": mname, "seed_id": r.seed_id, "group": r.group, "side": "B", "cond": cname, "variant": "std", "kind": "bridge",
           "category": r.category, "template": r.template, "site": "span", "alpha": alpha, "basis_id": cname.split("_")[0] + "_vector",
           "rank": 1, "gold_index": gold, "status": "ok"}
    try:
        clean = S.patched_scores(model, tok, sp["B"]["text"], sp["B"]["text"], pb, pb, cands, factory, first_ids)
        pat = S.patched_scores(model, tok, sp["A"]["text"], sp["B"]["text"], pa, pb, cands, factory, first_ids)
        if "error" in pat or "error" in clean:
            row.update({"status": "error", "error": pat.get("error") or clean.get("error")})
        else:
            Mc, Mp = S.margin(clean["s"], gold), S.margin(pat["s"], gold)
            row.update({"M_clean": Mc, "M_patched": Mp, "C_answer": Mp - Mc, "C_first": pat["first_logits"][gold] - clean["first_logits"][gold],
                        "patched_len": pat["patched_len"], "span_len_a": pat["span_len_a"], "span_len_b": pat["span_len_b"]})
    except Exception as ex:
        row.update({"status": "error", "error": str(ex)[:300]}); log.exception("g5 bridge %s %s", r.seed_id, cname)
    row["sec"] = time.time() - t0
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=C.FRESH_MODELS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.model, a.smoke)


if __name__ == "__main__":
    main()
