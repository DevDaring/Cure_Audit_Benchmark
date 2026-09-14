"""
g3_leace_conditioned.py -- A3 of Submission2/Next_Plan.md (ICLR-feedback cycle): does the
LEACE null depend on a rank-deficient covariance? GPU, one model at a time.

Two parts.

Part A (every model; feeds the pooled-audit re-read of g1 --extra-leace)
  The pooled audit fitted the swap-side LEACE on 300 fit pairs (about 660 span rows at
  d = 2304..4096). This part refits the SAME swap-side eraser at three sample sizes on the
  same fit split: 'swap_small' (50 pairs), 'swap_300' (the shipped size, re-drawn with this
  cycle's seed as a replication of the fit), 'swap_max' (every eligible fit-split pair). All
  three use the package's shrinkage (as the shipped fit did); 'swap_max_noshrink' is added
  without it. Each eraser records the number of rows, the covariance's effective rank at the
  middle layer and the LEACE fitted rank. g1 --extra-leace then reads the patching effect
  under each on the 160 pooled test seeds.

Part B (the two fresh-replication models)
  A coherent, well-conditioned gender eraser: label = referent gender (F vs M) at every
  descriptor token (names, man/woman, boy/girl, male/female) of every BBQ Gender_identity
  item whose two entities are one F and one M, excluding every context used by the pooled
  audit or the fresh replication. Items are split by template into a FIT pool (2/3) and an
  EVAL pool (1/3). 'gender_large' is fitted on every fit-pool row (thousands of rows, N of the
  order of d), 'gender_small' on 61 rows drawn from the same pool (the F3 size). Held-out
  probe accuracy before/after, residual covariance and effective ranks are recorded, and the
  cosine between each eraser's erased direction and the contrast-derived basis, per layer.
  A fresh F<->M swap evaluation set (ambiguous items, gold = unknown, one entity's name or
  descriptor swapped for the other gender's, built with the final cycle's pair rule) is drawn
  from the EVAL templates, two per template, and every side is evaluated under
    B, S1 (span erasure), MC (F3 mean difference), LC_f3 (F3 LEACE, 61 rows),
    LC_gender_small, LC_gender_large,
  with the complete-answer readouts of f2_runner.eval_side and the C_answer bridge.

Outputs: g3_leace_<model>_<tag>.npz, g3_leace_<model>.json, g3_gender_manifest_<model>.csv,
g3_rows_<model>.parquet (resumable). Usage: python g3_leace_conditioned.py --model M [--smoke]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import time

import numpy as np
import pandas as pd

import nn_common as C
import f1_data as D             # noqa: E402
import f1_scoring as S          # noqa: E402
import f2_runner as F2          # noqa: E402
import p1_intervention as I     # noqa: E402
import p1_pilot as P1           # noqa: E402
import p2_controlled_erasure as P2   # noqa: E402
import p2_leace_faithful as LF  # noqa: E402

log = C.log
ROW_KEY = ["model_name", "seed_id", "side", "cond", "variant", "kind"]
SMALL_ROWS = 61                 # the F3 fit size
FM_WORDS = {"F": ("woman", "girl", "female", "she"), "M": ("man", "boy", "male", "he")}


# ---------------------------------------------------------------------------
# shared: fit + describe an eraser
# ---------------------------------------------------------------------------

def effective_rank(X: np.ndarray, rel_tol: float = 1e-6) -> int:
    """Rank of the centred covariance: eigenvalues above rel_tol x the largest (d x d eigh,
    cheap even for N in the tens of thousands)."""
    Xc = (X - X.mean(0, keepdims=True)).astype(np.float64)
    cov = Xc.T @ Xc / max(1, len(Xc) - 1)
    w = np.linalg.eigvalsh(cov)
    return int((w > rel_tol * w.max()).sum()) if w.size else 0


def describe_fit(tag: str, erasers: dict, X: dict, z: np.ndarray, tr: np.ndarray, te: np.ndarray, layers_probe: list[int],
                 target: dict | None) -> dict:
    import f3_baseline as F3
    rec = {"tag": tag, "n_rows_fit": int(tr.sum()), "n_rows_heldout": int(te.sum()), "d": int(next(iter(X.values())).shape[1]),
           "class_counts_fit": {"0": int((z[tr] == 0).sum()), "1": int((z[tr] == 1).sum())}, "per_layer": []}
    for l in layers_probe:
        e = erasers[l]
        Xl_tr, Xl_te = F3.apply_leace_np(X[l][tr], e), F3.apply_leace_np(X[l][te], e)
        row = {"layer": int(l), "leace_rank": int(e["actual_rank"]), "cov_effective_rank_fit": effective_rank(X[l][tr]),
               "probe_pre": F3.probe_accuracy(X[l][tr], z[tr], X[l][te], z[te], C.RANDOM_SEED) if te.sum() else None,
               "probe_post": F3.probe_accuracy(Xl_tr, z[tr], Xl_te, z[te], C.RANDOM_SEED) if te.sum() else None,
               "max_abs_cov_post_fit": float(np.abs((Xl_tr - Xl_tr.mean(0)).T @ (z[tr] - z[tr].mean()) / len(z[tr])).max())}
        if target is not None and l in target:
            u = LF.erased_subspace_rows(e["proj_left"])
            v = np.asarray(target[l][0], np.float64); v = v / np.linalg.norm(v)
            row["abs_cos_to_contrast_basis"] = float(np.abs(u.astype(np.float64) @ v).max()) if len(u) else 0.0
        rec["per_layer"].append(row)
    return rec


def save_erasers(mname: str, tag: str, erasers: dict) -> None:
    np.savez(C.OUT / ("g3_leace_%s_%s.npz" % (mname, tag)), **LF.erasers_to_npz_dict(erasers))


# ---------------------------------------------------------------------------
# Part A: swap-side LEACE at three sample sizes (pooled audit fit split)
# ---------------------------------------------------------------------------

def part_a(mname: str, model, tok, rec: dict, smoke: bool) -> None:
    c, _ = P1._pentad_c()
    all_pairs = P1.fit_pairs(mname, 10 ** 6, C.RANDOM_SEED)
    if smoke:
        all_pairs = all_pairs[:12]
    diffs, acts, labels, info = P2.collect_fit_activations(model, tok, "chat", all_pairs, c)
    rec["part_a"] = {"n_eligible_fit_pairs": len(all_pairs), "collect_info": info}
    layers = sorted(acts); mid = layers[len(layers) // 2]
    dev = str(next(model.parameters()).device)
    # rows come in (A, B) prompt order; a pair contributes 2 * n_span rows. One fifth of the
    # pairs is held out for the probes of every fit; the fits draw from the remaining pairs.
    items = LF.resolve_fit_items(tok, "chat", all_pairs, c, P1.system_prompt())
    pair_of_row = np.concatenate([[k // 2] * len(it["pos"]) for k, it in enumerate(items)]) if items else np.array([], int)
    n_pairs_used = len(items) // 2
    rng = random.Random(C.RANDOM_SEED)
    order = list(range(n_pairs_used)); rng.shuffle(order)
    n_held = max(1, n_pairs_used // 5)
    held, pool = order[:n_held], order[n_held:]
    te = np.isin(pair_of_row, held)
    sizes = {"swap_small": min(50, len(pool)), "swap_300": min(300, len(pool)), "swap_max": len(pool)}
    if smoke:
        sizes = {k: min(v, 4) for k, v in sizes.items()}
    rec["part_a"].update({"n_pairs_used": n_pairs_used, "n_pairs_heldout": n_held, "sizes": sizes})
    target, _ = F2.load_target_basis(mname)
    for tag, n in sizes.items():
        tr = np.isin(pair_of_row, pool[:n])
        for shrink in ((True, False) if tag == "swap_max" else (True,)):
            t = tag if shrink else tag + "_noshrink"
            t0 = time.time()
            er = LF.fit_leace_erasers({l: acts[l][tr] for l in layers}, labels[tr], shrinkage=shrink, device=dev)
            d = describe_fit(t, er, acts, labels, tr, te, [layers[len(layers) // 4], mid, layers[3 * len(layers) // 4]], target)
            d.update({"n_pairs": int(n), "shrinkage": shrink, "sec_fit": time.time() - t0, "label": "0 = side A, 1 = side B of the swap"})
            rec["part_a"][t] = d
            save_erasers(mname, t, er)
            log.info("g3 A %s %s: %d pairs, %d rows, mid-layer cov rank %d, probe pre/post %s/%s", mname, t, n, int(tr.sum()),
                     d["per_layer"][1]["cov_effective_rank_fit"], d["per_layer"][1]["probe_pre"], d["per_layer"][1]["probe_post"])


# ---------------------------------------------------------------------------
# Part B: coherent gender eraser and a fresh F<->M evaluation set
# ---------------------------------------------------------------------------

def gender_pool() -> tuple[pd.DataFrame, dict]:
    bbq = pd.read_parquet(C.N.BBQ_RAW)
    g = bbq[bbq.category == "Gender_identity"]
    _, touched_ctx, notes = D.touched_templates(bbq)
    man = pd.read_csv(C.FINAL / "source_manifest.csv")
    fresh_ctx = {D.norm(x) for x in pd.concat([man.context_A, man.context_B])}
    rows = []
    for r in g.itertuples(index=False):
        e = D.entities(r)
        if len(e["ent"]) != 2 or {gg for _, _, gg in e["ent"]} != {"F", "M"}:
            continue
        if D.norm(r.context) in touched_ctx or D.norm(r.context) in fresh_ctx:
            continue
        rows.append(r)
    df = pd.DataFrame(rows)
    return df, {"n_items": len(df), "n_templates": int(df.question_index.nunique()) if len(df) else 0, **notes}


def split_templates(df: pd.DataFrame) -> tuple[set, set]:
    ts = sorted(df.question_index.astype(str).unique())
    rng = random.Random(C.RANDOM_SEED); rng.shuffle(ts)
    n_fit = len(ts) // 2                      # the fit pool is capped far below its size; more eval templates = more clusters
    return set(ts[:n_fit]), set(ts[n_fit:])


def collect_gender(model, tok, df: pd.DataFrame, max_items: int) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """Descriptor-token activations (chat format, unedited) labelled by referent gender."""
    X, z, tpl = {}, [], []
    n_prompts = 0; n_rows = 0
    for r in df.itertuples(index=False):
        if n_prompts >= max_items:
            break
        e = D.entities(r)
        up = D.prompt_text(r.context, r.question, [r.ans0, r.ans1, r.ans2])
        text = S.chat_text(tok, up)
        spans = []
        for _, desc, grp in e["ent"]:
            pos = S.demographic_span(tok, text, up, desc)["positions"]
            if pos:
                spans.append((pos, 0 if grp == "M" else 1))
        if not spans:
            continue
        _, hs = I.forward_hidden(model, tok, text, None, None)
        for pos, lab in spans:
            for l in hs:
                X.setdefault(l, []).append(hs[l][pos, :].numpy())
            z += [lab] * len(pos); tpl += [str(r.question_index)] * len(pos); n_rows += len(pos)
        n_prompts += 1
    X = {l: np.concatenate(v, 0).astype(np.float32) for l, v in X.items()}
    return X, np.asarray(z), np.asarray(tpl), {"n_prompts": n_prompts, "n_rows": n_rows}


def _sub_word(text: str, old: str, new: str) -> str:
    def rep(m):
        w = m.group(0)
        return new[0].upper() + new[1:] if w[0].isupper() and not new[0].isupper() else new
    return re.sub(r"(?<![A-Za-z0-9])" + re.escape(old) + r"(?![A-Za-z0-9])", rep, text, flags=re.I)


def build_fm_eval_set(df_eval: pd.DataFrame, n_per_template: int, max_seeds: int) -> tuple[pd.DataFrame, dict]:
    """F<->M swap pairs (gold = unknown) on the evaluation templates. Every BBQ Gender_identity
    item pairs one female with one male entity, so no sibling item flips the gender of one
    entity while keeping the other (the final cycle's pair rule yields M->M and F->F swaps only).
    The variant is therefore built the way the pooled audit built its subvariants: the swapped
    entity's descriptor is replaced, in the context and in its option, by an opposite-gender
    descriptor drawn from the same template's own items (a name from the template's name list,
    or the man/woman-type word); everything else is fixed, so the diff is one span."""
    amb = df_eval[df_eval.context_condition == "ambig"]
    names = {}
    for r in amb.itertuples(index=False):
        for _, d, g in D.entities(r)["ent"]:
            names.setdefault((str(r.question_index), g), set()).add(d)
    rng = random.Random(C.RANDOM_SEED)
    cands, excl, seen = [], collections.Counter(), set()
    for r in amb.itertuples(index=False):
        e = D.entities(r)
        if len(e["ent"]) != 2 or e["unknown"] is None or int(r.label) != e["unknown"]:
            excl["shape_or_gold_not_unknown"] += 1; continue
        opts = [r.ans0, r.ans1, r.ans2]
        if len({D.norm(o) for o in opts}) != 3:
            excl["degenerate_options"] += 1; continue
        for i1, d1, g1 in e["ent"]:
            i2, d2, g2 = next(x for x in e["ent"] if x[0] != i1)
            other = "F" if g1 == "M" else "M"
            pool = sorted(x for x in names.get((str(r.question_index), other), set())
                          if D.norm(x) not in (D.norm(d1), D.norm(d2)) and D.whole_word_count(r.context, x) == 0)
            if not pool:
                excl["no_opposite_gender_descriptor_in_template"] += 1; continue
            if D.whole_word_count(r.context, d1) != 1 or D.whole_word_count(opts[i1], d1) < 1 or D.whole_word_count(r.context, d2) < 1:
                excl["descriptor_not_exactly_once"] += 1; continue
            d_new = pool[rng.randrange(len(pool))]
            ctx_b = _sub_word(r.context, d1, d_new)
            opts_b = list(opts); opts_b[i1] = _sub_word(opts[i1], d1, d_new)
            ops = D.word_diff(r.context, ctx_b)
            if len(ops) != 1 or ops[0][0] != "replace" or len({D.norm(o) for o in opts_b}) != 3:
                excl["diff_not_single_replacement"] += 1; continue
            key = (str(r.question_index), D.norm(d1), D.norm(d_new), D.norm(d2))
            if key in seen:
                excl["duplicate_descriptor_triple_in_template"] += 1; continue
            seen.add(key)
            sem = ["", "", ""]; sem[i1] = "ent1"; sem[e["unknown"]] = "unknown"; sem[i2] = "ent2"
            sid = "fm_" + hashlib.sha1(("bbq|%s|%d|%s|%s" % (r.category, int(r.example_id), d1, d_new)).encode()).hexdigest()[:10]
            cands.append({
                "seed_id": sid, "kind": "fm_swap", "category": r.category, "question_index": str(r.question_index),
                "question_polarity": r.question_polarity, "example_id_A": int(r.example_id), "example_id_B": -1,
                "context_A": r.context, "context_B": ctx_b, "question": r.question,
                "options_A": json.dumps(opts), "options_B": json.dumps(opts_b), "semantic_ids": json.dumps(sem),
                "descriptor_A": d1, "descriptor_B": d_new, "group_A": g1, "group_B": other, "entity2_descriptor": d2,
                "gold_A": opts[e["unknown"]], "gold_B": opts_b[e["unknown"]], "gold_index": int(e["unknown"]),
                "prompt_A": D.prompt_text(r.context, r.question, opts), "prompt_B": D.prompt_text(ctx_b, r.question, opts_b),
                "n_context_replacements": 1,
                "stereotyped_groups": json.dumps(list(dict(r.additional_metadata).get("stereotyped_groups", []))) if r.additional_metadata is not None else "[]",
                "template": "%s|%s" % (r.category, r.question_index), "group": "fm_eval", "in_tier_100": True, "in_diag_32": False,
                "f3_eligible": True})
    by_t = {}
    for x in cands:
        by_t.setdefault(x["template"], []).append(x)
    out = []
    for t in sorted(by_t):
        xs = by_t[t]; rng.shuffle(xs)
        # alternate the flipped gender so both directions are present in every template
        f_first = [x for x in xs if x["group_A"] == "F"]; m_first = [x for x in xs if x["group_A"] == "M"]
        mix = [x for pair in zip(f_first, m_first) for x in pair] + f_first[len(m_first):] + m_first[len(f_first):]
        out += mix[:n_per_template]
    rng.shuffle(out)
    out = out[:max_seeds]
    info = {"n_candidates": len(cands), "n_templates_with_candidates": len(by_t), "excluded": dict(excl),
            "construction": "descriptor substitution (opposite-gender descriptor of the same template), as the pooled audit's subvariants"}
    return pd.DataFrame(out), info


def spans_for(tok, man: pd.DataFrame) -> dict:
    out = {"seeds": {}}
    for r in man.itertuples(index=False):
        rec = {}
        for side in ("A", "B"):
            up = getattr(r, "prompt_%s" % side); desc = getattr(r, "descriptor_%s" % side)
            text = S.chat_text(tok, up)
            demo = S.demographic_span(tok, text, up, desc)
            rec[side] = {"text": text, "demo": demo["positions"], "ne": [], "ne_reason": "not_used",
                         "n_prefill": len(tok(text, add_special_tokens=True)["input_ids"]), "structural_failure": not demo["positions"]}
        out["seeds"][r.seed_id] = rec
    return out


def part_b(mname: str, model, tok, rec: dict, smoke: bool) -> None:
    df, notes = gender_pool()
    fit_t, eval_t = split_templates(df)
    df_fit = df[df.question_index.astype(str).isin(fit_t)]
    df_eval = df[df.question_index.astype(str).isin(eval_t)]
    rec["part_b"] = {"pool": notes, "n_fit_templates": len(fit_t), "n_eval_templates": len(eval_t),
                     "n_fit_items": int(len(df_fit)), "n_eval_items": int(len(df_eval))}
    prev = rec.get("part_b_done") or {}
    fits, md = {}, {}
    if prev.get("fits") and prev.get("split") == sorted(fit_t) and all((C.OUT / ("g3_leace_%s_%s.npz" % (mname, t))).exists() for t in ("gender_large", "gender_small")) \
            and (C.OUT / ("g3_md_%s_gender.npz" % mname)).exists():
        for t in ("gender_large", "gender_small"):
            fits[t] = LF.erasers_from_npz(np.load(C.OUT / ("g3_leace_%s_%s.npz" % (mname, t))))
        zz = np.load(C.OUT / ("g3_md_%s_gender.npz" % mname)); md = {int(k[3:]): zz[k] for k in zz.files if k.startswith("md_")}
        rec["part_b"].update(prev["fits"])
        log.info("g3 B %s: fits reused from disk", mname)
    else:
        fits, md = _part_b_fits(mname, model, tok, rec, df_fit, smoke)
        rec["part_b_done"] = {"split": sorted(fit_t), "fits": {k: rec["part_b"][k] for k in ("collect", "gender_large", "gender_small", "md_gender_abs_cos_to_contrast_basis")}}
        C.write_json(rec, C.OUT / ("g3_leace_%s.json" % mname))
    _part_b_eval(mname, model, tok, rec, df_eval, fits, md, smoke)


def _part_b_fits(mname: str, model, tok, rec: dict, df_fit: pd.DataFrame, smoke: bool) -> tuple[dict, dict]:
    X, z, tpl, cinfo = collect_gender(model, tok, df_fit, max_items=(20 if smoke else 1500))
    rec["part_b"]["collect"] = cinfo
    layers = sorted(X); dev = str(next(model.parameters()).device)
    # held-out templates inside the fit pool for the probes (1/4 of the fit templates)
    ft = sorted(set(tpl.tolist())); rng = random.Random(C.RANDOM_SEED); rng.shuffle(ft)
    held = set(ft[:max(1, len(ft) // 4)])
    te = np.isin(tpl, list(held)); tr = ~te
    target, _ = F2.load_target_basis(mname)
    probe_layers = [layers[len(layers) // 4], layers[len(layers) // 2], layers[3 * len(layers) // 4]]
    fits = {}
    idx_tr = np.where(tr)[0]
    small = np.zeros_like(tr); small[np.random.default_rng(C.RANDOM_SEED).choice(idx_tr, size=min(SMALL_ROWS, len(idx_tr)), replace=False)] = True
    for tag, mask in (("gender_large", tr), ("gender_small", small)):
        t0 = time.time()
        er = LF.fit_leace_erasers({l: X[l][mask] for l in layers}, z[mask], shrinkage=True, device=dev)
        d = describe_fit(tag, er, X, z, mask, te, probe_layers, target)
        d.update({"sec_fit": time.time() - t0, "label": "0 = male referent, 1 = female referent (descriptor tokens)"})
        rec["part_b"][tag] = d; fits[tag] = er
        save_erasers(mname, tag, er)
        log.info("g3 B %s %s: %d rows, mid-layer cov rank %d, probe pre/post %s/%s", mname, tag, int(mask.sum()),
                 d["per_layer"][1]["cov_effective_rank_fit"], d["per_layer"][1]["probe_pre"], d["per_layer"][1]["probe_post"])
    # mean-difference direction of the large pool (for the cosine table and an MC_gender condition)
    md = {}
    for l in layers:
        m = X[l][tr][z[tr] == 1].mean(0) - X[l][tr][z[tr] == 0].mean(0); md[l] = (m / np.linalg.norm(m))[None, :].astype(np.float32)
    np.savez(C.OUT / ("g3_md_%s_gender.npz" % mname), **{"md_%d" % l: md[l] for l in layers})
    rec["part_b"]["md_gender_abs_cos_to_contrast_basis"] = {str(l): float(abs(float(md[l][0] @ (target[l][0] / np.linalg.norm(target[l][0]))))) for l in probe_layers}
    del X
    return fits, md


def _part_b_eval(mname: str, model, tok, rec: dict, df_eval: pd.DataFrame, fits: dict, md: dict, smoke: bool) -> None:
    # ---- fresh F<->M evaluation set
    ev, ev_info = build_fm_eval_set(df_eval, n_per_template=(1 if smoke else 6), max_seeds=(2 if smoke else 72))
    rec["part_b"]["eval_set"] = {"n_seeds": int(len(ev)), "n_templates": int(ev.template.nunique()) if len(ev) else 0,
                                 "flipped_F_to_M": int((ev.group_A == "F").sum()) if len(ev) else 0, **ev_info}
    if ev.empty:
        log.warning("g3 B %s: empty F<->M evaluation set (%s)", mname, ev_info); return
    ev.to_csv(C.OUT / ("g3_gender_manifest_%s.csv" % mname), index=False)
    sp_all = spans_for(tok, ev)
    bases = F2.Bases(mname)
    f3_leace, f3_md = bases.leace, bases.md
    conds = {"B": (None, None), "S1": (None, None), "MC": ("md", f3_md), "LC_f3": ("leace", f3_leace),
             "LC_gender_small": ("leace", fits["gender_small"]), "LC_gender_large": ("leace", fits["gender_large"]),
             "MC_gender": ("md", md)}
    path = C.OUT / ("g3_rows_%s.parquet" % mname)
    done = C.done_keys(path, ROW_KEY)
    alphas = {}
    n = 0
    for r in ev.itertuples(index=False):
        sp = sp_all["seeds"][r.seed_id]
        rows = []
        for cname, (slot, obj) in conds.items():
            base_cond = {"B": "B", "S1": "S1", "MC": "MC", "MC_gender": "MC"}.get(cname, "LC")
            if slot == "leace":
                bases.leace = obj
            elif slot == "md":
                bases.md = obj
            if slot and obj is None:
                continue
            for side in ("A", "B"):
                if (mname, r.seed_id, side, cname, "std", "score") in done:
                    continue
                if sp[side]["structural_failure"]:
                    rows.append({"model_name": mname, "seed_id": r.seed_id, "group": "fm_eval", "side": side, "cond": cname, "variant": "std",
                                 "kind": "score", "status": "structural_failure", "category": r.category, "template": r.template})
                    continue
                row = F2.eval_side(model, tok, bases, alphas, r, side, sp[side], base_cond)
                row["cond"] = cname; rows.append(row)
            if cname in ("B", "S1", "LC_gender_large", "LC_f3") and (mname, r.seed_id, "B", cname, "std", "bridge") not in done \
                    and not (sp["A"]["structural_failure"] or sp["B"]["structural_failure"]):
                row = F2.eval_bridge(model, tok, bases, alphas, r, sp, base_cond); row["cond"] = cname; rows.append(row)
        if rows:
            C.append_rows(path, rows); n += len(rows)
    bases.leace, bases.md = f3_leace, f3_md
    log.info("g3 B %s: %d rows on the F<->M evaluation set", mname, n)


def run(mname: str, smoke: bool, parts: str) -> None:
    C.ensure_out()
    cfg, model, tok = C.load_model(mname)
    rec = C.read_json(C.OUT / ("g3_leace_%s.json" % mname), {}) or {}
    rec.update({"model": mname, "utc": C.utc_now(), "concept_erasure_version": LF.leace_version()})
    if "a" in parts:
        have = [k for k, v in (rec.get("part_a") or {}).items() if isinstance(v, dict) and "per_layer" in v]
        if {"swap_small", "swap_300", "swap_max", "swap_max_noshrink"} <= set(have) and all((C.OUT / ("g3_leace_%s_%s.npz" % (mname, t))).exists() for t in have):
            log.info("g3 A %s: already complete, skipped", mname)
        else:
            part_a(mname, model, tok, rec, smoke)
            rec["part_a"]["complete"] = True
            rec["part_a"]["tags"] = [k for k, v in rec["part_a"].items() if isinstance(v, dict) and "per_layer" in v]
            C.write_json(rec, C.OUT / ("g3_leace_%s.json" % mname))
    if "b" in parts and mname in C.FRESH_MODELS:
        part_b(mname, model, tok, rec, smoke)
        C.write_json(rec, C.OUT / ("g3_leace_%s.json" % mname))
    (C.OUT / ("DONE_g3_%s.txt" % mname)).write_text(C.utc_now(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=C.ALL_MODELS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--parts", default="ab", help="a = swap-side refits (pooled audit), b = gender eraser + F<->M set (fresh models)")
    a = ap.parse_args()
    run(a.model, a.smoke, a.parts)


if __name__ == "__main__":
    main()
