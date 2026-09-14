"""
g2_alignment.py -- A2 of Submission2/Next_Plan.md (ICLR-feedback cycle): the span-length
mismatch and the truncation rule, on the fresh replication. GPU, one model at a time.

The bridge of the final cycle (f2_runner.eval_bridge / f1_scoring.patched_scores) injects a's
span states into b's span position by position and truncates to the shorter span, so when b's
demographic term has more tokens than a's, the extra tokens of b stay unpatched. This stage
re-reads C_answer on every pair whose two spans differ in token length (about two fifths of
the fresh pairs) under three alignment rules, for the no-edit and span-erasure conditions:

  truncate   the frozen rule (reproduces the stored value)
  mean-pad   every position of b's span is patched; positions beyond a's span length receive
             the mean of a's span vectors at that layer
  last-pad   as mean-pad, with the last vector of a's span instead of the mean

When a's span is the longer one, all three rules patch all of b's positions with a's first
len(b) vectors (nothing of b is left unpatched); such pairs are kept so the strata are complete.

Rows: results/feedback_20260915/g2_alignment_<model>.parquet, one per (seed, cond, rule);
resumable. Usage: python g2_alignment.py --model gemma-2-2b-it [--smoke] [--all-pairs]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import nn_common as C
import f1_scoring as S          # noqa: E402
import f2_runner as F2          # noqa: E402
import p1_intervention as I     # noqa: E402

log = C.log
ROW_KEY = ["model_name", "seed_id", "cond", "rule"]
RULES = ("truncate", "mean-pad", "last-pad")


def patched_scores_aligned(model, tok, text_a: str, text_b: str, pos_a: list[int], pos_b: list[int],
                           cands_b: list[str], edit_factory, first_ids_b: list[int], rule: str) -> dict:
    import torch
    dec = I._decoder_layers(model)
    L = list(range(len(dec)))
    na, nb = len(pos_a), len(pos_b)
    if na == 0 or nb == 0:
        return {"error": "empty span"}
    if rule == "truncate":
        n = min(na, nb); pa, pb = pos_a[:n], pos_b[:n]
    else:
        pb = list(pos_b); pa = pos_a[:min(na, nb)]
    e_a = edit_factory() if edit_factory else None
    _, hs_a = I.forward_hidden(model, tok, text_a, e_a, pos_a)
    src = {}
    for l in L:
        v = hs_a[l][pa, :].clone()
        if rule != "truncate" and nb > na:
            fill = hs_a[l][pos_a, :].mean(0, keepdim=True) if rule == "mean-pad" else hs_a[l][pos_a[-1:], :]
            v = torch.cat([v, fill.repeat(nb - na, 1)], dim=0)
        src[l] = v
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    handles = []

    def make_patch(l):
        s = src[l].to(dev).to(dt)

        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] > 1:
                hs[:, pb, :] = s
            return output
        return hook

    e_b = edit_factory() if edit_factory else None
    try:
        for l in L:
            handles.append(dec[l].register_forward_hook(make_patch(l)))
        out = S.score_candidates(model, tok, text_b, cands_b, e_b, pb, first_ids_b)
    finally:
        for h in handles:
            h.remove()
    out.update({"patched_len": len(pb), "span_len_a": na, "span_len_b": nb, "rule": rule})
    return out


def run(mname: str, smoke: bool, all_pairs: bool) -> None:
    C.ensure_out()
    path = C.OUT / ("g2_alignment_%s.parquet" % mname)
    man = F2.manifest(); man = man[man.group == "final"]
    sp_all = C.read_json(C.FINAL / ("spans_%s.json" % mname))
    if sp_all is None:
        raise SystemExit("spans_%s.json missing" % mname)
    cfg, model, tok = C.load_model(mname)
    bases = F2.Bases(mname)
    cal = C.read_json(C.FINAL / ("calibration_%s.json" % mname), {"alphas": {}})
    done = C.done_keys(path, ROW_KEY)
    t0 = time.time(); n = 0; n_pairs = 0
    for r in man.itertuples(index=False):
        sp = sp_all["seeds"][r.seed_id]
        pa, pb = sp["A"]["demo"], sp["B"]["demo"]
        if not pa or not pb:
            continue
        unequal = len(pa) != len(pb)
        if not (unequal or all_pairs):
            continue
        n_pairs += 1
        if smoke and n_pairs > 2:
            break
        options = json.loads(r.options_B); gold = int(r.gold_index)
        cands = S.candidates(options, "text"); first_ids = S.first_token_ids(tok, sp["B"]["text"], options)
        rows = []
        for cond in ("B", "S1"):
            factory, site, pkey, alpha, bid = F2.make_edit(model, bases, cond, cal["alphas"])
            for rule in RULES:
                if (mname, r.seed_id, cond, rule) in done:
                    continue
                row = {"model_name": mname, "seed_id": r.seed_id, "category": r.category, "template": r.template, "cond": cond, "rule": rule,
                       "alpha": alpha, "basis_id": bid, "gold_index": gold, "unequal_span_len": unequal, "b_longer": len(pb) > len(pa), "status": "ok"}
                try:
                    clean = patched_scores_aligned(model, tok, sp["B"]["text"], sp["B"]["text"], pb, pb, cands, factory, first_ids, "truncate")
                    pat = patched_scores_aligned(model, tok, sp["A"]["text"], sp["B"]["text"], pa, pb, cands, factory, first_ids, rule)
                    if "error" in pat or "error" in clean:
                        row.update({"status": "error", "error": pat.get("error") or clean.get("error")})
                    else:
                        Mc, Mp = S.margin(clean["s"], gold), S.margin(pat["s"], gold)
                        row.update({"M_clean": Mc, "M_patched": Mp, "C_answer": Mp - Mc,
                                    "C_first": pat["first_logits"][gold] - clean["first_logits"][gold],
                                    "patched_len": pat["patched_len"], "span_len_a": pat["span_len_a"], "span_len_b": pat["span_len_b"],
                                    "s_clean": json.dumps(clean["s"]), "s_patched": json.dumps(pat["s"])})
                except Exception as ex:
                    row.update({"status": "error", "error": str(ex)[:300]}); log.exception("g2 %s %s %s %s", mname, r.seed_id, cond, rule)
                row["utc"] = C.utc_now()
                rows.append(row)
        if rows:
            C.append_rows(path, rows); n += len(rows)
    log.info("g2 %s: %d pairs, %d new rows in %.1f min", mname, n_pairs, n, (time.time() - t0) / 60)
    (C.OUT / ("DONE_g2_%s.txt" % mname)).write_text(C.utc_now(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=C.FRESH_MODELS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--all-pairs", action="store_true", help="also the equal-length pairs (the rules coincide there; a consistency check)")
    a = ap.parse_args()
    run(a.model, a.smoke, a.all_pairs)


if __name__ == "__main__":
    main()
