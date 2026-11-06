"""
h7_token_scores.py -- per-token option log-probabilities, to explain why the Llama-3.1-8B
sensitivity reduction reverses under length-normalised scoring (rebuttal cycle, answers JwYi
Q2 and the PAT construct-validity point). GPU, after h1a (reads each cell's alpha from its rows).

The manuscript stored only the sum and the mean of each option's token log-probabilities. This
stage keeps the token-level vector for the final seeds under B, P_span@1, P_ne@1 and CV_span@1,
so a change in T can be attributed to (a) the descriptor's own tokens in the option, (b) the
JSON prefix and suffix shared by every option, (c) the remaining option tokens.

Rows: results/rebuttal_20261107/h7_tokens_<model>.parquet. No generation.
Usage: python h7_token_scores.py --model llama-3.1-8b-instruct [--smoke]
"""

from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd

import hh_common as H

log = H.log
CONDS = {"B": ("none", None, "none"), "P_span@1": ("proj", "target", "span"), "P_ne@1": ("proj", "target", "ne"), "CV_span@1": ("steer", "cv", "span")}


def token_scores(model, tok, text: str, cands: list[str], edit, positions: list[int]) -> list[dict]:
    """One teacher-forced pass per candidate batch: token ids, strings and log-probabilities."""
    import torch
    import p1_intervention as I
    dev = next(model.parameters()).device
    base = tok(text, add_special_tokens=True)["input_ids"]
    T0 = len(base)
    rows = []
    for c in cands:
        ids = tok(text + c, add_special_tokens=True)["input_ids"]
        if ids[:T0] != base:
            ids = base + tok(c, add_special_tokens=False)["input_ids"]
        rows.append(ids)
    maxlen = max(len(r) for r in rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    input_ids = torch.full((len(rows), maxlen), pad, dtype=torch.long)
    attn = torch.zeros_like(input_ids)
    for i, r in enumerate(rows):
        input_ids[i, :len(r)] = torch.tensor(r); attn[i, :len(r)] = 1
    enc = {"input_ids": input_ids.to(dev), "attention_mask": attn.to(dev)}
    if edit is not None:
        edit.set_targets([list(positions)] * len(rows), [0] * len(rows))
    with torch.no_grad(), (edit if edit is not None else I.no_edit()):
        logits = model(**enc).logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    out = []
    for i, r in enumerate(rows):
        tgt = torch.tensor(r[T0:], device=dev)
        lp = logp[i, T0 - 1:len(r) - 1, :].gather(-1, tgt[:, None]).squeeze(-1)
        out.append({"ids": r[T0:], "tokens": [tok.decode([t]) for t in r[T0:]], "lp": [float(x) for x in lp.tolist()]})
    return out


def descriptor_mask(tokens: list[str], desc: str) -> list[int]:
    """1 for the candidate tokens that lie inside the descriptor's characters."""
    text = "".join(tokens)
    m = re.search(r"(?<![A-Za-z0-9])" + re.escape(desc) + r"(?![A-Za-z0-9])", text, flags=re.I)
    if m is None:
        return [0] * len(tokens)
    mask, pos = [], 0
    for t in tokens:
        a, b = pos, pos + len(t)
        mask.append(int(b > m.start() and a < m.end()))
        pos = b
    return mask


def run(mname: str, smoke: bool) -> None:
    import f1_scoring as S
    H.ensure_out()
    path = H.OUT / ("h7_tokens_%s.parquet" % mname)
    man = H.fresh_manifest(); man = man[man.group == "final"]
    if smoke:
        man = man.head(2)
    sp_all = H.load_spans(mname)
    h1 = pd.read_parquet(H.OUT / ("h1_dose_%s.parquet" % mname), columns=["seed_id", "side", "cond", "kind", "status", "alpha"])
    h1 = h1[(h1.kind == "score") & (h1.status == "ok")]
    alpha = {(r.seed_id, r.side, r.cond): float(r.alpha) for r in h1.itertuples(index=False)}
    cfg, model, tok = H.load_model(mname)
    target, _ = H.target_basis(mname)
    bases = {"target": target, "cv": H.cv_vector(mname)}
    done = H.done_keys(path, ["model_name", "seed_id", "side", "cond"])
    t0 = time.time(); n = 0
    for r in man.itertuples(index=False):
        rows = []
        for side in ("A", "B"):
            s = sp_all["seeds"][r.seed_id][side]
            options = json.loads(getattr(r, "options_%s" % side)); desc = getattr(r, "descriptor_%s" % side)
            cands = S.candidates(options, "text")
            for cond, (op, bname, site) in CONDS.items():
                if (mname, r.seed_id, side, cond) in done or (bname and bases[bname] is None) or (site == "ne" and not s.get("ne")) or not s["demo"]:
                    continue
                a = alpha.get((r.seed_id, side, cond), 1.0 if cond == "P_span@1" else None)
                if op != "none" and a is None:
                    continue
                edit = H.make_factory(model, op, bases[bname], a, site)() if op != "none" else None
                pos = H.positions_for_site(s, site) if op != "none" else []
                ts = token_scores(model, tok, s["text"], cands, edit, pos)
                rows.append({"model_name": mname, "seed_id": r.seed_id, "template": r.template, "category": r.category, "side": side, "cond": cond, "alpha": a if op != "none" else 0.0,
                             "gold_index": int(r.gold_index), "descriptor": desc, "tokens": json.dumps([x["tokens"] for x in ts]), "lp": json.dumps([x["lp"] for x in ts]),
                             "desc_mask": json.dumps([descriptor_mask(x["tokens"], desc) for x in ts]), "utc": H.utc_now()})
        if rows:
            H.append_rows(path, rows); n += len(rows)
    log.info("h7 %s: %d new rows in %.1f min", mname, n, (time.time() - t0) / 60)
    (H.OUT / ("DONE_h7_%s.txt" % mname)).write_text(H.utc_now(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=H.FRESH_MODELS + H.REPLICATION_MODELS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.model, a.smoke)


if __name__ == "__main__":
    main()
