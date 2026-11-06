"""
h4_rewrite.py -- the prompt-level baseline: neutralise the demographic term in the text instead
of editing activations (rebuttal cycle, answers uG1a W3 'replace the term with a neutral word
such as person'). GPU, one model at a time, all four models.

For every seed of the fresh set and of the relevant-information control set, the entity-1
descriptor is neutralised in the user prompt (context and option line) under a rule read off
the option that names the entity (hh_common.rewrite_rule):
  name       'Dorothy'                      -> 'The person'
  head       'The elderly woman'            -> 'The person'
  attribute  'The individual with no scars' -> 'The individual with an unspecified feature'
  modifier   'The Jewish man'               -> 'The unspecified man'
The entity stays identifiable from the context, so a disambiguated control item stays answerable.
Both variants of a pair are rewritten with their own descriptor; when the two rewritten
prompts coincide the answer sensitivity T is zero by construction, which the row records
(identical_after_rewrite). A pair whose rewritten options are not three distinct strings is
recorded as degenerate and not scored. Conditions: RW (rewritten prompt, no edit) and B (the
original prompt, no edit; a reproduction of the final cycle's rows on this GPU).

Rows: results/rebuttal_20261107/h4_rewrite_<model>.parquet.
Usage: python h4_rewrite.py --model gemma-2-2b-it [--smoke]
"""

from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np

import hh_common as H

log = H.log


def rewrite_side(r, side: str) -> dict:
    options = json.loads(getattr(r, "options_%s" % side))
    desc = getattr(r, "descriptor_%s" % side)
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(desc) + r"(?![A-Za-z0-9])", re.I)
    opt_with = [o for o in options if pat.search(o)]
    if not opt_with:
        return {"ok": False, "reason": "descriptor not in any option"}
    rule, repl = H.rewrite_rule(opt_with[0], desc)
    other = str(r.entity2_descriptor)
    new_opts = [H.apply_rewrite(o, desc, rule, repl, protect=other) for o in options]
    new_prompt = H.apply_rewrite(getattr(r, "prompt_%s" % side), desc, rule, repl, protect=other)
    norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
    degenerate = len({norm(o) for o in new_opts}) != len(new_opts) or any(not norm(o) for o in new_opts)
    return {"ok": not degenerate, "reason": "degenerate options" if degenerate else "", "rule": rule, "replacement": repl, "options": new_opts,
            "prompt": new_prompt, "n_desc_hits_prompt": len(pat.findall(getattr(r, "prompt_%s" % side)))}


def run(mname: str, smoke: bool) -> None:
    import f1_scoring as S
    H.ensure_out()
    path = H.OUT / ("h4_rewrite_%s.parquet" % mname)
    man = H.fresh_manifest()
    sp_all = H.load_spans(mname)
    cfg, model, tok = H.load_model(mname)
    done = H.done_keys(path)
    t0 = time.time(); n = 0
    for group in ("final", "control"):
        sub = man[man.group == group]
        if smoke:
            sub = sub.head(2)
        for r in sub.itertuples(index=False):
            sp = sp_all["seeds"][r.seed_id]
            rw = {side: rewrite_side(r, side) for side in ("A", "B")}
            identical = bool(rw["A"].get("ok") and rw["B"].get("ok") and rw["A"]["prompt"] == rw["B"]["prompt"])
            rows = []
            for side in ("A", "B"):
                s = sp[side]
                for cond in ("B", "RW"):
                    if (mname, r.seed_id, side, cond, "std", "score") in done:
                        continue
                    meta = {"cond": cond, "cell": cond, "op": "none", "site": "none", "level": 0.0, "alpha": 0.0, "feasible": True, "basis_id": "none", "rank": 0,
                            "rewrite_rule": rw[side].get("rule"), "rewrite_replacement": rw[side].get("replacement"), "identical_after_rewrite": identical,
                            "rewrite_prompt": rw[side].get("prompt"), "n_desc_hits_prompt": rw[side].get("n_desc_hits_prompt")}
                    if cond == "RW":
                        if not rw[side].get("ok"):
                            rows.append({"model_name": mname, "seed_id": r.seed_id, "group": r.group, "side": side, "cond": cond, "variant": "std", "kind": "score",
                                         "status": "degenerate", "error": rw[side].get("reason"), "category": r.category, "template": r.template, **meta})
                            continue
                        text = S.chat_text(tok, rw[side]["prompt"])
                        row = H.eval_cell(model, tok, mname, r, side, s, None, [], meta, generate=True, options=rw[side]["options"], text=text, prompt=rw[side]["prompt"])
                    else:
                        row = H.eval_cell(model, tok, mname, r, side, s, None, [], meta, generate=True)
                    rows.append(row)
            if rows:
                H.append_rows(path, rows); n += len(rows)
    log.info("h4 %s: %d new rows in %.1f min", mname, n, (time.time() - t0) / 60)
    (H.OUT / ("DONE_h4_%s.txt" % mname)).write_text(H.utc_now(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=H.FRESH_MODELS + H.REPLICATION_MODELS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.model, a.smoke)


if __name__ == "__main__":
    main()
