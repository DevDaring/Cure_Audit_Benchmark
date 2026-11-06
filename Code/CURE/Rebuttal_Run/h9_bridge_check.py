"""
h9_bridge_check.py -- why does the edited C_answer differ per seed from September? (rebuttal cycle,
reproduction diagnostic). GPU. Computes C_answer under the span erasure on the same seeds in three ways:

  sept       f1_scoring.patched_scores: patch hooks registered first, so the edit acts on the injected
             state; the edit touches only the span truncated to the shorter length
  order      this cycle's order (edit first, injection wins), edit on the truncated span
  this       this cycle's bridge (edit first, injection wins), edit on the whole term in both runs

'sept' must equal the stored September value if the instrument is unchanged. 'order' against 'sept'
isolates the hook order; 'this' against 'order' isolates the whole-term edit on unequal-length pairs.

Output: results/rebuttal_20261107/h9_bridge_<model>.csv. Usage: python h9_bridge_check.py --model gemma-2-2b-it --n 40
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import hh_common as H


def main() -> None:
    import f1_scoring as S
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    m = a.model
    man = H.fresh_manifest(); man = man[man.group == "final"]
    old = pd.read_parquet(H.FINAL / ("per_item_%s.parquet" % m))
    old = old[(old.kind == "bridge") & (old.status == "ok") & (old.cond == "S1")].set_index("seed_id").C_answer
    sp_all = H.load_spans(m)
    cfg, model, tok = H.load_model(m)
    target, _ = H.target_basis(m)
    fac = H.make_factory(model, "proj", target, 1.0, "span")
    rows = []
    for r in man.head(a.n).itertuples(index=False):
        sp = sp_all["seeds"][r.seed_id]
        pa, pb = sp["A"]["demo"], sp["B"]["demo"]
        if not pa or not pb:
            continue
        n = min(len(pa), len(pb))
        opts = json.loads(r.options_B); gold = int(r.gold_index); c = S.candidates(opts, "text"); ta, tb = sp["A"]["text"], sp["B"]["text"]
        cl = S.patched_scores(model, tok, tb, tb, pb, pb, c, fac, None); pt = S.patched_scores(model, tok, ta, tb, pa, pb, c, fac, None)
        sept = S.margin(pt["s"], gold) - S.margin(cl["s"], gold)
        cl = H.score_under_patch(model, tok, tb, tb, pb, pb, c, fac, fac, pb, pb, None); pt = H.score_under_patch(model, tok, ta, tb, pa, pb, c, fac, fac, pa[:n], pb[:n], None)
        cl2 = H.score_under_patch(model, tok, tb, tb, pb, pb, c, fac, fac, pb[:n], pb[:n], None)
        order = S.margin(pt["s"], gold) - S.margin(cl2["s"], gold)
        pt3 = H.score_under_patch(model, tok, ta, tb, pa, pb, c, fac, fac, pa, pb, None)
        this = S.margin(pt3["s"], gold) - S.margin(cl["s"], gold)
        rows.append({"seed_id": r.seed_id, "equal": len(pa) == len(pb), "stored": float(old.get(r.seed_id, np.nan)), "sept": sept, "order": order, "this": this})
    d = pd.DataFrame(rows)
    d.to_csv(H.K.RESULTS / "rebuttal_20261107" / ("h9_bridge_%s.csv" % m), index=False)
    corr = lambda x, y: float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
    print("pairs", len(d), "equal", int(d.equal.sum()))
    print("sept vs stored: corr %.5f, max |diff| %.4f" % (corr(d.sept, d.stored), (d.sept - d.stored).abs().max()))
    for lab, x in (("equal", d[d.equal]), ("unequal", d[~d.equal])):
        print("%-8s order vs sept: corr %.4f median|diff| %.4f | this vs order: corr %.4f median|diff| %.4f | this vs sept: corr %.4f | mean|C| sept %.3f order %.3f this %.3f" % (
            lab, corr(x.order, x.sept), (x.order - x.sept).abs().median(), corr(x.this, x.order), (x.this - x.order).abs().median(), corr(x.this, x.sept), x.sept.abs().mean(), x.order.abs().mean(), x["this"].abs().mean()))


if __name__ == "__main__":
    main()
