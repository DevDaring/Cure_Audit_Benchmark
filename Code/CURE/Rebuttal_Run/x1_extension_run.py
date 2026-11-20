"""
x1_extension_run.py -- scores the extension items under no edit, the matched-position erasure and the
contrast vector at the erasure's energy (results/extension_20261120/x_protocol.json). GPU, one model.

Reuses h1_dose_site.run unchanged, with three switches: the item list is x_manifest.csv, the spans are
resolved for those items by the frozen rule of h5_replication_prep.resolve_spans, and the cells are
restricted to B, P_span@1 and CV_span@1. Patching passes are skipped: these readouts do not need them.
Kernels as the rebuttal cycle (bfloat16, flash attention 2.8.3), so the new rows compare with the old.

Smoke (--smoke): two new pairs and the two inspected 'repro' pairs, whose stored option scores must be
reproduced (checked on the author's machine before the full run).

Rows: results/<HH_OUT_DIR>/h1_dose_<model>.parquet; spans x_spans_<model>.json.
Usage: HH_OUT_DIR=extension_20261120 python x1_extension_run.py --model gemma-2-2b-it [--smoke]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import hh_common as H

MAN = H.K.RESULTS / "extension_20261120" / "x_manifest.csv"
REB = H.K.RESULTS / "rebuttal_20261107"


def manifest() -> pd.DataFrame:
    return pd.read_csv(MAN)


def cv_vector(model: str):
    """The same oriented contrast vectors the rebuttal cycle used: g5 for Gemma-2-2B and
    Llama-3.1-8B, h5 for Qwen2.5-7B and Phi-4-mini."""
    p = H.FB / ("g5_vectors_%s.npz" % model)
    if not p.exists():
        p = REB / ("h5_vectors_%s.npz" % model)
    z = np.load(p)
    return {int(k[3:]): np.asarray(z[k], np.float32) for k in z.files if k.startswith("CV_")}


def spans_for(mname: str, tok, smoke: bool) -> dict:
    import f1_scoring as S
    p = H.OUT / ("x_spans_%s.json" % mname)
    sp = H.read_json(p) if p.exists() else None
    man = manifest()
    if smoke:
        man = pd.concat([man[man.group == "ext"].head(2), man[man.group == "repro"]])
    if sp and set(man.seed_id) <= set(sp["seeds"]):
        return sp
    out = {"model": mname, "rule": "h5_replication_prep.resolve_spans, unchanged", "seeds": {}}
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
                         "n_prefill": len(tok(text, add_special_tokens=True)["input_ids"]), "structural_failure": (not demo["positions"]) or ne["reason"] != "ok"}
        out["seeds"][r.seed_id] = rec
    H.write_json(out, p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=H.ALL_MODELS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    import h1_dose_site as D
    H.ensure_out()
    loaded = H.load_model(a.model)            # the same loader, tokenizer and kernels as the rebuttal cycle
    H.load_model = lambda m: loaded           # h1 reuses this load instead of loading again
    sp = spans_for(a.model, loaded[2], a.smoke)
    man = manifest()
    if a.smoke:
        man = pd.concat([man[man.group == "ext"].head(2), man[man.group == "repro"]])
    H.fresh_manifest = lambda: man
    H.load_spans = lambda m: sp
    H.cv_vector = cv_vector
    D.CELLS = {"CV_span": ("steer", "span", (1.0,))}
    D.PASS_LEVELS = {"a": (1.0,)}
    D.MATCHED_TO = {}
    groups = ["ext", "repro"] if a.smoke else ["ext"]
    D.run(a.model, "a", groups, smoke=False)
    (H.OUT / ("DONE_x1_%s.txt" % a.model)).write_text(H.utc_now(), encoding="utf-8")


if __name__ == "__main__":
    main()
