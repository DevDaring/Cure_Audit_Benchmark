"""
h8_quant_check.py -- how much of a small edit survives the model's numeric precision? (rebuttal
cycle, design check before the full runs). GPU.

The models run in bfloat16 (8 significand bits: 0.4-0.8 per cent relative spacing). The hook computes
delta in bfloat16 and writes h - delta back in bfloat16, so a component whose delta is under half a
unit in the last place rounds back to the original value. The energy bookkeeping records the nominal
delta. This stage measures, per cell and per prompt, the nominal energy against the energy actually
applied (||h' - h||^2 after the write) and the applied displacement's component along the edit
direction, at the strengths the smoke rows used. Also times a float32 copy of the model and compares
its unedited and span-erasure scores with the bfloat16 ones.

Output: results/<dir>/h8_quant_<model>.json. Usage: python h8_quant_check.py --model gemma-2-2b-it
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import hh_common as H
import h1_dose_site as H1

log = H.log


def measure(model, tok, text, rows, alpha, op, pos):
    import f1_scoring as S
    e = H.DoseEdit(model, rows, alpha, mode=op)
    S.score_candidates(model, tok, text, [S.PREFIX + "x" + S.SUFFIX], e, pos, None)
    return e.applied_summary()


def run(mname: str, seeds: int) -> None:
    import torch
    import f1_scoring as S
    H.ensure_out()
    src = H.K.RESULTS / ("rebuttal_smoke_%s" % mname)
    d = pd.read_parquet(src / ("h1_dose_%s.parquet" % mname))
    d = d[(d.kind == "score") & (d.status == "ok") & (d.group == "final")]
    H.OUT = src                        # the smoke directory holds this model's spans and vector when h5 made them
    sp_all = H.load_spans(mname)
    cfg, model, tok = H.load_model(mname)
    target, _ = H.target_basis(mname)
    bases = {"target": target, "random": H.random_direction(target), "random_k": H.random_subspace(target, 64), "cv": H.cv_vector(mname)}
    man = H.fresh_manifest()
    rec = {"model": mname, "dtype_loaded": str(next(model.parameters()).dtype), "cells": {}, "note": "ratios of applied to nominal energy; along = squared component of the applied displacement along the edit direction, over nominal"}
    for cond in sorted(d.cond.unique()):
        if cond == "B" or "@E(" in cond:
            continue
        cell = cond.split("@")[0]
        if cell not in H1.CELLS:
            continue
        op, site, _ = H1.CELLS[cell]
        rows = H1.rows_for(cell, bases)
        out = []
        for x in d[d.cond == cond].head(seeds * 2).itertuples(index=False):
            s = sp_all["seeds"][x.seed_id][x.side]
            pos = H.positions_for_site(s, site)
            if not pos:
                continue
            out.append(measure(model, tok, s["text"], rows, float(x.alpha), op, pos))
        if out:
            rec["cells"][cond] = {"median_alpha": float(d[d.cond == cond].alpha.median()), "n": len(out), **{k: float(np.median([o[k] for o in out])) for k in out[0]}}
    # ---- float32 copy: timing and agreement
    r0 = man[man.group == "final"].iloc[0]; s0 = sp_all["seeds"][r0.seed_id]["A"]
    cands = S.candidates(json.loads(r0.options_A), "text")
    def score():
        t0 = time.time()
        b = S.score_candidates(model, tok, s0["text"], cands, None, [], None)["s"]
        e = H.DoseEdit(model, target, 1.0, mode="proj")
        s1 = S.score_candidates(model, tok, s0["text"], cands, e, s0["demo"], None)["s"]
        import p1_intervention as I
        t1 = time.time()
        I.generate_under_edit(model, tok, [s0["text"]], H.DoseEdit(model, target, 1.0, mode="proj"), [s0["demo"]], max_new_tokens=48)
        return b, s1, t1 - t0, time.time() - t1
    b16, s16, t16, g16 = score()
    rec["sec_bf16"] = {"score_pair": t16, "generate_48": g16}
    try:
        torch.cuda.empty_cache()
        model = model.to(torch.float32)
        target32 = target
        b32, s32, t32, g32 = score()
        rec["sec_float32"] = {"score_pair": t32, "generate_48": g32}
        rec["float32_vs_bf16_max_abs_score_dev"] = {"unedited": float(np.max(np.abs(np.array(b16) - np.array(b32)))), "span_erasure": float(np.max(np.abs(np.array(s16) - np.array(s32))))}
        rec["gpu_mem_gb_float32"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
        cells32 = {}
        for cond in ("P_all@1", "P_span@0.1", "P_all@0.1", "R_all@1", "CV_all@1"):
            if cond not in rec["cells"]:
                continue
            cell = cond.split("@")[0]
            op, site, _ = H1.CELLS[cell]
            rows = H1.rows_for(cell, bases)
            out = []
            for x in d[d.cond == cond].head(seeds * 2).itertuples(index=False):
                s = sp_all["seeds"][x.seed_id][x.side]
                out.append(measure(model, tok, s["text"], rows, float(x.alpha), op, H.positions_for_site(s, site)))
            cells32[cond] = {k: float(np.median([o[k] for o in out])) for k in out[0]}
        rec["cells_float32"] = cells32
    except Exception as ex:
        rec["float32_error"] = str(ex)[:300]
    out_dir = H.K.RESULTS / "rebuttal_20261107"; out_dir.mkdir(parents=True, exist_ok=True)
    H.write_json(rec, out_dir / ("h8_quant_%s.json" % mname))
    print(json.dumps(rec, indent=1)[:6000])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=H.ALL_MODELS)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    run(a.model, a.seeds)


if __name__ == "__main__":
    main()
