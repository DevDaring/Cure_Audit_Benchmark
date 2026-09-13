"""
reparse_generations.py -- re-derive the generation readout columns of every per-item file
from the stored raw generations, so rows produced before and after a parser change are
scored identically.

Why: the answer parser originally required a closing brace; generations capped at
max_new_tokens often truncate the JSON inside the rationale field, so the (complete) answer
field was discarded and the row marked invalid ("unmapped"). Llama-3.1-8B was affected on
96% of pilot rows. Every row stores gen_raw_{A,B}, seed_id and subvariant_{A,B}; prompts and
gold answers come from the pentad, so the readout is a pure function that can be recomputed.

Recomputed per side X in {A, B}: gen_idx_X, gen_parse_X, gen_valid_X, gen_correct_X; and
gen_flip_AB. Rows without gen_raw_* (P3 files, status rows) are left unchanged. A column
gen_reparsed_utc records when.

Usage
  python reparse_generations.py                       # every per-item file under results/v2
  python reparse_generations.py --dir results/v2_qwen2.5-7b-instruct
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common as K
import p1_pilot as P1

log = K.setup_logging("cure.ext.reparse")
FILES = ["pilot_per_item.parquet", "p2_per_item.parquet"]


def _pentad_index() -> dict:
    pen = pd.read_parquet(K.PENTAD_CLEAN)
    pen = pen[pen["slot"] == "c"]
    return {(r["seed_id"], r["subvariant"]): (str(r["prompt_text"]), str(r.get("gold_answer", "")))
            for _, r in pen.iterrows()}


def reparse_file(path: Path, idx: dict) -> dict:
    df = pd.read_parquet(path)
    if "gen_raw_A" not in df.columns:
        return {"file": path.name, "skipped": "no generation columns"}
    before_valid = df["gen_valid_A"].map(lambda v: bool(v) if v is not None and v == v else False).mean()
    changed = 0
    for i, r in df.iterrows():
        if not isinstance(r.get("gen_raw_A"), str):
            continue
        new = {}
        for side in ("A", "B"):
            key = (r["seed_id"], r.get(f"subvariant_{side}"))
            if key not in idx:
                continue
            prompt, gold = idx[key]
            raw = r.get(f"gen_raw_{side}")
            if not isinstance(raw, str):
                continue
            oi, why = P1.parse_answer(raw, prompt)
            gi = P1.gold_index(prompt, gold)
            new[f"gen_idx_{side}"] = oi
            new[f"gen_parse_{side}"] = why
            new[f"gen_valid_{side}"] = oi is not None
            new[f"gen_correct_{side}"] = (oi == gi) if (oi is not None and gi is not None) else None
        if "gen_idx_A" in new and "gen_idx_B" in new:
            new["gen_flip_AB"] = (new["gen_idx_A"] != new["gen_idx_B"]) if (new["gen_valid_A"] and new["gen_valid_B"]) else None
        for k, v in new.items():
            old = r.get(k)
            if not ((old is None or (isinstance(old, float) and np.isnan(old))) and v is None) and old != v:
                changed += 1
            df.at[i, k] = v
    for c in ("gen_idx_A", "gen_idx_B"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["gen_reparsed_utc"] = K.utc_now()
    df.to_parquet(path, index=False)
    after_valid = df["gen_valid_A"].map(lambda v: bool(v) if v is not None and v == v else False).mean()
    return {"file": path.name, "rows": len(df), "cells_changed": changed,
            "valid_A_before": round(float(before_valid), 3), "valid_A_after": round(float(after_valid), 3)}


def reparse_dir(d: Path) -> list[dict]:
    idx = _pentad_index()
    out = []
    for name in FILES:
        p = d / name
        if p.exists():
            rec = reparse_file(p, idx)
            log.info("%s: %s", K.rel(p), rec)
            out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="one results directory (default: results/v2)")
    args = ap.parse_args()
    d = Path(args.dir).resolve() if args.dir else K.RESULTS / "v2"
    for rec in reparse_dir(d):
        print(rec)


if __name__ == "__main__":
    main()
