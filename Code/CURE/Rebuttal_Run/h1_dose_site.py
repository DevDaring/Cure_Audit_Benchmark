"""
h1_dose_site.py -- site x operator x amount on the fresh replication, energies matched PER PROMPT
SIDE (rebuttal cycle; answers dfbY W1/Q1 and JwYi Q3/Q6). GPU, one model at a time.

For every seed and side of the fresh set and of the relevant-information control set, the rank-one
span erasure at alpha = 1 fixes the reference energy E_S1(side). Every other cell is solved for the
strength at which it removes f * E_S1(side) on that very prompt. Three kinds of cell:

  equal total energy  f = 1 (and 0.1): the amount the span erasure removes, spent elsewhere
  dose curve          P_span at f in {1, 0.3, 0.1, 0.03}; P_all at f in {10, 1, 0.1}; CV_span at {1, 0.3, 0.1}
  equal strength      P_all and P_ne at alpha = 1 ('@a1'): the manuscript's every-position condition,
                      whose energy per token is close to the span erasure's. Total-energy and
                      per-token matching answer different questions, so both are reported.

  P   projection of the contrast-derived rank-one basis    span, ne (control span), last, all
  R   projection of the fixed random rank-one direction    all
  Rk  projection of a fixed random rank-64 subspace        span, f = 1 only. EXPLORATORY: a rank-one random
      direction cannot reach E_S1 at the span, and raising the rank confounds direction with rank
  CV  additive contrast vector (RepE / ActAdd style)       span, last, all

  sites  span  the demographic term's tokens     ne   a non-demographic span of equal length
         last  the last prompt token             all  every prefill position
  Every site is prefill-only: no cell edits generated tokens, so all cells spend their energy in the
  same pass. The manuscript's generation-wrapper policy, which also edits decode steps, is not rerun.

A projection cannot exceed its alpha = 1 energy. A cell that cannot reach its target (P_last at f = 1,
possibly P_ne) is run at alpha = 1, flagged feasible = False with its achieved fraction, and the span
erasure is scaled DOWN per side to that achieved energy ('P_span@E(P_last)', 'P_span@E(P_ne)'), so the
site comparison stays at equal energy on every prompt whatever the feasibility.

Readouts: complete-answer scores (T, length-normalised T, option accuracy), generation accuracy and
validity (all levels except 0.03), energy; on the fresh set the answer-level patching effect C_answer
for the equal-energy and equal-strength cells, the edit at its own site on every pass and the patch at
the span (hh_common.bridge_cell).

Rows: results/rebuttal_20261107/h1_dose_<model>.parquet (resumable; key seed, side, cond, kind).
Usage: python h1_dose_site.py --model gemma-2-2b-it --pass a|b [--smoke]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import hh_common as H

log = H.log
A1 = "a1"                     # equal strength: alpha = 1, no energy target
CELLS = {                     # cell: (op, site, levels)
    "P_span": ("proj", "span", (1.0, 0.3, 0.1, 0.03)),
    "P_ne": ("proj", "ne", (1.0, 0.1, A1)),
    "P_last": ("proj", "last", (1.0, 0.1)),
    "P_all": ("proj", "all", (10.0, 1.0, 0.1, A1)),
    "R_all": ("proj", "all", (1.0, 0.1)),
    "Rk_span": ("proj", "span", (1.0,)),
    "CV_span": ("steer", "span", (1.0, 0.3, 0.1)),
    "CV_last": ("steer", "last", (1.0,)),
    "CV_all": ("steer", "all", (1.0,)),
}
PASS_LEVELS = {"a": (10.0, 1.0, 0.1, A1), "b": (0.3, 0.03)}
NO_GENERATION = (0.03,)
NO_BRIDGE_CELLS = ("Rk_span",)
BRIDGE_LEVELS = (1.0, 0.1, A1)
MATCHED_TO = {"P_last": "P_span@E(P_last)", "P_ne": "P_span@E(P_ne)"}


def cond_name(cell: str, level) -> str:
    return "%s@%s" % (cell, level if level == A1 else "%g" % level)


def rows_for(cell: str, bases: dict) -> dict:
    if cell.startswith("Rk_"):
        return bases["random_k"]
    if cell.startswith("R_"):
        return bases["random"]
    if cell.startswith("CV_"):
        return bases["cv"]
    return bases["target"]


def basis_id(cell: str) -> str:
    return {"P": "target_rank1", "R": "random_rank1_seed%d" % H.N.RANDOM_SEED_FINAL, "Rk": "random_rank64_seed%d" % H.RANDOM_SEED,
            "CV": "CV_vector"}[cell.split("_")[0]]


class AlphaStore:
    """alpha, feasibility and per-pass energy of every evaluated (seed, side, cond), from the rows on
    disk and from this run, so bridges and matched cells survive a restart."""

    def __init__(self, df: pd.DataFrame):
        self.d = {}
        if len(df):
            for r in df[(df.kind == "score") & (df.status == "ok")].itertuples(index=False):
                self.d[(r.seed_id, r.side, r.cond)] = {"alpha": float(r.alpha), "energy": float(r.energy_per_pass), "feasible": bool(r.feasible)}

    def put(self, row: dict) -> None:
        if row.get("status") == "ok" and row.get("kind") == "score":
            self.d[(row["seed_id"], row["side"], row["cond"])] = {"alpha": float(row["alpha"]), "energy": float(row["energy_per_pass"]), "feasible": bool(row["feasible"])}

    def get(self, seed: str, side: str, cond: str) -> dict | None:
        return self.d.get((seed, side, cond))


def run(mname: str, which: str, groups: list[str], smoke: bool) -> None:
    H.ensure_out()
    path = H.OUT / ("h1_dose_%s.parquet" % mname)
    man = H.fresh_manifest()
    sp_all = H.load_spans(mname)
    cfg, model, tok = H.load_model(mname)
    target, prov = H.target_basis(mname)
    bases = {"target": target, "random": H.random_direction(target), "random_k": H.random_subspace(target, 64), "cv": H.cv_vector(mname)}
    if bases["cv"] is None:
        raise SystemExit("no steering vector for %s" % mname)
    levels = PASS_LEVELS[which]
    existing = H.read_rows(path)
    done = set(tuple(r) for r in existing[H.ROW_KEY].itertuples(index=False, name=None)) if len(existing) else set()
    store = AlphaStore(existing)
    H.write_json({"model": mname, "cells": {k: [v[0], v[1], list(v[2])] for k, v in CELLS.items()}, "pass_levels": PASS_LEVELS, "no_generation_levels": list(NO_GENERATION),
                  "match_tol": H.MATCH_TOL, "basis": prov, "model_cfg": {k: cfg.get(k) for k in ("hf_id", "model_class", "attn_implementation", "dtype")}, "utc": H.utc_now()},
                 H.OUT / ("h1_protocol_%s_%s.json" % (mname, which)))
    t_start = time.time(); n_new = 0

    for group in groups:
        sub = man[man.group == group]
        if smoke:
            sub = sub.head(2)
        for r in sub.itertuples(index=False):
            sp = sp_all["seeds"][r.seed_id]
            rows = []

            def keep(row, key):
                rows.append(row); store.put(row); done.add(key)

            for side in ("A", "B"):
                s = sp[side]
                if not s["demo"]:
                    continue
                k = (mname, r.seed_id, side, "B", "std", "score")
                if k not in done:
                    meta = {"cond": "B", "cell": "B", "op": "none", "site": "none", "level": 0.0, "alpha": 0.0, "feasible": True, "target_energy": 0.0, "e1": 0.0,
                            "rel_err": 0.0, "n_eval": 0, "basis_id": "none", "rank": 0, "E_S1_side": np.nan}
                    keep(H.eval_cell(model, tok, mname, r, side, s, None, [], meta, generate=True), k)
                ref_cond = cond_name("P_span", 1.0)
                k = (mname, r.seed_id, side, ref_cond, "std", "score")
                if k not in done:
                    meta = {"cond": ref_cond, "cell": "P_span", "op": "proj", "site": "span", "level": 1.0, "alpha": 1.0, "feasible": True, "target_energy": np.nan, "e1": np.nan,
                            "rel_err": 0.0, "n_eval": 0, "basis_id": basis_id("P_span"), "rank": 1, "E_S1_side": np.nan}
                    row = H.eval_cell(model, tok, mname, r, side, s, H.make_factory(model, "proj", target, 1.0, "span"), H.positions_for_site(s, "span"), meta, generate=True)
                    if row["status"] == "ok":
                        row["E_S1_side"] = row["target_energy"] = row["e1"] = row["energy_per_pass"]
                    keep(row, k)
                ref = store.get(r.seed_id, side, ref_cond)
                if ref is None or ref["energy"] <= 0:
                    log.warning("%s %s %s: no reference energy; side skipped", mname, r.seed_id, side); continue
                E_S1 = ref["energy"]
                make_span = (lambda a: H.make_factory(model, "proj", target, a, "span"))
                pos_span = H.positions_for_site(s, "span")
                e1_cache = {"P_span": E_S1}
                for cell, (op, site, cell_levels) in CELLS.items():
                    if site == "ne" and (s.get("structural_failure") or not s.get("ne")):
                        continue
                    pos = H.positions_for_site(s, site)
                    rws = rows_for(cell, bases)
                    make_at = (lambda a, rws=rws, op=op, site=site: H.make_factory(model, op, rws, a, site))
                    for lvl in cell_levels:
                        if lvl not in levels or (cell == "P_span" and lvl == 1.0):
                            continue
                        cond = cond_name(cell, lvl)
                        k = (mname, r.seed_id, side, cond, "std", "score")
                        if k in done:
                            continue
                        rank = 64 if cell.startswith("Rk_") else 1
                        base = {"cond": cond, "cell": cell, "op": op, "site": site, "basis_id": basis_id(cell), "rank": rank, "E_S1_side": E_S1}
                        try:
                            if lvl == A1:
                                meta = {**base, "level": np.nan, "alpha": 1.0, "feasible": True, "target_energy": np.nan, "e1": np.nan, "rel_err": np.nan, "n_eval": 0, "calib_reason": "equal strength"}
                            else:
                                cal = H.calibrate_alpha(model, tok, s["text"], pos, make_at, lvl * E_S1, e1=e1_cache.get(cell), capped=(op != "steer"))
                                e1_cache.setdefault(cell, cal["e1"])
                                meta = {**base, "level": lvl, "alpha": cal["alpha"], "feasible": cal["feasible"], "target_energy": cal["target"], "e1": cal["e1"], "rel_err": cal["rel_err"],
                                        "n_eval": cal["n_eval"], "calib_reason": cal.get("reason", "")}
                        except Exception as ex:
                            log.exception("calibration failed %s %s %s", r.seed_id, side, cond)
                            keep({"model_name": mname, "seed_id": r.seed_id, "group": r.group, "side": side, "cond": cond, "variant": "std", "kind": "score", "status": "error",
                                  "error": "calibration: " + str(ex)[:200], "category": r.category, "template": r.template}, k)
                            continue
                        keep(H.eval_cell(model, tok, mname, r, side, s, make_at(meta["alpha"]), pos, meta, generate=(lvl not in NO_GENERATION)), k)
                # ---- the span erasure scaled down to what the site could achieve at f = 1
                if 1.0 in levels:
                    for cell, matched in MATCHED_TO.items():
                        got = store.get(r.seed_id, side, cond_name(cell, 1.0))
                        k = (mname, r.seed_id, side, matched, "std", "score")
                        if got is None or k in done:
                            continue
                        cal = H.calibrate_alpha(model, tok, s["text"], pos_span, make_span, min(got["energy"], E_S1), e1=E_S1)
                        meta = {"cond": matched, "cell": "P_span", "op": "proj", "site": "span", "level": cal["target"] / E_S1, "alpha": cal["alpha"], "feasible": cal["feasible"],
                                "target_energy": cal["target"], "e1": cal["e1"], "rel_err": cal["rel_err"], "n_eval": cal["n_eval"], "basis_id": basis_id("P_span"), "rank": 1,
                                "E_S1_side": E_S1, "matched_to": cond_name(cell, 1.0), "calib_reason": cal.get("reason", "")}
                        keep(H.eval_cell(model, tok, mname, r, side, s, make_span(cal["alpha"]), pos_span, meta, generate=True), k)
            # ---- bridges: fresh set; the edit at its own site on both sides, the patch at the span
            if group == "final" and sp["A"]["demo"] and sp["B"]["demo"]:
                plan = []
                if 1.0 in levels:
                    plan.append(("B", "B", "none", "none", 0.0))
                    plan += [(m, "P_span", "proj", "span", None) for m in MATCHED_TO.values()]
                for cell, (op, site, cell_levels) in CELLS.items():
                    if cell in NO_BRIDGE_CELLS:
                        continue
                    plan += [(cond_name(cell, l), cell, op, site, l) for l in cell_levels if l in levels and l in BRIDGE_LEVELS]
                for cond, cell, op, site, lvl in plan:
                    k = (mname, r.seed_id, "B", cond, "std", "bridge")
                    if k in done:
                        continue
                    if cell == "B":
                        fa = fb = None; pa = pb = []
                        meta = {"cond": "B", "cell": "B", "op": "none", "site": "none", "level": 0.0, "alpha": np.nan, "alpha_A": np.nan, "alpha_B": np.nan, "feasible": True, "basis_id": "none", "rank": 0}
                    else:
                        ca, cb = store.get(r.seed_id, "A", cond), store.get(r.seed_id, "B", cond)
                        if ca is None or cb is None:
                            continue
                        rws = rows_for(cell, bases)
                        fa = H.make_factory(model, op, rws, ca["alpha"], site); fb = H.make_factory(model, op, rws, cb["alpha"], site)
                        pa = H.positions_for_site(sp["A"], site); pb = H.positions_for_site(sp["B"], site)
                        meta = {"cond": cond, "cell": cell, "op": op, "site": site, "level": lvl if isinstance(lvl, float) else np.nan, "alpha": cb["alpha"], "alpha_A": ca["alpha"],
                                "alpha_B": cb["alpha"], "feasible": bool(ca["feasible"] and cb["feasible"]), "basis_id": basis_id(cell), "rank": 1}
                    rows.append(H.bridge_cell(model, tok, mname, r, sp, fa, fb, pa, pb, meta)); done.add(k)
            if rows:
                for x in rows:
                    x.setdefault("hh_seed", H.RANDOM_SEED)
                H.append_rows(path, rows); n_new += len(rows)
            if smoke:
                log.info("smoke %s %s: %d rows", mname, r.seed_id, len(rows))
    log.info("h1 %s pass %s: %d new rows in %.1f min", mname, which, n_new, (time.time() - t_start) / 60)
    (H.OUT / ("DONE_h1%s_%s.txt" % (which, mname))).write_text(H.utc_now(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=H.ALL_MODELS)
    ap.add_argument("--pass", dest="which", default="a", choices=["a", "b"])
    ap.add_argument("--groups", default="final,control")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.model, a.which, [g for g in a.groups.split(",") if g], a.smoke)


if __name__ == "__main__":
    main()
