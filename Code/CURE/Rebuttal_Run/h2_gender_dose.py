"""
h2_gender_dose.py -- LEACE and mean-difference baselines against the erasure at EQUAL energy
on the gender-flip set (rebuttal cycle, answers JwYi 'compare at similar strengths' and uG1a
W2). GPU, one of the two fresh models at a time.

The feedback cycle's Table 25 compared each eraser at its own alpha = 1, at which the coherent
gender LEACE removes about one to two per cent of the span erasure's energy. Here every
baseline is run at its native strength and the span erasure is scaled DOWN per side to that
baseline's energy, so each pair of rows differs in the operator only:

  B                     no edit
  P_span@{1,0.3,0.1,0.03}  the span erasure at fractions of its own energy (dose curve)
  LEACE_gl@native       official LEACE, gender label, large pool (g3), alpha = 1
  P_span@E(LEACE_gl)    the erasure matched to that row's energy on the same side
  LEACE_f3@native       the manuscript's man/woman LEACE (61 rows)      + P_span@E(LEACE_f3)
  MDg_span@native       mean-difference projection, gender label, large pool + P_span@E(MDg)
  MDf3_span@native      the manuscript's man/woman mean difference (45 pairs) + P_span@E(MDf3)
  CV_span@1             the contrast steering vector at E_S1 (operator comparison at equal energy)

Readouts: T, length-normalised T, option accuracy, generation accuracy, energy, and C_answer
(bridge) for every cell. Rows: results/rebuttal_20261107/h2_gender_<model>.parquet.
Usage: python h2_gender_dose.py --model gemma-2-2b-it [--smoke]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import hh_common as H

log = H.log
P_LEVELS = (1.0, 0.3, 0.1, 0.03)


def spans_for(tok, man):
    import f1_scoring as S
    out = {"seeds": {}}
    for r in man.itertuples(index=False):
        rec = {}
        for side in ("A", "B"):
            up = getattr(r, "prompt_%s" % side); desc = getattr(r, "descriptor_%s" % side)
            text = S.chat_text(tok, up)
            demo = S.demographic_span(tok, text, up, desc)
            rec[side] = {"text": text, "demo": demo["positions"], "ne": [], "n_prefill": len(tok(text, add_special_tokens=True)["input_ids"]),
                         "structural_failure": not demo["positions"]}
        out["seeds"][r.seed_id] = rec
    return out


def run(mname: str, smoke: bool) -> None:
    H.ensure_out()
    path = H.OUT / ("h2_gender_%s.parquet" % mname)
    man = H.gender_manifest(mname)
    if man is None:
        raise SystemExit("no gender-flip manifest for %s" % mname)
    if smoke:
        man = man.head(2)
    cfg, model, tok = H.load_model(mname)
    sp_all = spans_for(tok, man)
    target, prov = H.target_basis(mname)
    ops = {"LEACE_gl": ("leace", H.leace_erasers(mname, "gender_large")), "LEACE_f3": ("leace", H.leace_erasers(mname, "f3")),
           "MDg_span": ("proj", H.md_basis(mname, "gender")), "MDf3_span": ("proj", H.md_basis(mname, "f3")), "CV_span": ("steer", H.cv_vector(mname))}
    ops = {k: v for k, v in ops.items() if v[1] is not None}
    H.write_json({"model": mname, "baselines": sorted(ops), "p_levels": list(P_LEVELS), "basis": prov, "n_seeds": int(len(man)),
                  "model_cfg": {k: cfg.get(k) for k in ("hf_id", "model_class", "attn_implementation", "dtype")}, "utc": H.utc_now()},
                 H.OUT / ("h2_protocol_%s.json" % mname))
    existing = H.read_rows(path)
    done = set(tuple(r) for r in existing[H.ROW_KEY].itertuples(index=False, name=None)) if len(existing) else set()
    alphas = {}
    if len(existing):
        for x in existing[(existing.kind == "score") & (existing.status == "ok")].itertuples(index=False):
            alphas[(x.seed_id, x.side, x.cond)] = (float(x.alpha), float(x.energy_per_pass), bool(x.feasible))
    t0 = time.time(); n_new = 0
    for r in man.itertuples(index=False):
        sp = sp_all["seeds"][r.seed_id]
        rows = []
        plan = []           # (cond, op, rows, alpha_or_None(target cond), site)
        for side in ("A", "B"):
            s = sp[side]
            if s["structural_failure"]:
                continue
            pos = H.positions_for_site(s, "span")

            def do(cond, op, rws, alpha, meta_extra, generate=True):
                key = (mname, r.seed_id, side, cond, "std", "score")
                if key in done:
                    return
                fac = H.make_factory(model, op, rws, alpha, "span") if op != "none" else None
                meta = {"cond": cond, "cell": cond.split("@")[0], "op": op, "site": "span" if op != "none" else "none", "alpha": alpha, "rank": 0 if op == "none" else 1}
                meta.update(meta_extra)
                row = H.eval_cell(model, tok, mname, r, side, s, fac, pos if op != "none" else [], meta, generate=generate)
                rows.append(row); done.add(key)
                if row["status"] == "ok":
                    alphas[(r.seed_id, side, cond)] = (float(alpha), float(row["energy_per_pass"]), bool(meta.get("feasible", True)))

            do("B", "none", None, 0.0, {"level": 0.0, "feasible": True, "target_energy": 0.0, "basis_id": "none"})
            do("P_span@1", "proj", target, 1.0, {"level": 1.0, "feasible": True, "target_energy": np.nan, "basis_id": "target_rank1"})
            ref = alphas.get((r.seed_id, side, "P_span@1"))
            if ref is None or ref[1] <= 0:
                continue
            E_S1 = ref[1]
            e1_target = E_S1
            make_P = (lambda a: H.make_factory(model, "proj", target, a, "span"))
            for lvl in P_LEVELS[1:]:
                cond = "P_span@%g" % lvl
                if (mname, r.seed_id, side, cond, "std", "score") in done:
                    continue
                cal = H.calibrate_alpha(model, tok, s["text"], pos, make_P, lvl * E_S1, e1=e1_target)
                do(cond, "proj", target, cal["alpha"], {"level": lvl, "feasible": cal["feasible"], "target_energy": cal["target"], "rel_err": cal["rel_err"],
                                                        "n_eval": cal["n_eval"], "basis_id": "target_rank1", "E_S1_side": E_S1}, generate=(lvl in (1.0, 0.1)))
            for bname, (op, rws) in ops.items():
                native = "%s@native" % bname if op != "steer" else "CV_span@1"
                if op == "steer":
                    cal = H.calibrate_alpha(model, tok, s["text"], pos, (lambda a, rws=rws: H.make_factory(model, "steer", rws, a, "span")), E_S1, capped=False)
                    do(native, "steer", rws, cal["alpha"], {"level": 1.0, "feasible": cal["feasible"], "target_energy": cal["target"], "rel_err": cal["rel_err"],
                                                          "n_eval": cal["n_eval"], "basis_id": "CV_vector_feedback_cycle", "E_S1_side": E_S1})
                    continue
                do(native, op, rws, 1.0, {"level": np.nan, "feasible": True, "target_energy": np.nan, "basis_id": bname, "E_S1_side": E_S1})
                base = alphas.get((r.seed_id, side, native))
                if base is None:
                    continue
                matched = "P_span@E(%s)" % bname
                if (mname, r.seed_id, side, matched, "std", "score") in done:
                    continue
                cal = H.calibrate_alpha(model, tok, s["text"], pos, make_P, base[1], e1=e1_target)
                do(matched, "proj", target, cal["alpha"], {"level": base[1] / E_S1 if E_S1 else np.nan, "feasible": cal["feasible"], "target_energy": cal["target"],
                                                            "rel_err": cal["rel_err"], "n_eval": cal["n_eval"], "basis_id": "target_rank1", "E_S1_side": E_S1,
                                                            "matched_to": native})
        # ---- bridges for every cell present on both sides
        if not (sp["A"]["structural_failure"] or sp["B"]["structural_failure"]):
            conds = sorted({c for (sid, side, c) in alphas if sid == r.seed_id} | {"B"})
            for cond in conds:
                if (mname, r.seed_id, "B", cond, "std", "bridge") in done:
                    continue
                if cond == "B":
                    fa = fb = None; meta = {"cond": "B", "cell": "B", "op": "none", "site": "none", "alpha": np.nan, "alpha_A": np.nan, "alpha_B": np.nan, "feasible": True}
                else:
                    ca, cb = alphas.get((r.seed_id, "A", cond)), alphas.get((r.seed_id, "B", cond))
                    if ca is None or cb is None:
                        continue
                    cell = cond.split("@")[0]
                    if cell == "P_span":
                        op, rws = "proj", target
                    else:
                        op, rws = ops[cell]
                    fa = H.make_factory(model, op, rws, ca[0], "span"); fb = H.make_factory(model, op, rws, cb[0], "span")
                    meta = {"cond": cond, "cell": cell, "op": op, "site": "span", "alpha": cb[0], "alpha_A": ca[0], "alpha_B": cb[0], "feasible": bool(ca[2] and cb[2])}
                rows.append(H.bridge_cell(model, tok, mname, r, sp, fa, fb, sp["A"]["demo"] if fa else [], sp["B"]["demo"] if fb else [], meta))
                done.add((mname, r.seed_id, "B", cond, "std", "bridge"))
        if rows:
            H.append_rows(path, rows); n_new += len(rows)
    log.info("h2 %s: %d new rows in %.1f min", mname, n_new, (time.time() - t0) / 60)
    (H.OUT / ("DONE_h2_%s.txt" % mname)).write_text(H.utc_now(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=H.FRESH_MODELS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.model, a.smoke)


if __name__ == "__main__":
    main()
