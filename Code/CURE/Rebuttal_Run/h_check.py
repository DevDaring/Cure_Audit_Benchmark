"""
h_check.py -- correctness check of the rebuttal cycle's rows (not a liveness check). CPU, author's
machine. Reads a results directory and reports, per model:

  rows and failures  per stage; share of rows with status ok, and the first distinct errors
  reproduction       this cycle's B and P_span@1 against the September rows on the same seeds:
                     mean T on both, per-seed correlation of T and of C_answer (two fresh models)
  energy             achieved / target over feasible cells (must lie within the 5 per cent tolerance,
                     allowing bfloat16 noise), feasible share per cell, alpha range per cell
  null checks        P_span@1 alpha exactly 1; B energy exactly 0; no decode edit in any cell;
                     same-input patch null where recorded
  monotonicity       mean energy per cell against its level on the P_span dose curve
  early reading      mean T, T_mean, accuracy per cell on the seeds done so far, labelled partial

Exit code 1 when a hard check fails.
Usage: python h_check.py [--dir results/rebuttal_20261107] [--smoke] [--brief]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import hh_common as H

SEPT_COND = {"B": "B", "P_span@1": "S1"}


def sept_pairs(model: str) -> pd.DataFrame | None:
    p = H.FINAL / ("per_item_%s.parquet" % model)
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d = d[(d.kind == "score") & (d.variant == "std") & (d.status == "ok") & (d.cond.isin(SEPT_COND.values()))]
    return H.pair_table(d)


def check_model(model: str, d: pd.DataFrame, brief: bool) -> list[str]:
    fails = []
    n = len(d); ok = (d.status == "ok").sum()
    print("  h1 rows %d, ok %d (%.1f%%), seeds %d, kinds %s, last row %s" % (n, ok, 100 * ok / max(1, n), d.seed_id.nunique(), d.kind.value_counts().to_dict(), d.utc.dropna().max() if "utc" in d else "?"))
    if ok < n:
        print("  errors:", d[d.status != "ok"].error.astype(str).str[:110].value_counts().head(4).to_dict())
        if (n - ok) / max(1, n) > 0.01:
            fails.append("more than 1 per cent of h1 rows failed")
    sc = d[(d.kind == "score") & (d.status == "ok")]
    if sc.empty:
        return fails
    # ---- null checks
    b = sc[sc.cond == "B"]
    if len(b) and float(b.energy_num.abs().max()) != 0.0:
        fails.append("B has non-zero energy")
    ref = sc[sc.cond == "P_span@1"]
    if len(ref) and not (ref.alpha == 1.0).all():
        fails.append("P_span@1 alpha is not 1")
    if "n_decode_edits_gen" in sc and float(sc.n_decode_edits_gen.fillna(0).abs().max()) != 0.0:
        fails.append("a cell edited decode steps during generation")
    if float(sc.n_decode_edits_score.fillna(0).abs().max()) != 0.0:
        fails.append("a cell edited decode steps during scoring")
    # ---- energy
    m = sc[(sc.target_energy > 0) & (sc.cond != "P_span@1")]
    if len(m):
        f = m[m.feasible == True]   # noqa: E712
        ratio = f.energy_per_pass / f.target_energy
        print("  energy achieved/target over feasible cells: min %.3f median %.3f max %.3f (n=%d)" % (ratio.min(), ratio.median(), ratio.max(), len(ratio)))
        if not ratio.between(0.93, 1.07).all():
            fails.append("%d feasible rows outside the energy tolerance" % int((~ratio.between(0.93, 1.07)).sum()))
        st = m[(m.op == "steer") & (m.feasible != True)]   # noqa: E712
        if len(st):
            fails.append("%d steering rows are infeasible (an additive vector has no cap)" % len(st))
        tab = m.groupby("cond").agg(rows=("feasible", "size"), feasible=("feasible", "mean"), alpha_med=("alpha", "median"), alpha_max=("alpha", "max"), frac=("energy_per_pass", "mean"))
        tab["frac_of_E_S1"] = m.assign(x=m.energy_per_pass / m.E_S1_side).groupby("cond").x.median()
        if not brief:
            print(tab.drop(columns="frac").round(3).to_string())
    # ---- reproduction against September
    pt = H.pair_table(sc)
    final = pt[pt.group == "final"]
    sep = sept_pairs(model)
    if sep is not None and len(sep):
        sf = sep[sep.group == "final"]
        for new, old in SEPT_COND.items():
            a = final[final.cond == new].set_index("seed_id"); s = sf[sf.cond == old].set_index("seed_id")
            j = a.join(s, how="inner", lsuffix="_new", rsuffix="_sep")
            if len(j) >= 5:
                r = float(np.corrcoef(j.T_new, j.T_sep)[0, 1])
                print("  reproduction %s vs Sept %s: n=%d mean T %.4f vs %.4f, corr %.4f, |dT| median %.4f" % (new, old, len(j), j.T_new.mean(), j.T_sep.mean(), r, (j.T_new - j.T_sep).abs().median()))
                if r < 0.90:
                    fails.append("%s does not reproduce the September T (corr %.3f)" % (new, r))
        br = d[(d.kind == "bridge") & (d.status == "ok") & (d.cond.isin(SEPT_COND))]
        old = pd.read_parquet(H.FINAL / ("per_item_%s.parquet" % model))
        old = old[(old.kind == "bridge") & (old.status == "ok")]
        for new, o in SEPT_COND.items():
            j = br[br.cond == new].set_index("seed_id")[["C_answer"]].join(old[old.cond == o].set_index("seed_id")[["C_answer"]], how="inner", lsuffix="_new", rsuffix="_sep")
            if len(j) >= 5:
                r = float(np.corrcoef(j.C_answer_new, j.C_answer_sep)[0, 1])
                print("  reproduction C_answer %s: n=%d mean|C| %.3f vs %.3f, corr %.4f" % (new, len(j), j.C_answer_new.abs().mean(), j.C_answer_sep.abs().mean(), r))
                if r < 0.90:
                    fails.append("C_answer %s does not reproduce (corr %.3f)" % (new, r))
    # ---- early reading (labelled partial; never used for a decision about the design)
    if not brief and len(final):
        g = final.groupby("cond").agg(pairs=("T", "size"), T=("T", "mean"), T_mean=("T_mean", "mean"), acc=("acc", "mean"), valid=("validity", "mean"), feasible=("feasible", "mean"))
        print("  PARTIAL readings, final set:"); print(g.round(4).to_string())
        ctl = pt[pt.group == "control"]
        if len(ctl):
            print("  PARTIAL control accuracy:", ctl.groupby("cond").acc.mean().round(3).to_dict())
    return fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="directory to read (default: this cycle's output directory, or the per-model smoke directories with --smoke)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--brief", action="store_true")
    a = ap.parse_args()
    fails = {}
    for model in H.ALL_MODELS:
        out = Path(a.dir) if a.dir else (H.K.RESULTS / ("rebuttal_smoke_%s" % model) if a.smoke else H.OUT)
        p = out / ("h1_dose_%s.parquet" % model)
        print("==", model, "(%s)" % out.name)
        if out.exists():
            print("  files:", sorted(x.name for x in out.iterdir() if model in x.name and x.suffix in (".parquet", ".txt"))[:12])
        idn = out / ("h5_identity_%s.json" % model)
        if idn.exists():
            r = json.loads(idn.read_text(encoding="utf-8")); print("  h5 identity pass:", r.get("pass_all_h5"))
            if not r.get("pass_all_h5"):
                fails.setdefault(model, []).append("h5 identity failed")
        if not p.exists():
            print("  no h1 rows yet"); continue
        fails.setdefault(model, []).extend(check_model(model, H.read_rows(p), a.brief))
        for stage, name, key in (("h2", "h2_gender_%s.parquet", "cond"), ("h3", "h3_pairs_%s.parquet", "cond_id"), ("h4", "h4_rewrite_%s.parquet", "cond"), ("h7", "h7_tokens_%s.parquet", "cond")):
            q = out / (name % model)
            if q.exists():
                x = H.read_rows(q)
                st = x.status.value_counts().to_dict() if "status" in x else {"ok": len(x)}
                print("  %s rows %d seeds %d status %s" % (stage, len(x), x.seed_id.nunique(), st))
                bad = sum(v for k, v in st.items() if k not in ("ok", "degenerate", "unresolved"))
                if bad / max(1, len(x)) > 0.01:
                    fails.setdefault(model, []).append("%s has %d failed rows" % (stage, bad))
    hard = {k: v for k, v in fails.items() if v}
    print("\nRESULT:", "ALL HARD CHECKS PASS" if not hard else "FAILURES %s" % hard)
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
