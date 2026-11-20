"""
x2_extension_analysis.py -- the declared tests of results/extension_20261120/x_protocol.json. CPU.

Primary family (Holm over three): for T, T_mean and flip, the per-template mean over the four models of
(B - P_span@1), one-sided (greater), studentised bootstrap over templates (Efron and Tibshirani 1993;
Holm 1979). Secondary: per model and readout, B - P_span@1 and B - CV_span@1 (one-sided, Holm over the
four models), and P_span@1 - CV_span@1 (two-sided, descriptive). The inspected 160-pair set is combined
only in an exploratory block.

Output: results/extension_20261120/x_tests.csv and x_report.txt. Usage: python x2_extension_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hh_common as H  # noqa: E402

X = H.K.RESULTS / "extension_20261120"
REB = H.K.RESULTS / "rebuttal_20261107"
MODELS = ["llama-3.1-8b-instruct", "gemma-2-2b-it", "qwen2.5-7b-instruct", "phi-4-mini-instruct"]
READOUTS = ("T", "T_mean", "flip")
lines = []


def say(s=""):
    print(s); lines.append(s)


def holm(ps):
    order = np.argsort(ps); out = [0.0] * len(ps); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i])); out[i] = run
    return out


def per_pair(d: pd.DataFrame, group: str) -> pd.DataFrame:
    """One row per (seed, cond): T, T_mean and the flip indicator."""
    s = d[(d.kind == "score") & (d.status == "ok") & (d.group == group) & d.cond.isin(["B", "P_span@1", "CV_span@1"])]
    pt = H.pair_table(s)
    recs = []
    for (seed, cond), g in s.groupby(["seed_id", "cond"]):
        a, b = g[g.side == "A"], g[g.side == "B"]
        if len(a) and len(b) and bool(a.iloc[0].gen_valid) and bool(b.iloc[0].gen_valid):
            recs.append({"seed_id": seed, "cond": cond, "flip": float(a.iloc[0].gen_idx != b.iloc[0].gen_idx)})
    f = pd.DataFrame(recs)
    out = pt[["seed_id", "cond", "template", "T", "T_mean"]].merge(f, on=["seed_id", "cond"], how="left")
    return out


def diffs(pp: pd.DataFrame, readout: str, plus: str, minus: str) -> pd.DataFrame:
    w = pp.pivot_table(index=["seed_id", "template"], columns="cond", values=readout, aggfunc="first").reset_index()
    if plus not in w or minus not in w:
        return pd.DataFrame(columns=["template", "d"])
    w = w.dropna(subset=[plus, minus])
    return pd.DataFrame({"template": w.template, "d": w[plus] - w[minus]})


def test(dd: pd.DataFrame, alternative: str) -> dict:
    r = H.paired(dd.d.to_numpy(), dd.template.to_numpy(), alternative=alternative)
    return r


def main() -> None:
    pp = {}
    for m in MODELS:
        p = X / ("h1_dose_%s.parquet" % m)
        if p.exists():
            pp[m] = per_pair(H.read_rows(p), "ext")
    say("extension tests; models with rows: %s" % [m for m in pp])
    for m, x in pp.items():
        say("  %-24s pairs %d templates %d" % (m, x.seed_id.nunique(), x.template.nunique()))
    rows = []
    # primary: pooled over the four models, per template
    if len(pp) == 4:
        prim = []
        for ro in READOUTS:
            per_t = []
            for m in MODELS:
                dd = diffs(pp[m], ro, "B", "P_span@1")
                per_t.append(dd.groupby("template").d.mean().rename(m))
            t = pd.concat(per_t, axis=1).dropna()
            r = H.paired(t.mean(axis=1).to_numpy(), t.index.to_numpy(), alternative="greater")
            prim.append({"family": "primary", "model_name": "pooled", "readout": ro, "statement": "B - P_span@1 > 0 (mean over four models)", **r})
        for r, ph in zip(prim, holm([r["p"] for r in prim])):
            r["p_holm"] = ph
        rows += prim
    for fam, plus, minus, alt in (("per_model_erasure", "B", "P_span@1", "greater"), ("per_model_vector", "B", "CV_span@1", "greater"),
                                  ("erasure_vs_vector", "P_span@1", "CV_span@1", "two-sided")):
        for ro in READOUTS:
            fr = []
            for m, x in pp.items():
                dd = diffs(x, ro, plus, minus)
                if len(dd) < 4:
                    continue
                fr.append({"family": fam, "model_name": m, "readout": ro, "statement": "%s - %s%s" % (plus, minus, " > 0" if alt == "greater" else ""), **test(dd, alt)})
            for r, ph in zip(fr, holm([r["p"] for r in fr]) if fr else []):
                r["p_holm"] = ph
            rows += fr
    for r in rows:
        greater = r["statement"].endswith("> 0")
        if r["p_holm"] < 0.05 and greater:
            r["verdict"] = "supported"
        elif greater and r["hi"] < 0:
            r["verdict"] = "reversal"
        elif not greater:
            r["verdict"] = "descriptive"
        else:
            r["verdict"] = "limited"
    t = pd.DataFrame(rows)
    if len(t):
        t.to_csv(X / "x_tests.csv", index=False)
        say("")
        for r in t.itertuples(index=False):
            say("%-18s %-24s %-7s %-36s est %+.4f [%+.4f, %+.4f] n=%d p_holm %.4f %s" % (r.family, r.model_name.split("-")[0], r.readout, r.statement, r.estimate, r.lo, r.hi, r.n_pairs, r.p_holm, r.verdict))
    # exploratory: the inspected 160 pairs combined with the new pairs, per model
    say("\nexploratory: inspected 160 pairs combined with the new pairs (B - P_span@1, one-sided)")
    for m, x in pp.items():
        old = per_pair(H.read_rows(REB / ("h1_dose_%s.parquet" % m)), "final")
        both = pd.concat([old, x])
        for ro in READOUTS:
            r = test(diffs(both, ro, "B", "P_span@1"), "greater")
            say("  %-24s %-7s est %+.4f [%+.4f, %+.4f] p %.4f n=%d" % (m, ro, r["estimate"], r["lo"], r["hi"], r["p"], r["n_pairs"]))
    (X / "x_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
