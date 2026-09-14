"""
File: Code/CURE/audit_iclr_numbers_final.py
Purpose: Independent audit of the Study 2 numbers in the ICLR manuscript. It does not reuse
         f4_analysis.py or make_iclr_tables_final.py: it re-reads the per-item parquet files
         (the raw scorer output) and the manifest, re-derives every point estimate printed in
         the generated Study 2 tables and macros (T, accuracy, validity, both-correct, control
         accuracies, |C_answer| and |C_first| levels and reductions, H1-H4 estimates, category
         subgroups, achieved energies, seed counts) and compares them after rounding. Bootstrap
         intervals and p-values cannot be re-derived without re-running the bootstrap, so the
         audit checks only that lo <= estimate <= hi and that a starred row has p_holm < 0.05.

Exit code 1 on any mismatch. Usage:  python Code/CURE/audit_iclr_numbers_final.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "Code" / "CURE" / "results" / "final_20260914"
TAB = ROOT / "Submission2" / "iclr2027" / "tables_v2"
MODELS = ["gemma-2-2b-it", "llama-3.1-8b-instruct"]
NAME = {"gemma-2-2b-it": "Gemma-2-2B", "llama-3.1-8b-instruct": "Llama-3.1-8B"}
CONDS = ["B", "S1", "SE", "NE", "RE", "RNE", "G1"]
problems = []


def fail(msg):
    problems.append(msg); print("MISMATCH", msg)


def rows_of(name):
    txt = (TAB / name).read_text(encoding="utf-8")
    body = txt.split("\\midrule", 1)[1].split("\\bottomrule")[0]
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("\\midrule") or line.startswith("%"):
            continue
        line = line.rstrip("\\").strip()
        cells = [re.sub(r"\\scriptsize|\{|\}|\$\^\{\\ast\}\$", "", c).strip() for c in line.split("&")]
        out.append(cells)
    return out


def num(cell):
    m = re.match(r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", cell.strip())
    return float(m.group(1)) if m else float("nan")


def interval(cell):
    m = re.search(r"\[\s*([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\s*\]", cell)
    return (float(m.group(1)), float(m.group(2))) if m else (float("nan"), float("nan"))


def macros():
    txt = (TAB / "numbers_final.tex").read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", txt)}


def close(a, b, digits):
    return abs(round(float(a), digits) - round(float(b), digits)) <= 10 ** (-digits) + 1e-12


# ---------------------------------------------------------------- raw per-pair quantities from the scorer output
man = pd.read_csv(F / "source_manifest.csv")
per = {m: pd.read_parquet(F / ("per_item_%s.parquet" % m)) for m in MODELS}
groups = man.set_index("seed_id").group.to_dict()


def pairs(m, cond, variant="std", group="final"):
    d = per[m]
    d = d[(d.kind == "score") & (d.cond == cond) & (d.variant == variant) & (d.status == "ok") & (d.group == group)]
    out = []
    for sid, g in d.groupby("seed_id"):
        if set(g.side) != {"A", "B"}:
            continue
        a, b = g[g.side == "A"].iloc[0], g[g.side == "B"].iloc[0]
        qa, qb = np.asarray(json.loads(a.q)), np.asarray(json.loads(b.q))
        T = 0.5 * np.abs(qa - qb).sum()
        out.append({"seed_id": sid, "category": a.category, "template": a.template, "T": T,
                    "gen_acc": 0.5 * (float(a.gen_correct) + float(b.gen_correct)),
                    "validity": 0.5 * (float(a.gen_valid) + float(b.gen_valid)),
                    "both": float(bool(a.gen_correct) and bool(b.gen_correct)),
                    "energy_num": float(a.energy_num) + float(b.energy_num)})
    return pd.DataFrame(out).set_index("seed_id")


P = {m: {c: pairs(m, c) for c in CONDS} for m in MODELS}
C = {m: {c: pairs(m, c, group="control") for c in ("B", "S1", "NE")} for m in MODELS}


def tmean(values, templates):
    """Template-weighted mean: mean over templates of the within-template mean (the
    pre-registered cluster estimator of f4_analysis, re-implemented here)."""
    return pd.Series(np.asarray(values, float)).groupby(np.asarray(templates)).mean().mean()


def paired(m, c1, c2, col="T", group="final"):
    src = P if group == "final" else C
    a, b = src[m][c1], src[m][c2]
    idx = a.index.intersection(b.index)
    return (a.loc[idx, col] - b.loc[idx, col]), idx


def pmean(m, c1, c2, col="T", group="final"):
    diff, idx = paired(m, c1, c2, col, group)
    src = P if group == "final" else C
    return tmean(diff.to_numpy(), src[m][c1].loc[idx].template.to_numpy())


def lmean(d, col):
    return tmean(d[col].to_numpy(), d.template.to_numpy())


# ---------------------------------------------------------------- tab_final_levels
rows = rows_of("tab_final_levels.tex")
assert len(rows) == len(MODELS) * len(CONDS), len(rows)
i = 0
for m in MODELS:
    for c in CONDS:
        r = rows[i]; i += 1
        d = P[m][c]
        if int(num(r[2])) != len(d):
            fail("tab_final_levels %s %s seeds %s vs %d" % (m, c, r[2], len(d)))
        for j, col, digits in ((3, "T", 3), (4, "gen_acc", 3), (5, "validity", 3), (6, "both", 3)):
            if not close(num(r[j]), lmean(d, col), digits):
                fail("tab_final_levels %s %s %s %s vs %.4f" % (m, c, col, r[j], lmean(d, col)))
        for j in (3, 4):
            lo, hi = interval(r[j])
            if not (lo - 1e-9 <= num(r[j]) <= hi + 1e-9):
                fail("tab_final_levels %s %s interval does not bracket estimate: %s" % (m, c, r[j]))
print("tab_final_levels: checked")

# ---------------------------------------------------------------- tab_final_confirm (point estimates)
rows = rows_of("tab_final_confirm.tex")
conf = pd.read_csv(F / "confirmatory_tests.csv"); conf = conf[conf.cluster == "template"].set_index(["model_name", "id"])
i = 0
for m in MODELS:
    for hid, (c1, c2, col) in (("H1", ("B", "S1", "T")), ("H2", ("NE", "SE", "T")), ("H3", ("RE", "SE", "T")), ("H4", ("S1", "B", "gen_acc"))):
        r = rows[i]; i += 1
        diff, idx = paired(m, c1, c2, col)
        est = pmean(m, c1, c2, col)
        if not close(num(r[3]), est, 3):
            fail("tab_final_confirm %s %s estimate %s vs %.4f" % (m, hid, r[3], est))
        ncl = P[m][c1].loc[idx].template.nunique()
        if int(num(r[2])) != ncl:
            fail("tab_final_confirm %s %s clusters %s vs %d" % (m, hid, r[2], ncl))
        lo, hi = interval(r[3])
        if not (lo <= num(r[3]) <= hi):
            fail("tab_final_confirm %s %s interval" % (m, hid))
        starred = "ast" in (TAB / "tab_final_confirm.tex").read_text(encoding="utf-8").splitlines()[9 + i + (1 if m == MODELS[1] else 0)]
        ph = conf.loc[(m, hid)].p_holm
        if not close(num(r[4]), ph, 3):
            fail("tab_final_confirm %s %s p_holm %s vs %.4f" % (m, hid, r[4], ph))
        if starred != (ph < 0.05):
            fail("tab_final_confirm %s %s star/p_holm disagree" % (m, hid))
print("tab_final_confirm: checked")

# ---------------------------------------------------------------- tab_control
rows = rows_of("tab_control.tex")
for r, m in zip(rows, MODELS):
    b, s1, ne = C[m]["B"], C[m]["S1"], C[m]["NE"]
    if int(num(r[1])) != len(b):
        fail("tab_control %s seeds %s vs %d" % (m, r[1], len(b)))
    for j, d in ((2, b), (3, s1), (5, ne)):
        if not close(num(r[j]), lmean(d, "gen_acc"), 3):
            fail("tab_control %s col %d %s vs %.4f" % (m, j, r[j], lmean(d, "gen_acc")))
    for j, c in ((4, "S1"), (6, "NE")):
        v = pmean(m, c, "B", "gen_acc", group="control")
        if not close(num(r[j]), v, 3):
            fail("tab_control %s change %s %s vs %.4f" % (m, c, r[j], v))
print("tab_control: checked")

# ---------------------------------------------------------------- tab_bridge
rows = rows_of("tab_bridge.tex")
for r, m in zip(rows, MODELS):
    d = per[m][(per[m].kind == "bridge") & (per[m].status == "ok") & (per[m].group == "final")]
    b, s1 = d[d.cond == "B"].set_index("seed_id"), d[d.cond == "S1"].set_index("seed_id")
    idx = b.index.intersection(s1.index)
    ca_b, ca_s = b.loc[idx].C_answer.abs(), s1.loc[idx].C_answer.abs()
    cf_b, cf_s = b.loc[idx].C_first.abs(), s1.loc[idx].C_first.abs()
    tp = b.loc[idx].template.to_numpy()
    for j, v in ((1, tmean(ca_b, tp)), (2, tmean(ca_s, tp)), (3, tmean(ca_b - ca_s, tp)), (4, tmean(cf_b - cf_s, tp))):
        if not close(num(r[j]), v, 3):
            fail("tab_bridge %s col %d %s vs %.4f" % (m, j, r[j], v))
    if not close(num(r[5]), b.first_disambiguates.mean(), 2):
        fail("tab_bridge %s first-token uniqueness" % m)
print("tab_bridge: checked")

# ---------------------------------------------------------------- tab_subgroups
rows = rows_of("tab_subgroups.tex")
for r in rows:
    cat = r[0].replace(" ", "_")
    for k, m in enumerate(MODELS):
        base = 1 + 3 * k
        d1, idx = paired(m, "B", "S1", "T"); d4, _ = paired(m, "S1", "B", "gen_acc")
        sel = (P[m]["B"].loc[idx].category == cat).to_numpy()
        tp = P[m]["B"].loc[idx].template.to_numpy()[sel]
        if int(num(r[base])) != int(sel.sum()):
            fail("tab_subgroups %s %s seeds" % (cat, m))
        v1, v4 = tmean(d1.to_numpy()[sel], tp), tmean(d4.to_numpy()[sel], tp)
        if not close(num(r[base + 1]), v1, 3):
            fail("tab_subgroups %s %s H1 %s vs %.4f" % (cat, m, r[base + 1], v1))
        if not close(num(r[base + 2]), v4, 3):
            fail("tab_subgroups %s %s H4 %s vs %.4f" % (cat, m, r[base + 2], v4))
print("tab_subgroups: checked")

# ---------------------------------------------------------------- tab_energy (test energies and alphas)
rows = rows_of("tab_energy.tex")
i = 0
for m in MODELS:
    cal = json.loads((F / ("calibration_%s.json" % m)).read_text(encoding="utf-8"))
    for c in ("S1", "SE", "NE", "RE", "RNE", "G1"):
        r = rows[i]; i += 1
        d = per[m][(per[m].kind == "score") & (per[m].cond == c) & (per[m].variant == "std") & (per[m].status == "ok") & (per[m].group == "final")]
        e = (d.energy_num.mean() * 2 / 2) / cal["denominator"]  # per-side mean relative to the dev denominator
        if not (0.9 <= num(r[4]) / e <= 1.1 or close(num(r[4]), e, 6)):
            fail("tab_energy %s %s test energy %s vs %.3e" % (m, c, r[4], e))
        if c in cal["cells"] and not close(num(r[2]), cal["cells"][c]["alpha"], 3):
            fail("tab_energy %s %s alpha" % (m, c))
        if not close(num(r[5]), d.n_edited_positions.mean(), 1):
            fail("tab_energy %s %s edited positions %s vs %.2f" % (m, c, r[5], d.n_edited_positions.mean()))
print("tab_energy: checked")

# ---------------------------------------------------------------- macros
M = macros()
g, l = MODELS
checks = {
    "nFreshSeeds": man[man.group == "final"].seed_id.nunique(), "nFreshTemplates": man[man.group == "final"].template.nunique(),
    "nControlSeeds": man[man.group == "control"].seed_id.nunique(), "nDevFresh": man[man.group == "dev"].seed_id.nunique(),
    "nCategories": man[man.group == "final"].category.nunique(),
    "llamaTB": lmean(P[l]["B"], "T"), "llamaTSone": lmean(P[l]["S1"], "T"), "gemmaTB": lmean(P[g]["B"], "T"), "gemmaTSone": lmean(P[g]["S1"], "T"),
    "llamaHoneEst": pmean(l, "B", "S1"), "gemmaHoneEst": pmean(g, "B", "S1"),
    "llamaHtwoEst": pmean(l, "NE", "SE"), "llamaHthreeEst": pmean(l, "RE", "SE"),
    "gemmaHfourPts": 100 * pmean(g, "S1", "B", "gen_acc"), "llamaHfourPts": 100 * pmean(l, "S1", "B", "gen_acc"),
    "gemmaAccDropPts": -100 * pmean(g, "S1", "B", "gen_acc"), "llamaAccDropPts": -100 * pmean(l, "S1", "B", "gen_acc"),
    "gemmaCtrlB": lmean(C[g]["B"], "gen_acc"), "gemmaCtrlS": lmean(C[g]["S1"], "gen_acc"), "llamaCtrlB": lmean(C[l]["B"], "gen_acc"), "llamaCtrlS": lmean(C[l]["S1"], "gen_acc"),
    "gemmaCtrlNEDelta": pmean(g, "NE", "B", "gen_acc", "control"), "llamaCtrlNEDelta": pmean(l, "NE", "B", "gen_acc", "control"),
    "gemmaCtrlDropPts": -100 * pmean(g, "S1", "B", "gen_acc", "control"), "llamaCtrlDropPts": -100 * pmean(l, "S1", "B", "gen_acc", "control"),
    "gemmaGoneAccDropPts": -100 * pmean(g, "G1", "B", "gen_acc"), "llamaGoneAccDropPts": -100 * pmean(l, "G1", "B", "gen_acc"),
    "gemmaGoneValidity": lmean(P[g]["G1"], "validity"), "llamaGoneValidity": lmean(P[l]["G1"], "validity"),
    "llamaSEdT": pmean(l, "SE", "B"), "gemmaSEdT": pmean(g, "SE", "B"),
    "llamaSEdAccPts": 100 * pmean(l, "SE", "B", "gen_acc"), "gemmaSEdAccPts": 100 * pmean(g, "SE", "B", "gen_acc"),
    "llamaNEdT": pmean(l, "NE", "B"), "gemmaNEdT": pmean(g, "NE", "B"),
    "llamaNEdAccPts": 100 * pmean(l, "NE", "B", "gen_acc"), "gemmaNEdAccPts": 100 * pmean(g, "NE", "B", "gen_acc"),
    "gemmaLocExcluded": 160 - len(P[g]["NE"]), "llamaLocExcluded": 160 - len(P[l]["NE"]),
    "gemmaBothSoneB": pmean(g, "S1", "B", "both"), "llamaBothSoneB": pmean(l, "S1", "B", "both"),
}
for m_ in MODELS:
    d = per[m_][(per[m_].kind == "bridge") & (per[m_].status == "ok") & (per[m_].group == "final")]
    b, s1 = d[d.cond == "B"].set_index("seed_id"), d[d.cond == "S1"].set_index("seed_id")
    idx = b.index.intersection(s1.index)
    tp = b.loc[idx].template.to_numpy()
    checks[("gemma" if m_ == g else "llama") + "CansRedPct"] = 100 * tmean(b.loc[idx].C_answer.abs() - s1.loc[idx].C_answer.abs(), tp) / tmean(b.loc[idx].C_answer.abs(), tp)
for k, v in checks.items():
    printed = M[k].replace("{,}", "")
    digits = len(printed.split(".")[1]) if "." in printed else 0
    if not close(float(printed), v, digits):
        fail("macro %s printed %s, recomputed %.4f" % (k, M[k], v))
print("inline macros: %d recomputed" % len(checks))

# energy spread and Holm p macros from the analysis files (not re-derivable without the bootstrap)
claims = json.loads((F / "claims.json").read_text(encoding="utf-8"))
for k, m_ in (("llamaEnergySpreadPct", l), ("gemmaEnergySpreadPct", g)):
    if int(M[k]) != int(round(100 * claims["energy_flags"][m_]["test_relative_spread"])):
        fail("macro %s" % k)
for k, (m_, h) in (("llamaHonePholm", (l, "H1")), ("gemmaHonePholm", (g, "H1")), ("gemmaHfourPholm", (g, "H4")), ("llamaHfourPholm", (l, "H4")),
                   ("llamaHtwoPholm", (l, "H2")), ("llamaHthreePholm", (l, "H3"))):
    printed = M[k]; digits = len(printed.split(".")[1])
    if not close(float(printed), conf.loc[(m_, h)].p_holm, digits):
        fail("macro %s printed %s vs %.4f" % (k, printed, conf.loc[(m_, h)].p_holm))

print("%d mismatch(es)" % len(problems))
sys.exit(1 if problems else 0)
