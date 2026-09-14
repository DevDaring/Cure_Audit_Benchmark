"""
File: Code/CURE/audit_iclr_numbers.py
Purpose: Independent audit of the ICLR manuscript's numbers. It does not reuse the generator's
         code paths: it re-reads the raw result files with its own selectors, re-derives every
         cell of every generated table and every inline macro, and compares them with what the
         generated fragments print (after rounding). It also cross-checks the headline
         quantities against Submission2/Evidence_Contract.md, which p5_measured_ledger.py wrote
         from the same result files by a separate route.

Exit code 1 on any mismatch. Usage:  python Code/CURE/audit_iclr_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "Code" / "CURE" / "results" / "v2"
TAB = ROOT / "Submission2" / "iclr2027" / "tables_v2"
CONTRACT = ROOT / "Submission2" / "Evidence_Contract.md"
MODELS = ["llama-3.1-8b-instruct", "qwen2.5-7b-instruct", "gemma-2-2b-it", "phi-4-mini-instruct"]
NAME = {"llama-3.1-8b-instruct": "Llama-3.1-8B", "qwen2.5-7b-instruct": "Qwen2.5-7B",
        "gemma-2-2b-it": "Gemma-2-2B", "phi-4-mini-instruct": "Phi-4-mini"}
problems = []


def fail(msg):
    problems.append(msg); print("MISMATCH", msg)


def table_rows(name: str) -> list[list[str]]:
    """Rows of a generated table as lists of cell strings (LaTeX stripped, intervals kept)."""
    txt = (TAB / name).read_text(encoding="utf-8")
    body = txt.split("\\midrule", 1)[1].split("\\bottomrule")[0]
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("\\midrule") or line.startswith("\\cmidrule"):
            continue
        line = re.sub(r"\\multirow\{\d+\}\{\*\}\{([^}]*)\}", r"\1", line)
        line = line.replace("\\\\", "").replace("{\\scriptsize", "").replace("$^{\\ast}$", "*")
        cells = [re.sub(r"[{}$]", "", c).strip() for c in line.split("&")]
        rows.append(cells)
    return rows


def nums(cell: str) -> list[float]:
    return [float(x) for x in re.findall(r"[+-]?\d+(?:\.\d+)?", cell.replace(",", ""))]


def same(a: float, b: float, nd: int) -> bool:
    return abs(round(a, nd) - round(b, nd)) < 10 ** (-nd) * 0.51 or abs(a - b) < 10 ** (-nd) * 0.51


# ------------------------------------------------------------------ raw data (own selectors)
summ = pd.read_csv(V2 / "p2_summary.csv")
summ = summ[(summ.stratum == "pooled") & (summ.pair_type == "demographic")]
conf = pd.read_csv(V2 / "p2_confirmatory.csv")
cap = pd.read_parquet(V2 / "p2_capability.parquet")
cap = cap[(cap.status == "ok") & (cap.phase == "test")]
pil = pd.read_parquet(V2 / "pilot_per_item.parquet")
p3 = {k: pd.read_csv(V2 / ("p3_%s_summary.csv" % k)) for k in ("magnitude", "depth", "massive")}
p4m = pd.read_csv(V2 / "p4_forecast_metrics.csv")
p4i = pd.read_csv(V2 / "p4_incremental_tests.csv")


def srow(m, cid, phase="test"):
    r = summ[(summ.model_name == m) & (summ.cond_id == cid) & (summ.phase == phase)]
    assert len(r) == 1, (m, cid, phase, len(r))
    return r.iloc[0]


def probe(m, cid, p, col):
    r = cap[(cap.model_name == m) & (cap.cond_id == cid) & (cap.probe == p)]
    assert len(r) == 1, (m, cid, p)
    return float(r[col].iloc[0])


# ------------------------------------------------------------------ tab_confirm
rows = table_rows("tab_confirm.tex")
for m, row in zip(MODELS, rows):
    assert row[0] == NAME[m], row
    k = srow(m, "cure_centred_svd_r1_a1")
    c = conf[(conf.model_name == m) & (conf.phase == "test")].set_index("id")
    checks = [("|C| before", nums(row[1])[0], k.absC_pre_mean, 3),
              ("D", nums(row[2])[0], c.loc["C1", "point"], 3), ("D lo", nums(row[2])[1], c.loc["C1", "lo"], 3), ("D hi", nums(row[2])[2], c.loc["C1", "hi"], 3),
              ("R%", nums(row[3])[0], 100 * k.R, 1),
              ("dacc", nums(row[4])[0], c.loc["C2", "point"], 3),
              ("D vs rand", nums(row[5])[0], c.loc["C3", "point"], 3),
              ("dacc vs rand", nums(row[6])[0], c.loc["C4", "point"], 3)]
    for lab, got, exp, nd in checks:
        if not same(got, exp, nd):
            fail("tab_confirm %s %s: table %s vs data %s" % (NAME[m], lab, got, exp))
    for i, cid in ((2, "C1"), (4, "C2"), (5, "C3"), (6, "C4")):
        star = "*" in row[i]
        if star != bool(c.loc[cid, "reject_0.05_holm"]):
            fail("tab_confirm %s %s Holm star %s vs data %s" % (NAME[m], cid, star, c.loc[cid, "reject_0.05_holm"]))
print("tab_confirm: %d rows checked" % len(rows))

# ------------------------------------------------------------------ tab_controls (figure source) and fig data
COND = {"span erasure, rank 1": "cure_centred_svd_r1_a1", "span erasure, rank 2": "cure_centred_svd_r2_a1",
        "span erasure, rank 4": "cure_centred_svd_r4_a1", "uncentred SVD, rank 1": "uncentred_svd_r1_a1",
        "mean-difference direction": "mean_difference_r1_a1", "LEACE, frozen fit": "leace_faithful_r1_a1",
        "LEACE, sequential fit": "leace_sequential_r1_a1", "random subspace, rank 1": "random_ortho_1_r1_a1",
        "random subspace, rank 4": "random_ortho_1_r4_a1", "neutral contrast, rank 1": "neutral_contrast_r1_a1"}
n = 0
for row in table_rows("tab_controls.tex"):
    cid = COND[row[0]]
    for m, cell in zip(MODELS, row[1:]):
        k = srow(m, cid); got = nums(cell)
        for lab, g, e in (("D", got[0], k.D), ("lo", got[1], k.D_lo), ("hi", got[2], k.D_hi)):
            n += 1
            if not same(g, e, 3):
                fail("tab_controls %s %s %s: %s vs %s" % (row[0], NAME[m], lab, g, e))
print("tab_controls / fig_conditions data: %d values checked" % n)

# ------------------------------------------------------------------ tab_global
GL = {"unedited": "unedited_r0_a0", "span-fitted erasure, rank 1": "cure_centred_svd_r1_a1", "span-fitted erasure, rank 4": "cure_centred_svd_r4_a1",
      "LEACE, frozen fit": "leace_faithful_r1_a1", "random subspace, rank 1": "random_ortho_1_r1_a1", "random subspace, rank 4": "random_ortho_1_r4_a1"}
n = 0
for row in table_rows("tab_global.tex"):
    cid = GL[row[0]]
    for i, m in enumerate(MODELS):
        mm, pp = nums(row[1 + 2 * i])[0], nums(row[2 + 2 * i])[0]
        e_mm, e_pp = probe(m, cid, "mmlu_200", "value"), probe(m, cid, "wikitext2_20k", "perplexity")
        n += 2
        if not same(mm, e_mm, 3):
            fail("tab_global %s %s MMLU %s vs %s" % (row[0], NAME[m], mm, e_mm))
        if not (same(pp, e_pp, 1) if e_pp < 1000 else same(pp, e_pp, 0)):
            fail("tab_global %s %s ppl %s vs %s" % (row[0], NAME[m], pp, e_pp))
print("tab_global: %d values checked" % n)

# ------------------------------------------------------------------ tab_protocol
demo = pil[(pil.pair_type == "demographic") & (pil.fmt == "chat")]
ident = pil[pil.pair_type == "identity"]
for m, row in zip(MODELS, table_rows("tab_protocol.tex")):
    d = demo[demo.model_name == m]
    for j, c in enumerate(("unedited", "span_seq_r1", "last_token_r1")):
        dc = d[d.condition == c]
        absC = dc.absC.mean()
        val = pd.concat([dc.gen_valid_A, dc.gen_valid_B]).fillna(False).astype(bool).mean()
        if not same(nums(row[1 + j])[0], absC, 3):
            fail("tab_protocol %s |C| %s: %s vs %s" % (NAME[m], c, row[1 + j], absC))
        if not same(nums(row[4 + j])[0], val, 2):
            fail("tab_protocol %s validity %s: %s vs %s" % (NAME[m], c, row[4 + j], val))
    fr = ident[(ident.model_name == m) & (ident.condition == "span_frozen_r1")].absC.max()
    if not same(nums(row[7])[0], fr, 2):
        fail("tab_protocol %s frozen identity max: %s vs %s" % (NAME[m], row[7], fr))
    seq0 = ident[(ident.model_name == m) & (ident.condition == "span_seq_r1")].absC.max()
    if seq0 != 0:
        fail("identity pairs under sequential convention not zero on %s: %s" % (NAME[m], seq0))
print("tab_protocol: 4 rows checked; sequential identity |C| = 0 on all models")

# ------------------------------------------------------------------ tab_accounts
ACC = {"span erasure, frozen bases": ("magnitude", "targeted"), "random subspace, development-fixed strength": ("magnitude", "random_1_matched"),
       "span erasure, sequential refit": ("depth", "full_sequential"), "erasure, massive coordinates restored": ("massive", "restore_massive"),
       "erasure, all coordinates restored": ("massive", "restore_all")}
n = 0
readout = ""
for row in table_rows("tab_accounts.tex"):
    readout, lab = (row[0] or readout), row[1]          # multirow leaves the readout cell blank below its first row
    key, sign = ("absC_change_vs_unedited", 1.0) if "C" in readout else ("acc_loss_vs_unedited", -1.0)
    acct, c = ACC[lab]
    df = p3[acct]; df = df[df.phase == "test"]
    for m, cell in zip(MODELS, row[2:]):
        r = df[(df.model_name == m) & (df.condition == c)].iloc[0]
        got = nums(cell); exp = sorted([sign * r[key + "_lo"], sign * r[key + "_hi"]])
        n += 3
        if not same(got[0], sign * r[key], 3) or not same(got[1], exp[0], 3) or not same(got[2], exp[1], 3):
            fail("tab_accounts %s / %s %s: %s vs %s [%s, %s]" % (readout, lab, NAME[m], cell, sign * r[key], exp[0], exp[1]))
print("tab_accounts: %d values checked (accuracy column = minus the stored loss)" % n)

# ------------------------------------------------------------------ tab_forecast
m4 = p4m[(p4m.phase == "test") & (p4m.target == "new_error")]
i4 = p4i[(p4i.phase == "test") & (p4i.target == "new_error") & (p4i.predictor == "4_baseline_plus_audit") & (p4i.versus == "2_baseline")]
for m, row in zip(MODELS, table_rows("tab_forecast.tex")):
    b = m4[(m4.model_name == m) & (m4.predictor == "2_baseline")].iloc[0]
    a = m4[(m4.model_name == m) & (m4.predictor == "3_audit_only")].iloc[0]
    d = i4[i4.model_name == m].iloc[0]
    for lab, got, exp, nd in (("n", nums(row[1])[0], b.n, 0), ("prev", nums(row[2])[0], b.prevalence, 3),
                              ("base", nums(row[3])[0], b.auroc, 3), ("audit", nums(row[4])[0], a.auroc, 3),
                              ("incr", nums(row[5])[0], d.delta, 3), ("incr lo", nums(row[5])[1], d.delta_lo, 3), ("incr hi", nums(row[5])[2], d.delta_hi, 3)):
        if not same(got, exp, nd):
            fail("tab_forecast %s %s: %s vs %s" % (NAME[m], lab, got, exp))
    if bool(d.excludes_zero):
        fail("tab_forecast %s: increment interval excludes zero in data; text says it includes zero" % NAME[m])
print("tab_forecast: 4 rows checked; no increment interval excludes zero")

# ------------------------------------------------------------------ tab_behaviour / tab_strength / tab_split / tab_token
for m, row in zip(MODELS, table_rows("tab_behaviour.tex")):
    u, k = srow(m, "identity_alpha0_r1_a0"), srow(m, "cure_centred_svd_r1_a1")
    for lab, got, exp in (("val0", nums(row[1])[0], u.validity), ("val1", nums(row[2])[0], k.validity),
                          ("acc0", nums(row[3])[0], u.gen_acc_attempted), ("acc1", nums(row[4])[0], k.gen_acc_attempted),
                          ("flip0", nums(row[5])[0], u.flip_attempted), ("flip1", nums(row[6])[0], k.flip_attempted),
                          ("dflip", nums(row[7])[0], k.flip_attempted - u.flip_attempted),
                          ("opt0", nums(row[8])[0], u.opt_acc), ("opt1", nums(row[9])[0], k.opt_acc)):
        if not same(got, exp, 3):
            fail("tab_behaviour %s %s: %s vs %s" % (NAME[m], lab, got, exp))
    if not same(nums(row[10])[0], k.W, 2):
        fail("tab_behaviour %s W: %s vs %s" % (NAME[m], row[10], k.W))
print("tab_behaviour: 4 rows checked")
for row in table_rows("tab_strength.tex"):
    a = row[0].replace("\\alpha=", "").strip()
    cid = {"0.25": "cure_centred_svd_r1_a0.25", "0.5": "cure_centred_svd_r1_a0.5", "1": "cure_centred_svd_r1_a1"}[a]
    for i, m in enumerate(MODELS):
        k = srow(m, cid, "dev")
        if not same(nums(row[1 + 2 * i])[0], k.D, 3) or not same(nums(row[2 + 2 * i])[0], k.d_gen_acc_attempted, 3):
            fail("tab_strength alpha=%s %s: %s / %s vs %s / %s" % (a, NAME[m], row[1 + 2 * i], row[2 + 2 * i], k.D, k.d_gen_acc_attempted))
print("tab_strength: 3 rows checked")
sm = pd.read_csv(ROOT / "Code" / "CURE" / "results" / "reanalysis_v2" / "split_manifest_summary.csv")
for row in table_rows("tab_split.tex"):
    src = {"BBQ": "bbq", "CrowS-Pairs": "crows_pairs", "StereoSet": "stereoset", "All": None}[row[0]]
    sub = sm if src is None else sm[sm.seed_source == src]
    for j, split in enumerate(("fit", "dev", "test")):
        if nums(row[1 + j])[0] != sub[sub.split == split].n_seeds.sum():
            fail("tab_split %s %s: %s vs %s" % (row[0], split, row[1 + j], sub[sub.split == split].n_seeds.sum()))
print("tab_split: checked")
tt = pd.read_csv(ROOT / "Code" / "CURE" / "results" / "reanalysis" / "target_token_audit.csv").set_index("source")
for row in table_rows("tab_token.tex"):
    src = {"BBQ": "bbq", "CrowS-Pairs": "crows_pairs", "StereoSet": "stereoset", "All": "all"}[row[0]]
    if nums(row[1])[0] != tt.loc[src, "n"] or nums(row[2])[0] != tt.loc[src, "n_discriminative"] or not same(nums(row[3])[0], tt.loc[src, "frac_ambiguous"], 2):
        fail("tab_token %s: %s vs %s" % (row[0], row[1:], tt.loc[src].tolist()))
print("tab_token: checked")

# ------------------------------------------------------------------ inline macros
mac = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", (TAB / "numbers_v2.tex").read_text(encoding="utf-8")))
R1 = {m: srow(m, "cure_centred_svd_r1_a1").R for m in MODELS}
R4 = {m: srow(m, "cure_centred_svd_r4_a1").R for m in MODELS}
ratio = {m: probe(m, "cure_centred_svd_r1_a1", "wikitext2_20k", "perplexity") / probe(m, "unedited_r0_a0", "wikitext2_20k", "perplexity") for m in MODELS}
drop = {m: probe(m, "unedited_r0_a0", "mmlu_200", "value") - probe(m, "cure_centred_svd_r1_a1", "mmlu_200", "value") for m in MODELS}
rr = max(probe(m, c, "wikitext2_20k", "perplexity") / probe(m, "unedited_r0_a0", "wikitext2_20k", "perplexity") for m in MODELS for c in ("random_ortho_1_r1_a1", "random_ortho_1_r4_a1"))
rm = max(abs(probe(m, "unedited_r0_a0", "mmlu_200", "value") - probe(m, c, "mmlu_200", "value")) for m in MODELS for c in ("random_ortho_1_r1_a1", "random_ortho_1_r4_a1"))
proto = json.loads((V2 / "p2_protocol.json").read_text(encoding="utf-8"))
expect = {"RminPct": ("%.0f" % (100 * min(R1.values()))), "RmaxPct": ("%.0f" % (100 * max(R1.values()))),
          "RfourMinPct": ("%.0f" % (100 * min(R4.values()))), "RfourMaxPct": ("%.0f" % (100 * max(R4.values()))),
          "pplRatioMin": "%.1f" % min(ratio.values()), "pplRatioMax": "%.0f" % max(ratio.values()),
          "pplRatioRandMax": "%.2f" % rr, "mmluRandMaxAbs": "%.3f" % rm,
          "mmluDropMax": "%.2f" % max(drop.values()), "mmluDropMin": "%.2f" % min(drop.values()),
          "nTestSeeds": str(proto["n_test_seeds"]), "nDevSeeds": str(proto["n_dev_seeds"]), "nFitPairs": str(proto["n_fit_pairs"]),
          "nMaxNewTokens": str(proto["max_new_tokens"]), "nSeedsTotal": str(int(sm.n_seeds.sum())),
          "gemmaAccGain": "%.3f" % srow("gemma-2-2b-it", "cure_centred_svd_r1_a1").d_gen_acc_attempted,
          "validityMin": "%.2f" % min(srow(m, "cure_centred_svd_r1_a1").validity for m in MODELS),
          "auditOnlyMax": "%.2f" % m4[m4.predictor == "3_audit_only"].auroc.max(),
          "incrMaxAbs": "%.3f" % i4.delta.abs().max(),
          "frozenIdentityMax": "%.2f" % max(ident[(ident.model_name == m) & (ident.condition == "span_frozen_r1")].absC.max() for m in MODELS)}
for k, e in expect.items():
    if mac.get(k) != e:
        fail("macro %s: paper %s vs recomputed %s" % (k, mac.get(k), e))
print("inline macros: %d recomputed" % len(expect))

# ------------------------------------------------------------------ Evidence Contract cross-check (independent route)
# The contract is regenerated by Code/CURE/Next_Run/f5_contract.py from claims_f0.json; its L1
# entries carry the Study 1 rank-one span-erasure D and interval, which are recomputed here.
ct = CONTRACT.read_text(encoding="utf-8")
for m in MODELS:
    blk = ct.split("### L1_%s" % m)
    if len(blk) < 2:
        fail("contract has no L1 entry for %s" % m); continue
    mm = re.search(r"\*\*Estimate\.\*\* ([+-]\d\.\d+) \[([+-]\d\.\d+), ([+-]\d\.\d+)\]", blk[1].split("### ")[0])
    k = srow(m, "cure_centred_svd_r1_a1")
    if not mm:
        fail("contract L1 estimate for %s not parsed" % m); continue
    if not same(float(mm.group(1)), k.D, 3) or not same(float(mm.group(2)), k.D_lo, 3) or not same(float(mm.group(3)), k.D_hi, 3):
        fail("contract L1 vs data %s: %s vs (%s, %s, %s)" % (m, mm.groups(), k.D, k.D_lo, k.D_hi))
print("Evidence Contract L1 entries agree with the raw files")

print("%d mismatch(es)" % len(problems))
sys.exit(1 if problems else 0)
