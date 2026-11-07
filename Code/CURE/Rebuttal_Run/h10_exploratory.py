"""
h10_exploratory.py -- readouts added AFTER the declared tests were seen; EXPLORATORY and labelled so
in any report (rebuttal cycle). CPU, stored rows only.

  flips      share of pairs whose generated answer differs between the two variants: a behavioural
             sensitivity that uses no option scoring, so no sum-versus-mean choice can move it
  unknown    share of generated answers that are the 'unknown' option, on the fresh and the control
             sets: does a cell lower sensitivity by pushing every answer to one option?
  frontier   per cell, T reduction against control-set accuracy loss (option scoring, so f = 0.03 counts)
  noise      correlation of per-seed changes in |C_answer| and T, raw and corrected for the pair-level
             noise in C_answer that h9 measures (attenuation, Spearman 1904)
  rules      the reduction in |C_answer| under September's rule and under the whole-term rule (h9)
  direction  same energy, same sites: contrast direction against a random one, P_all@1 vs R_all@1

Output: results/rebuttal_20261107/h_exploratory.txt. Usage: python h10_exploratory.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import hh_common as H

MODELS = ["llama-3.1-8b-instruct", "gemma-2-2b-it", "qwen2.5-7b-instruct", "phi-4-mini-instruct"]
CONDS = ["B", "P_span@1", "P_span@0.3", "P_span@0.1", "P_span@0.03", "CV_span@1", "CV_span@0.3", "CV_span@0.1", "P_ne@1", "P_all@1", "R_all@1", "Rk_span@1"]
UNK = ("unknown", "can't answer", "cannot answer", "undetermined", "can't be determined", "cannot be determined", "not enough info", "not answerable", "not known", "not enough information", "unclear")
out = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)


def unknown_index(options: list[str]) -> int | None:
    for i, o in enumerate(options):
        t = o.strip().lower()
        if any(t == u or t.startswith(u) for u in UNK):
            return i
    return None


for m in MODELS:
    d = pd.read_parquet(H.OUT / ("h1_dose_%s.parquet" % m))
    s = d[(d.kind == "score") & (d.status == "ok")].copy()
    s["unk_idx"] = [unknown_index(json.loads(o)) for o in s.options]
    say("\n=====", m)
    rows = []
    for grp in ("final", "control"):
        for c in CONDS:
            g = s[(s.group == grp) & (s.cond == c)]
            if g.empty:
                continue
            a, b = g[g.side == "A"].set_index("seed_id"), g[g.side == "B"].set_index("seed_id")
            j = a[["gen_idx", "gen_valid", "template"]].join(b[["gen_idx", "gen_valid"]], rsuffix="_b", how="inner")
            v = j[(j.gen_valid == True) & (j.gen_valid_b == True)]   # noqa: E712
            flip = float((v.gen_idx != v.gen_idx_b).mean()) if len(v) else np.nan
            ga = g[g.gen_valid == True]   # noqa: E712
            unk = float((ga.gen_idx == ga.unk_idx).mean()) if len(ga) else np.nan
            sua = float((g.argmax == g.unk_idx).mean())
            rows.append({"group": grp, "cond": c, "flip": flip, "n_valid_pairs": len(v), "gen_unknown": unk, "score_unknown": sua, "score_acc": float(g.score_correct.mean()),
                         "gen_acc": float(g.gen_correct.dropna().astype(float).mean()) if g.gen_correct.notna().any() else np.nan})
    say(pd.DataFrame(rows).round(3).to_string(index=False))
    # flip reduction with template-cluster test
    pt = s[s.group == "final"]
    w = {}
    for c in ("B", "P_span@1", "CV_span@1", "P_ne@1", "P_all@1", "R_all@1"):
        g = pt[pt.cond == c]
        a, b = g[g.side == "A"].set_index("seed_id"), g[g.side == "B"].set_index("seed_id")
        j = a[["gen_idx", "gen_valid", "template"]].join(b[["gen_idx", "gen_valid"]], rsuffix="_b", how="inner")
        j = j[(j.gen_valid == True) & (j.gen_valid_b == True)]   # noqa: E712
        w[c] = (j.gen_idx != j.gen_idx_b).astype(float).rename(c); tm = j.template
    tmpl = pt.drop_duplicates("seed_id").set_index("seed_id").template
    for c in ("P_span@1", "CV_span@1", "P_ne@1", "P_all@1", "R_all@1"):
        x = pd.concat([w["B"], w[c]], axis=1).dropna()
        r = H.paired((x["B"] - x[c]).to_numpy(), tmpl.reindex(x.index).to_numpy())
        say("  flip B - %-10s %+.4f [%+.4f, %+.4f] p %.4f n=%d" % (c, r["estimate"], r["lo"], r["hi"], r["p"], r["n_pairs"]))
    # direction at equal energy and site
    pp = H.pair_table(pt)
    wa = pp.pivot_table(index="seed_id", columns="cond", values="acc", aggfunc="first"); wa["template"] = tmpl
    wt = pp.pivot_table(index="seed_id", columns="cond", values="T", aggfunc="first"); wt["template"] = tmpl
    for lab, w_ in (("acc", wa), ("T", wt)):
        j = w_[["P_all@1", "R_all@1", "template"]].dropna()
        r = H.paired((j["R_all@1"] - j["P_all@1"]).to_numpy(), j.template.to_numpy())
        say("  direction everywhere, %-3s R_all@1 - P_all@1: %+.4f [%+.4f, %+.4f] p %.4f" % (lab, r["estimate"], r["lo"], r["hi"], r["p"]))
    # noise and the T-C correlation
    br = d[(d.kind == "bridge") & (d.status == "ok")].pivot_table(index="seed_id", columns="cond", values="C_answer", aggfunc="first")
    dc = br["B"].abs() - br["P_span@1"].abs()
    dt = wt["B"] - wt["P_span@1"]
    j = pd.concat([dc.rename("dc"), dt.rename("dt")], axis=1).dropna()
    r_raw = float(np.corrcoef(j.dc, j.dt)[0, 1])
    h9 = pd.read_csv(H.OUT / ("h9_bridge_%s.csv" % m))
    e = h9[h9.equal]
    noise_var = float(((e.order - e.sept) ** 2).mean()) / 2.0      # two noisy readings of one quantity on equal-length pairs
    rel = max(0.0, 1.0 - noise_var / float(j.dc.var()))
    say("  corr(dC, dT) raw %.3f; noise var in C %.3f, var(dC) %.3f, reliability %.3f, corrected %.3f" % (r_raw, noise_var, j.dc.var(), rel, r_raw / np.sqrt(rel) if rel > 0 else np.nan))
    # the reduction under both rules
    bb = br["B"].abs().reindex(h9.seed_id).to_numpy()
    say("  R_C: September's rule %.1f%%, whole-term %.1f%% (equal pairs %.1f / %.1f, unequal %.1f / %.1f)" % (
        100 * (1 - h9.sept.abs().mean() / np.nanmean(bb)), 100 * (1 - h9["this"].abs().mean() / np.nanmean(bb)),
        100 * (1 - e.sept.abs().mean() / np.nanmean(bb[h9.equal])), 100 * (1 - e["this"].abs().mean() / np.nanmean(bb[h9.equal])),
        100 * (1 - h9[~h9.equal].sept.abs().mean() / np.nanmean(bb[~h9.equal])), 100 * (1 - h9[~h9.equal]["this"].abs().mean() / np.nanmean(bb[~h9.equal]))))
    x = s[(s.target_energy > 0) & (s.feasible == True) & (s.cond != "P_span@1")]   # noqa: E712
    q = x.energy_per_pass / x.target_energy
    say("  scored rows outside 5%%: %d of %d; along/nominal P_all@1 %.3f, P_all@0.1 %.3f" % ((~q.between(.95, 1.05)).sum(), len(q), s[s.cond == "P_all@1"].along_over_nominal.median(), s[s.cond == "P_all@0.1"].along_over_nominal.median()))
(H.OUT / "h_exploratory.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
