"""
File: Code/CURE/reanalyse.py
Purpose: Re-derive the prognosis result on the correct unit and with an honest treatment of
         its outcome variable, and reconcile every count the manuscript reports.

Three problems with the analysis as shipped, all visible in the saved outputs:

  1. UNIT. The manuscript reports "508 pairs per model". The parquet holds 508 unique
     seed_id values, one row each. The unit is the SEED, which is the stronger claim: pairs
     from one seed share a template and are not independent, so a pair-level n would have
     been pseudoreplicated. Reporting seeds fixes the wording and keeps the inference valid.

  2. CENSORED ORDINAL OUTCOME. repair_rank takes five values, {1, 2, 4, 8, 9}. The tested
     ranks are 1, 2, 4 and 8; 9 is a sentinel meaning "not repaired at any tested rank", so
     the outcome is right-censored. Pearson correlation and R^2 on such a variable read as
     more precise than the measurement supports. Spearman and Kendall are appropriate
     because they use only order, and the sentinel preserves order. This module reports
     those, plus a threshold formulation that avoids the sentinel's magnitude entirely:
     how well does the audit score separate seeds needing a high rank from the rest?

  3. NO OUT-OF-SAMPLE CHECK. The correlation is computed on the same seeds throughout. A
     split-half evaluation, repeated over many splits, shows whether the relation holds on
     seeds the fit never saw.

No model inference is run; everything reads saved outputs.

Implements / builds on / cites:
  - Spearman (1904). "The proof and measurement of association between two things."
  - Kendall (1938). "A new measure of rank correlation." Biometrika 30(1/2).
  - Efron & Tibshirani (1993). An Introduction to the Bootstrap.

Usage:
  python Code/CURE/reanalyse.py

Part of the CURE codebase (Springer Machine Learning submission).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, mannwhitneyu
from sklearn.metrics import roc_auc_score

log = logging.getLogger("cure_reanalyse")

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "Code" / "CURE" / "results"
OUT = R / "reanalysis"
MODELS = ["llama-3.1-8b-instruct", "qwen2.5-7b-instruct",
          "gemma-2-2b-it", "phi-4-mini-instruct"]
DISPLAY = {"llama-3.1-8b-instruct": "Llama-3.1-8B", "qwen2.5-7b-instruct": "Qwen2.5-7B",
           "gemma-2-2b-it": "Gemma-2-2B", "phi-4-mini-instruct": "Phi-4-mini"}
TESTED_RANKS = [1, 2, 4, 8]
SENTINEL = 9
RANDOM_SEED = 20260816
N_BOOT = 5000
N_SPLITS = 200


def _prog(model: str) -> pd.DataFrame:
    return pd.read_parquet(R / f"cure_prognosis_{model}.parquet")


# ---------------------------------------------------------------------------
# 1. The prognosis, stated on the right unit and with the censoring visible
# ---------------------------------------------------------------------------
def prognosis() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for m in MODELS:
        d = _prog(m)
        n_seeds = int(d.seed_id.nunique())
        assert len(d) == n_seeds, f"{m}: rows != unique seeds; unit is not the seed"

        x = d.audit_score.to_numpy(float)
        y = d.repair_rank.to_numpy(float)
        censored = int((d.repair_rank == SENTINEL).sum())

        rho, p_rho = spearmanr(x, y)
        tau, p_tau = kendalltau(x, y)

        # Bootstrap over seeds for an interval on the rank correlation.
        draws = np.empty(N_BOOT)
        for b in range(N_BOOT):
            take = rng.integers(0, n_seeds, n_seeds)
            draws[b] = spearmanr(x[take], y[take]).statistic
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])

        # Threshold formulation: does the audit separate the hard-to-repair seeds? This
        # uses only the order of the outcome, so the sentinel's magnitude never enters.
        hard = (d.repair_rank >= 8).astype(int).to_numpy()
        auc = float(roc_auc_score(hard, x)) if len(np.unique(hard)) > 1 else np.nan
        if len(np.unique(hard)) > 1:
            u, p_u = mannwhitneyu(x[hard == 1], x[hard == 0], alternative="greater")
        else:
            p_u = np.nan

        rows.append({
            "model_name": m, "n_seeds": n_seeds,
            "n_censored": censored, "frac_censored": censored / n_seeds,
            "spearman": float(rho), "spearman_p": float(p_rho),
            "spearman_ci_lo": float(lo), "spearman_ci_hi": float(hi),
            "kendall": float(tau), "kendall_p": float(p_tau),
            "frac_hard": float(hard.mean()),
            "auc_hard": auc, "auc_p": float(p_u) if p_u == p_u else np.nan,
        })
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "prognosis_seed_level.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 2. Does the prognosis hold on seeds the fit never saw?
# ---------------------------------------------------------------------------
def prognosis_heldout() -> pd.DataFrame:
    """Split-half: rank-correlate on one half, then score the other half with that ordering.

    A rank correlation has no parameters to carry across a split, so the transferable claim
    is the threshold one: fit the audit-score cut that best separates hard-to-repair seeds on
    the training half, and apply that cut unchanged to the test half.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for m in MODELS:
        d = _prog(m)
        x = d.audit_score.to_numpy(float)
        hard = (d.repair_rank >= 8).astype(int).to_numpy()
        n = len(d)
        if len(np.unique(hard)) < 2:
            continue
        rhos, aucs, accs = [], [], []
        for _ in range(N_SPLITS):
            idx = rng.permutation(n)
            a, b = idx[: n // 2], idx[n // 2:]
            if len(np.unique(hard[a])) < 2 or len(np.unique(hard[b])) < 2:
                continue
            # Threshold chosen on the training half only.
            grid = np.quantile(x[a], np.linspace(0.05, 0.95, 37))
            best_t, best_j = grid[0], -np.inf
            for t in grid:
                pred = (x[a] >= t).astype(int)
                tpr = pred[hard[a] == 1].mean()
                fpr = pred[hard[a] == 0].mean()
                if tpr - fpr > best_j:
                    best_j, best_t = tpr - fpr, t
            accs.append(float(((x[b] >= best_t).astype(int) == hard[b]).mean()))
            aucs.append(float(roc_auc_score(hard[b], x[b])))
            rhos.append(float(spearmanr(x[b], d.repair_rank.to_numpy(float)[b]).statistic))
        if not accs:
            continue
        rows.append({
            "model_name": m, "n_splits": len(accs),
            "heldout_spearman_mean": float(np.mean(rhos)),
            "heldout_auc_mean": float(np.mean(aucs)),
            "heldout_auc_sd": float(np.std(aucs)),
            "heldout_accuracy_mean": float(np.mean(accs)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        OUT.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT / "prognosis_heldout.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 3. The rank curve, reported as measured rather than as a chosen point
# ---------------------------------------------------------------------------
def rank_curve() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        p = R / f"cure_rankcurve_{m}.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        for rank, v in j.get("curve", {}).items():
            rows.append({
                "model_name": m, "rank": int(rank),
                "bias_removed": v.get("reduced_validation"),
                "utility_cost": v.get("utility_cost"),
                "erased_acc": v.get("erased_acc"),
                "is_operating_rank": int(rank) == j.get("operating_rank"),
                "utility_budget": j.get("max_utility_cost"),
            })
    out = pd.DataFrame(rows)
    if len(out):
        OUT.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT / "rank_curve.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 4. Reconcile the counts the manuscript reports
# ---------------------------------------------------------------------------
def provenance() -> pd.DataFrame:
    checks = []

    def add(q, claimed, actual, src, note=""):
        checks.append({"quantity": q, "claimed": claimed, "actual": actual,
                       "match": claimed == actual, "source": src, "note": note})

    for m in MODELS:
        d = _prog(m)
        add(f"prognosis n, {DISPLAY[m]}", 508, int(len(d)),
            f"cure_prognosis_{m}.parquet", "manuscript says pairs; these are seeds")

    f = pd.read_parquet(R / f"cure_final_{MODELS[0]}.parquet")
    add("head-to-head pair set", 1000, int(f.n_pairs.max()),
        f"cure_final_{MODELS[0]}.parquet")
    add("methods compared", 9, int(f.method.nunique()),
        f"cure_final_{MODELS[0]}.parquet", "CURE plus eight baselines")

    t = pd.read_parquet(R / f"tacl_extra_{MODELS[0]}.parquet")
    add("held-out test pairs", 600, int(t.n_test_pairs.max()),
        f"tacl_extra_{MODELS[0]}.parquet")
    add("behavioural seeds", 160, int(t.n_acc_seeds.max()),
        f"tacl_extra_{MODELS[0]}.parquet")

    out = pd.DataFrame(checks)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "provenance_audit.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 5. What token does the commutator actually score, and can it tell options apart?
# ---------------------------------------------------------------------------
def target_token_audit() -> pd.DataFrame:
    """Characterise the scalar the audit reads.

    erase.py reads the logit of ONE token at the final sequence position. That token comes
    from _get_bias_answer in the audit code: the first word of the gold answer, falling back
    to the demographic term when the gold answer is literally "unknown".

    Two consequences matter for how the quantity may be described.

      * The gold answer is usually several words long, so the scored token is a fragment of
        it rather than the answer.
      * If that fragment also begins a competing option, a change in its logit cannot say
        which option the model moved towards. Items whose options are minimally different
        sentences share their opening word almost by construction.

    This function measures both, so the manuscript can state what is measured instead of the
    looser "the gold-answer logit".
    """
    import re as _re

    pen = pd.read_parquet(ROOT / "Code/audit/Dataset/seeds/pentad_dataset_clean.parquet")
    surface = pen[pen.slot == "a"].set_index("seed_id")
    subs = pen[pen.slot == "c"]
    opt_re = _re.compile(r"^\(([A-C])\)\s*(.+?)\s*$", _re.M)

    def scored(g: pd.DataFrame) -> tuple[str, str]:
        gold = g["gold_answer"].dropna().unique()
        if len(gold):
            gv = str(gold[0]).strip()
            if gv and gv.lower() != "unknown":
                return "gold-first-word", (gv.split()[0] if gv.split() else gv)
        for tok in g["swap_token"].dropna().astype(str):
            tok = tok.strip()
            if tok and tok.lower() not in {"none", "nan", ""}:
                return "swap-token", (tok.split()[0] if tok.split() else tok)
        return "empty", ""

    rows = []
    for sid, g in subs.groupby("seed_id"):
        if sid not in surface.index:
            continue
        branch, tok = scored(g)
        opts = [m.group(2) for m in opt_re.finditer(str(surface.loc[sid].prompt_text))]
        if not opts or not tok:
            continue
        firsts = [o.split()[0] if o.split() else o for o in opts]
        rows.append({
            "seed_id": sid, "source": surface.loc[sid].seed_source,
            "branch": branch, "token": tok,
            "n_options": len(opts),
            "n_sharing": sum(1 for f in firsts if f == tok),
        })
    d = pd.DataFrame(rows)
    d["discriminative"] = d.n_sharing == 1

    per_src = (d.groupby("source")["discriminative"]
                 .agg(n="size", n_discriminative="sum").reset_index())
    per_src["frac_ambiguous"] = 1 - per_src.n_discriminative / per_src.n
    total = pd.DataFrame([{"source": "all", "n": len(d),
                           "n_discriminative": int(d.discriminative.sum()),
                           "frac_ambiguous": float(1 - d.discriminative.mean())}])
    out = pd.concat([per_src, total], ignore_index=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "target_token_audit.csv", index=False)
    d.to_csv(OUT / "target_token_per_seed.csv", index=False)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    prov = provenance()
    print("\n=== PROVENANCE ===")
    print(prov.to_string(index=False))
    bad = prov[~prov["match"]]
    print(f"\n{len(bad)} mismatch(es)" if len(bad) else "\nall counts reconcile")

    pr = prognosis()
    print("\n=== PROGNOSIS, SEED LEVEL, CENSORING VISIBLE ===")
    print(pr[["model_name", "n_seeds", "frac_censored", "spearman",
              "spearman_ci_lo", "spearman_ci_hi", "kendall",
              "frac_hard", "auc_hard"]].round(3).to_string(index=False))

    ho = prognosis_heldout()
    print("\n=== PROGNOSIS OUT OF SAMPLE (split-half) ===")
    print(ho.round(3).to_string(index=False) if len(ho) else "(not computable)")

    rc = rank_curve()
    print("\n=== RANK CURVE ===")
    print(rc.round(3).to_string(index=False) if len(rc) else "(absent)")

    tt = target_token_audit()
    print("\n=== WHAT TOKEN THE COMMUTATOR SCORES ===")
    print(tt.round(3).to_string(index=False))
    print("a seed is 'ambiguous' when the scored token also begins a competing option, "
          "so its logit cannot say which option the model moved towards")


if __name__ == "__main__":
    main()
