"""
p4_risk_forecast.py -- optional independent risk forecast (Next_Plan.md, P4). CPU only.

Purpose: decide, per model and per example, whether the pre-edit audit score |C_pre| forecasts
INDEPENDENT damage from the CURE edit better than cheap confidence and size baselines, on
held-out seed groups, without threshold leakage. Section 2.6 of the plan explains why the
shipped prognosis is circular: its target is |C| crossing TAU after the edit, a function of the
predictor itself. Here the primary target is behavioural and does not mention TAU.

Design (Next_Plan.md P4, item by item)
  Unit        one prompt (seed_id, subvariant) of the P2 evaluation, never a pair and never a
              seed aggregate; sides A and B of every P2 pair are unstacked and de-duplicated.
              Seeds are the cluster for cross-validation and for every bootstrap.
  Treatment   the cure_centred_svd edit at the FROZEN operating alpha and rank (from
              p2_matched_alphas.json, else --alpha/--rank, else the single point present).
  Targets     new_error   PRIMARY. Among prompts the unedited model answered correctly and
                          validly, y = 1 if the edited answer is wrong OR invalid (output
                          invalidity is an adverse outcome). Generation correctness when P2
                          recorded it (gen_correct_*, gen_valid_*), else option correctness.
              prob_loss   SECONDARY, continuous: p_gold(unedited) - p_gold(edited) from the
                          canonical option distribution, over prompts with a gold option.
              audit_cross SECONDARY ONLY: among prompts whose worst pair has |C_pre| > TAU,
                          y = 1 if the worst edited pair still has |C_post| > TAU. Predictor 3 is
                          the score itself here; the caveat is written into every such row.
  Predictors  1  constant base rate (train prevalence / train mean); the no-edit policy is the
                 coverage-zero point of the coverage curve.
              2  pre-edit answer margin + option entropy + benchmark category + prompt token
                 length: logistic regression (Ridge for prob_loss) on standardised features.
              3  audit score |C_pre| alone (worst pair containing the prompt): AUROC from the raw
                 score, probabilities by Platt scaling fitted on the training folds.
              4  predictor-2 features + audit score.
              2b / 4b  the same plus the planned removed-energy estimate when P2 recorded it
                 (energy_total_A / energy_mean_A on the edited rows: a deterministic function of
                 the unedited state and the frozen basis, one forward pass, cheaper than the
                 three-pass audit). The audit must beat 2b as well as 2 when 2b exists.
  Protocol    GroupKFold by seed_id (5 folds) on the manifest DEV seeds for every predictor and
              target (pooled out-of-fold metrics and per-fold AUROC); then ONE fit on all DEV
              rows and ONE evaluation on the manifest TEST seeds. Nothing is chosen on TEST.
  Metrics     AUROC (Hanley and McNeil 1982), average precision with prevalence, Brier score
              (Brier 1950) with the constant-predictor Brier and the skill score, calibration
              slope and intercept (Cox 1958; Steyerberg 2009), calibration-in-the-large, ECE
              (10 bins); Spearman, R^2 and RMSE for the continuous target. Utility loss versus
              retained editing coverage: decline the fraction q of prompts with the highest
              forecast risk, report the share of new errors avoided, the accuracy loss that
              remains, the precision of the declined set and the share of initially failing
              (|C_pre| > TAU) prompts still edited.
  Incremental value  paired seed-cluster bootstrap (Efron and Tibshirani 1993, 2000 draws) of
              AUROC(4) - AUROC(2) on TEST (and 4b - 2b, 3 - 2, 4 - 3), 95% percentile interval.
  Stop condition (the plan's): if predictor 4 does not beat predictor 2 on TEST AUROC with a 95%
              interval excluding zero (and 4b does not beat 2b when energy exists), the report
              says "REMOVE the practical-forecast claim".
  Audit cost  from P2's timing columns when present (sec_audit versus sec_options and
              sec_generation on the same items) and from the pass counts: the audit needs three
              forward passes per pair (source, patched, clean) plus span resolution; the
              baseline features need one batched option-scoring pass, which the edited
              evaluation performs anyway.
  Scope       forecasting is per example WITHIN a model. No cross-model model is fitted to the
              four aggregate model points, and none should be read from these outputs.

Inputs (the script exits with a message if a required file is absent)
  results/v2/p2_per_item.parquet   long layout, one row per (model_name, condition, seed_id,
                                   subvariant_A, subvariant_B [, alpha, rank]) with p1_pilot-style
                                   side columns: opt_argmax_{A,B}, opt_margin_{A,B},
                                   opt_entropy_{A,B}, opt_probs_{A,B} (json list), gold_idx_{A,B},
                                   opt_correct_{A,B}, absC; optional gen_valid_{A,B},
                                   gen_correct_{A,B}, n_tokens_{A,B}, energy_total_A,
                                   energy_mean_A, sec_audit, sec_options, sec_generation.
  results/v2/p2_matched_alphas.json   optional operating point {model: {condition: {alpha, rank}}}
  reanalysis_v2/split_manifest.csv    seed_id, split in {fit, dev, test}
  Code/audit/Dataset/seeds/pentad_dataset_clean.parquet  prompt text for the token-length
                                   fallback (cached tokenizer if available, else word count)

Outputs (results/v2/)
  p4_forecast_metrics.csv   one row per (model, target, predictor, phase in {dev_cv, test})
  p4_coverage_curve.csv     utility-loss versus coverage rows per (model, target, predictor)
  p4_report.md              per-model tables, incremental tests, audit cost, the decision line

Usage
  python p4_risk_forecast.py --models qwen2.5-7b-instruct llama-3.1-8b-instruct
  python p4_risk_forecast.py --models gemma-2-2b-it --alpha 1.0 --rank 1

Builds on
  - Brier (1950) Verification of forecasts expressed in terms of probability. Mon. Weather Rev.
  - Platt (1999) Probabilistic outputs for support vector machines (sigmoid calibration).
  - Cox (1958) Two further applications of a model for binary regression, Biometrika
    (calibration slope and intercept); Steyerberg (2009) Clinical Prediction Models.
  - Hanley and McNeil (1982) The meaning and use of the area under a ROC curve, Radiology.
  - Davis and Goadrich (2006) The relationship between precision-recall and ROC curves, ICML.
  - Efron and Tibshirani (1993) An Introduction to the Bootstrap.
  - Pedregosa et al. (2011) scikit-learn: GroupKFold, LogisticRegression, Ridge, metrics.

Part of the CURE codebase (ICLR 2027, Submission2). No GPU.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as K

log = K.setup_logging("cure.ext.p4")

# Copied from Code/audit/config.py (OSM_MODELS[*]["hf_id"]) so the CPU stage never imports the
# audit config or reads the .env; used only to try a locally cached tokenizer for token length.
HF_IDS = {"llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
          "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
          "gemma-2-2b-it": "google/gemma-2-2b-it",
          "phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct"}

BASE_FEATURES = ["margin_pre", "entropy_pre", "n_tokens", "benchmark"]
PREDICTORS = {"1_base_rate": [],
              "2_baseline": BASE_FEATURES,
              "3_audit_only": ["audit_pre"],
              "4_baseline_plus_audit": BASE_FEATURES + ["audit_pre"]}
PREDICTORS_ENERGY = {"2b_baseline_plus_energy": BASE_FEATURES + ["planned_energy"],
                     "4b_baseline_audit_energy": BASE_FEATURES + ["audit_pre", "planned_energy"]}
INCREMENTAL_PAIRS = [("4_baseline_plus_audit", "2_baseline"), ("3_audit_only", "2_baseline"),
                     ("4_baseline_plus_audit", "3_audit_only"), ("4b_baseline_audit_energy", "2b_baseline_plus_energy")]
TARGETS = ("new_error", "prob_loss", "audit_cross")
COVERAGE_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)
N_FOLDS = 5
REQUIRED = ["model_name", "condition", "seed_id", "subvariant_A", "subvariant_B", "absC",
            "opt_margin_A", "opt_entropy_A", "opt_probs_A", "gold_idx_A", "opt_correct_A"]


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(K.REPO))
    except ValueError:
        return str(path)


def wcsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", _rel(path), len(df))


def require(path: Path, hint: str) -> None:
    if not path.exists():
        log.error("missing %s. %s", path, hint)
        sys.exit(2)


def load_p2(out: Path) -> pd.DataFrame:
    p = out / "p2_per_item.parquet"
    require(p, "P4 reads P2's per-item results: run p2 first.")
    d = pd.read_parquet(p)
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        log.error("p2_per_item.parquet lacks required columns %s (has %s)", missing, sorted(d.columns)[:40])
        sys.exit(2)
    if "error" in d:
        d = d[d["error"].isna()]
    return d


def read_matched(out: Path) -> dict:
    p = out / "p2_matched_alphas.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _parse_cond_id(cid: str) -> tuple[float | None, int | None]:
    m = re.search(r"_r(\d+)_a([0-9.]+)", str(cid))
    return (float(m.group(2)), int(m.group(1))) if m else (None, None)


def matched_point(matched: dict, model: str, cond: str) -> dict | None:
    """Read P2's p2_matched_alphas.json: {model: {"target_condition": "<family>_r<r>_a<a>",
    "controls": {family: {"alpha", "rank", ...}}}}. Older flat layouts are still accepted."""
    v = matched.get(model)
    if isinstance(v, (int, float)):
        return {"alpha": float(v)}
    if not isinstance(v, dict):
        return None
    tgt = str(v.get("target_condition", ""))
    if tgt.startswith(cond):
        a, r = _parse_cond_id(tgt)
        return {"alpha": 1.0 if a is None else a, "rank": 1 if r is None else r}
    c = (v.get("controls") or {}).get(cond)
    if isinstance(c, dict) and "alpha" in c:
        return {"alpha": float(c["alpha"]), "rank": int(c.get("rank", 1))}
    legacy = v.get(cond)
    if isinstance(legacy, (int, float)):
        return {"alpha": float(legacy)}
    return legacy if isinstance(legacy, dict) else None


def operating_point(d: pd.DataFrame, model: str, cond: str, matched: dict, cli_alpha, cli_rank) -> tuple[float | None, int | None, str]:
    """(alpha, rank, note) of the frozen treatment. CLI > p2_matched_alphas.json > the single
    point present in the parquet's TEST phase (P2 runs one point on test) > the single point
    overall; several points without a choice is an error."""
    ent = matched_point(matched, model, cond)
    alpha = cli_alpha if cli_alpha is not None else (ent or {}).get("alpha")
    rank = cli_rank if cli_rank is not None else (ent or {}).get("rank")
    src = "cli" if (cli_alpha is not None or cli_rank is not None) else ("p2_matched_alphas.json" if ent else "parquet")
    rows = d[(d["model_name"] == model) & (d["condition"] == cond)]
    pts = set()
    if "alpha" in rows and "rank" in rows:
        def points(r):
            return set(zip(pd.to_numeric(r["alpha"], errors="coerce").round(4), pd.to_numeric(r["rank"], errors="coerce")))
        pts = points(rows)
        if alpha is None and rank is None:
            test_pts = points(rows[rows["phase"] == "test"]) if "phase" in rows else set()
            if len(test_pts) == 1:
                alpha, rank = next(iter(test_pts)); src = "single point in the parquet test phase"
            elif len(pts) == 1:
                alpha, rank = next(iter(pts)); src = "single point in parquet"
            elif len(pts) > 1:
                log.error("%s/%s has several (alpha, rank) points %s: pass --alpha and --rank or provide "
                          "p2_matched_alphas.json", model, cond, sorted(pts)); sys.exit(2)
    return (None if alpha is None else float(alpha)), (None if rank is None else int(rank)), \
        "%s (points present: %s)" % (src, sorted(pts) if pts else "no alpha/rank columns")


# ---------------------------------------------------------------------------
# examples: one row per prompt
# ---------------------------------------------------------------------------

def _p_gold(probs_json, gi) -> float:
    try:
        p = json.loads(probs_json); gi = int(gi)
        return float(p[gi]) if 0 <= gi < len(p) else float("nan")
    except (TypeError, ValueError, json.JSONDecodeError):
        return float("nan")


def _bool(s: pd.Series) -> pd.Series:
    return s.map({True: 1.0, False: 0.0, 1: 1.0, 0: 0.0}).astype(float)


def unstack_sides(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Per-prompt readouts from pair rows: sides A and B unstacked, de-duplicated on
    (seed_id, subvariant), plus the worst (max) |C| over the pairs containing the prompt. Every
    value column is suffixed with `prefix` so pre and post frames merge without collisions."""
    parts = []
    for side in ("A", "B"):
        if f"opt_correct_{side}" not in rows:
            continue
        p = pd.DataFrame({"seed_id": rows["seed_id"].values, "subvariant": rows[f"subvariant_{side}"].values,
                          f"margin_{prefix}": pd.to_numeric(rows[f"opt_margin_{side}"], errors="coerce").values,
                          f"entropy_{prefix}": pd.to_numeric(rows[f"opt_entropy_{side}"], errors="coerce").values,
                          f"opt_correct_{prefix}": _bool(rows[f"opt_correct_{side}"]).values,
                          f"p_gold_{prefix}": [_p_gold(a, b) for a, b in zip(rows[f"opt_probs_{side}"], rows[f"gold_idx_{side}"])],
                          f"absC_{prefix}": pd.to_numeric(rows["absC"], errors="coerce").values})
        for opt in ("gen_valid", "gen_correct"):
            if f"{opt}_{side}" in rows:
                p[f"{opt}_{prefix}"] = _bool(rows[f"{opt}_{side}"]).values
        for tokc in (f"n_tokens_{side}", f"prompt_tokens_{side}"):
            if tokc in rows:
                p[f"n_tokens_{prefix}"] = pd.to_numeric(rows[tokc], errors="coerce").values; break
        energy = pd.Series(np.nan, index=range(len(rows)))
        for ec in (f"energy_total_{side}", f"energy_mean_{side}", "energy_gen", "energy_audit"):
            if ec in rows:                      # per-side energy when P2 recorded it, gaps filled pair-level
                energy = energy.fillna(pd.Series(pd.to_numeric(rows[ec], errors="coerce").values))
        p[f"planned_energy_{prefix}"] = energy.values
        parts.append(p)
    long = pd.concat(parts, ignore_index=True)
    agg = {c: "first" for c in long.columns if c not in ("seed_id", "subvariant", f"absC_{prefix}")}
    agg[f"absC_{prefix}"] = "max"
    return long.groupby(["seed_id", "subvariant"], as_index=False).agg(agg)


def _cached_tokenizer(repo_id: str):
    """Tokenizer from the local HF cache only; falls back to the snapshot path when the cached
    snapshot holds tokenizer files without config.json (as test_p1_cpu.load_cached_tokenizer)."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
    except Exception:
        from huggingface_hub import scan_cache_dir
        for r in scan_cache_dir().repos:
            if r.repo_id == repo_id:
                snap = sorted(r.revisions, key=lambda x: x.last_modified)[-1].snapshot_path
                return AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        raise


def token_lengths(model: str, ex: pd.DataFrame) -> tuple[pd.Series, str]:
    """Prompt token length: P2's column if present, else a locally cached tokenizer on the pentad
    prompt text, else a whitespace word count (the source is recorded in the report)."""
    if "n_tokens_pre" in ex and ex["n_tokens_pre"].notna().mean() > 0.9:
        return ex["n_tokens_pre"], "p2 column"
    pen = pd.read_parquet(K.PENTAD_CLEAN, columns=["seed_id", "slot", "subvariant", "prompt_text"])
    pen = pen[pen["slot"] == "c"]
    text = ex[["seed_id", "subvariant"]].merge(pen[["seed_id", "subvariant", "prompt_text"]], how="left",
                                               on=["seed_id", "subvariant"])["prompt_text"].fillna("")
    try:
        tok = _cached_tokenizer(HF_IDS[model])
        return pd.Series([len(tok(t, add_special_tokens=True)["input_ids"]) for t in text], index=ex.index), "cached tokenizer"
    except Exception as exc:                       # offline cache absent: honest proxy
        log.warning("tokenizer unavailable for %s (%s); using whitespace word count", model, str(exc)[:80])
        return pd.Series([len(t.split()) for t in text], index=ex.index), "whitespace word count (proxy)"


def build_examples(d: pd.DataFrame, model: str, args, manifest: pd.DataFrame, matched: dict) -> tuple[pd.DataFrame, dict]:
    dm = d[d["model_name"] == model]
    pre = dm[dm["condition"] == args.unedited_condition]
    post = dm[dm["condition"] == args.target_condition]
    alpha, rank, note = operating_point(d, model, args.target_condition, matched, args.alpha, args.rank)
    if "alpha" in post and alpha is not None:
        post = post[np.isclose(pd.to_numeric(post["alpha"], errors="coerce"), alpha, atol=1e-4)]
    if "rank" in post and rank is not None:
        post = post[pd.to_numeric(post["rank"], errors="coerce") == rank]
    info = {"alpha": alpha, "rank": rank, "operating_point_source": note, "n_pre_rows": len(pre), "n_post_rows": len(post)}
    if not len(pre) or not len(post):
        return pd.DataFrame(), info
    ex = unstack_sides(pre, "pre").merge(unstack_sides(post, "post"), on=["seed_id", "subvariant"], how="inner")
    if "planned_energy_post" in ex:                 # recorded on the edited rows; a deterministic
        ex["planned_energy"] = ex["planned_energy_post"]   # function of the unedited state and the basis
    ex["benchmark"] = ex["seed_id"].map(K.benchmark_of)
    ex = ex.merge(manifest[["seed_id", "split"]], on="seed_id", how="left")
    info["n_prompts_all"] = len(ex)
    info["n_prompts_outside_dev_test"] = int((~ex["split"].isin(["dev", "test"])).sum())
    ex = ex[ex["split"].isin(["dev", "test"])].reset_index(drop=True)
    ex["n_tokens"], info["token_length_source"] = token_lengths(model, ex)
    ex["audit_pre"] = ex["absC_pre"]
    info["has_generation"] = "gen_correct_pre" in ex and "gen_correct_post" in ex
    info["energy_coverage"] = float(ex["planned_energy"].notna().mean()) if "planned_energy" in ex else 0.0
    info["has_energy"] = info["energy_coverage"] > 0.9
    return ex, info


def make_targets(ex: pd.DataFrame, has_gen: bool) -> dict:
    """{target: (population mask, y, kind)}; definitions in the module docstring."""
    if has_gen:
        c_pre, c_post = ex["gen_correct_pre"], ex["gen_correct_post"]
        v_pre = ex["gen_valid_pre"] if "gen_valid_pre" in ex else pd.Series(1.0, index=ex.index)
        v_post = ex["gen_valid_post"] if "gen_valid_post" in ex else pd.Series(1.0, index=ex.index)
    else:
        c_pre, c_post = ex["opt_correct_pre"], ex["opt_correct_post"]
        v_pre = v_post = pd.Series(1.0, index=ex.index)
    pop_ne = (c_pre == 1) & (v_pre == 1)
    y_ne = (~((c_post == 1) & (v_post == 1))).astype(float)
    pop_pl = ex["p_gold_pre"].notna() & ex["p_gold_post"].notna()
    y_pl = ex["p_gold_pre"] - ex["p_gold_post"]
    pop_ac = ex["absC_pre"] > K.TAU
    y_ac = (ex["absC_post"] > K.TAU).astype(float)
    return {"new_error": (pop_ne, y_ne, "binary"), "prob_loss": (pop_pl, y_pl, "continuous"),
            "audit_cross": (pop_ac, y_ac, "binary")}


# ---------------------------------------------------------------------------
# models and metrics
# ---------------------------------------------------------------------------

def design(ex: pd.DataFrame, features: list[str], benchmarks: list[str]) -> np.ndarray:
    cols = []
    for f in features:
        if f == "benchmark":
            for b in benchmarks:
                cols.append((ex["benchmark"] == b).astype(float).to_numpy())
        else:
            cols.append(pd.to_numeric(ex[f], errors="coerce").fillna(pd.to_numeric(ex[f], errors="coerce").median()).to_numpy())
    return np.column_stack(cols) if cols else np.zeros((len(ex), 0))


def fit_predict(kind: str, features: list[str], tr: pd.DataFrame, y_tr: np.ndarray, te: pd.DataFrame,
                benchmarks: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probability or continuous prediction, ranking score) on `te`."""
    if not features:
        c = float(np.mean(y_tr))
        return np.full(len(te), c), np.full(len(te), c)
    Xtr, Xte = design(tr, features, benchmarks), design(te, features, benchmarks)
    if kind == "binary":
        if len(np.unique(y_tr)) < 2:
            c = float(np.mean(y_tr)); return np.full(len(te), c), np.full(len(te), c)
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        mdl.fit(Xtr, y_tr.astype(int))
        p = mdl.predict_proba(Xte)[:, 1]
        score = Xte[:, 0] if features == ["audit_pre"] else p        # raw audit score ranks predictor 3
        return p, score
    mdl = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    mdl.fit(Xtr, y_tr)
    yhat = mdl.predict(Xte)
    return yhat, yhat


def fast_auc(y, s) -> float:
    """Rank-based AUROC (Hanley and McNeil 1982), ties averaged."""
    from scipy.stats import rankdata
    y = np.asarray(y, float); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Logistic recalibration of logit(p) on y (Cox 1958): slope 1 and intercept 0 is ideal."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    if np.std(z) < 1e-9 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # sklearn/scipy 'iprint' solver-option notice
        m = LogisticRegression(penalty=None, max_iter=2000).fit(z.reshape(-1, 1), np.asarray(y, int))
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def constant_overrides(m: dict, kind: str) -> dict:
    """A constant predictor ranks nothing: AUROC 0.5 and AP = prevalence by definition, and no
    calibration slope. Out-of-fold constants differ slightly across folds, which would otherwise
    print a meaningless rank statistic."""
    if kind == "binary":
        m.update({"auroc": 0.5, "avg_precision": m["prevalence"], "cal_slope": float("nan"), "cal_intercept": float("nan"),
                  "fold_stat_mean": 0.5, "fold_stat_sd": 0.0})
    else:
        m.update({"spearman": float("nan"), "pearson": float("nan"), "fold_stat_mean": float("nan"), "fold_stat_sd": float("nan")})
    return m


def ece(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y, float); p = np.asarray(p, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def binary_metrics(y, p, score, prev_train: float) -> dict:
    y = np.asarray(y, float); p = np.asarray(p, float); score = np.asarray(score, float)
    both = len(np.unique(y)) > 1
    brier = float(np.mean((p - y) ** 2)); brier_c = float(np.mean((prev_train - y) ** 2))
    slope, inter = calibration_slope_intercept(y, p)
    return {"n": int(len(y)), "prevalence": float(y.mean()),
            "auroc": fast_auc(y, score) if both else float("nan"),
            "avg_precision": float(average_precision_score(y, score)) if both else float("nan"),
            "brier": brier, "brier_constant": brier_c,
            "brier_skill": (1 - brier / brier_c) if brier_c > 0 else float("nan"),
            "cal_slope": slope, "cal_intercept": inter, "cal_in_large": float(p.mean() - y.mean()),
            "ece": ece(y, p)}


def continuous_metrics(y, yhat) -> dict:
    from scipy.stats import pearsonr, spearmanr
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss_res = float(((y - yhat) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    const = np.std(yhat) < 1e-12 or len(y) < 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = float("nan") if const else float(spearmanr(y, yhat).correlation)
        pr = float("nan") if const else float(pearsonr(y, yhat)[0])
    return {"n": int(len(y)), "target_mean": float(y.mean()), "spearman": rho, "pearson": pr,
            "r2": (1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "rmse": float(np.sqrt(ss_res / len(y))), "mae": float(np.abs(y - yhat).mean())}


def _cluster_indices(clusters) -> list[np.ndarray]:
    codes, uniq = pd.factorize(np.asarray(clusters), sort=False)
    return [np.where(codes == i)[0] for i in range(len(uniq))]


def paired_bootstrap(y, s_a, s_b, clusters, stat, n_boot: int = K.N_BOOT, seed: int = K.RANDOM_SEED_V2) -> tuple[float, float, float]:
    """Seed-cluster percentile bootstrap of stat(y, s_a) - stat(y, s_b) on the same resamples."""
    y = np.asarray(y, float); s_a = np.asarray(s_a, float); s_b = np.asarray(s_b, float)
    idx_by = _cluster_indices(clusters)
    point = stat(y, s_a) - stat(y, s_b)
    if len(idx_by) < 2:
        return float(point), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    k = len(idx_by)
    for b in range(n_boot):
        idx = np.concatenate([idx_by[i] for i in rng.integers(0, k, size=k)])
        draws[b] = stat(y[idx], s_a[idx]) - stat(y[idx], s_b[idx])
    if not np.isfinite(draws).any():
        return float(point), float("nan"), float("nan")
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def single_bootstrap(y, s, clusters, stat, n_boot: int = K.N_BOOT, seed: int = K.RANDOM_SEED_V2) -> tuple[float, float]:
    y = np.asarray(y, float); s = np.asarray(s, float)
    idx_by = _cluster_indices(clusters)
    if len(idx_by) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    k = len(idx_by)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = np.concatenate([idx_by[i] for i in rng.integers(0, k, size=k)])
        draws[b] = stat(y[idx], s[idx])
    if not np.isfinite(draws).any():
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def _spearman_stat(y, s) -> float:
    from scipy.stats import spearmanr
    if np.std(s) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(y, s).correlation)


# ---------------------------------------------------------------------------
# protocol: dev cross-validation, one test evaluation, coverage curve
# ---------------------------------------------------------------------------

def predictor_table(has_energy: bool) -> dict:
    preds = dict(PREDICTORS)
    if has_energy:
        preds.update(PREDICTORS_ENERGY)
    return preds


def cv_dev(dev: pd.DataFrame, y: pd.Series, kind: str, preds: dict, benchmarks: list[str]) -> tuple[list[dict], dict]:
    """GroupKFold by seed on DEV: pooled out-of-fold metrics and per-fold AUROC / Spearman."""
    groups = dev["seed_id"].to_numpy()
    n_groups = len(np.unique(groups))
    n_splits = min(N_FOLDS, n_groups)
    rows, oof = [], {}
    if n_splits < 2:
        return rows, oof
    gkf = GroupKFold(n_splits=n_splits)
    yv = y.to_numpy(float)
    for name, feats in preds.items():
        p_oof = np.full(len(dev), np.nan); s_oof = np.full(len(dev), np.nan); per_fold = []
        prev_tr = []
        for tr_i, te_i in gkf.split(dev, yv, groups):
            p, s = fit_predict(kind, feats, dev.iloc[tr_i], yv[tr_i], dev.iloc[te_i], benchmarks)
            p_oof[te_i] = p; s_oof[te_i] = s
            prev_tr.append(float(yv[tr_i].mean()))
            per_fold.append(fast_auc(yv[te_i], s) if kind == "binary" else _spearman_stat(yv[te_i], s))
        oof[name] = (p_oof, s_oof)
        m = binary_metrics(yv, p_oof, s_oof, float(np.mean(prev_tr))) if kind == "binary" else continuous_metrics(yv, p_oof)
        finite = [v for v in per_fold if np.isfinite(v)]
        m.update({"predictor": name, "features": "+".join(feats) or "constant (AUROC 0.5 by definition)", "phase": "dev_cv",
                  "n_folds": n_splits, "n_seeds": int(n_groups),
                  "fold_stat_mean": float(np.mean(finite)) if finite else float("nan"),
                  "fold_stat_sd": float(np.std(finite)) if finite else float("nan"),
                  "fold_stat": "auroc" if kind == "binary" else "spearman"})
        rows.append(constant_overrides(m, kind) if not feats else m)
    return rows, oof


def final_test(dev: pd.DataFrame, y_dev: pd.Series, test: pd.DataFrame, y_test: pd.Series, kind: str, preds: dict,
               benchmarks: list[str]) -> tuple[list[dict], dict]:
    """One fit on all DEV rows, one evaluation on TEST rows, with seed-cluster AUROC intervals."""
    rows, out = [], {}
    yd, yt = y_dev.to_numpy(float), y_test.to_numpy(float)
    stat = fast_auc if kind == "binary" else _spearman_stat
    for name, feats in preds.items():
        p, s = fit_predict(kind, feats, dev, yd, test, benchmarks)
        out[name] = (p, s)
        m = binary_metrics(yt, p, s, float(yd.mean())) if kind == "binary" else continuous_metrics(yt, p)
        lo, hi = single_bootstrap(yt, s, test["seed_id"].to_numpy(), stat) if feats else (float("nan"), float("nan"))
        m.update({"predictor": name, "features": "+".join(feats) or "constant (AUROC 0.5 by definition)", "phase": "test",
                  "n_seeds": int(test["seed_id"].nunique()), "n_train": int(len(dev)), "train_prevalence": float(yd.mean()),
                  ("auroc_lo" if kind == "binary" else "spearman_lo"): lo, ("auroc_hi" if kind == "binary" else "spearman_hi"): hi})
        rows.append(constant_overrides(m, kind) if not feats else m)
    return rows, out


def incremental_rows(y, preds_out: dict, clusters, kind: str, phase: str) -> list[dict]:
    stat = fast_auc if kind == "binary" else _spearman_stat
    rows = []
    for a, b in INCREMENTAL_PAIRS:
        if a in preds_out and b in preds_out:
            d, lo, hi = paired_bootstrap(y, preds_out[a][1], preds_out[b][1], clusters, stat)
            rows.append({"phase": phase, "predictor": a, "versus": b, "stat": "auroc" if kind == "binary" else "spearman",
                         "delta": d, "delta_lo": lo, "delta_hi": hi, "excludes_zero": bool(np.isfinite(lo) and lo > 0),
                         "n": int(len(y))})
    return rows


def coverage_curve(y, p, failing, model: str, target: str, predictor: str) -> list[dict]:
    """Decline the q fraction of prompts with the highest forecast risk; edit the rest."""
    y = np.asarray(y, float); p = np.asarray(p, float); failing = np.asarray(failing, float)
    n = len(y); order = np.argsort(-p, kind="stable")
    n_adv = y.sum(); n_fail = failing.sum()
    rows = []
    for q in COVERAGE_GRID:
        k = int(round(q * n))
        declined, edited = order[:k], order[k:]
        rows.append({"model_name": model, "target": target, "predictor": predictor, "decline_frac": float(q),
                     "coverage": float(1 - q), "n_declined": k,
                     "adverse_avoided_frac": float(y[declined].sum() / n_adv) if n_adv > 0 else float("nan"),
                     "loss_remaining_rate": float(y[edited].sum() / n),
                     "precision_declined": float(y[declined].mean()) if k > 0 else float("nan"),
                     "suppression_coverage": float(failing[edited].sum() / n_fail) if n_fail > 0 else float("nan")})
    return rows


def random_reference_curve(y, model: str, target: str) -> list[dict]:
    """The expected curve of a random ordering: the no-edit policy is q = 1, edit-all is q = 0."""
    y = np.asarray(y, float); n = len(y); prev = float(y.mean())
    return [{"model_name": model, "target": target, "predictor": "random_order_expected", "decline_frac": float(q),
             "coverage": float(1 - q), "n_declined": int(round(q * n)), "adverse_avoided_frac": float(q),
             "loss_remaining_rate": prev * (1 - float(q)), "precision_declined": prev if q > 0 else float("nan"),
             "suppression_coverage": 1 - float(q)} for q in COVERAGE_GRID]


def audit_cost(d: pd.DataFrame, model: str, args) -> dict:
    dm = d[d["model_name"] == model]
    pre = dm[dm["condition"] == args.unedited_condition]; post = dm[dm["condition"] == args.target_condition]
    out = {"audit_forward_passes_per_pair": 3, "baseline_feature_passes_per_prompt": 1,
           "note": "the audit needs the source, patched and clean passes per pair (p1_intervention.commutator); the "
                   "baseline features come from the one batched option-scoring pass the evaluation performs anyway"}
    for col, key in (("sec_audit", "sec_audit_unedited_mean"), ("sec_options", "sec_options_unedited_mean"),
                     ("sec_generation", "sec_generation_unedited_mean")):
        if col in pre:
            out[key] = float(pd.to_numeric(pre[col], errors="coerce").mean())
    for col, key in (("sec_options", "sec_options_edited_mean"), ("sec_generation", "sec_generation_edited_mean"),
                     ("sec_audit", "sec_audit_edited_mean")):
        if col in post:
            out[key] = float(pd.to_numeric(post[col], errors="coerce").mean())
    if "sec_audit_unedited_mean" in out and "sec_options_unedited_mean" in out and out["sec_options_unedited_mean"] > 0:
        out["audit_seconds_over_option_pass_seconds"] = out["sec_audit_unedited_mean"] / out["sec_options_unedited_mean"]
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _f(v, nd: int = 3) -> str:
    try:
        return "nan" if v is None or not np.isfinite(float(v)) else f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def report_model(model: str, info: dict, metrics: pd.DataFrame, incr: pd.DataFrame, cov: pd.DataFrame, cost: dict,
                 has_energy: bool) -> tuple[list[str], str]:
    L = ["## %s" % K.DISPLAY.get(model, model), "",
         "Treatment: %s at alpha = %s, rank = %s (%s). Prompts: %d dev, %d test; %d prompts outside dev/test dropped. "
         "Outcome correctness: %s. Token length: %s. Planned-energy predictor: %s." % (
             info.get("target_condition"), _f(info.get("alpha")), info.get("rank"), info.get("operating_point_source"),
             info.get("n_dev", 0), info.get("n_test", 0), info.get("n_prompts_outside_dev_test", 0),
             "generation (gen_correct, gen_valid)" if info.get("has_generation") else "canonical option scoring (no generation columns in P2)",
             info.get("token_length_source"),
             "available" if has_energy else "unavailable (recorded for %.0f%% of prompts; needs more than 90%%)"
             % (100 * info.get("energy_coverage", 0.0))), ""]
    decision = "REMOVE the practical-forecast claim"
    for target in TARGETS:
        mt = metrics[metrics["target"] == target]
        mt = mt[mt["predictor"].astype(str) != "n/a"] if "predictor" in mt else mt
        if not len(mt) or "n" not in mt.columns or mt["n"].isna().all():
            note = ""
            deg = metrics[(metrics["target"] == target) & (metrics.get("predictor", pd.Series(dtype=str)).astype(str) == "n/a")]
            if len(deg) and "note" in deg:
                note = ": " + str(deg["note"].iloc[0])
            L += ["### %s" % target, "", "no rows (degenerate population or single class%s)" % note, ""]; continue
        kind = "continuous" if target == "prob_loss" else "binary"
        L += ["### target = %s (%s)" % (target, kind), ""]
        if target == "audit_cross":
            L.append("Secondary target only. Predictor 3 is the score whose threshold crossing defines this target; the "
                     "AUROC is partly mechanical (Next_Plan.md Section 2.6) and cannot substitute for damage.")
            L.append("")
        if kind == "binary":
            L += ["| predictor | phase | n | prevalence | AUROC [95% CI] | AP | Brier | Brier const | skill | cal slope | cal int | ECE | fold AUROC mean (sd) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for _, r in mt.iterrows():
                ci = " [%s, %s]" % (_f(r.get("auroc_lo")), _f(r.get("auroc_hi"))) if r["phase"] == "test" else ""
                fold = "%s (%s)" % (_f(r.get("fold_stat_mean")), _f(r.get("fold_stat_sd"))) if r["phase"] == "dev_cv" else "-"
                L.append("| %s | %s | %d | %s | %s%s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["predictor"], r["phase"], r["n"], _f(r["prevalence"]), _f(r["auroc"]), ci, _f(r["avg_precision"]),
                    _f(r["brier"]), _f(r["brier_constant"]), _f(r["brier_skill"]), _f(r["cal_slope"]), _f(r["cal_intercept"]),
                    _f(r["ece"]), fold))
        else:
            L += ["| predictor | phase | n | target mean | Spearman [95% CI] | Pearson | R^2 | RMSE | MAE | fold Spearman mean (sd) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for _, r in mt.iterrows():
                ci = " [%s, %s]" % (_f(r.get("spearman_lo")), _f(r.get("spearman_hi"))) if r["phase"] == "test" else ""
                fold = "%s (%s)" % (_f(r.get("fold_stat_mean")), _f(r.get("fold_stat_sd"))) if r["phase"] == "dev_cv" else "-"
                L.append("| %s | %s | %d | %s | %s%s | %s | %s | %s | %s | %s |" % (
                    r["predictor"], r["phase"], r["n"], _f(r["target_mean"]), _f(r["spearman"]), ci, _f(r["pearson"]),
                    _f(r["r2"]), _f(r["rmse"]), _f(r["mae"]), fold))
        it = incr[incr["target"] == target] if len(incr) and "target" in incr else incr.iloc[0:0]
        if len(it):
            L += ["", "Incremental value (paired seed-cluster bootstrap of the difference, 95% percentile interval):", "",
                  "| phase | predictor | versus | stat | delta | lo | hi | excludes zero |", "|---|---|---|---|---|---|---|---|"]
            for _, r in it.iterrows():
                L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (r["phase"], r["predictor"], r["versus"], r["stat"],
                                                                        _f(r["delta"]), _f(r["delta_lo"]), _f(r["delta_hi"]), r["excludes_zero"]))
        ct = cov[(cov["target"] == target) & cov["decline_frac"].isin([0.1, 0.2, 0.3])]
        if len(ct):
            L += ["", "Utility loss versus retained coverage on TEST (decline the highest-risk fraction; the no-edit policy is "
                  "coverage 0, the edit-all policy is coverage 1):", "",
                  "| predictor | decline | coverage | new errors avoided | loss remaining (rate) | precision of declined | suppression coverage |",
                  "|---|---|---|---|---|---|---|"]
            for _, r in ct.sort_values(["predictor", "decline_frac"]).iterrows():
                L.append("| %s | %s | %s | %s | %s | %s | %s |" % (r["predictor"], _f(r["decline_frac"], 2), _f(r["coverage"], 2),
                                                                  _f(r["adverse_avoided_frac"]), _f(r["loss_remaining_rate"]),
                                                                  _f(r["precision_declined"]), _f(r["suppression_coverage"])))
        L.append("")
        if target == "new_error":
            t42 = it[(it["phase"] == "test") & (it["predictor"] == "4_baseline_plus_audit") & (it["versus"] == "2_baseline")]
            t4b = it[(it["phase"] == "test") & (it["predictor"] == "4b_baseline_audit_energy") & (it["versus"] == "2b_baseline_plus_energy")]
            beats_2 = bool(len(t42) and t42["excludes_zero"].iloc[0])
            beats_2b = (not has_energy) or bool(len(t4b) and t4b["excludes_zero"].iloc[0])
            if beats_2 and beats_2b:
                decision = ("KEEP, narrowly: on TEST the audit adds AUROC %s [%s, %s] over the confidence/size baseline"
                            "%s for the primary behavioural target on %s. Scope: per-example forecasting within this model at "
                            "the frozen treatment; not a cross-model claim." % (
                                _f(t42["delta"].iloc[0]), _f(t42["delta_lo"].iloc[0]), _f(t42["delta_hi"].iloc[0]),
                                (" and %s [%s, %s] over the baseline-plus-energy predictor" % (
                                    _f(t4b["delta"].iloc[0]), _f(t4b["delta_lo"].iloc[0]), _f(t4b["delta_hi"].iloc[0]))) if has_energy and len(t4b) else "",
                                K.DISPLAY.get(model, model)))
            else:
                why = "no TEST comparison was possible (degenerate target)" if not len(t42) else \
                    "AUROC(4) - AUROC(2) = %s [%s, %s] on TEST%s" % (
                        _f(t42["delta"].iloc[0]), _f(t42["delta_lo"].iloc[0]), _f(t42["delta_hi"].iloc[0]),
                        ("; AUROC(4b) - AUROC(2b) = %s [%s, %s]" % (_f(t4b["delta"].iloc[0]), _f(t4b["delta_lo"].iloc[0]),
                                                                     _f(t4b["delta_hi"].iloc[0]))) if len(t4b) else "")
                decision = ("REMOVE the practical-forecast claim for %s: the audit adds no held-out prediction of new errors "
                            "beyond confidence and size (%s; the interval does not exclude zero). Retain the ordinal audit "
                            "analysis as a limited diagnostic only." % (K.DISPLAY.get(model, model), why))
    L += ["### Audit cost", ""]
    L.append("Forward passes: audit %d per pair versus %d batched option pass per prompt for the baseline features. %s" % (
        cost["audit_forward_passes_per_pair"], cost["baseline_feature_passes_per_prompt"],
        "; ".join("%s = %s" % (k, _f(v)) for k, v in cost.items() if k.startswith("sec_") or k.startswith("audit_seconds")) or
        "P2 recorded no timing columns."))
    L += ["", "**Decision (plan stop condition): %s**" % decision, ""]
    return L, decision


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_model(d: pd.DataFrame, model: str, args, manifest: pd.DataFrame, matched: dict):
    ex, info = build_examples(d, model, args, manifest, matched)
    info["target_condition"] = args.target_condition
    if not len(ex):
        log.warning("%s: no usable prompts (pre rows %s, post rows %s)", model, info.get("n_pre_rows"), info.get("n_post_rows"))
        return info, [], [], []
    dev, test = ex[ex["split"] == "dev"].reset_index(drop=True), ex[ex["split"] == "test"].reset_index(drop=True)
    info.update({"n_dev": len(dev), "n_test": len(test), "n_dev_seeds": int(dev["seed_id"].nunique()),
                 "n_test_seeds": int(test["seed_id"].nunique())})
    benchmarks = sorted(ex["benchmark"].unique())
    preds = predictor_table(info["has_energy"])
    targets = make_targets(ex, info["has_generation"])
    metric_rows, incr_rows, cov_rows = [], [], []
    for tname, (pop, y, kind) in targets.items():
        dmask, tmask = pop & (ex["split"] == "dev"), pop & (ex["split"] == "test")
        ed, et = ex[dmask].reset_index(drop=True), ex[tmask].reset_index(drop=True)
        yd, yt = y[dmask].reset_index(drop=True), y[tmask].reset_index(drop=True)
        base = {"model_name": model, "target": tname, "kind": kind, "n_population_dev": len(ed), "n_population_test": len(et),
                "caveat": ("predictor 3 is the score whose TAU crossing defines the target; partly mechanical"
                           if tname == "audit_cross" else "")}
        degenerate = (kind == "binary" and (yd.nunique() < 2 or yt.nunique() < 2)) or len(ed) < 10 or len(et) < 10
        if degenerate:
            metric_rows.append({**base, "predictor": "n/a", "phase": "n/a",
                                "note": "degenerate: fewer than 10 rows or a single class in dev or test"})
            log.warning("%s/%s degenerate (dev %d, test %d)", model, tname, len(ed), len(et)); continue
        rows, oof = cv_dev(ed, yd, kind, preds, benchmarks)
        metric_rows += [{**base, **r} for r in rows]
        incr_rows += [{**base, **r} for r in incremental_rows(yd.to_numpy(float), oof, ed["seed_id"].to_numpy(), kind, "dev_cv")]
        rows, out = final_test(ed, yd, et, yt, kind, preds, benchmarks)
        metric_rows += [{**base, **r} for r in rows]
        incr_rows += [{**base, **r} for r in incremental_rows(yt.to_numpy(float), out, et["seed_id"].to_numpy(), kind, "test")]
        if kind == "binary":
            failing = (et["absC_pre"] > K.TAU).to_numpy(float)
            for name, (p, s) in out.items():
                if name != "1_base_rate":
                    cov_rows += coverage_curve(yt.to_numpy(float), s, failing, model, tname, name)
            cov_rows += random_reference_curve(yt.to_numpy(float), model, tname)
    return info, metric_rows, incr_rows, cov_rows


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Design")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=K.MODELS)
    ap.add_argument("--target-condition", default="cure_centred_svd")
    ap.add_argument("--unedited-condition", default="unedited")
    ap.add_argument("--alpha", type=float, default=None, help="frozen operating strength (else p2_matched_alphas.json)")
    ap.add_argument("--rank", type=int, default=None, help="frozen operating rank (else p2_matched_alphas.json)")
    ap.add_argument("--out-dir", default=None, help="override results/v2 (smoke tests only)")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    # scipy >= 1.18 dropped the 'iprint' option that sklearn's lbfgs wrapper still passes; the
    # notice has no effect on the fit and would otherwise print once per logistic regression.
    warnings.filterwarnings("ignore", message="Unknown solver options")
    out = Path(args.out_dir) if args.out_dir else K.OUT_V2
    out.mkdir(parents=True, exist_ok=True)
    d = load_p2(out)
    manifest = pd.read_csv(K.OUT_P0 / "split_manifest.csv")[["seed_id", "split"]]
    matched = read_matched(out)
    metrics, incr, cov, report = [], [], [], []
    report += ["# P4 independent risk forecast", "",
               "Generated %s. Per-example forecasting within each model at the frozen cure_centred_svd treatment. Primary "
               "target: new error (wrong or invalid) after editing among initially correct prompts. Dev: GroupKFold by seed "
               "(%d folds); test: one evaluation. No cross-model model is fitted to the four aggregate model points."
               % (K.utc_now(), N_FOLDS), ""]
    decisions = {}
    for m in args.models:
        if not (d["model_name"] == m).any():
            log.warning("no P2 rows for %s; skipped", m); report += ["## %s" % K.DISPLAY.get(m, m), "", "no P2 rows", ""]; continue
        info, mr, ir, cr = run_model(d, m, args, manifest, matched)
        metrics += mr; incr += ir; cov += cr
        mdf, idf, cdf = pd.DataFrame(mr), pd.DataFrame(ir), pd.DataFrame(cr)
        if not len(mdf):
            report += ["## %s" % K.DISPLAY.get(m, m), "", "no usable prompts: %s" % json.dumps(info, default=K._json_default), ""]
            continue
        if not len(idf):
            idf = pd.DataFrame(columns=["target", "phase", "predictor", "versus", "stat", "delta", "delta_lo", "delta_hi", "excludes_zero"])
        if not len(cdf):
            cdf = pd.DataFrame(columns=["target", "predictor", "decline_frac", "coverage", "adverse_avoided_frac",
                                        "loss_remaining_rate", "precision_declined", "suppression_coverage"])
        lines, decision = report_model(m, info, mdf, idf, cdf, audit_cost(d, m, args), info.get("has_energy", False))
        report += lines; decisions[m] = decision
    wcsv(pd.DataFrame(metrics), out / "p4_forecast_metrics.csv")
    wcsv(pd.DataFrame(incr), out / "p4_incremental_tests.csv")
    wcsv(pd.DataFrame(cov), out / "p4_coverage_curve.csv")
    report += ["## Summary of decisions", ""] + ["- %s: %s" % (K.DISPLAY.get(m, m), v) for m, v in decisions.items()] + [""]
    if decisions and all(v.startswith("REMOVE") for v in decisions.values()):
        report.append("Overall: REMOVE the practical-forecast claim; retain the ordinal audit analysis as a clearly limited diagnostic.")
    elif decisions:
        report.append("Overall: the forecast claim can be kept only for the models listed as KEEP, with the stated scope; "
                      "state the models where it was removed.")
    (out / "p4_report.md").write_text("\n".join(report), encoding="utf-8")
    log.info("wrote %s", _rel(out / "p4_report.md"))
    print("\n".join(l for l in report if l.startswith("- ") or l.startswith("Overall")))


if __name__ == "__main__":
    main()
