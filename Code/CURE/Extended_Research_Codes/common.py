"""
common.py -- shared paths, constants and helpers for the Extended Research Codes.

Implements the P0 contract of Submission2/Next_Plan.md: every quantity is read from a named
artifact, every output states its definition and denominator, and nothing here mutates an
existing result file. GPU stages (P1+) import the same paths so the whole package agrees on
where inputs live and where outputs go.

Layout
  inputs   Code/CURE/results/*.parquet|json          (existing, never modified)
           Code/audit/results/cdva_results.parquet
           Code/audit/results/tist/e0/cdva_degeneracy_mask.parquet
           Code/audit/Dataset/seeds/pentad_dataset{,_clean}.parquet
  outputs  Code/CURE/results/reanalysis_v2/            (P0, CPU-only)
           Code/CURE/results/v2/                       (P1 to P4, GPU)

Part of the CURE codebase (ICLR 2027 submission, Submission2).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                    # Code/CURE/Extended_Research_Codes
CURE = HERE.parent                                        # Code/CURE
CODE = CURE.parent                                        # Code
REPO = CODE.parent                                        # repo root
AUDIT = CODE / "audit"

# Make the existing CURE package importable (config_cure, erase, experiments, ...) and, through
# config_cure, the audit stack (load_osm, osm_behavioral, utils_attention, cdva_patching).
for p in (str(CURE), str(AUDIT), str(AUDIT / "GPU_CPU")):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = CURE / "results"
OUT_P0 = RESULTS / "reanalysis_v2"
# EXT_V2_DIR lets several VMs (one per model) write disjoint output directories, e.g.
# results/v2_qwen2.5-7b-instruct, that are merged into results/v2 afterwards
# (Extended_Research_Codes/merge_v2.py). Default is the single shared directory.
OUT_V2 = RESULTS / os.environ.get("EXT_V2_DIR", "v2")
LOGS = CURE / "logs"

CDVA_PATH = AUDIT / "results" / "cdva_results.parquet"
MASK_PATH = AUDIT / "results" / "tist" / "e0" / "cdva_degeneracy_mask.parquet"
PENTAD_RAW = AUDIT / "Dataset" / "seeds" / "pentad_dataset.parquet"
PENTAD_CLEAN = AUDIT / "Dataset" / "seeds" / "pentad_dataset_clean.parquet"
BEHAVIORAL_PATH = AUDIT / "results" / "behavioral_results.parquet"

MODELS = ["llama-3.1-8b-instruct", "qwen2.5-7b-instruct",
          "gemma-2-2b-it", "phi-4-mini-instruct"]
DISPLAY = {"llama-3.1-8b-instruct": "Llama-3.1-8B", "qwen2.5-7b-instruct": "Qwen2.5-7B",
           "gemma-2-2b-it": "Gemma-2-2B", "phi-4-mini-instruct": "Phi-4-mini"}
PATCHING_LIB = {"llama-3.1-8b-instruct": "transformer_lens", "qwen2.5-7b-instruct": "nnsight",
                "gemma-2-2b-it": "transformer_lens", "phi-4-mini-instruct": "nnsight"}

# Values copied from Code/CURE/config_cure.py so P0 never needs the .env or the audit config
# import. If config_cure changes these, p0_claims_ledger flags the drift.
TAU = 0.7644                 # audit threshold compared against raw |C| by CURE
ERASE_RANKS = [1, 2, 4, 8]
SENTINEL_RANK = 9            # "not repaired at any tested rank"
SUBSPACE_PAIRS = 600
SWEEP_SUBSET = 1000
RANDOM_SEED_CURE = 20260101  # config_cure.RANDOM_SEED default
MAX_UTILITY_COST = 0.15

RANDOM_SEED_V2 = 20260912    # every new stochastic step in this package
N_BOOT = 2000

log = logging.getLogger("cure.ext")


def setup_logging(name: str) -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(LOGS / "extended_research.log", encoding="utf-8")],
    )
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def rel(path: Path) -> str:
    """Repo-relative for logging when inside the repo, absolute otherwise (scratch dirs)."""
    try:
        return str(Path(path).relative_to(REPO))
    except ValueError:
        return str(path)


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", rel(path), len(df))
    return path


def write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    log.info("wrote %s", rel(path))
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# result readers (never write)
# ---------------------------------------------------------------------------

def read_sweep(model: str) -> pd.DataFrame:
    """cure_recovery_sweep_<model>.parquet: 1000 pairs x ranks {1,2,4,8}."""
    d = pd.read_parquet(RESULTS / f"cure_recovery_sweep_{model}.parquet")
    d["benchmark"] = d["seed_id"].astype(str).str.split("_", n=1).str[0]
    return d


def read_prognosis(model: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS / f"cure_prognosis_{model}.parquet")


def read_final(model: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS / f"cure_final_{model}.parquet")


def read_heldout(model: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS / f"tacl_extra_{model}.parquet")


def read_rankcurve(model: str) -> dict:
    p = RESULTS / f"cure_rankcurve_{model}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def read_cdva() -> pd.DataFrame:
    return pd.read_parquet(CDVA_PATH)


def read_mask() -> pd.DataFrame:
    return pd.read_parquet(MASK_PATH)


def benchmark_of(seed_id: str) -> str:
    return str(seed_id).split("_", 1)[0]


# ---------------------------------------------------------------------------
# The pair loader and the stratified subset, replicated WITHOUT a model so the fit / sweep /
# evaluation pair sets of the shipped runs can be reconstructed exactly on CPU. The logic is a
# line-for-line copy of experiments._load_pairs (minus tokenizer positions, which do not affect
# set membership) and experiments.stratified_subset. p0_data_integrity verifies the
# reconstruction against the saved sweep parquet before using it.
# ---------------------------------------------------------------------------

def _bias_answer(c_variants: pd.DataFrame) -> str:
    """Copy of cdva_patching._get_bias_answer (priority: gold first word, else swap token)."""
    if "gold_answer" in c_variants.columns:
        gold_vals = c_variants["gold_answer"].dropna().unique()
        if len(gold_vals) > 0:
            gold = str(gold_vals[0]).strip()
            if gold and gold.lower() != "unknown":
                return gold.split()[0] if gold.split() else gold
    if "swap_token" in c_variants.columns:
        for tok in c_variants["swap_token"].dropna().astype(str):
            tok = tok.strip()
            if tok and tok.lower() not in {"none", "nan", ""}:
                return tok.split()[0] if tok.split() else tok
    return ""


def load_pairs_cpu(model: str, pentad_path: Path = PENTAD_RAW) -> list[dict]:
    """Same filter and order as experiments._load_pairs, on CPU. pentad_path lets the caller
    compare the shipped loader (raw pentad, no mask) with the clean pentad."""
    cdva = read_cdva()
    cdva = cdva[(cdva["model_name"] == model)
                & (cdva["position_fallback_used"] == False)        # noqa: E712
                & (cdva["success_flag"] == True)].copy()           # noqa: E712
    pentad = pd.read_parquet(pentad_path)
    pc = pentad[pentad["slot"] == "c"]
    look = {(r["seed_id"], r["subvariant"]): (str(r["prompt_text"]), str(r.get("swap_token", "")))
            for _, r in pc.iterrows()}
    bias_by_seed = {sid: _bias_answer(g) for sid, g in pc.groupby("seed_id")}
    pairs = []
    for _, pr in cdva.iterrows():
        sid = pr["seed_id"]
        ka = (sid, pr["pair_A_subvariant"]); kb = (sid, pr["pair_B_subvariant"])
        if ka not in look or kb not in look:
            continue
        pa, swa = look[ka]; pb, swb = look[kb]
        bias = bias_by_seed.get(sid, "")
        if not bias or not pa.strip() or not pb.strip() or pa.strip() == pb.strip():
            continue
        pairs.append({
            "seed_id": sid, "subvariant_A": pr["pair_A_subvariant"],
            "subvariant_B": pr["pair_B_subvariant"],
            "orig_delta": float(pr.get("delta_logit", float("nan"))),
            "benchmark": benchmark_of(sid),
        })
    return pairs


def stratified_subset_cpu(pairs: list[dict], n: int, seed: int) -> list[dict]:
    """Copy of experiments.stratified_subset."""
    if not n or n >= len(pairs):
        return list(pairs)
    rng = random.Random(seed)
    by_src: dict[str, list] = {}
    for p in pairs:
        by_src.setdefault(str(p["seed_id"]).split("_", 1)[0], []).append(p)
    out = []
    total = len(pairs)
    for grp in by_src.values():
        k = max(1, round(n * len(grp) / total))
        rng.shuffle(grp)
        out.extend(grp[:k])
    rng.shuffle(out)
    return out[:n]


def pair_key(p: dict) -> tuple:
    return (p["seed_id"], p["subvariant_A"], p["subvariant_B"])


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def seed_cluster_bootstrap(df: pd.DataFrame, stat_fn, cluster_col: str = "seed_id",
                           n_boot: int = N_BOOT, seed: int = RANDOM_SEED_V2,
                           alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap that resamples whole clusters (all pairs of a seed move together),
    per Efron and Tibshirani (1993). Returns (point, lo, hi).

    Generic but slow path: stat_fn(DataFrame) -> float, one pandas concat per draw. Use
    ratio_bootstrap() below for any statistic that is a ratio of two sums over rows, which
    covers every metric in this package and runs in vectorised numpy."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(cluster_col, sort=False)]
    point = float(stat_fn(df))
    if len(groups) < 2:
        return point, float("nan"), float("nan")
    draws = np.empty(n_boot)
    idx = np.arange(len(groups))
    for b in range(n_boot):
        take = rng.choice(idx, size=len(groups), replace=True)
        draws[b] = stat_fn(pd.concat([groups[i] for i in take], ignore_index=True))
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def ratio_bootstrap(num: np.ndarray, den: np.ndarray, clusters: np.ndarray,
                    transform=None, n_boot: int = N_BOOT, seed: int = RANDOM_SEED_V2,
                    alpha: float = 0.05) -> tuple[float, float, float]:
    """Seed-clustered percentile bootstrap for a statistic of the form
        transform( sum(num) / sum(den) )
    over rows, resampling clusters with replacement. Any mean of a per-row quantity is
    num=quantity, den=1; a ratio of means is num=post, den=pre. `transform` defaults to the
    identity (e.g. pass lambda r: 1 - r for R = 1 - mean(post)/mean(pre)).

    Runs in O(n_boot x n_clusters) numpy, no pandas in the loop."""
    num = np.asarray(num, float); den = np.asarray(den, float)
    clusters = np.asarray(clusters)
    transform = transform or (lambda r: r)
    codes, uniq = pd.factorize(clusters, sort=False)
    k = len(uniq)
    cnum = np.bincount(codes, weights=num, minlength=k)
    cden = np.bincount(codes, weights=den, minlength=k)
    point = float(transform(cnum.sum() / cden.sum())) if cden.sum() != 0 else float("nan")
    if k < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    take = rng.integers(0, k, size=(n_boot, k))
    bn = cnum[take].sum(axis=1)
    bd = cden[take].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = transform(bn / bd)
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def n_clusters(df: pd.DataFrame, cluster_col: str = "seed_id") -> int:
    return int(df[cluster_col].nunique())
