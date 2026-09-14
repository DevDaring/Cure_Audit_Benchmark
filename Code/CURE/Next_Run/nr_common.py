"""
nr_common.py -- shared paths, protocol record, hashing and statistics for the final research
cycle of Submission2 (Submission2/Next_Plan.md, 14 September 2026, phases F0-F4).

Reuses Code/CURE/Extended_Research_Codes (common paths, HookedEdit, readouts, LEACE helper)
and never modifies anything under results/v2 or results/reanalysis_v2. Every new output is
written under results/final_20260914/ with the protocol hash of the run that produced it.

Layout of results/final_20260914/ (planned outputs of Next_Plan.md Section 11)
  final_protocol.json      frozen choices, versions, hashes, budget cap and rate
  source_manifest.csv      every fresh / control / development seed with its provenance
  exclusions.csv           every candidate item excluded before inference and why
  basis_manifest.json      the reused rank-one bases and their fit provenance
  runtime_budget.json      measured seconds per item and the GPU-hour accounting
  calibration.json         energy denominators, targets, alphas, achieved energies (dev)
  identity_checks.json     hook checks per model
  per_item.parquet         one row per model x seed x side x condition (resumable)
  raw_generations.jsonl    the raw generated strings
  achieved_energy.parquet  per row energy bookkeeping (test)
  baseline_validation.csv  F3 coherent-label LEACE / mean-difference diagnostics
  legacy_behaviour.csv     F0 reanalysis of results/v2 generations
  energy_audit.csv         F0 audit of the P3 'matched' rows
  confirmatory_tests.csv   H1-H4 x 2 models, Holm over eight
  preservation_tests.csv   one-sided non-inferiority tests, family of six
  secondary_results.csv    prespecified secondary estimates with intervals
  number_provenance.csv    every reported number -> file
  claims.json              endpoint / status / permitted wording per claim
  provenance_audit.txt     F0 narrative (plain text; markdown is never pushed)
  COMPLETION.txt           the closing record

Part of the CURE codebase (ICLR 2027, Submission2).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                      # Code/CURE/Next_Run
CURE = HERE.parent
EXT = CURE / "Extended_Research_Codes"
for p in (str(EXT), str(CURE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common as K                                          # noqa: E402  (Extended_Research_Codes)

REPO = K.REPO
AUDIT = K.AUDIT
V2 = K.RESULTS / "v2"
P0 = K.RESULTS / "reanalysis_v2"
FINAL = K.RESULTS / os.environ.get("NR_FINAL_DIR", "final_20260914")
BBQ_RAW = AUDIT / "cache" / "datasets" / "bbq_raw.parquet"
SEEDS_POOL = AUDIT / "Dataset" / "seeds" / "seeds.parquet"
DEV_SEEDS_POOL = AUDIT / "Dataset" / "seeds" / "dev_seeds.parquet"
PENTAD = K.PENTAD_RAW

FINAL_MODELS = ["gemma-2-2b-it", "llama-3.1-8b-instruct"]   # Next_Plan.md Section 4
ALL_MODELS = K.MODELS
DISPLAY = K.DISPLAY

RANDOM_SEED_FINAL = 20260914      # every stochastic choice of this cycle
N_BOOT_FINAL = 10000              # Section 9: at least 10,000 resamples
MAX_NEW_TOKENS = 48               # existing generation limit, reused
FMT = "chat"
MATCH_TOL = 0.05                  # Section 7.2: 5% relative development matching tolerance
CAUTION_MISMATCH = 0.10           # Section 7.2: >10% test mismatch is a caution flag
ACC_MARGIN_PP = 2.0               # Section 9: proposed non-inferiority margin, percentage points
GPU_HOUR_CAP = 24.0               # Section 4 planning ceiling

CONDITIONS = ["B", "S1", "SE", "NE", "RE", "RNE", "G1"]     # Section 7.1
BASELINE_CONDITIONS = ["LC", "MC"]                          # Section 8: coherent-label LEACE, mean difference
CONTROL_CONDITIONS = ["B", "S1", "NE"]                      # Section 6.1: relevant-information group

log = logging.getLogger("cure.final")


def setup_logging(name: str = "cure.final") -> logging.Logger:
    K.LOGS.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(name)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); lg.addHandler(sh)
        fh = logging.FileHandler(K.LOGS / "final_20260914.log", encoding="utf-8"); fh.setFormatter(fmt); lg.addHandler(fh)
        lg.setLevel(logging.INFO)
    return lg


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    return K.sha256(p)


def write_json(obj, path: Path) -> Path:
    return K.write_json(obj, path)


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    return K.write_csv(df, path)


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def protocol_hash(d: dict) -> str:
    """Hash of the frozen protocol choices (sorted JSON); every result row carries it."""
    return sha256_text(json.dumps(d, sort_keys=True, default=K._json_default))[:16]


# ---------------------------------------------------------------------------
# the frozen protocol record
# ---------------------------------------------------------------------------

def protocol_path(model: str | None = None) -> Path:
    """One protocol file per model on the VMs (two VMs must never commit the same file);
    f4_analysis merges them into final_protocol.json on the author's machine."""
    m = model or os.environ.get("NR_MODEL")
    return FINAL / ("final_protocol_%s.json" % m if m else "final_protocol.json")


def load_protocol(model: str | None = None) -> dict:
    return read_json(protocol_path(model), {})


def save_protocol(proto: dict, model: str | None = None) -> None:
    proto = dict(proto)
    core = {k: v for k, v in proto.items() if not k.startswith("_") and k not in ("protocol_hash", "saved_utc")}
    proto["protocol_hash"] = protocol_hash(core)
    proto["saved_utc"] = utc_now()
    write_json(proto, protocol_path(model))


# ---------------------------------------------------------------------------
# resumable per-item store (atomic parquet append, as in Extended_Research_Codes)
# ---------------------------------------------------------------------------

def read_parquet_retry(path: Path, tries: int = 5) -> pd.DataFrame:
    for i in range(tries):
        try:
            return pd.read_parquet(path)
        except Exception as ex:                      # a writer may be mid-replace
            if i == tries - 1:
                raise
            log.warning("read %s failed (%s); retry", K.rel(path), str(ex)[:80]); time.sleep(1 + i)
    return pd.DataFrame()


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        old = read_parquet_retry(path)
        new = pd.concat([old, new], ignore_index=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    new.to_parquet(tmp, index=False)
    for i in range(5):
        try:
            os.replace(tmp, path); return
        except PermissionError:
            time.sleep(0.5 + i)
    os.replace(tmp, path)


def done_keys(path: Path, cols: list[str]) -> set:
    if not path.exists():
        return set()
    d = read_parquet_retry(path)
    return set(tuple(r) for r in d[cols].itertuples(index=False, name=None))


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=K._json_default) + "\n")


# ---------------------------------------------------------------------------
# statistics (Section 9): seed-cluster studentized bootstrap, Holm, non-inferiority
# ---------------------------------------------------------------------------

def holm(pvals: list[float]) -> list[float]:
    """Holm (1979) step-down adjusted p-values (NaN stays NaN and is not counted)."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    m = int(ok.sum())
    adj = np.full(len(p), np.nan)
    if m == 0:
        return adj.tolist()
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    run = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        run = max(run, val)
        adj[i] = run
    return adj.tolist()


def seed_level_means(df: pd.DataFrame, value_col: str, cluster_col: str = "seed_id") -> np.ndarray:
    """One number per source seed: the mean of the row-level quantity within the seed (both
    sides, and any diagnostic permutation, stay together)."""
    g = df.groupby(cluster_col, sort=True)[value_col].mean()
    return g.to_numpy(float)


def studentized_paired_test(x: np.ndarray, n_boot: int = N_BOOT_FINAL, seed: int = RANDOM_SEED_FINAL,
                            null_value: float = 0.0, alternative: str = "two-sided") -> dict:
    """Null-centred studentized bootstrap of a mean of seed-level paired differences.

    t_obs = (mean - null) / se; bootstrap statistics t* = (mean* - mean) / se* are centred
    at the observed mean; two-sided p = (#|t*| >= |t_obs| + 1) / (B + 1); one-sided
    'greater' p = (#t* >= t_obs + 1) / (B + 1). Percentile interval of the mean is also
    returned. Degenerate samples (n < 2 or zero variance) return p = NaN and are flagged, never
    turned into a rejection."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    out = {"n_seeds": n, "mean": float(x.mean()) if n else float("nan"), "null_value": null_value,
           "alternative": alternative, "n_boot": n_boot, "degenerate": False}
    if n < 2 or float(x.std(ddof=1)) == 0.0:
        out.update({"se": float("nan"), "t": float("nan"), "p": float("nan"), "lo": float("nan"), "hi": float("nan"),
                    "degenerate": True, "degenerate_reason": "n<2" if n < 2 else "zero variance"})
        return out
    rng = np.random.default_rng(seed)
    mean = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    t_obs = (mean - null_value) / se
    take = rng.integers(0, n, size=(n_boot, n))
    xb = x[take]
    mb = xb.mean(axis=1); sb = xb.std(axis=1, ddof=1) / np.sqrt(n)
    zero = sb == 0
    tb = np.where(zero, 0.0, (mb - mean) / np.where(zero, 1.0, sb))
    if alternative == "two-sided":
        p = (float((np.abs(tb) >= abs(t_obs)).sum()) + 1.0) / (n_boot + 1.0)
    elif alternative == "greater":
        p = (float((tb >= t_obs).sum()) + 1.0) / (n_boot + 1.0)
    else:
        p = (float((tb <= t_obs).sum()) + 1.0) / (n_boot + 1.0)
    lo, hi = np.percentile(mb, [2.5, 97.5])
    out.update({"se": float(se), "t": float(t_obs), "p": float(p), "lo": float(lo), "hi": float(hi),
                "n_zero_variance_resamples": int(zero.sum())})
    return out


def paired_interval(x: np.ndarray, n_boot: int = N_BOOT_FINAL, seed: int = RANDOM_SEED_FINAL) -> tuple[float, float, float]:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan")), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    take = rng.integers(0, len(x), size=(n_boot, len(x)))
    mb = x[take].mean(axis=1)
    return float(x.mean()), float(np.percentile(mb, 2.5)), float(np.percentile(mb, 97.5))
