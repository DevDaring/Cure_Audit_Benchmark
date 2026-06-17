"""
gpu_common.py -- shared utilities for the GPU_Remaining run.

Completes the GPU-only review items on a single rented GPU:
  T0.2  CDVA patch-site recovery baseline   (validates the causal instrument)
  T0.1  temperature sweep at T=1.0          (FM4 robustness)
  T1.2  randomized option-order rerun       (FM4 vs positional bias)

Design goals:
  - Reuse the EXISTING MIRAGE code (config.OSM_MODELS, GPU_CPU.load_osm.load_model,
    GPU_CPU.utils_attention.patch_activation, GPU_CPU.osm_behavioral.evaluate_osm_model)
    so behaviour is identical to the production run.
  - Resume-safe: every task writes incrementally; completed (model, task) units are skipped.
  - Checkpoint to GitHub (branch gpu-results) every 15 minutes, so a lost VM resumes
    from the last push.
"""

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent                  # Code/audit/GPU_Remaining
MIRAGE = HERE.parent                                    # Code/audit
REPO = MIRAGE.parent.parent                             # repo root (Audit_Benchmark)
sys.path.insert(0, str(MIRAGE))                         # so `import config`, `from GPU_CPU...` work

RESULTS = HERE / "results"
LOGS = HERE / "logs"
RESULTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

PENTAD_PATH = MIRAGE / "Dataset" / "seeds" / "pentad_dataset.parquet"
CDVA_PATH = MIRAGE / "results" / "cdva_results.parquet"
BEHAVIORAL_PATH = MIRAGE / "results" / "behavioral_results.parquet"

GIT_BRANCH = "main"   # push results into the VM-only results/ folder on main; clone resumes
PUSH_INTERVAL_S = 15 * 60


def setup_logging(name: str) -> logging.Logger:
    LOGS.mkdir(exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(LOGS / f"{name}.log")])
    return logging.getLogger(name)


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (no external dep)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_pentad() -> pd.DataFrame:
    return pd.read_parquet(PENTAD_PATH)


def load_cdva_pairs() -> pd.DataFrame:
    """Production CDVA pairs (located, successful) -- the pairs to re-measure for recovery."""
    df = pd.read_parquet(CDVA_PATH)
    return df[(df["position_fallback_used"] == False) & (df["success_flag"] == True)].copy()  # noqa: E712


# ----------------------------- GitHub checkpointing -----------------------------

def _run(cmd: list[str], cwd: Path = REPO, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=check)


def git_configure(token: str, user_name: str = "MIRAGE GPU Runner",
                  user_email: str = "koushikdeb2009@gmail.com") -> None:
    """Point the origin remote at a token URL on the current (main) branch."""
    _run(["git", "config", "user.name", user_name])
    _run(["git", "config", "user.email", user_email])
    _run(["git", "config", "pull.rebase", "true"])
    remote = f"https://{token}@github.com/DevDaring/Cure_Audit_Benchmark.git"
    _run(["git", "remote", "set-url", "origin", remote])


def push_checkpoint(message: str) -> bool:
    """Force-add results+logs (results/ is gitignored) and push to main. Returns True on push."""
    rel_results = "Code/audit/GPU_Remaining/results"
    rel_logs = "Code/audit/GPU_Remaining/logs"
    _run(["git", "add", "-f", rel_results, rel_logs])
    # never push the dry-run TEST results -- only real main-run results belong on GitHub
    _run(["git", "reset", "-q", "--", rel_results + "/dryrun"])
    status = _run(["git", "status", "--porcelain"]).stdout.strip()
    if not status:
        return False
    _run(["git", "commit", "-m", message])
    # pull --rebase first so VM/local pushes don't collide (they touch different files)
    _run(["git", "pull", "--rebase", "origin", GIT_BRANCH])
    ok = _run(["git", "push", "origin", GIT_BRANCH]).returncode == 0
    if not ok:
        _run(["git", "pull", "--rebase", "origin", GIT_BRANCH])
        ok = _run(["git", "push", "origin", GIT_BRANCH]).returncode == 0
    return ok


class CheckpointPusher(threading.Thread):
    """Background thread: push results+logs to GitHub every PUSH_INTERVAL_S seconds."""

    def __init__(self, log: logging.Logger, interval_s: int = PUSH_INTERVAL_S):
        super().__init__(daemon=True)
        self.log = log
        self.interval_s = interval_s
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                pushed = push_checkpoint(f"gpu-results: checkpoint {ts}")
                self.log.info("checkpoint push: %s", "pushed" if pushed else "no changes")
            except Exception as exc:  # never let the pusher kill the run
                self.log.warning("checkpoint push failed: %s", exc)

    def stop_and_flush(self, message: str) -> None:
        self._stop.set()
        try:
            push_checkpoint(message)
        except Exception as exc:
            self.log.warning("final push failed: %s", exc)
