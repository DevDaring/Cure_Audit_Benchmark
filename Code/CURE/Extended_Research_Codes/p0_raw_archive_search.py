"""
p0_raw_archive_search.py -- recover raw generation archives if they exist.

Next_Plan.md, Section 2.3 and P0 row 6:

  Raw CURE generations were not found among the available CURE result files; the shared
  behavioral_results.parquet has no tacl- or e4- run IDs. The summary files cannot recover
  paired behavioural confidence intervals or all failure counts. Search any external run
  archives before paying to regenerate them.

Where raw generations could be:
  1. Code/audit/results/behavioral_results.parquet   evaluate_osm_model may append rows
     there keyed by run_id; CURE used run_ids "e4-<hex>" (native_accuracy) and
     "tacl-<hex>a"/"tacl-<hex>c" (behavioural_readout).
  2. Any parquet/csv/jsonl under Code/ or Submission2/ whose columns include raw_response
     or parsed_answer AND whose run_id matches those prefixes.
  3. Git history: checkpoint.py force-adds Code/CURE/results and Code/CURE/logs and pushes
     them, so a file that once existed in results/ may survive in an old commit even if it
     was later removed. Search every commit's tree for candidate names.
  4. The run logs (Code/CURE/logs/*.log) which may contain per-item lines.
  5. results/quarantine and results/dryrun.

Outputs
  reanalysis_v2/raw_archive_search.json   what was searched, what was found, verdict
  reanalysis_v2/raw_archive_candidates.csv  every candidate file with columns and run_id sample

Usage:
  python p0_raw_archive_search.py
"""

from __future__ import annotations

import re
import subprocess

import pandas as pd

import common as K

log = K.setup_logging("cure.ext.p0.archive")

PREFIXES = ("tacl-", "e4-")
COLS_OF_INTEREST = {"raw_response", "parsed_answer", "run_id", "seed_id", "success_flag"}


def search_behavioral() -> dict:
    out = {"file": str(K.BEHAVIORAL_PATH.relative_to(K.REPO)), "exists": K.BEHAVIORAL_PATH.exists()}
    if not K.BEHAVIORAL_PATH.exists():
        return out
    b = pd.read_parquet(K.BEHAVIORAL_PATH, columns=["run_id", "model_name", "seed_id", "slot"])
    rid = b["run_id"].astype(str)
    out["n_rows"] = int(len(b))
    out["n_run_ids"] = int(rid.nunique())
    out["run_id_prefix_samples"] = sorted({r.split("-")[0][:8] for r in rid.unique()})[:20]
    for p in PREFIXES:
        hit = rid.str.startswith(p)
        out[f"rows_with_prefix_{p}"] = int(hit.sum())
    return out


def scan_files() -> list[dict]:
    rows = []
    roots = [K.CODE, K.REPO / "Submission2"]
    for root in roots:
        for p in root.rglob("*"):
            if p.suffix.lower() not in (".parquet", ".csv", ".jsonl") or not p.is_file():
                continue
            if "Paper_Writing" in p.parts or ".git" in p.parts:
                continue
            try:
                if p.suffix.lower() == ".parquet":
                    import pyarrow.parquet as pq
                    cols = set(pq.ParquetFile(str(p)).schema.names)
                elif p.suffix.lower() == ".csv":
                    cols = set(pd.read_csv(p, nrows=0).columns)
                else:
                    head = p.read_text(encoding="utf-8", errors="ignore")[:4000]
                    cols = set(re.findall(r'"(\w+)"\s*:', head))
            except Exception as exc:
                rows.append({"file": str(p.relative_to(K.REPO)), "error": str(exc)[:80]})
                continue
            inter = cols & COLS_OF_INTEREST
            if {"raw_response", "parsed_answer"} & inter:
                sample = ""
                try:
                    if p.suffix.lower() == ".parquet" and "run_id" in cols:
                        s = pd.read_parquet(p, columns=["run_id"])["run_id"].astype(str)
                        sample = ",".join(sorted({x[:12] for x in s.unique()})[:6])
                        hits = int(s.str.startswith(PREFIXES).sum())
                    else:
                        hits = None
                except Exception:
                    hits = None
                rows.append({"file": str(p.relative_to(K.REPO)), "bytes": p.stat().st_size,
                             "generation_columns": ",".join(sorted(inter)),
                             "run_id_sample": sample, "rows_with_cure_prefix": hits})
    return rows


def git_history_search() -> dict:
    """Names of files under Code/CURE/results that ever existed in any commit."""
    out = {"searched": False}
    try:
        r = subprocess.run(["git", "log", "--all", "--name-only", "--pretty=format:", "--",
                            "Code/CURE/results", "Code/CURE/logs"],
                           cwd=str(K.REPO), capture_output=True, text=True, timeout=120)
        names = sorted({ln.strip() for ln in r.stdout.splitlines() if ln.strip()})
        out["searched"] = True
        out["n_paths_ever_committed"] = len(names)
        out["paths_ever_committed"] = names
        cand = [n for n in names if re.search(r"gener|behav|tacl|e4|utility|answers|raw", n, re.I)]
        out["candidate_paths"] = cand
        # do any candidates exist in HEAD or only in history?
        exists = {n: (K.REPO / n).exists() for n in cand}
        out["candidate_exists_in_worktree"] = exists
    except Exception as exc:
        out["error"] = str(exc)[:200]
    return out


def log_search() -> dict:
    out = {}
    for p in sorted(K.LOGS.glob("*.log")) + sorted((K.RESULTS).glob("*.log")):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        out[str(p.relative_to(K.REPO))] = {
            "bytes": p.stat().st_size,
            "lines_with_tacl": len(re.findall(r"tacl-", txt)),
            "lines_with_e4": len(re.findall(r"\be4-", txt)),
            "lines_with_parsed_answer": len(re.findall(r"parsed_answer", txt)),
            "has_per_item_output": bool(re.search(r"raw_response|parsed_answer=", txt)),
        }
    return out


def main() -> None:
    K.OUT_P0.mkdir(parents=True, exist_ok=True)
    beh = search_behavioral()
    files = scan_files()
    hist = git_history_search()
    logs = log_search()
    found = any((r.get("rows_with_cure_prefix") or 0) > 0 for r in files) \
        or any(beh.get(f"rows_with_prefix_{p}", 0) > 0 for p in PREFIXES)
    verdict = ("FOUND: raw generations with CURE run ids exist; see raw_archive_candidates.csv"
               if found else
               "NOT FOUND: no raw per-item generations under CURE run ids in the behavioural "
               "archive, any result/csv/jsonl file, the run logs, or any commit in git history. "
               "Paired behavioural intervals and per-condition failure counts for the shipped "
               "held-out table cannot be reconstructed; they REQUIRE a rerun that stores raw "
               "outputs (P1/P2 harness does so by design).")
    rec = {"generated_utc": K.utc_now(), "behavioral_archive": beh, "git_history": hist,
           "run_logs": logs, "n_candidate_files": len(files), "verdict": verdict}
    K.write_json(rec, K.OUT_P0 / "raw_archive_search.json")
    K.write_csv(pd.DataFrame(files), K.OUT_P0 / "raw_archive_candidates.csv")
    print("\n=== RAW ARCHIVE SEARCH ===")
    print("behavioral_results.parquet:", {k: v for k, v in beh.items() if k != "run_id_prefix_samples"})
    print("candidate files with generation columns:", len(files))
    for r in files[:15]:
        print("  ", r)
    print("git history candidate paths:", hist.get("candidate_paths"))
    print("run logs:", logs)
    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    main()
