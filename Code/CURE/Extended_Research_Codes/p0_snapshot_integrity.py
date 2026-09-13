"""
p0_snapshot_integrity.py -- preserve the existing results and correct the per-rank identity.

Next_Plan.md, Section 2.8 and P0 row 1:

  integrity.py matches cure_recovery_sweep_* to the "cure_recovery" family, whose primary
  key omits erase_rank. Applying its deduplication would discard 3,000 of the 4,000 valid
  rows per model. Correct the key and preserve a snapshot before any rerun.

What this does (nothing here mutates a result file):
  1. SHA-256 every file under Code/CURE/results/ and the audit inputs the package reads, so a
     later run can prove its inputs were the ones analysed.          -> input_checksums.csv
  2. Define the corrected primary keys. The bug has two parts: the key for the sweep family
     lacks erase_rank, and integrity._family matches by *prefix in dict order*, so
     "cure_recovery_sweep_x" hits "cure_recovery" first. The fix lists the longer prefix
     first and includes erase_rank (and condition) in its key.
  3. Dry-run both key sets over every result parquet and report how many rows each WOULD
     drop, without writing.                                           -> dedup_dry_run.csv
  4. Expose run_integrity_safe(), a drop-in for integrity.run() that (a) uses the corrected
     keys and (b) copies every parquet to results/snapshots/<utc>/ before it touches
     anything. GPU runners in this package call this, never integrity.run().

Usage:
  python p0_snapshot_integrity.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

import common as K

log = K.setup_logging("cure.ext.p0.integrity")

# Shipped keys, copied from Code/CURE/integrity.py for the side-by-side dry run.
SHIPPED_PRIMARY_KEYS = {
    "cure_recovery": ["model_name", "seed_id", "subvariant_A", "subvariant_B", "condition"],
    "cure_utility": ["model_name", "erase_rank", "metric"],
    "cure_baselines": ["method", "model_name"],
    "cure_prognosis": ["model_name", "seed_id"],
    "subspace": ["model_name", "axis"],
}

# Corrected keys. Order matters: longer prefixes first, because family lookup is a prefix
# match in insertion order.
CORRECTED_PRIMARY_KEYS = {
    "cure_recovery_sweep": ["model_name", "seed_id", "subvariant_A", "subvariant_B",
                            "condition", "erase_rank"],
    "cure_recovery": ["model_name", "seed_id", "subvariant_A", "subvariant_B",
                      "condition", "erase_rank"],
    "cure_utility": ["model_name", "erase_rank", "metric"],
    "cure_baselines": ["method", "model_name"],
    "cure_final": ["method", "model_name"],
    "cure_prognosis": ["model_name", "seed_id"],
    "tacl_extra": ["method", "model_name"],
    "subspace": ["model_name", "axis"],
}


def family(name: str, keys: dict) -> str | None:
    for fam in keys:                       # insertion order, as integrity._family does
        if name.startswith(fam):
            return fam
    return None


def checksums() -> pd.DataFrame:
    rows = []
    targets = sorted(K.RESULTS.glob("*.parquet")) + sorted(K.RESULTS.glob("*.json"))
    targets += [K.CDVA_PATH, K.MASK_PATH, K.PENTAD_RAW, K.PENTAD_CLEAN, K.BEHAVIORAL_PATH]
    for p in targets:
        if not p.exists():
            rows.append({"file": str(p.relative_to(K.REPO)), "bytes": None,
                         "sha256": None, "exists": False})
            continue
        rows.append({"file": str(p.relative_to(K.REPO)), "bytes": p.stat().st_size,
                     "sha256": K.sha256(p), "exists": True})
    df = pd.DataFrame(rows)
    df["recorded_utc"] = K.utc_now()
    return df


def dedup_dry_run() -> pd.DataFrame:
    rows = []
    for p in sorted(K.RESULTS.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception as exc:                      # report, never quarantine here
            rows.append({"file": p.name, "rows": None, "error": str(exc)[:120]})
            continue
        for label, keys in (("shipped", SHIPPED_PRIMARY_KEYS), ("corrected", CORRECTED_PRIMARY_KEYS)):
            fam = family(p.name, keys)
            cols = keys.get(fam, [])
            usable = bool(fam) and all(c in df.columns for c in cols)
            would_drop = int(len(df) - len(df.drop_duplicates(subset=cols))) if usable else 0
            rows.append({"file": p.name, "rows": len(df), "key_set": label, "family": fam,
                         "key_columns": "+".join(cols) if usable else "(family unmatched or column missing)",
                         "rows_would_drop": would_drop,
                         "rows_would_keep": len(df) - would_drop})
    return pd.DataFrame(rows)


def snapshot(reason: str = "manual") -> Path:
    """Copy every result parquet/json into results/snapshots/<utc>/ and return the folder."""
    dest = K.RESULTS / "snapshots" / f"{K.utc_now().replace(':', '')}_{reason}"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in list(K.RESULTS.glob("*.parquet")) + list(K.RESULTS.glob("*.json")):
        shutil.copy2(p, dest / p.name)
        n += 1
    log.info("snapshot: %d files -> %s", n, dest.relative_to(K.REPO))
    return dest


def run_integrity_safe(results_dir: Path | None = None, mutate: bool = False) -> dict:
    """Corrected replacement for integrity.run().

    mutate=False (default): report only. mutate=True: snapshot first, then dedup with the
    corrected keys. Corrupt files are reported, not moved."""
    results_dir = results_dir or K.RESULTS
    report = {"ts": K.utc_now(), "mutate": mutate, "checked": [], "deduped": {}, "corrupt": []}
    if mutate:
        report["snapshot"] = str(snapshot("pre_dedup"))
    for p in sorted(results_dir.glob("*.parquet")):
        report["checked"].append(p.name)
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            report["corrupt"].append({"file": p.name, "reason": str(exc)[:160]})
            continue
        fam = family(p.name, CORRECTED_PRIMARY_KEYS)
        cols = CORRECTED_PRIMARY_KEYS.get(fam, [])
        if fam and all(c in df.columns for c in cols):
            before = len(df)
            dd = df.drop_duplicates(subset=cols, keep="last")
            if len(dd) < before:
                report["deduped"][p.name] = before - len(dd)
                if mutate:
                    dd.to_parquet(p, index=False)
    K.write_json(report, K.OUT_P0 / "integrity_report_v2.json")
    return report


def main() -> None:
    K.OUT_P0.mkdir(parents=True, exist_ok=True)
    cs = checksums()
    K.write_csv(cs, K.OUT_P0 / "input_checksums.csv")
    dd = dedup_dry_run()
    K.write_csv(dd, K.OUT_P0 / "dedup_dry_run.csv")
    run_integrity_safe(mutate=False)

    print("\n=== DEDUP DRY RUN (rows each key set WOULD drop; nothing written) ===")
    view = dd[dd["rows_would_drop"].notna()] if "rows_would_drop" in dd else dd
    print(view[["file", "rows", "key_set", "family", "rows_would_drop"]].to_string(index=False))
    sweep = view[(view["file"].str.startswith("cure_recovery_sweep")) & (view["key_set"] == "shipped")]
    if len(sweep):
        print("\nshipped key would delete %d of %d sweep rows across %d models"
              % (int(sweep["rows_would_drop"].sum()), int(sweep["rows"].sum()), len(sweep)))
    corr = view[(view["file"].str.startswith("cure_recovery_sweep")) & (view["key_set"] == "corrected")]
    print("corrected key would delete %d" % int(corr["rows_would_drop"].sum()))


if __name__ == "__main__":
    main()
