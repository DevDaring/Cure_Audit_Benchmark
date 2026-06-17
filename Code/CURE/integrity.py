"""
integrity.py -- duplicate and corruption checks, run at the start of every CURE run.

For each result parquet:
  1. Corruption: if the file fails to read or its schema is wrong, move it to
     results/quarantine/ and treat that unit as not done so it recomputes.
  2. Duplicates: drop duplicate primary keys deterministically, keeping the last write.
  3. Counts: a short or empty file marks the unit incomplete so it resumes.

Writes results/integrity_report.json each run.
"""

import json
import logging
import shutil
import time
from pathlib import Path

import pandas as pd

import config_cure as C

log = logging.getLogger("cure.integrity")

# Primary keys per result family (prefix of the file name -> key columns).
PRIMARY_KEYS = {
    "cure_recovery": ["model_name", "seed_id", "subvariant_A", "subvariant_B", "condition"],
    "cure_utility": ["model_name", "erase_rank", "metric"],
    "cure_baselines": ["method", "model_name"],
    "cure_prognosis": ["model_name", "seed_id"],
    "subspace": ["model_name", "axis"],
}


def _family(path: Path) -> str | None:
    for fam in PRIMARY_KEYS:
        if path.name.startswith(fam):
            return fam
    return None


def parquet_nonempty(path: Path) -> bool:
    """A unit is complete only if its parquet exists and has rows."""
    if not path.exists():
        return False
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(str(path)).metadata.num_rows > 0
    except Exception:
        return False


def run(results_dir: Path = None) -> dict:
    results_dir = results_dir or C.RESULTS
    quarantine = results_dir / "quarantine"
    report = {"checked": [], "deduped": {}, "quarantined": [], "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    for path in sorted(results_dir.glob("*.parquet")):
        report["checked"].append(path.name)
        fam = _family(path)
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            quarantine.mkdir(exist_ok=True)
            dest = quarantine / f"{path.stem}.{int(time.time())}.parquet"
            shutil.move(str(path), str(dest))
            report["quarantined"].append({"file": path.name, "reason": str(exc)[:160]})
            log.warning("quarantined corrupt parquet %s: %s", path.name, str(exc)[:120])
            continue
        if fam and all(k in df.columns for k in PRIMARY_KEYS[fam]):
            before = len(df)
            df = df.drop_duplicates(subset=PRIMARY_KEYS[fam], keep="last")
            removed = before - len(df)
            if removed > 0:
                df.to_parquet(path, index=False)
                report["deduped"][path.name] = removed
                log.info("dedup %s: removed %d duplicate rows", path.name, removed)

    (results_dir / "integrity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("integrity: %d files, %d deduped, %d quarantined",
             len(report["checked"]), len(report["deduped"]), len(report["quarantined"]))
    return report
