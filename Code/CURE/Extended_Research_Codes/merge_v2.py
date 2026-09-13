"""
merge_v2.py -- combine the per-model output directories written by one VM each
(results/v2_<model>/, see bootstrap_extended.sh) into the single results/v2/ that P4, the
claims ledger and the manuscript tables read.

Rules, by file kind
  *.parquet with the same name in several directories   concatenated (rows carry model_name),
                                                        exact duplicate rows dropped
  p2_matched_alphas.json                                union of the per-model dicts
  *_protocol.json                                       "models" sub-dicts unioned; other top-level
                                                        keys taken from the first directory and
                                                        compared, differences listed in
                                                        merge_manifest.json
  p2_dev_seeds.csv, p2_test_seeds.csv, pilot_seeds.csv  must agree on seed_id across VMs
                                                        (the manifest is deterministic); copied once
  model-specific files (name contains the model name)   copied as they are
  *.md                                                  skipped; every report is rebuilt afterwards
                                                        by the stage's own --report-only /
                                                        --summarise-only mode
  other CSVs with the same name                         concatenated as a fallback (they are
                                                        regenerated too)

After merging, the summaries and reports are regenerated from the merged rows:
  p1_pilot.py --report-only, p2_controlled_erasure.py --summarise-only,
  p3_explanation.py --account <a> --report-only for every account that has rows.

Usage
  python merge_v2.py                  # results/v2_* -> results/v2, then rebuild reports
  python merge_v2.py --no-reports
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

import common as K

log = K.setup_logging("cure.ext.merge")
OUT = K.RESULTS / "v2"


def per_model_dirs() -> dict[str, Path]:
    dirs = {}
    for d in sorted(K.RESULTS.glob("v2_*")):
        if d.is_dir() and not d.name.startswith("v2_smoke_"):
            dirs[d.name[3:]] = d
    return dirs


def _same_seed_sets(paths: list[Path]) -> bool:
    sets = [set(pd.read_csv(p)["seed_id"]) for p in paths]
    return all(s == sets[0] for s in sets)


def merge(dirs: dict[str, Path]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": {m: str(d) for m, d in dirs.items()}, "files": {}, "warnings": []}
    by_name: dict[str, list[tuple[str, Path]]] = {}
    for m, d in dirs.items():
        for p in sorted(d.rglob("*")):
            if p.is_file():
                by_name.setdefault(str(p.relative_to(d)), []).append((m, p))

    for rel, items in sorted(by_name.items()):
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        name = Path(rel).name
        if name.endswith(".md"):
            manifest["files"][rel] = "skipped (report, regenerated)"
            continue
        model_specific = any(m in name for m in dirs)
        if model_specific or len(items) == 1:
            m, p = items[0]
            shutil.copy2(p, dst)
            manifest["files"][rel] = "copied from %s" % m
            continue
        if name.endswith(".parquet"):
            frames = [pd.read_parquet(p).assign(_src_vm=m) for m, p in items]
            df = pd.concat(frames, ignore_index=True)
            n0 = len(df)
            df = df.drop_duplicates(subset=[c for c in df.columns if c != "_src_vm"]).drop(columns="_src_vm")
            df.to_parquet(dst, index=False)
            manifest["files"][rel] = "concatenated %d rows from %d VMs (%d duplicates dropped)" % (len(df), len(items), n0 - len(df))
        elif name in ("p2_dev_seeds.csv", "p2_test_seeds.csv", "pilot_seeds.csv"):
            if not _same_seed_sets([p for _, p in items]):
                manifest["warnings"].append("%s differs across VMs; copied from %s" % (rel, items[0][0]))
            shutil.copy2(items[0][1], dst)
            manifest["files"][rel] = "identical across VMs, copied once" if not manifest["warnings"] or rel not in manifest["warnings"][-1] else "DIFFERS, first copy kept"
        elif name == "p2_matched_alphas.json":
            merged = {}
            for m, p in items:
                merged.update(json.loads(p.read_text(encoding="utf-8")))
            K.write_json(merged, dst)
            manifest["files"][rel] = "dict union over %s" % ", ".join(sorted(merged))
        elif name.endswith("_protocol.json"):
            base = json.loads(items[0][1].read_text(encoding="utf-8"))
            models = dict(base.get("models", {}))
            diffs = []
            for m, p in items[1:]:
                other = json.loads(p.read_text(encoding="utf-8"))
                models.update(other.get("models", {}))
                for k, v in other.items():
                    if k != "models" and base.get(k) != v:
                        diffs.append({"vm": m, "key": k})
            base["models"] = models
            base["merged_from"] = sorted(dirs)
            if diffs:
                base["top_level_differences"] = diffs
            K.write_json(base, dst)
            manifest["files"][rel] = "models unioned (%d); %d top-level differences" % (len(models), len(diffs))
        elif name.endswith(".csv"):
            frames = []
            for _, p in items:
                try:
                    frames.append(pd.read_csv(p))
                except pd.errors.EmptyDataError:
                    pass
            df = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()
            K.write_csv(df, dst)
            manifest["files"][rel] = "concatenated (regenerated by report modes)"
        elif name.endswith(".json"):
            merged = {}
            for m, p in items:
                merged[m] = json.loads(p.read_text(encoding="utf-8"))
            K.write_json(merged, dst)
            manifest["files"][rel] = "keyed by model"
        else:
            shutil.copy2(items[0][1], dst)
            manifest["files"][rel] = "copied from %s (%d VMs had it)" % (items[0][0], len(items))
    K.write_json(manifest, OUT / "merge_manifest.json")
    return manifest


def rebuild_reports() -> None:
    here = Path(__file__).resolve().parent
    py = sys.executable
    if (OUT / "pilot_per_item.parquet").exists():
        subprocess.call([py, "p1_pilot.py", "--report-only"], cwd=here)
    if (OUT / "p2_per_item.parquet").exists():
        subprocess.call([py, "p2_controlled_erasure.py", "--summarise-only"], cwd=here)
    for acc in ("magnitude", "depth", "massive"):
        if (OUT / f"p3_{acc}_per_item.parquet").exists():
            subprocess.call([py, "p3_explanation.py", "--account", acc, "--report-only"], cwd=here)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-reports", action="store_true")
    ap.add_argument("--no-reparse", action="store_true", help="skip re-deriving the generation readout from gen_raw")
    args = ap.parse_args()
    dirs = per_model_dirs()
    if not dirs:
        log.error("no results/v2_* directories to merge"); sys.exit(2)
    if not args.no_reparse:
        import reparse_generations as RP
        for m, d in dirs.items():
            RP.reparse_dir(d)          # same parser for every row, whenever it was produced
    log.info("merging %s", ", ".join(dirs))
    man = merge(dirs)
    for w in man["warnings"]:
        log.warning(w)
    log.info("%d files merged into %s", len(man["files"]), K.rel(OUT))
    if not args.no_reports:
        rebuild_reports()


if __name__ == "__main__":
    main()
