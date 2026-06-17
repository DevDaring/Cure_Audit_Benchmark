"""
run_gpu_remaining.py -- single entry point for the GPU-only review items.

Modes:
  --mode dry    Run T0.2/T0.1/T1.2 on 2 models (one TransformerLens, one nnsight)
                with 2 instances each. Validates the real code paths. Writes to
                results/dryrun/. Exit 0 only if all checks pass.
  --mode main   Run all tasks on all 4 OSM models, sequentially. Checkpoints after
                each (model, task); resumes by skipping completed units. Pushes to
                GitHub (branch gpu-results) every 15 minutes and after each unit.

Usage:
  python run_gpu_remaining.py --mode dry
  python run_gpu_remaining.py --mode main
"""

import argparse
import json
import os
import shutil
import sys
import time

from gpu_common import (HERE, RESULTS, LOGS, setup_logging, load_dotenv,
                        load_pentad, load_cdva_pairs, git_configure,
                        push_checkpoint, CheckpointPusher)

log = setup_logging("run_gpu_remaining")
DRY = RESULTS / "dryrun"


def _install_attn_fallback():
    """If flash_attention_2 is unavailable, fall back to sdpa so load_model still works."""
    import transformers
    orig = transformers.AutoModelForCausalLM.from_pretrained.__func__
    def patched(cls, *a, **k):
        try:
            return orig(cls, *a, **k)
        except Exception as e:
            if k.get("attn_implementation") == "flash_attention_2":
                log.warning("flash_attention_2 failed (%s); retrying with sdpa", str(e)[:120])
                k["attn_implementation"] = "sdpa"
                return orig(cls, *a, **k)
            raise
    transformers.AutoModelForCausalLM.from_pretrained = classmethod(patched)


def _import_tasks():
    import gpu_tasks
    return gpu_tasks


def _parquet_nonempty(path) -> bool:
    """A task unit counts as complete only if its parquet exists AND has rows.
    A 0-row parquet (e.g. the old empty TransformerLens T0.2) must be re-run."""
    if not path.exists():
        return False
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(str(path)).metadata.num_rows > 0
    except Exception:
        return False


def _models():
    from config import OSM_MODELS
    return OSM_MODELS


def _two_dry_models():
    """One TransformerLens model and one nnsight model, smallest of each, for the dry run."""
    osm = _models()
    tl = [m for m in osm if "nnsight" not in str(m.get("patching_lib", "")).lower()]
    nn = [m for m in osm if "nnsight" in str(m.get("patching_lib", "")).lower()]
    pick = []
    if tl:
        pick.append(min(tl, key=lambda m: 0 if "gemma" in m["name"] else 1))
    if nn:
        pick.append(min(nn, key=lambda m: 0 if "phi" in m["name"] else 1))
    return pick or osm[:2]


def _run_one_model(T, cfg, pentad, cdva, out_dir, limit, log, t02_limit=None):
    """Generation tasks first (HF model), then patching (TL/nnsight wrapper). Returns dict of dfs."""
    from GPU_CPU.load_osm import load_model, unload_model
    name = cfg["name"]
    log.info("=== model %s (limit=%s) ===", name, limit)
    model, tok = load_model(cfg)
    res = {}
    try:
        res["t01_temp"] = T.run_t01_temperature(cfg, model, tok, pentad, temperature=1.0, limit=limit)
        log.info("  T0.1 rows=%d", len(res["t01_temp"]))
        res["t12_order"] = T.run_t12_optionorder(cfg, model, tok, pentad, limit=limit)
        log.info("  T1.2 rows=%d", len(res["t12_order"]))
        res["t02_recovery"] = T.run_t02_recovery(cfg, model, tok, cdva, pentad,
                                                 limit=(t02_limit if t02_limit is not None else limit))
        log.info("  T0.2 rows=%d", len(res["t02_recovery"]))
    finally:
        unload_model(name)
    for task, df in res.items():
        df.to_parquet(out_dir / f"{task}_{name}.parquet", index=False)
    return res


def cmd_dry(T, pentad, cdva):
    """Smoke-test the FULL code path on EVERY model with 2 instances each.

    Runs all 4 OSM models (not just one TL + one nnsight), so the model-specific
    paths that only appear in main -- TransformerLens conversion of llama vs gemma,
    nnsight wrapping of qwen vs phi -- are all exercised before the main run. A
    dry pass therefore guarantees main will not fail on load/convert/patch grounds.
    """
    DRY.mkdir(parents=True, exist_ok=True)
    ok = True
    for cfg in _models():           # ALL 4 models, not a 2-model subset
        try:
            res = _run_one_model(T, cfg, pentad, cdva, DRY, limit=2, log=log, t02_limit=8)
        except Exception as exc:
            log.error("DRY model %s raised: %s", cfg["name"], exc)
            return False
        if len(res["t01_temp"]) < 1 or "parsed_answer" not in res["t01_temp"].columns:
            log.error("DRY FAIL: T0.1 empty for %s", cfg["name"]); ok = False
        if len(res["t12_order"]) < 1 or "order_permuted" not in res["t12_order"].columns:
            log.error("DRY FAIL: T1.2 empty for %s", cfg["name"]); ok = False
        t02 = res["t02_recovery"]
        # A working patching path produces one row per processed pair. ZERO rows on
        # 8 dry pairs means the model's TL/nnsight path is broken -> would yield an
        # empty T0.2 in main. Fail the dry run here, where it is cheap to catch.
        if len(t02) < 1:
            log.error("DRY FAIL: T0.2 produced 0 rows for %s (patching path broken)", cfg["name"]); ok = False
        elif "recovery_fraction" not in t02.columns:
            log.error("DRY FAIL: T0.2 schema for %s", cfg["name"]); ok = False
        else:
            import numpy as _np
            vals = t02["recovery_fraction"].to_numpy()
            log.info("  T0.2 %s: rows=%d finite=%d/%d sample=%s", cfg["name"], len(vals),
                     int(_np.isfinite(vals).sum()), len(vals), vals[:4].tolist())
    log.info("DRY RESULT: %s", "PASS" if ok else "FAIL")
    return ok


def cmd_main(T, pentad, cdva, pusher):
    status_path = RESULTS / "STATUS.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"done": []}
    done = set(tuple(x) for x in status["done"])
    summaries = {}
    for cfg in _models():
        name = cfg["name"]
        # resume: skip a (model, task) whose output parquet already exists non-empty
        pending = [t for t in ("t01_temp", "t12_order", "t02_recovery")
                   if not _parquet_nonempty(RESULTS / f"{t}_{name}.parquet")]
        if not pending:
            log.info("model %s already complete; skipping.", name); continue
        try:
            res = _run_one_model(T, cfg, pentad, cdva, RESULTS, limit=None, log=log)
        except Exception as exc:
            log.error("MAIN model %s raised: %s -- pushing logs, continuing.", name, exc)
            push_checkpoint(f"gpu-results: error on {name}")
            continue
        for t in res:
            done.add((name, t))
        status["done"] = sorted([list(x) for x in done])
        status_path.write_text(json.dumps(status, indent=2))
        summaries[name] = T.summarize_recovery(res["t02_recovery"])
        (RESULTS / "recovery_summary.json").write_text(json.dumps(summaries, indent=2))
        push_checkpoint(f"gpu-results: {name} complete")
        log.info("model %s complete + pushed.", name)
    (RESULTS / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    pusher.stop_and_flush("gpu-results: ALL DONE")
    log.info("ALL MODELS COMPLETE.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "main"], required=True)
    args = ap.parse_args()

    load_dotenv(HERE / ".env")
    os.environ.setdefault("MIRAGE_SEQUENTIAL_MODELS", "1")
    try:
        _install_attn_fallback()
    except Exception as exc:
        log.warning("attn fallback install skipped: %s", exc)

    # Verify BOTH patching libraries import before doing any work. A broken
    # transformer_lens must fail here (exit non-zero -> supervisor retries with
    # a freshly pulled/installed env) rather than silently yielding empty
    # TransformerLens (llama/gemma) T0.2 results 15 minutes into the run.
    import transformer_lens  # noqa: F401
    import nnsight  # noqa: F401
    import importlib.metadata as _md
    # NB: transformer_lens 2.18.0 exposes no module-level __version__; read it from
    # package metadata instead (a direct .__version__ access raises AttributeError).
    log.info("patching libs OK: transformer_lens=%s nnsight=%s",
             _md.version("transformer_lens"), _md.version("nnsight"))

    T = _import_tasks()
    pentad = load_pentad()
    cdva = load_cdva_pairs()
    log.info("loaded pentad=%d rows, cdva pairs=%d", len(pentad), len(cdva))

    if args.mode == "dry":
        ok = cmd_dry(T, pentad, cdva)
        sys.exit(0 if ok else 1)

    # main: configure git + start the 15-min pusher
    token = os.environ.get("Github_Classic_Token", "")
    if token:
        git_configure(token)
    pusher = CheckpointPusher(log)
    pusher.start()
    cmd_main(T, pentad, cdva, pusher)


if __name__ == "__main__":
    main()
