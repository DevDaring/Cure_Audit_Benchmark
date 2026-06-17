"""
File: GPU_CPU/run_gpu_pipeline.py
Purpose: Production entry-point for the full MIRAGE GPU pipeline.
         Runs after the dataset has been built (Dataset/seeds/pentad_dataset.parquet exists).

Steps:
  1. Load OSM models (sequential on <48 GB VRAM, simultaneous on 80 GB).
  2. run_osm_behavioral  — behavioral evaluation on the full pentad dataset.
  3. run_cdva_patching   — causal activation patching on counterfactual (c) variants.
  4. Unload models to free VRAM for CPU post-processing.
  5. run_cdva_calibration — tau threshold calibration on the dev set.

Sequential mode (A100 40 GB, L4 24 GB): load one model → behavioral → CDVA →
unload, then repeat for the next model.  Peak VRAM ≈ one model (~16 GB).

Both behavioral and CDVA functions include incremental-save / resume logic: if the
process is killed mid-run (e.g. an eviction), re-running this script will skip
already-completed rows and continue from the last checkpoint.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167

Part of the audit codebase (diagnosis half of CURE).
"""
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RESULTS_DIR, SEEDS_DIR, ensure_dirs
from logger_setup import setup_logging


def _free_vram() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_pipeline_for_models(
    pentad_df,
    models: dict[str, tuple[Any, Any]],
    run_id: str,
) -> tuple[Any, Any]:
    """Run behavioral + CDVA for the models currently loaded."""
    from GPU_CPU.osm_behavioral import run_osm_behavioral
    from GPU_CPU.cdva_patching import run_cdva

    behavioral_df = run_osm_behavioral(pentad_df, models, run_id)
    cdva_df = run_cdva(pentad_df, models, run_id)
    return behavioral_df, cdva_df


def _run_sequential(pentad_df, run_id: str, t0: float) -> tuple[Any, Any]:
    """Load → behavioral → CDVA → unload for each OSM model in turn.

    Behavioral and CDVA results accumulate across all 4 models — each call to
    run_osm_behavioral/run_cdva reads the existing checkpoint parquet and
    appends to it, so the final parquets contain all 4 models when done.
    The returned DataFrames are the latest full parquet reads after all models.
    """
    import pandas as pd
    from GPU_CPU.load_osm import load_model, unload_model

    for idx, model_cfg in enumerate(OSM_MODELS, start=1):
        name = model_cfg["name"]
        logger.info(
            "Step 1–3 (model %d/4): %s — load → behavioral → CDVA → unload ...",
            idx,
            name,
        )
        model, tokenizer = load_model(model_cfg)
        models = {name: (model, tokenizer)}

        _run_pipeline_for_models(pentad_df, models, run_id)

        unload_model(name)
        _free_vram()
        logger.info(
            "Model %d/4 (%s) complete (%.1f s elapsed).",
            idx,
            name,
            time.monotonic() - t0,
        )

    # Read back the complete accumulated parquets for calibration / reporting.
    from config import RESULTS_DIR
    beh_path = RESULTS_DIR / "behavioral_results.parquet"
    cdva_path = RESULTS_DIR / "cdva_results.parquet"
    behavioral_df = pd.read_parquet(beh_path) if beh_path.exists() else pd.DataFrame()
    cdva_df = pd.read_parquet(cdva_path) if cdva_path.exists() else pd.DataFrame()
    logger.info(
        "Sequential loop complete: %d behavioral rows, %d CDVA rows across all models.",
        len(behavioral_df), len(cdva_df),
    )
    return behavioral_df, cdva_df


def _run_simultaneous(pentad_df, run_id: str, t0: float) -> tuple[Any, Any]:
    """Load all 4 OSM models at once (~42 GB; A100 80 GB)."""
    from GPU_CPU.load_osm import load_all_osm_models, unload_model

    logger.info("Step 1/4: Loading all 4 OSM models simultaneously (~42 GB) ...")
    models = load_all_osm_models()
    logger.info(
        "Step 1/4 done: %d models loaded (%.1f s elapsed)",
        len(models),
        time.monotonic() - t0,
    )

    logger.info(
        "Step 2–3/4: Behavioral + CDVA — %d models × %d rows ...",
        len(models),
        len(pentad_df),
    )
    behavioral_df, cdva_df = _run_pipeline_for_models(pentad_df, models, run_id)

    logger.info("Step 4/4: Unloading all OSM models ...")
    for model_cfg in OSM_MODELS:
        unload_model(model_cfg["name"])
    _free_vram()

    return behavioral_df, cdva_df


def main() -> bool:
    run_id = setup_logging()
    logger.info("=== MIRAGE GPU Pipeline (run_id=%s) ===", run_id)
    ensure_dirs()
    t0 = time.monotonic()

    # ── 0. Verify pentad dataset exists ──────────────────────────────────
    pentad_path = SEEDS_DIR / "pentad_dataset.parquet"
    if not pentad_path.exists():
        logger.error(
            "Pentad dataset not found at %s. "
            "Run 'python run_dataset.py' first to build the dataset.",
            pentad_path,
        )
        return False

    import pandas as pd

    pentad_df = pd.read_parquet(pentad_path)

    # Hard gate — refuse GPU work on partial or invalid pentad.
    from Dataset.validate_pentad import assert_production_ready, write_pentad_manifest

    assert_production_ready(pentad_df)
    write_pentad_manifest(pentad_df)

    from GPU_CPU.pipeline_guards import clear_stale_gpu_results_if_pentad_changed

    state_dir = Path(os.environ.get("STATE_DIR", "/data/state"))
    clear_stale_gpu_results_if_pentad_changed(state_dir)

    logger.info(
        "Pentad dataset loaded: %d rows | %d unique seeds",
        len(pentad_df),
        pentad_df["seed_id"].nunique(),
    )

    from GPU_CPU.load_osm import get_gpu_vram_gb, use_sequential_loading

    sequential = use_sequential_loading()
    vram_gb = get_gpu_vram_gb()
    if sequential:
        logger.info(
            "Sequential model loading enabled (GPU VRAM=%.1f GB; "
            "one model at a time, peak ~16 GB).",
            vram_gb,
        )
        behavioral_df, cdva_df = _run_sequential(pentad_df, run_id, t0)
    else:
        logger.info(
            "Simultaneous model loading enabled (GPU VRAM=%.1f GB; all 4 models resident).",
            vram_gb,
        )
        behavioral_df, cdva_df = _run_simultaneous(pentad_df, run_id, t0)

    logger.info(
        "Behavioral + CDVA done: %d behavioral rows, %d CDVA rows (%.1f s)",
        len(behavioral_df) if behavioral_df is not None else 0,
        len(cdva_df) if cdva_df is not None else 0,
        time.monotonic() - t0,
    )
    logger.info("VRAM freed. Running CDVA tau calibration on CPU ...")

    # ── 5. CDVA tau calibration (uses dev-seed subset of full results) ─────
    from GPU_CPU.cdva_calibration import calibrate_tau

    dev_seeds_path = SEEDS_DIR / "dev_seeds.parquet"
    if (
        dev_seeds_path.exists()
        and behavioral_df is not None
        and cdva_df is not None
        and len(behavioral_df) > 0
        and len(cdva_df) > 0
    ):
        dev_seeds_df = pd.read_parquet(dev_seeds_path)
        dev_ids = (
            set(dev_seeds_df["seed_id"].tolist())
            if "seed_id" in dev_seeds_df.columns
            else set()
        )
        dev_beh = (
            behavioral_df[behavioral_df["seed_id"].isin(dev_ids)]
            if dev_ids
            else behavioral_df
        )
        dev_cdva = cdva_df[cdva_df["seed_id"].isin(dev_ids)] if dev_ids else cdva_df
        if len(dev_beh) > 0:
            calibrate_tau(dev_beh, dev_cdva)
        else:
            logger.warning("No dev-seed rows in behavioral results; tau calibration skipped.")
    else:
        logger.warning("dev_seeds.parquet not found; skipping tau calibration.")

    elapsed = time.monotonic() - t0
    logger.info(
        "=== GPU PIPELINE COMPLETE in %.1f s (%.1f h) ===",
        elapsed,
        elapsed / 3600,
    )
    logger.info("  Results in: %s", RESULTS_DIR)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
