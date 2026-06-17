"""
File: Dry_Run/dry_run_dataset.py
Purpose: Sanity check for the Dataset/ pipeline on one seed only.
         Tests all dataset downloads, seed sampling, pentad generation
         (deterministic slots only), and validation.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import validate_all_keys
from logger_setup import setup_logging

logger = logging.getLogger(__name__)

_RESULTS: dict[str, str] = {}


def _mark(component: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    _RESULTS[component] = f"{status}  {detail}".strip()
    log = logger.info if passed else logger.error
    log("  [%s] %s  %s", status, component, detail)


def _test_env_keys() -> bool:
    missing = validate_all_keys()
    if missing:
        _mark("ENV_KEYS", False, f"Missing: {missing}")
        return False
    _mark("ENV_KEYS", True, "All required keys present")
    return True


def _test_bbq_download() -> bool:
    try:
        from Dataset.download_bbq import download_bbq, validate_bbq
        df = download_bbq()
        validate_bbq(df)
        _mark("BBQ_DOWNLOAD", True, f"{len(df)} rows")
        return True
    except Exception as exc:
        _mark("BBQ_DOWNLOAD", False, str(exc))
        return False


def _test_crows_download() -> bool:
    try:
        from Dataset.download_crows_pairs import download_crows_pairs, validate_crows_pairs
        df = download_crows_pairs()
        validate_crows_pairs(df)
        _mark("CROWS_DOWNLOAD", True, f"{len(df)} rows")
        return True
    except Exception as exc:
        _mark("CROWS_DOWNLOAD", False, str(exc))
        return False


def _test_stereoset_download() -> bool:
    try:
        from Dataset.download_stereoset import download_stereoset, validate_stereoset
        df = download_stereoset()
        validate_stereoset(df)
        _mark("STEREOSET_DOWNLOAD", True, f"{len(df)} rows")
        return True
    except Exception as exc:
        _mark("STEREOSET_DOWNLOAD", False, str(exc))
        return False


def _test_winobias_download() -> bool:
    try:
        from Dataset.download_winobias import download_winobias, validate_winobias
        df = download_winobias()
        validate_winobias(df)
        _mark("WINOBIAS_DOWNLOAD", True, f"{len(df)} rows")
        return True
    except Exception as exc:
        _mark("WINOBIAS_DOWNLOAD", False, str(exc))
        return False


def _test_seed_sampling() -> bool:
    try:
        from Dataset.sample_seeds import sample_seeds
        main_seeds, dev_seeds = sample_seeds()
        assert len(main_seeds) > 0, "No main seeds sampled"
        assert main_seeds["seed_id"].is_unique, "Duplicate seed_ids"
        _mark("SEED_SAMPLING", True, f"main={len(main_seeds)} dev={len(dev_seeds)}")
        return True
    except Exception as exc:
        _mark("SEED_SAMPLING", False, str(exc))
        return False


def _test_pentad_generation_one_seed() -> bool:
    try:
        import pandas as pd
        from Dataset.sample_seeds import sample_seeds
        from Dataset.pentad_generator import generate_pentad_deterministic
        import numpy as np

        main_seeds, _ = sample_seeds()
        one_seed = main_seeds.head(1)
        rng = np.random.default_rng(seed=20260101)
        rows = generate_pentad_deterministic(one_seed, rng)
        assert len(rows) >= 7, f"Expected >= 7 rows for one seed, got {len(rows)}"
        _mark("PENTAD_GEN_ONE_SEED", True, f"{len(rows)} probe variants")
        return True
    except Exception as exc:
        _mark("PENTAD_GEN_ONE_SEED", False, str(exc))
        return False


def _test_parquet_roundtrip() -> bool:
    try:
        import io
        import pandas as pd
        import pyarrow as pa

        df = pd.DataFrame({
            "seed_id": ["test_001"],
            "prompt_id": ["test_001_a_surface"],
            "slot": ["a"],
            "prompt_text": ["A test prompt."],
            "success_flag": [True],
            "parsed_confidence": [0.9],
        })
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        df2 = pd.read_parquet(buf)
        assert list(df2.columns) == list(df.columns), "Column mismatch"
        assert df2["parsed_confidence"].dtype == float, "Type mismatch"
        _mark("PARQUET_ROUNDTRIP", True, "Types preserved")
        return True
    except Exception as exc:
        _mark("PARQUET_ROUNDTRIP", False, str(exc))
        return False


def run() -> bool:
    """Run all Dataset dry-run checks. Returns True if all pass."""
    run_id = setup_logging()
    logger.info("=== Dataset Dry Run (run_id=%s) ===", run_id)

    checks = [
        _test_env_keys,
        _test_bbq_download,
        _test_crows_download,
        _test_stereoset_download,
        _test_winobias_download,
        _test_seed_sampling,
        _test_pentad_generation_one_seed,
        _test_parquet_roundtrip,
    ]

    all_pass = True
    for check_fn in checks:
        try:
            result = check_fn()
            all_pass = all_pass and result
        except Exception:
            logger.error("Unhandled exception in %s:\n%s", check_fn.__name__, traceback.format_exc())
            all_pass = False

    _print_summary()
    return all_pass


def _print_summary() -> None:
    logger.info("\n=== Dataset Dry Run Summary ===")
    for component, status in _RESULTS.items():
        logger.info("  %-40s %s", component, status)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run()
    sys.exit(0 if success else 1)
