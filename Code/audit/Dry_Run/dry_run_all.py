"""
File: Dry_Run/dry_run_all.py
Purpose: Master dry-run entry point. Runs all three sub-dry-runs
         sequentially and exits 0 only if all pass.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167

Part of the audit codebase (diagnosis half of CURE).
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from logger_setup import setup_logging

logger = logging.getLogger(__name__)


def _run_module(module_path: str, label: str, **kwargs) -> bool:
    try:
        module_file = Path(__file__).parent / module_path
        import importlib.util
        spec = importlib.util.spec_from_file_location(label, str(module_file))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {module_file}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        result = mod.run(**kwargs)
        return bool(result)
    except Exception:
        logger.error("Dry run %s crashed:\n%s", label, traceback.format_exc())
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="MIRAGE master dry run")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU_CPU dry run (for CPU-only machines).")
    parser.add_argument("--only", choices=["dataset", "gpu_cpu", "cpu_only"], help="Run one sub-dry-run only.")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=2,
        help="Number of probe seeds for the GPU_CPU sub-dry-run (default: 2).",
    )
    args = parser.parse_args()

    run_id = setup_logging()
    logger.info("=== MIRAGE Master Dry Run (run_id=%s) ===", run_id)

    phases: list[tuple[str, str]] = [
        ("dry_run_dataset.py", "DATASET"),
        ("dry_run_gpu_cpu.py", "GPU_CPU"),
        ("dry_run_cpu_only.py", "CPU_ONLY"),
    ]

    if args.only == "dataset":
        phases = [phases[0]]
    elif args.only == "gpu_cpu":
        phases = [phases[1]]
    elif args.only == "cpu_only":
        phases = [phases[2]]
    elif args.skip_gpu:
        phases = [phases[0], phases[2]]

    results: dict[str, bool] = {}
    for module_file, label in phases:
        logger.info("\n--- Running phase: %s ---", label)
        kwargs = {"n_seeds": args.n_seeds} if label == "GPU_CPU" else {}
        passed = _run_module(module_file, label, **kwargs)
        results[label] = passed
        status = "PASS" if passed else "FAIL"
        logger.info("--- Phase %s: %s ---\n", label, status)

    logger.info("=== Master Dry Run Final Summary ===")
    all_pass = True
    for label, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info("  %-15s %s", label, status)
        if not passed:
            all_pass = False

    if all_pass:
        logger.info("\nAll dry runs PASSED. MIRAGE environment is ready.")
        sys.exit(0)
    else:
        logger.error("\nOne or more dry runs FAILED. Review logs above.")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
