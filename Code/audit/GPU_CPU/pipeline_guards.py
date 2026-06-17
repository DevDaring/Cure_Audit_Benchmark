"""
GPU pipeline guards — ensure results match the current validated pentad.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR, SEEDS_DIR

logger = logging.getLogger(__name__)

_MANIFEST_PATH = SEEDS_DIR / "pentad_manifest.json"
_STALE_MARKERS = (
    RESULTS_DIR / "behavioral_results.parquet",
    RESULTS_DIR / "cdva_results.parquet",
    RESULTS_DIR / "tau_calibration.json",
)


def clear_stale_gpu_results_if_pentad_changed(state_dir: Path | None = None) -> None:
    """
    Remove GPU result files when pentad_dataset.parquet changed since last run.

    Prevents scoring against behavioral outputs from a prior broken dataset.
    """
    from Dataset.validate_pentad import _PENTAD_PATH, pentad_file_sha256

    if not _PENTAD_PATH.exists():
        return

    current_sha = pentad_file_sha256(_PENTAD_PATH)
    stored_sha = ""
    if _MANIFEST_PATH.exists():
        try:
            with open(_MANIFEST_PATH) as fh:
                stored_sha = json.load(fh).get("pentad_sha256", "")
        except Exception:
            stored_sha = ""

    if stored_sha and stored_sha == current_sha:
        logger.info("Pentad SHA unchanged — keeping existing GPU results.")
        return

    removed: list[str] = []
    for path in _STALE_MARKERS:
        if path.exists():
            path.unlink()
            removed.append(path.name)

    if state_dir is not None:
        for marker in ("GPU_PIPELINE_OK",):
            m = state_dir / marker
            if m.exists():
                m.unlink()
                removed.append(marker)

    if removed:
        logger.warning(
            "Pentad changed (sha %s → %s). Cleared stale artifacts: %s",
            stored_sha[:12] or "none",
            current_sha[:12],
            removed,
        )
