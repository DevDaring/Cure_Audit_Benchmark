"""
File: regenerate_api_slots.py
Purpose: Re-generate slot (d) and (e) DeepSeek prompts after deterministic
         slots (a/b/c) have been patched.

Usage:
    python regenerate_api_slots.py
    python regenerate_api_slots.py --keep-checkpoint
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SEEDS_DIR
from logger_setup import setup_logging

logger = logging.getLogger(__name__)

_PENTAD_PATH = SEEDS_DIR / "pentad_dataset.parquet"
_SEEDS_PATH = SEEDS_DIR / "seeds.parquet"
_CTX_CHECKPOINT = SEEDS_DIR / "context_shift_checkpoint.json"
_COT_CHECKPOINT = SEEDS_DIR / "cot_attack_checkpoint.json"


def _save_partial_pentad(det_df: pd.DataFrame, api_rows: list[dict]) -> None:
    """Persist det + partial API rows so a crash during slot-e does not lose slot-d."""
    if not api_rows:
        return
    combined = pd.concat(
        [det_df, pd.DataFrame(api_rows)],
        ignore_index=True,
    ).sort_values(["seed_id", "slot", "subvariant"]).reset_index(drop=True)
    combined.to_parquet(_PENTAD_PATH, index=False)
    logger.info(
        "Incremental pentad save: %d rows (%d api).",
        len(combined),
        len(api_rows),
    )


def _clear_api_checkpoints() -> None:
    for cp in (_CTX_CHECKPOINT, _COT_CHECKPOINT):
        if cp.exists():
            cp.unlink()
            logger.info("Removed checkpoint: %s", cp.name)


def main() -> bool:
    parser = argparse.ArgumentParser(description="Regenerate DeepSeek slots d and e")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Reuse checkpoint (only safe if slot-a text unchanged)",
    )
    args = parser.parse_args()

    setup_logging()

    if not _PENTAD_PATH.exists():
        logger.error("pentad_dataset.parquet not found at %s", _PENTAD_PATH)
        return False
    if not _SEEDS_PATH.exists():
        logger.error("seeds.parquet not found at %s", _SEEDS_PATH)
        return False

    pentad_df = pd.read_parquet(_PENTAD_PATH)
    seeds_df = pd.read_parquet(_SEEDS_PATH)
    seeds_df = seeds_df[
        seeds_df["seed_source"].astype(str).str.lower().isin({"bbq", "crows_pairs", "stereoset"})
    ].reset_index(drop=True)

    det_df = pentad_df[pentad_df["slot"].isin(["a", "b", "c"])].copy()
    det_df = det_df[
        det_df["seed_source"].astype(str).str.lower().isin({"bbq", "crows_pairs", "stereoset"})
    ]
    old_api = pentad_df[pentad_df["slot"].isin(["d", "e"])].copy()
    logger.info("Loaded pentad: %d rows (%d det, %d api)", len(pentad_df), len(det_df), len(old_api))

    slot_a = det_df[det_df["slot"] == "a"][["seed_id", "prompt_text", "gold_answer"]].rename(
        columns={"prompt_text": "slot_a_prompt"}
    )
    enriched = seeds_df.merge(slot_a, on="seed_id", how="inner")
    logger.info("Enriched %d seeds with patched slot-a prompts", len(enriched))

    if len(enriched) == 0:
        logger.error("No seeds matched patched slot-a rows — run patch_det_slots.py first.")
        return False

    if args.dry_run:
        logger.info("[dry-run] Would regenerate d/e for %d seeds", len(enriched))
        return True

    clear = not args.keep_checkpoint
    if clear:
        _clear_api_checkpoints()
    elif _CTX_CHECKPOINT.exists() or _COT_CHECKPOINT.exists():
        logger.info("Keeping existing API slot checkpoints (--keep-checkpoint).")

    from Dataset.context_shift_drafter import draft_context_shifts
    from Dataset.cot_attack_generator import generate_cot_attacks

    logger.info("Regenerating slot (d) via DeepSeek ...")
    d_rows = draft_context_shifts(
        enriched,
        clear_checkpoint=clear,
        remove_checkpoint_on_success=False,
    )
    _save_partial_pentad(det_df, d_rows)

    logger.info("Regenerating slot (e) via DeepSeek ...")
    e_rows = generate_cot_attacks(
        enriched,
        clear_checkpoint=clear,
        remove_checkpoint_on_success=False,
    )

    combined = pd.concat(
        [det_df, pd.DataFrame(d_rows), pd.DataFrame(e_rows)],
        ignore_index=True,
    ).sort_values(["seed_id", "slot", "subvariant"]).reset_index(drop=True)

    logger.info("Combined rows: %d (expected %d)", len(combined), len(enriched) * 12)
    combined.to_parquet(_PENTAD_PATH, index=False)

    from Dataset.validate_pentad import run_all_validations, write_pentad_manifest

    try:
        run_all_validations(combined)
        write_pentad_manifest(combined)
    except Exception as exc:
        logger.error("Validation FAILED after API regeneration: %s", exc)
        return False

    _clear_api_checkpoints()
    logger.info("API slot regeneration complete and validated.")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
