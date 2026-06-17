"""
Patch only slot-b (iso-control) rows in pentad_dataset.parquet.

Preserves slot-a, slot-c, and all DeepSeek slots d/e — no API calls required when
slot-a text is unchanged.

Usage:
    python patch_slot_b_only.py
    python patch_slot_b_only.py --dry-run
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
_AUDIT_SOURCES = frozenset({"bbq", "crows_pairs", "stereoset"})


def main() -> bool:
    parser = argparse.ArgumentParser(description="Patch slot-b only in pentad dataset")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    setup_logging()

    if not _PENTAD_PATH.exists():
        logger.error("pentad_dataset.parquet not found: %s", _PENTAD_PATH)
        return False
    if not _SEEDS_PATH.exists():
        logger.error("seeds.parquet not found: %s", _SEEDS_PATH)
        return False

    from Dataset.pentad_generator import _build_slot_b, _load_equiv_sets
    from Dataset.validate_pentad import validate_slot_b_grammar

    df = pd.read_parquet(_PENTAD_PATH)
    seeds = pd.read_parquet(_SEEDS_PATH)
    seeds = seeds[seeds["seed_source"].astype(str).str.lower().isin(_AUDIT_SOURCES)]
    seed_lookup = {row["seed_id"]: row.to_dict() for _, row in seeds.iterrows()}
    equiv_sets = _load_equiv_sets()

    slot_b_old = df[df["slot"] == "b"].copy()
    logger.info("Patching slot-b for %d seeds (keeping a/c/d/e unchanged).", len(slot_b_old))

    new_b_rows: list[dict] = []
    for _, old_row in slot_b_old.iterrows():
        seed_id = old_row["seed_id"]
        seed_dict = seed_lookup.get(seed_id)
        if not seed_dict:
            logger.warning("seed_id=%s not in seeds.parquet — keeping old slot-b.", seed_id)
            new_b_rows.append(old_row.to_dict())
            continue

        slot_b = _build_slot_b(seed_dict, equiv_sets)
        new_row = old_row.to_dict()
        new_row["prompt_text"] = slot_b["prompt_text"]
        new_b_rows.append(new_row)

    new_b_df = pd.DataFrame(new_b_rows)
    other = df[df["slot"] != "b"]
    combined = pd.concat([other, new_b_df], ignore_index=True)
    combined = combined.sort_values(["seed_id", "slot", "subvariant"]).reset_index(drop=True)

    changed = (
        slot_b_old.set_index("prompt_id")["prompt_text"]
        != new_b_df.set_index("prompt_id")["prompt_text"]
    ).sum()
    logger.info("Slot-b rows changed: %d / %d", changed, len(slot_b_old))

    if args.dry_run:
        logger.info("[dry-run] Not saving.")
        return True

    from Dataset.validate_pentad import (
        assert_production_ready,
        run_all_validations,
        validate_slot_b_grammar,
        write_pentad_manifest,
    )

    validate_slot_b_grammar(combined)
    has_api = bool((combined["slot"].isin(["d", "e"])).any())
    if has_api:
        assert_production_ready(combined)
    else:
        run_all_validations(combined, require_api_slots=False)
        logger.warning(
            "Pentad is det-only (no d/e). Run regenerate_api_slots.py before GPU work."
        )

    combined.to_parquet(_PENTAD_PATH, index=False)
    if has_api:
        write_pentad_manifest(combined)
    logger.info("Slot-b patch complete. Pentad production-ready.")
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
