"""
Minimal PAV validator — structural axioms A1–A6 only (no GPU, no API keys).

Scores benchmark construction quality Q(B) from pentad_dataset.parquet.
Use before GPU work to audit probe-algebraic structure without running models.

Usage:
    python pav_validate.py
    python pav_validate.py --path Dataset/seeds/pentad_dataset.parquet
    python pav_validate.py --det-only   # allow pentad without d/e slots
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SEEDS_DIR
from Dataset.gold_utils import is_scorable_gold
from Dataset.validate_pentad import (
    _AUDIT_SOURCES,
    validate_b_differs_from_a,
    validate_c_variants_distinct,
    validate_completeness,
    validate_deepseek_embeds_slot_a,
    validate_mcq_options_present,
    validate_no_sentinel_prompts,
    validate_schema,
    validate_slot_b_grammar,
)

logger = logging.getLogger(__name__)

_AXIOM_CHECKS: list[tuple[str, str, object]] = [
    ("A1", "Gold coherence", None),  # handled per-seed below
    ("A2", "Swap coherence (distinct slot-c)", validate_c_variants_distinct),
    ("A3", "Probe closure (d/e embed slot-a)", validate_deepseek_embeds_slot_a),
    ("A4", "Iso legibility (slot-b differs from slot-a)", validate_b_differs_from_a),
    ("A5", "Slot completeness (12 prompts per seed)", validate_completeness),
    ("A6", "Grammar legibility (slot-b)", validate_slot_b_grammar),
]


def _check_a1_gold_coherence(df: pd.DataFrame) -> list[str]:
    """A1: slot-b gold matches slot-a gold for scorable items."""
    problems: list[str] = []
    audit = df[df["seed_source"].astype(str).str.lower().isin(_AUDIT_SOURCES)]
    for seed_id, group in audit.groupby("seed_id"):
        a = group[(group["slot"] == "a") & (group["subvariant"] == "surface")]
        b = group[(group["slot"] == "b") & (group["subvariant"] == "iso_control")]
        if a.empty or b.empty:
            continue
        gold_a = str(a.iloc[0].get("gold_answer", ""))
        gold_b = str(b.iloc[0].get("gold_answer", ""))
        if not is_scorable_gold(gold_a, str(a.iloc[0].get("seed_source", ""))):
            continue
        if gold_a.strip().lower() != gold_b.strip().lower():
            problems.append(f"{seed_id}: gold(a)={gold_a!r} != gold(b)={gold_b!r}")
    return problems


def validate_structural_pav(
    df: pd.DataFrame,
    *,
    require_api_slots: bool = True,
) -> tuple[float, dict[str, list[str]]]:
    """
    Run structural axioms A1–A6. Returns (Q(B), defects_by_axiom).

    Q(B) = 1 - (seeds_with_any_defect / n_audit_seeds).
    """
    audit = df[df["seed_source"].astype(str).str.lower().isin(_AUDIT_SOURCES)]
    n_seeds = int(audit["seed_id"].nunique())
    if n_seeds == 0:
        raise ValueError("No audit-source rows in pentad.")

    validate_schema(audit)
    validate_no_sentinel_prompts(audit)
    validate_mcq_options_present(audit)

    defects: dict[str, list[str]] = {}

    a1 = _check_a1_gold_coherence(df)
    if a1:
        defects["A1"] = a1

    for axiom_id, _name, fn in _AXIOM_CHECKS:
        if axiom_id == "A1":
            continue
        if axiom_id == "A3" and not require_api_slots:
            continue
        if fn is validate_completeness:
            probs = validate_completeness(audit)
            if probs:
                defects[axiom_id] = probs
        else:
            try:
                fn(audit)  # type: ignore[misc]
            except ValueError as exc:
                defects[axiom_id] = [str(exc)]

    seeds_with_defect: set[str] = set()
    for probs in defects.values():
        for p in probs:
            sid = p.split(":")[0]
            if sid:
                seeds_with_defect.add(sid)

    q_b = 1.0 - (len(seeds_with_defect) / n_seeds if n_seeds else 0.0)
    return q_b, defects


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal PAV structural validator")
    parser.add_argument(
        "--path",
        type=Path,
        default=SEEDS_DIR / "pentad_dataset.parquet",
        help="Path to pentad_dataset.parquet",
    )
    parser.add_argument(
        "--det-only",
        action="store_true",
        help="Skip A3 (d/e embed check) when API slots not yet generated",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.path.exists():
        logger.error("Pentad not found: %s", args.path)
        return 1

    df = pd.read_parquet(args.path)
    audit = df[df["seed_source"].astype(str).str.lower().isin(_AUDIT_SOURCES)]
    n_seeds = audit["seed_id"].nunique()

    q_b, defects = validate_structural_pav(
        df,
        require_api_slots=not args.det_only,
    )

    print("\n=== PAV Structural Validation (A1–A6) ===")
    print(f"Audit seeds: {n_seeds}")
    print(f"Q(B) construction quality: {q_b:.3f}")
    print()

    if not defects:
        print("All structural axioms PASSED.")
        return 0

    print("Defects by axiom:")
    for axiom_id in sorted(defects):
        probs = defects[axiom_id]
        print(f"  {axiom_id}: {len(probs)} issue(s)")
        for p in probs[:5]:
            print(f"    - {p}")
        if len(probs) > 5:
            print(f"    ... and {len(probs) - 5} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
