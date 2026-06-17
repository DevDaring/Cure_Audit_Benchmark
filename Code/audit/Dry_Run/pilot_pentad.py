"""
File: Dry_Run/pilot_pentad.py
Purpose: 5-seed pilot run of the full pentad pipeline including DeepSeek API
         calls for slots (d) and (e). Prints all 12 template variants per seed
         so templates can be visually inspected before the full 668-seed run.

No hardcoded keys -- all credentials from .env via config.py.

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RANDOM_SEED
from logger_setup import setup_logging

logger = logging.getLogger(__name__)

_PILOT_N = 5       # seeds to test
_WRAP_WIDTH = 120  # terminal wrap for prompt preview


def _preview(label: str, text: str) -> None:
    prefix = f"    [{label}] "
    wrapped = textwrap.fill(
        text, width=_WRAP_WIDTH, initial_indent=prefix,
        subsequent_indent=" " * len(prefix),
    )
    print(wrapped)


def run() -> bool:
    setup_logging()
    print("\n" + "=" * 70)
    print(f"  MIRAGE Pentad Pilot — {_PILOT_N} seeds, all 5 slots")
    print("=" * 70)

    # ------------------------------------------------------------------ seeds
    from Dataset.sample_seeds import sample_seeds
    main_seeds, _ = sample_seeds()
    pilot_seeds = main_seeds.head(_PILOT_N).reset_index(drop=True)

    print(f"\nPilot seeds ({len(pilot_seeds)}):")
    for _, row in pilot_seeds.iterrows():
        src  = row.get("seed_source", "?")
        cat  = row.get("seed_category", "?")
        sid  = row.get("seed_id", "?")
        text = (row.get("question") or row.get("sent_more") or row.get("sentence", ""))
        print(f"  {sid}  [{src}/{cat}]  {str(text)[:80]!r}")

    # -------------------------------------------------------- deterministic
    from Dataset.pentad_generator import generate_pentad_deterministic
    rng = np.random.default_rng(seed=RANDOM_SEED)
    det_rows = generate_pentad_deterministic(pilot_seeds, rng)
    logger.info("Deterministic rows (a/b/c): %d", len(det_rows))

    # -------------------------------------------------------- slot d
    print("\n" + "-" * 70)
    print("  Slot (d) — context shift via DeepSeek (round-robin 2 keys)")
    print("-" * 70)
    from Dataset.context_shift_drafter import draft_context_shifts
    d_rows = draft_context_shifts(pilot_seeds)
    logger.info("Slot-d rows: %d  (expected %d)", len(d_rows), _PILOT_N * 2)

    for row in d_rows:
        print(f"\n  seed={row['seed_id']}  subvariant={row['subvariant']}")
        _preview(row["subvariant"], row["prompt_text"])

    # -------------------------------------------------------- slot e
    print("\n" + "-" * 70)
    print("  Slot (e) — CoT attack via DeepSeek (round-robin 2 keys)")
    print("-" * 70)
    from Dataset.cot_attack_generator import generate_cot_attacks
    e_rows = generate_cot_attacks(pilot_seeds)
    logger.info("Slot-e rows: %d  (expected %d)", len(e_rows), _PILOT_N * 3)

    for row in e_rows:
        print(f"\n  seed={row['seed_id']}  subvariant={row['subvariant']}")
        _preview(row["subvariant"], row["prompt_text"])

    # -------------------------------------------------------- assemble & validate
    import pandas as pd
    all_rows = det_rows + d_rows + e_rows
    df = pd.DataFrame(all_rows)

    print("\n" + "-" * 70)
    print("  Validation")
    print("-" * 70)

    from Dataset.validate_pentad import run_all_validations
    try:
        run_all_validations(df)
        print("  All validations PASSED")
    except Exception as exc:
        print(f"  VALIDATION FAILED: {exc}")
        return False

    # -------------------------------------------------------- per-slot summary
    print("\n  Slot distribution:")
    for slot in ["a", "b", "c", "d", "e"]:
        count = (df["slot"] == slot).sum()
        print(f"    slot {slot}: {count} prompts")

    print(f"\n  Total rows: {len(df)}  (expected {_PILOT_N * 12})")
    print(f"  Unique seed_ids: {df['seed_id'].nunique()}")
    print(f"  All prompt_ids unique: {df['prompt_id'].is_unique}")
    print(f"  Empty prompt_text: {(df['prompt_text'].str.strip() == '').sum()}")

    # -------------------------------------------------------- key-rotation audit
    print("\n  DeepSeek key rotation audit:")
    api_rows = [r for r in d_rows + e_rows]
    models_used = {r["generator_model"] for r in api_rows}
    print(f"    generator_model values: {models_used}")
    print(f"    total API-generated rows: {len(api_rows)}")

    passed = len(df) == _PILOT_N * 12 and df["prompt_id"].is_unique
    print("\n" + "=" * 70)
    print(f"  PILOT {'PASSED' if passed else 'FAILED'}")
    print("=" * 70 + "\n")
    return passed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = run()
    sys.exit(0 if ok else 1)
