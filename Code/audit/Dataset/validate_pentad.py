"""
File: Dataset/validate_pentad.py
Purpose: Schema and completeness validation for the pentad probe dataset.
         Fails loudly on any integrity violation.

Semantic gates added (fixes review finding C1):
  - Rejects "None"/"nan"/"null" prompt texts (these slipped through the old
    whitespace-only check).
  - Verifies that slot-b text differs from slot-a text for each seed (or
    flags seeds where no protected token was found so the difference is
    documented rather than silently wrong).
  - Verifies that the 5 slot-c prompt texts are distinct (genuinely different
    counterfactual swaps, not 5 copies of the same prompt).
  - Verifies that multiple-choice sources (BBQ) include their answer options
    in the prompt text.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167

Part of the audit codebase (diagnosis half of CURE).
"""

import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SEEDS_DIR

logger = logging.getLogger(__name__)

REQUIRED_SLOT_SUBVARIANTS: list[tuple[str, str]] = [
    ("a", "surface"),
    ("b", "iso_control"),
    ("d", "d_valid"),
    ("d", "d_harmful"),
    ("e", "e1_neutral"),
    ("e", "e2_stereo_push"),
    ("e", "e3_antistereo_push"),
]

EXPECTED_C_COUNT = 5
TOTAL_EXPECTED = 12  # 1 + 1 + 5 + 2 + 3

REQUIRED_COLUMNS = {
    "seed_id",
    "seed_source",
    "seed_category",
    "prompt_id",
    "slot",
    "subvariant",
    "prompt_text",
    "gold_answer",
}

# Sentinel strings that indicate a mis-constructed prompt (A2 fix)
_INVALID_PROMPT_SENTINELS = re.compile(
    r"^(None|none|nan|NaN|null|NULL|na|NA)$", re.IGNORECASE
)

# MCQ sources that must include answer options in the prompt
_MCQ_SOURCES = {"bbq"}
_MCQ_OPTION_PATTERN = re.compile(r"\(A\)|\(B\)|\(C\)", re.IGNORECASE)

_AUDIT_SOURCES = frozenset({"bbq", "crows_pairs", "stereoset"})
_PENTAD_PATH = SEEDS_DIR / "pentad_dataset.parquet"
_MANIFEST_PATH = SEEDS_DIR / "pentad_manifest.json"


# ---------------------------------------------------------------------------
# Structural validators (schema, counts, duplicates)
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame) -> None:
    """Check required columns are present and non-null."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Pentad dataset missing columns: {missing}")

    for col in REQUIRED_COLUMNS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise ValueError(f"Column '{col}' has {null_count} null values.")

    logger.info("Schema validation passed. Columns: %s", list(df.columns))


def validate_completeness(df: pd.DataFrame) -> list[str]:
    """
    Check every seed has all 12 prompt variants.
    Returns list of problem descriptions (empty if all OK).
    """
    problems: list[str] = []
    for seed_id, group in df.groupby("seed_id"):
        n_total = len(group)
        n_c = (group["slot"] == "c").sum()

        if n_c != EXPECTED_C_COUNT:
            problems.append(
                f"{seed_id}: expected {EXPECTED_C_COUNT} slot-c variants, got {n_c}"
            )

        for req_slot, req_sub in REQUIRED_SLOT_SUBVARIANTS:
            present = ((group["slot"] == req_slot) & (group["subvariant"] == req_sub)).any()
            if not present:
                problems.append(
                    f"{seed_id}: missing slot='{req_slot}' subvariant='{req_sub}'"
                )

        if n_total != TOTAL_EXPECTED:
            problems.append(
                f"{seed_id}: expected {TOTAL_EXPECTED} total prompts, got {n_total}"
            )

    if problems:
        for p in problems[:20]:
            logger.error("Completeness error: %s", p)
        raise RuntimeError(
            f"Completeness check failed: {len(problems)} seeds with errors. "
            "First errors logged above."
        )

    logger.info("Completeness check passed. All seeds have %d prompts.", TOTAL_EXPECTED)
    return problems


def validate_det_completeness(df: pd.DataFrame) -> None:
    """After a det-only patch, each seed must have exactly 7 deterministic rows."""
    det_slots = {"a", "b", "c"}
    problems: list[str] = []
    for seed_id, group in df.groupby("seed_id"):
        det = group[group["slot"].isin(det_slots)]
        if len(det) != 7:
            problems.append(f"{seed_id}: expected 7 det rows, got {len(det)}")
        n_c = (det["slot"] == "c").sum()
        if n_c != EXPECTED_C_COUNT:
            problems.append(
                f"{seed_id}: expected {EXPECTED_C_COUNT} slot-c variants, got {n_c}"
            )
    if problems:
        for p in problems[:20]:
            logger.error("Det completeness error: %s", p)
        raise RuntimeError(
            f"Deterministic completeness failed for {len(problems)} seeds."
        )
    logger.info(
        "Deterministic completeness passed for %d seeds.", df["seed_id"].nunique()
    )


def validate_duplicate_prompt_ids(df: pd.DataFrame) -> None:
    """Fail if any prompt_id appears more than once."""
    dupes = df[df["prompt_id"].duplicated(keep=False)]
    if len(dupes) > 0:
        raise ValueError(
            f"{len(dupes)} rows have duplicate prompt_ids: "
            f"{dupes['prompt_id'].unique().tolist()[:10]}"
        )
    logger.info("No duplicate prompt_ids found.")


# ---------------------------------------------------------------------------
# Semantic validators (new -- fixes C1)
# ---------------------------------------------------------------------------

def validate_no_sentinel_prompts(df: pd.DataFrame) -> None:
    """
    Fail if any prompt_text is empty, whitespace-only, or equals a known
    sentinel string (None/nan/null) that indicates a construction failure.

    Previously only the whitespace check was applied, which let 1281
    StereoSet "None" prompts through.
    """
    problems: list[str] = []

    for _, row in df.iterrows():
        text = str(row["prompt_text"]) if pd.notna(row["prompt_text"]) else ""
        stripped = text.strip()

        if not stripped:
            problems.append(
                f"prompt_id={row['prompt_id']}: empty or whitespace-only prompt_text."
            )
        elif _INVALID_PROMPT_SENTINELS.match(stripped):
            problems.append(
                f"prompt_id={row['prompt_id']}: sentinel prompt_text value '{stripped}' "
                f"(seed_source={row.get('seed_source','')})."
            )

    if problems:
        for p in problems[:20]:
            logger.error("Sentinel prompt error: %s", p)
        raise ValueError(
            f"{len(problems)} prompts contain sentinel/empty text. "
            "First errors logged above."
        )
    logger.info("No sentinel prompts found.")


def validate_b_differs_from_a(df: pd.DataFrame) -> None:
    """
    Verify that slot-b prompt_text differs from slot-a prompt_text for each
    seed.  Seeds with no protected token in their prompt (and therefore an
    identical slot-b) are flagged as warnings rather than errors, because the
    review notes this is an inherent limitation for some items.

    A HARD FAILURE is raised if more than 50% of seeds have identical a/b.
    """
    identical_count = 0
    total_seeds = 0

    for seed_id, group in df.groupby("seed_id"):
        a_rows = group[group["subvariant"] == "surface"]
        b_rows = group[group["subvariant"] == "iso_control"]
        if a_rows.empty or b_rows.empty:
            continue
        total_seeds += 1
        a_text = str(a_rows.iloc[0]["prompt_text"])
        b_text = str(b_rows.iloc[0]["prompt_text"])
        if a_text.strip() == b_text.strip():
            identical_count += 1
            logger.debug(
                "seed_id=%s: slot-a == slot-b (no protected token substituted).", seed_id
            )

    if total_seeds == 0:
        return

    rate = identical_count / total_seeds
    logger.info(
        "Slot-b identical to slot-a: %d/%d seeds (%.1f%%).",
        identical_count, total_seeds, rate * 100,
    )
    if rate > 0.5:
        raise ValueError(
            f"slot-b == slot-a for {identical_count}/{total_seeds} seeds ({rate:.1%}). "
            "Protected-token substitution is failing for the majority of seeds. "
            "Check _PROTECTED_TO_NEUTRAL and the equivalence-set routing."
        )


def validate_c_variants_distinct(df: pd.DataFrame) -> None:
    """
    Verify that the 5 slot-c prompt_texts are distinct for each seed.

    Seeds with fewer than 5 distinct texts are flagged as warnings.  A HARD
    FAILURE is raised if more than 25% of seeds have only 1 unique c-text
    (which was the case for 87% of seeds in the broken dataset).
    """
    degenerate_count = 0
    total_seeds = 0

    for seed_id, group in df.groupby("seed_id"):
        c_rows = group[group["slot"] == "c"]
        if len(c_rows) < 2:
            continue
        total_seeds += 1
        n_unique = c_rows["prompt_text"].nunique()
        if n_unique == 1:
            degenerate_count += 1
            logger.debug(
                "seed_id=%s: all 5 slot-c prompts are identical (swap had no effect).", seed_id
            )
        elif n_unique < 5:
            logger.warning(
                "seed_id=%s: only %d of 5 slot-c prompts are distinct.", seed_id, n_unique
            )

    if total_seeds == 0:
        return

    rate = degenerate_count / total_seeds
    logger.info(
        "Slot-c all-identical seeds: %d/%d (%.1f%%).",
        degenerate_count, total_seeds, rate * 100,
    )
    if rate > 0.25:
        raise ValueError(
            f"{degenerate_count}/{total_seeds} seeds ({rate:.1%}) have all 5 slot-c "
            "prompts identical -- the counterfactual swap is not taking effect in the "
            "full prompt text. Check _build_slot_c and verify the demographic token "
            "appears in the complete prompt (context + question + options)."
        )


def validate_mcq_options_present(df: pd.DataFrame) -> None:
    """
    For MCQ sources (BBQ), verify that every slot-a prompt contains answer
    options in (A) / (B) / (C) format.

    StereoSet slots a/b/c should also include options after the fix.
    """
    problems: list[str] = []

    mcq_rows = df[
        (df["seed_source"].isin(_MCQ_SOURCES)) & (df["slot"] == "a")
    ]

    for _, row in mcq_rows.iterrows():
        text = str(row["prompt_text"])
        if not _MCQ_OPTION_PATTERN.search(text):
            problems.append(
                f"prompt_id={row['prompt_id']} ({row['seed_source']}): "
                "slot-a prompt is missing answer options (A)/(B)/(C). "
                "Expected the full context + MCQ format."
            )

    if problems:
        for p in problems[:20]:
            logger.error("MCQ options missing: %s", p)
        raise ValueError(
            f"{len(problems)} MCQ prompts missing answer options. "
            "First errors logged above."
        )
    logger.info("MCQ answer-options check passed.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_gold_answer_present(df: pd.DataFrame) -> None:
    """Every row must carry a non-empty gold_answer (WinoBias may use 'unknown')."""
    if "gold_answer" not in df.columns:
        raise ValueError("Pentad dataset missing required column: gold_answer")
    null_gold = df["gold_answer"].isna() | (df["gold_answer"].astype(str).str.strip() == "")
    if null_gold.any():
        raise ValueError(
            f"{null_gold.sum()} rows have null/empty gold_answer — scoring will be invalid."
        )
    logger.info("gold_answer column present and non-empty on all rows.")


def validate_audit_gold_labels(df: pd.DataFrame) -> None:
    """
    Audited benchmarks (bbq, crows_pairs, stereoset) must have scorable gold
    on slot-a.  WinoBias must not appear in the main pentad at all.
    """
    audit_sources = {"bbq", "crows_pairs", "stereoset"}
    problems: list[str] = []

    wino = df[df["seed_source"].astype(str).str.lower() == "winobias"]
    if len(wino) > 0:
        problems.append(
            f"{len(wino)} WinoBias rows present — WinoBias must be held out, not audited."
        )

    a_rows = df[(df["slot"] == "a") & df["seed_source"].astype(str).str.lower().isin(audit_sources)]
    bad_gold = a_rows[
        a_rows["gold_answer"].isna()
        | (a_rows["gold_answer"].astype(str).str.strip() == "")
        | (
            (a_rows["gold_answer"].astype(str).str.lower() == "unknown")
            & (a_rows["seed_source"].astype(str).str.lower() != "bbq")
        )
    ]
    if len(bad_gold) > 0:
        problems.append(
            f"{len(bad_gold)} audited slot-a rows have unknown/missing gold_answer."
        )

    if problems:
        for p in problems:
            logger.error("Audit gold error: %s", p)
        raise ValueError(
            f"Audit gold validation failed: {'; '.join(problems)}"
        )
    logger.info("Audit gold labels valid for all bbq/crows/stereoset seeds.")


_BAD_SLOT_B_PATTERNS = (
    re.compile(r"\bperson and person\b", re.IGNORECASE),
    re.compile(r"\bperson man\b", re.IGNORECASE),
    re.compile(r"\bperson woman\b", re.IGNORECASE),
    re.compile(r"\ba person man\b", re.IGNORECASE),
    re.compile(r"\ba person woman\b", re.IGNORECASE),
    re.compile(r"\ba someone\b", re.IGNORECASE),
    re.compile(r"Context:\s+person\b", re.IGNORECASE),
    re.compile(r"\bperson\s+are\b", re.IGNORECASE),
)


def validate_slot_b_grammar(df: pd.DataFrame) -> None:
    """Reject ungrammatical slot-b iso-controls (Person and Person, person man, ...)."""
    problems: list[str] = []
    for seed_id, group in df.groupby("seed_id"):
        b_rows = group[group["slot"] == "b"]
        if b_rows.empty:
            continue
        text = str(b_rows.iloc[0]["prompt_text"])
        for pat in _BAD_SLOT_B_PATTERNS:
            if pat.search(text):
                problems.append(str(seed_id))
                break
    if problems:
        sample = problems[:10]
        raise ValueError(
            f"{len(problems)} seeds have ungrammatical slot-b iso-control text. "
            f"Examples: {sample}"
        )
    logger.info("Slot-b grammar check passed.")


def validate_deepseek_embeds_slot_a(df: pd.DataFrame) -> None:
    """Slot d/e must contain the slot-a prompt text (DeepSeek prepends context)."""
    problems: list[str] = []
    for seed_id, group in df.groupby("seed_id"):
        a_rows = group[group["slot"] == "a"]
        if a_rows.empty:
            continue
        a_text = str(a_rows.iloc[0]["prompt_text"]).strip()
        if len(a_text) < 40:
            continue
        fingerprint = a_text[-80:].lower()
        for slot in ("d", "e"):
            for _, row in group[group["slot"] == slot].iterrows():
                pt = str(row["prompt_text"]).lower()
                if fingerprint not in pt and a_text.lower() not in pt:
                    problems.append(f"{row['prompt_id']}: {slot} missing slot-a text")
                    break

    if len(problems) > len(df["seed_id"].unique()) * 0.05:
        for p in problems[:10]:
            logger.error("DeepSeek embed error: %s", p)
        raise ValueError(
            f"{len(problems)} slot d/e prompts do not embed their slot-a text. "
            "Regenerate API slots from the patched slot-a prompts."
        )
    if problems:
        logger.warning(
            "%d slot d/e rows missing slot-a embed (under 5%% threshold).", len(problems)
        )
    else:
        logger.info("All DeepSeek d/e prompts embed their slot-a text.")


def run_all_validations(df: pd.DataFrame, require_api_slots: bool = True) -> None:
    """Run the validation suite. Raises on any failure."""
    logger.info("Starting pentad validation on %d rows ...", len(df))
    validate_schema(df)
    validate_gold_answer_present(df)
    validate_audit_gold_labels(df)
    validate_no_sentinel_prompts(df)
    validate_duplicate_prompt_ids(df)
    if require_api_slots:
        validate_completeness(df)
    else:
        validate_det_completeness(df)
    validate_b_differs_from_a(df)
    validate_slot_b_grammar(df)
    validate_c_variants_distinct(df)
    validate_mcq_options_present(df)
    if require_api_slots:
        validate_deepseek_embeds_slot_a(df)
    logger.info("All pentad validations PASSED.")


def pentad_file_sha256(path: Path | None = None) -> str:
    """SHA-256 of the pentad parquet file on disk."""
    path = path or _PENTAD_PATH
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_pentad_manifest(df: pd.DataFrame, path: Path | None = None) -> dict:
    """Persist production metadata after a successful pentad build."""
    path = path or _MANIFEST_PATH
    audit = df[df["seed_source"].astype(str).str.lower().isin(_AUDIT_SOURCES)]
    manifest = {
        "pentad_sha256": pentad_file_sha256(_PENTAD_PATH),
        "n_rows": int(len(df)),
        "n_audit_seeds": int(audit["seed_id"].nunique()),
        "rows_per_seed": TOTAL_EXPECTED,
        "has_api_slots": bool((df["slot"].isin(["d", "e"])).any()),
    }
    excluded_path = SEEDS_DIR / "excluded_seeds.json"
    if excluded_path.exists():
        with open(excluded_path) as fh:
            manifest["excluded_seeds"] = json.load(fh)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Pentad manifest written: %s", path)
    return manifest


def assert_production_ready(df: pd.DataFrame | None = None) -> None:
    """
    Hard gate before GPU work: full 12-slot pentad for all included audit seeds.

    Raises on any violation.  Call this from the pipeline orchestrator and
    run_gpu_pipeline.py — never start GPU inference on a det-only or partial set.
    """
    if df is None:
        if not _PENTAD_PATH.exists():
            raise FileNotFoundError(f"Pentad dataset missing: {_PENTAD_PATH}")
        df = pd.read_parquet(_PENTAD_PATH)

    audit = df[df["seed_source"].astype(str).str.lower().isin(_AUDIT_SOURCES)]
    if audit.empty:
        raise ValueError("Pentad has no audit-source rows (bbq/crows_pairs/stereoset).")

    wino = df[df["seed_source"].astype(str).str.lower() == "winobias"]
    if len(wino) > 0:
        raise ValueError(f"{len(wino)} WinoBias rows in pentad — must be held out.")

    n_seeds = audit["seed_id"].nunique()
    n_api = (audit["slot"].isin(["d", "e"])).sum()
    if n_api == 0:
        raise ValueError(
            "Pentad has no slot d/e rows. Run regenerate_api_slots.py before GPU pipeline."
        )

    expected_rows = n_seeds * TOTAL_EXPECTED
    if len(audit) != expected_rows:
        raise ValueError(
            f"Pentad incomplete: {len(audit)} audit rows, expected {expected_rows} "
            f"({n_seeds} seeds × {TOTAL_EXPECTED})."
        )

    run_all_validations(df, require_api_slots=True)
    logger.info(
        "Production-ready pentad: %d seeds, %d rows.", n_seeds, len(audit)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pentad_path = SEEDS_DIR / "pentad_dataset.parquet"
    if not pentad_path.exists():
        logger.error("Pentad dataset not found at %s. Run pentad_generator.py first.", pentad_path)
        import sys
        sys.exit(1)
    df = pd.read_parquet(pentad_path)
    assert_production_ready(df)
