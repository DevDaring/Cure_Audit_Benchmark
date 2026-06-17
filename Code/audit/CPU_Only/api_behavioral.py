"""
File: CPU_Only/api_behavioral.py
Purpose: Behavioral evaluation across the 4 API models with retry/fallback
         policy specified in spec Section 4.2.

Temperature-variance pass (FM4):
  After the deterministic pass (sample_index=0, temperature=0.0), a second
  pass runs slot-a 5 times at temperature=0.7 (sample_index 1-5).  This
  provides data for FM4 (criterion leakage / answer variance under sampling).
  Previously this pass was absent for API models (review finding B5).

gold_answer is copied from pentad_df into every behavioral result row so
that scoring.py can compare parsed_answer to gold_answer without a join.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - Parrish et al. (2022). BBQ. Findings of ACL 2022.
  - Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
  - Nadeem et al. (2021). StereoSet. ACL-IJCNLP 2021.

Part of the audit codebase (diagnosis half of CURE).
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import API_CHECKPOINT_EVERY, API_MODELS, RESULTS_DIR, RESEARCH_SYSTEM_PROMPT, ensure_dirs
from parse_utils import parse_model_response
from results_utils import dedup_behavioral, reparse_failed_rows

logger = logging.getLogger(__name__)

_BEHAVIORAL_PATH = RESULTS_DIR / "behavioral_results.parquet"

_SYSTEM_PROMPT = RESEARCH_SYSTEM_PROMPT

# Number of stochastic samples for the FM4 variance pass (slot-a only)
_FM4_N_SAMPLES = 5
_FM4_TEMPERATURE = 0.7


def _parse_response(raw: str, prompt_id: str) -> tuple[dict | None, str]:
    """Parse a raw response with LOCAL methods first; the judge API is only a
    last resort when every deterministic/heuristic method has failed."""
    if not raw:
        return None, "failed"

    # All local parsing (json -> deterministic repair -> json_repair lib ->
    # quoted-lines -> regex) happens inside parse_model_response.
    success, answer, conf, rationale, method, reason = parse_model_response(raw)
    if success:
        return {"answer": answer, "confidence": conf, "rationale": rationale}, method

    # Last resort only: DeepSeek judge extracts the answer from output that no
    # local method could parse (GCP/Gemini judge retired — was rate-limited).
    from CPU_Only.judge_router import judge
    parsed, method = judge(raw, provider="deepseek")
    return parsed, method


def _cell_value(val: Any) -> Any:
    """Preserve NaN/None like GPU osm_behavioral rows (avoid str(nan))."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _build_messages(prompt_text: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]


def _call_api_model(
    model_cfg: dict,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Route to the appropriate client based on primary_route."""
    primary = model_cfg["primary_route"]
    model_id = model_cfg["model_id"]

    if primary == "bedrock":
        from CPU_Only.api_clients.bedrock_client import call_bedrock_with_fallback
        return call_bedrock_with_fallback(model_id, messages, max_tokens, temperature=temperature)
    elif primary == "megallm":
        from CPU_Only.api_clients.megallm_client import call_megallm_with_fallback
        return call_megallm_with_fallback(model_id, messages, max_tokens, temperature=temperature)
    elif primary == "mistral":
        from CPU_Only.api_clients.mistral_client import call_mistral_with_fallback
        # Pass the config model_id through so API-4 calls the declared model
        # (mistral-medium-latest); on Mistral-platform failure it falls back to
        # OpenRouter (mistralai/mistral-medium-3-5, same model).
        return call_mistral_with_fallback(messages, max_tokens, model_name=model_id, temperature=temperature)
    else:
        raise ValueError(f"Unknown primary_route: '{primary}'")


def _evaluate_single_prompt(
    model_cfg: dict,
    prompt_id: str,
    prompt_text: str,
    gold_answer: str,
    run_id: str,
    seed_id: str,
    seed_source: str,
    seed_category: str,
    seed_subcategory: str,
    slot: str,
    subvariant: str,
    max_tokens: int,
    sample_index: int,
    temperature: float,
) -> dict:
    """Run a single API call and return a result row dict."""
    messages = _build_messages(prompt_text)
    t0 = time.monotonic()
    api_result = _call_api_model(model_cfg, messages, max_tokens, temperature=temperature)
    latency_ms = int((time.monotonic() - t0) * 1000)

    model_name = model_cfg["name"]
    raw = api_result.get("raw_response", "")
    parsed = None
    parse_method = "failed"
    parsed_answer = ""
    parsed_confidence = 0.0
    parsed_rationale = ""
    success_flag = api_result.get("success_flag", False)
    failure_reason = api_result.get("failure_reason", "")

    if success_flag and raw:
        parsed, parse_method = _parse_response(raw, prompt_id)
        if parsed:
            parsed_answer = str(parsed.get("answer", ""))
            parsed_confidence = float(parsed.get("confidence", 0.0))
            parsed_rationale = str(parsed.get("rationale", ""))
            if not parsed_answer.strip():
                # Parsed/judged to an empty answer (e.g. a truncated response with
                # no extractable answer). Not a usable result -> mark as failure so
                # it is retried and excluded from scoring, never stored as a
                # successful empty/hallucinated row.
                parse_method = "failed"
                success_flag = False
                failure_reason = "empty_answer"
        else:
            parse_method = "failed"
            success_flag = False
            failure_reason = "parse_error"

    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed_id": seed_id,
        "seed_source": seed_source,
        "seed_category": seed_category,
        "seed_subcategory": seed_subcategory,
        "prompt_id": prompt_id,
        "slot": slot,
        "subvariant": subvariant,
        "model_name": model_name,
        "model_provider": model_cfg.get("primary_route", ""),
        "model_version": model_cfg.get("model_id", ""),
        "route_used": api_result.get("route_used", ""),
        "key_index": api_result.get("key_index", -1),
        "attempt_count": api_result.get("attempt_count", 1),
        "prompt_text": prompt_text,
        "gold_answer": gold_answer,
        "raw_response": raw,
        "parsed_answer": parsed_answer,
        "parsed_confidence": parsed_confidence,
        "parsed_rationale": parsed_rationale,
        "parse_method": parse_method,
        "success_flag": success_flag,
        "failure_reason": failure_reason,
        "latency_ms": api_result.get("latency_ms", latency_ms),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "sample_index": sample_index,
    }


def _save_checkpoint(
    working: pd.DataFrame,
    new_rows: list[dict],
    completed_keys: set[tuple],
) -> pd.DataFrame:
    """Append rows, dedup, persist, and update completed_keys."""
    if not new_rows:
        return working
    batch_df = pd.DataFrame(new_rows)
    working = pd.concat([working, batch_df], ignore_index=True)
    working = dedup_behavioral(working)
    working.to_parquet(_BEHAVIORAL_PATH, index=False)
    for _, row in batch_df[batch_df["success_flag"] == True].iterrows():  # noqa: E712
        completed_keys.add((row["prompt_id"], row["model_name"], int(row["sample_index"])))
    return working


def evaluate_api_model(
    model_cfg: dict,
    pentad_df: pd.DataFrame,
    run_id: str,
    completed_keys: set[tuple],
    working: pd.DataFrame,
    max_tokens: int = 256,
    sample_index: int = 0,
    temperature: float = 0.0,
) -> pd.DataFrame:
    """
    Evaluate a single API model on the pentad dataset (sequential, one call at a time).
    Checkpoints every API_CHECKPOINT_EVERY prompts so runs can be stopped and resumed.
    """
    model_name = model_cfg["name"]
    pending: list[dict] = []
    done = 0
    total = len(pentad_df)

    for i, (_, prow) in enumerate(pentad_df.iterrows()):
        prompt_id = prow["prompt_id"]
        slot = prow.get("slot", "")
        subvariant = prow.get("subvariant", "")

        if sample_index > 0 and slot != "a":
            continue

        if (prompt_id, model_name, sample_index) in completed_keys:
            continue

        prompt_text = str(prow.get("prompt_text", ""))
        if not prompt_text.strip() or prompt_text.strip().lower() == "none":
            continue

        row = _evaluate_single_prompt(
            model_cfg=model_cfg,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            gold_answer=str(prow.get("gold_answer", "")),
            run_id=run_id,
            seed_id=str(prow.get("seed_id", "")),
            seed_source=str(prow.get("seed_source", "")),
            seed_category=_cell_value(prow.get("seed_category")),
            seed_subcategory=_cell_value(prow.get("seed_subcategory")),
            slot=slot,
            subvariant=subvariant,
            max_tokens=max_tokens,
            sample_index=sample_index,
            temperature=temperature,
        )
        pending.append(row)
        done += 1

        if done % API_CHECKPOINT_EVERY == 0:
            working = _save_checkpoint(working, pending, completed_keys)
            pending = []
            logger.info(
                "API %s (sample_index=%d): %d prompts done (checkpoint saved).",
                model_name, sample_index, done,
            )

        if (i + 1) % 50 == 0:
            logger.info(
                "API %s (sample_index=%d): scanned %d/%d pentad rows.",
                model_name, sample_index, i + 1, total,
            )

    if pending:
        working = _save_checkpoint(working, pending, completed_keys)

    return working


def run_api_behavioral(
    pentad_df: pd.DataFrame,
    run_id: str,
    max_tokens: int = 256,
) -> pd.DataFrame:
    """
    Run behavioral evaluation for all 4 API models with resume logic.

    Sequential execution (one API call at a time) to avoid rate limits.
    Checkpoints every MIRAGE_API_CHECKPOINT_EVERY prompts (default 50).

    Appends to existing behavioral_results.parquet.
    """
    ensure_dirs()

    if _BEHAVIORAL_PATH.exists():
        existing = dedup_behavioral(reparse_failed_rows(pd.read_parquet(_BEHAVIORAL_PATH)))
        if len(existing) > 0:
            existing.to_parquet(_BEHAVIORAL_PATH, index=False)
        logger.info("Loaded %d existing results for resume (deduped).", len(existing))
    else:
        existing = pd.DataFrame()

    completed_keys: set[tuple] = set()
    if len(existing) > 0 and "success_flag" in existing.columns:
        for _, row in existing[existing["success_flag"] == True].iterrows():  # noqa: E712
            completed_keys.add((row["prompt_id"], row["model_name"], int(row["sample_index"])))

    working = existing

    for model_cfg in API_MODELS:
        model_name = model_cfg["name"]
        logger.info("API evaluation: model=%s (sequential) ...", model_name)

        working = evaluate_api_model(
            model_cfg, pentad_df, run_id, completed_keys, working,
            max_tokens=max_tokens, sample_index=0, temperature=0.0,
        )
        logger.info("  Deterministic pass complete for %s.", model_name)

        logger.info(
            "  FM4 variance pass: model=%s, %d samples at temp=%.1f ...",
            model_name, _FM4_N_SAMPLES, _FM4_TEMPERATURE,
        )
        for sample_idx in range(1, _FM4_N_SAMPLES + 1):
            working = evaluate_api_model(
                model_cfg, pentad_df, run_id, completed_keys, working,
                max_tokens=max_tokens,
                sample_index=sample_idx,
                temperature=_FM4_TEMPERATURE,
            )
            logger.info("    sample_index=%d complete for %s.", sample_idx, model_name)

    final = dedup_behavioral(working) if len(working) > 0 else pd.DataFrame()
    if len(final) > 0:
        final.to_parquet(_BEHAVIORAL_PATH, index=False)

    logger.info("API behavioral evaluation complete. Total rows: %d", len(final))
    return final
