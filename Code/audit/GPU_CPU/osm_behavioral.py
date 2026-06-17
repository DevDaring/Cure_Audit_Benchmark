"""
File: GPU_CPU/osm_behavioral.py
Purpose: Behavioral evaluation of all 4 OSM models across the full pentad
         probe set. Produces behavioral_results.parquet.

Parallelism strategy:
  - **80 GB GPU:** all 4 OSM models stay in VRAM; EVAL_BATCH_SIZE=8 batched forwards.
  - **≤40 GB GPU (sequential loading):** one model at a time; batch size auto-reduced
    to 4 (override with MIRAGE_EVAL_BATCH_SIZE).
  - Batch generation (batch_size 4–8) is used for all OSM models including nnsight
    (Qwen, Phi); outlines single-prompt path is disabled for speed.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - Parrish et al. (2022). BBQ. Findings of ACL 2022.
  - outlines: constrained JSON decoding. https://github.com/outlines-dev/outlines

Part of the audit codebase (diagnosis half of CURE).
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RESULTS_DIR, RESEARCH_SYSTEM_PROMPT, ensure_dirs
from parse_utils import parse_model_response
from results_utils import dedup_behavioral, reparse_failed_rows

logger = logging.getLogger(__name__)

# Number of prompts to forward in a single batched GPU call (80 GB, all models resident).
EVAL_BATCH_SIZE: int = 8


def _eval_batch_size() -> int:
    """Batch size for behavioral inference; lower on sequential / smaller GPUs."""
    override = os.environ.get("MIRAGE_EVAL_BATCH_SIZE", "").strip()
    if override.isdigit():
        return max(1, int(override))
    try:
        from GPU_CPU.load_osm import use_sequential_loading

        return 4 if use_sequential_loading() else EVAL_BATCH_SIZE
    except Exception:
        return EVAL_BATCH_SIZE

_BEHAVIORAL_PATH = RESULTS_DIR / "behavioral_results.parquet"

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["answer", "confidence", "rationale"],
}

_SYSTEM_PROMPT = RESEARCH_SYSTEM_PROMPT


def _save_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate (prefer success) and write behavioral_results.parquet."""
    clean = dedup_behavioral(df)
    clean.to_parquet(_BEHAVIORAL_PATH, index=False)
    return clean


def _completed_keys_from(df: pd.DataFrame) -> set[tuple]:
    if len(df) == 0 or "success_flag" not in df.columns:
        return set()
    done = df[df["success_flag"].astype(bool)]
    if len(done) == 0:
        return set()
    return set(
        zip(done["prompt_id"], done["model_name"], done["sample_index"].astype(int))
    )


def _generate_constrained(
    model: Any,
    tokenizer: Any,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Generate a response using outlines constrained JSON decoding.
    Falls back to unconstrained generation if outlines fails.
    Used for single-prompt paths (e.g. when batch_size=1).
    """
    try:
        import outlines  # type: ignore
        import outlines.generate as og

        gen = og.json(model, _JSON_SCHEMA)
        result = gen(prompt, max_tokens=max_tokens, temperature=temperature)
        return json.dumps(result)
    except Exception as exc:
        logger.debug("outlines constrained decode failed (%s), falling back.", exc)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        import torch
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _generate_constrained_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    temperature: float,
    max_tokens: int,
) -> list[str]:
    """
    Batch generation path.  Tokenises all prompts together with left-padding
    and runs a single model.generate() call, returning one decoded string per
    prompt.

    Outlines does not support batching, so this path always uses the raw
    model.generate() fallback.  At temperature=0 the output is fully
    deterministic, identical to the single-prompt unconstrained fallback.

    Parameters
    ----------
    prompts : list[str]
        Up to EVAL_BATCH_SIZE formatted prompt strings.

    Returns
    -------
    list[str]
        Decoded output strings, one per input prompt (same order).
    """
    import torch

    if not prompts:
        return []

    # Left-padding so all sequences in the batch end at the same position —
    # this is required for decoder-only causal models.
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                # Prevent degenerate repetition loops (common on small models,
                # causes generate() to hang at max_new_tokens=256).
                repetition_penalty=1.1,
                # Stop cleanly at EOS even before max_new_tokens is reached.
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        tokenizer.padding_side = orig_padding_side

    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
        for i in range(out.shape[0])
    ]


def _build_prompt(system: str, user: str, tokenizer: Any) -> str:
    """Build a chat-formatted prompt. Falls back when tokenizer rejects system role (Gemma)."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Gemma-2 and some tokenizers: "System role not supported"
            merged = f"{system.strip()}\n\n{user.strip()}"
            fallback = [{"role": "user", "content": merged}]
            return tokenizer.apply_chat_template(
                fallback, tokenize=False, add_generation_prompt=True
            )
    return f"<|system|>{system}\n<|user|>{user}\n<|assistant|>"


def evaluate_osm_model(
    model_cfg: dict,
    model: Any,
    tokenizer: Any,
    pentad_df: pd.DataFrame,
    run_id: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    sample_index: int = 0,
    batch_size: int = EVAL_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Evaluate a single OSM model on the full pentad dataset.

    Parallelism: prompts are processed in batches of `batch_size` using a
    single GPU forward pass per batch (via _generate_constrained_batch).
    At temperature=0 this is deterministically equivalent to single-prompt
    evaluation.

    Parameters
    ----------
    batch_size : int
        Number of prompts per GPU call.  Default is EVAL_BATCH_SIZE (8).
        Lower to 1 or 2 when debugging or on GPUs with <24 GB free VRAM.

    Returns
    -------
    pd.DataFrame
        Result rows conforming to the MIRAGE result schema.
    """
    from GPU_CPU.utils_attention import _get_token_position  # noqa: F401

    model_name = model_cfg["name"]
    model_provider = "hf"
    try:
        model_version = model.config._name_or_path
    except Exception:
        model_version = model_cfg["hf_id"]

    # Pre-filter: FM4 variance pass only needs slot-a.
    if sample_index > 0:
        pentad_df = pentad_df[pentad_df.get("slot", pd.Series(dtype=str)) == "a"].copy()

    # Drop empty prompts.
    pentad_df = pentad_df[pentad_df["prompt_text"].astype(str).str.strip() != ""].reset_index(drop=True)

    # Use standard batch generation for all models.
    # The outlines single-prompt path was ~8-12 s/prompt (vs ~0.5 s/prompt
    # in batch mode) due to per-call outlines setup; any marginal JSON-format
    # benefit from constrained decoding is outweighed by the 10–20× slowdown.
    # Batch generation handles JSON parsing via the same fallback parser, so
    # research-critical failure rates remain well below 5 %.
    use_constrained_single = False

    rows: list[dict] = []
    total = len(pentad_df)
    now_utc = datetime.now(timezone.utc).isoformat()

    for batch_start in range(0, total, batch_size):
        batch = pentad_df.iloc[batch_start : batch_start + batch_size]

        # Build formatted prompts for the whole batch.
        formatted_prompts = [
            _build_prompt(_SYSTEM_PROMPT, str(r["prompt_text"]), tokenizer)
            for _, r in batch.iterrows()
        ]

        t_start = time.monotonic()
        if use_constrained_single:
            raw_responses = []
            for fp in formatted_prompts:
                try:
                    raw_responses.append(
                        _generate_constrained(model, tokenizer, fp, temperature, max_tokens)
                    )
                except Exception as exc:
                    raw_responses.append(str(exc))
        else:
            try:
                raw_responses = _generate_constrained_batch(
                    model, tokenizer, formatted_prompts, temperature, max_tokens
                )
            except Exception as exc:
                logger.warning(
                    "OSM %s: batch generation failed (%s). Falling back to single-prompt.",
                    model_name, exc,
                )
                raw_responses = []
                for fp in formatted_prompts:
                    try:
                        raw_responses.append(
                            _generate_constrained(model, tokenizer, fp, temperature, max_tokens)
                        )
                    except Exception as e2:
                        raw_responses.append(str(e2))

        latency_ms_total = int((time.monotonic() - t_start) * 1000)
        per_prompt_ms = latency_ms_total // max(len(batch), 1)

        for j, (_, prow) in enumerate(batch.iterrows()):
            raw_response = raw_responses[j] if j < len(raw_responses) else ""
            success_flag, parsed_answer, parsed_confidence, parsed_rationale, parse_method, failure_reason = (
                parse_model_response(raw_response)
            )

            if not success_flag and j < len(formatted_prompts):
                for _ in range(2):
                    try:
                        raw_response = _generate_constrained(
                            model, tokenizer, formatted_prompts[j], temperature, max_tokens
                        )
                        success_flag, parsed_answer, parsed_confidence, parsed_rationale, parse_method, failure_reason = (
                            parse_model_response(raw_response)
                        )
                        if success_flag:
                            break
                    except Exception:
                        pass

            rows.append(
                {
                    "run_id": run_id,
                    "timestamp_utc": now_utc,
                    "seed_id": prow.get("seed_id", ""),
                    "seed_source": prow.get("seed_source", ""),
                    "seed_category": prow.get("seed_category", ""),
                    "seed_subcategory": prow.get("seed_subcategory", ""),
                    "prompt_id": prow["prompt_id"],
                    "slot": prow.get("slot", ""),
                    "subvariant": prow.get("subvariant", ""),
                    "gold_answer": str(prow.get("gold_answer", "")),
                    "model_name": model_name,
                    "model_provider": model_provider,
                    "model_version": model_version,
                    "route_used": "local",
                    "key_index": -1,
                    "attempt_count": 1,
                    "prompt_text": str(prow.get("prompt_text", "")),
                    "raw_response": raw_response,
                    "parsed_answer": parsed_answer,
                    "parsed_confidence": parsed_confidence,
                    "parsed_rationale": parsed_rationale,
                    "parse_method": parse_method,
                    "success_flag": success_flag,
                    "failure_reason": failure_reason,
                    "latency_ms": per_prompt_ms,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "sample_index": sample_index,
                }
            )

        if (batch_start + len(batch)) % 100 < batch_size:
            logger.info(
                "OSM %s: %d/%d prompts done (sample_index=%d, batch_size=%d).",
                model_name, batch_start + len(batch), total, sample_index, batch_size,
            )

    return pd.DataFrame(rows)


def run_osm_behavioral(
    pentad_df: pd.DataFrame,
    models: dict[str, tuple[Any, Any]],
    run_id: str,
    force: bool = False,
) -> pd.DataFrame:
    """
    Run behavioral evaluation for all OSM models.
    Uses resume logic: skips already-completed rows.
    """
    ensure_dirs()

    # Load existing results and collapse duplicates from prior restart loops.
    if _BEHAVIORAL_PATH.exists():
        existing = dedup_behavioral(reparse_failed_rows(pd.read_parquet(_BEHAVIORAL_PATH)))
        if len(existing) > 0:
            existing.to_parquet(_BEHAVIORAL_PATH, index=False)
        logger.info("Loaded %d existing behavioral results (deduped).", len(existing))
    else:
        existing = pd.DataFrame()

    completed_keys = _completed_keys_from(existing)
    working = existing
    batch_size = _eval_batch_size()

    for model_cfg in OSM_MODELS:
        model_name = model_cfg["name"]
        if model_name not in models:
            logger.warning("Model '%s' not loaded, skipping.", model_name)
            continue

        model, tokenizer = models[model_name]

        # Deterministic pass (temperature=0, sample_index=0)
        done_det = {pid for (pid, mn, si) in completed_keys if mn == model_name and si == 0}
        missing_det = pentad_df[~pentad_df["prompt_id"].isin(done_det)]
        if len(missing_det) > 0:
            logger.info(
                "OSM %s: deterministic pass on %d prompts ...", model_name, len(missing_det)
            )
            det_results = evaluate_osm_model(
                model_cfg,
                model,
                tokenizer,
                missing_det,
                run_id,
                temperature=0.0,
                sample_index=0,
                batch_size=batch_size,
            )
            working = pd.concat([working, det_results], ignore_index=True)
            working = _save_behavioral(working)
            completed_keys = _completed_keys_from(working)
            logger.info("  Saved deterministic results incrementally.")

        # Variance pass (temperature=0.7, sample_index=1-5)
        for si in range(1, 6):
            done_var = {pid for (pid, mn, sv) in completed_keys if mn == model_name and sv == si}
            missing_var = pentad_df[~pentad_df["prompt_id"].isin(done_var)]
            if len(missing_var) > 0:
                logger.info(
                    "OSM %s: variance pass sample_index=%d on %d prompts ...",
                    model_name, si, len(missing_var),
                )
                var_results = evaluate_osm_model(
                    model_cfg,
                    model,
                    tokenizer,
                    missing_var,
                    run_id,
                    temperature=0.7,
                    sample_index=si,
                    batch_size=batch_size,
                )
                working = pd.concat([working, var_results], ignore_index=True)
                working = _save_behavioral(working)
                completed_keys = _completed_keys_from(working)

    final = _save_behavioral(working) if len(working) > 0 else pd.DataFrame()
    logger.info("OSM behavioral evaluation complete. Total rows: %d", len(final))
    return final
