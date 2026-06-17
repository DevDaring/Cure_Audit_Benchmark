"""
File: CPU_Only/api_clients/megallm_client.py
Purpose: API-3 (gemini-2.5-flash) client with a three-tier provider chain, all
         calling the SAME model:
           1. LinkAPI gateway   (primary, "geminicheap" pricing group)
           2. OpenRouter        (secondary, google/gemini-2.5-flash — paid)
           3. MegaLLM gateway   (last resort)

All three are OpenAI-compatible endpoints, so each tier uses the openai SDK with
a different base URL/key. LinkAPI leads because MegaLLM ran out of credits for
gemini-2.5-flash during the production run; MegaLLM is kept last so it resumes
serving automatically if its wallet is topped up.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - LinkAPI gateway (OpenAI-compatible): https://docs.linkapi.ai/
  - OpenRouter Gemini 2.5 Flash: https://openrouter.ai/google/gemini-2.5-flash
  - MegaLLM API (OpenAI-compatible): https://docs.megallm.io/

Retry/fallback policy:
  1. LinkAPI, single key, 2 attempts.
  2. OpenRouter round-robin (2 keys) on google/gemini-2.5-flash.
  3. MegaLLM, single key, 2 attempts (last resort).
  4. Flag row on failure.

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
import time
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    LINKAPI_API_BASE_URL,
    LINKAPI_API_KEY,
    MEGALLM_API_BASE_URL,
    MEGALLM_API_KEY,
    OPENROUTER_API_BASE_URL,
    OPENROUTER_KEYS,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 60
_MEGALLM_ATTEMPTS = 2  # single key; initial call + one retry
_LINKAPI_ATTEMPTS = 2  # single key; initial call + one retry

# MegaLLM model_id -> OpenRouter model id (same model on the fallback route).
_OPENROUTER_MODEL_MAP = {
    "gemini-2.5-flash": "google/gemini-2.5-flash",
}


def _call_openai_compatible(
    base_url: str,
    key: str,
    model_id: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.0,
) -> str | None:
    """Single call against an OpenAI-compatible endpoint. Returns text or None."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=base_url, timeout=_TIMEOUT)
        call_kwargs: dict = {
            "model": model_id,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "timeout": _TIMEOUT,
        }
        if temperature > 0.0:
            call_kwargs["temperature"] = temperature
        response = client.chat.completions.create(**call_kwargs)
        choice = response.choices[0]
        # gemini-2.5-flash is a reasoning model: hidden "thinking" tokens count
        # against max_tokens. If the budget is exhausted the visible answer is
        # truncated (finish_reason="length") to a useless fragment. Treat that as
        # a failure (fall through / retry) instead of returning garbage that the
        # judge would then hallucinate an answer from.
        if getattr(choice, "finish_reason", None) == "length":
            logger.warning(
                "Truncated response (finish_reason=length, model=%s); treating as failure.",
                model_id,
            )
            return None
        return choice.message.content or ""
    except Exception as exc:
        logger.warning("MegaLLM/OpenRouter call failed (model=%s): %s", model_id, exc)
        return None


def call_megallm_with_fallback(
    model_id: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Call LinkAPI (primary), then OpenRouter (secondary), then MegaLLM (last)
    — all on the same model.

    Returns
    -------
    dict with keys:
        raw_response, route_used, key_index, attempt_count, success_flag,
        failure_reason, latency_ms
    """
    attempt_count = 0
    t0 = time.monotonic()

    # gemini-2.5-flash spends hidden reasoning tokens per call (commonly 250-450,
    # but adversarial CoT/context prompts can exceed 2000), which count against
    # max_tokens; too small a budget truncates the answer. Reasoning cannot be
    # disabled via these gateways (reasoning_effort / thinking_budget are
    # ignored). Raising the ceiling never makes the model reason more -- it only
    # turns truncations into complete answers (and avoids the slow fallback
    # chain) -- so give a large headroom.
    max_tokens = max(max_tokens, 8192)

    # Primary: LinkAPI gateway (single key, geminicheap group). Leads because
    # MegaLLM ran out of gemini credits mid-run; see module docstring.
    if LINKAPI_API_KEY:
        for _ in range(_LINKAPI_ATTEMPTS):
            attempt_count += 1
            raw = _call_openai_compatible(
                LINKAPI_API_BASE_URL, LINKAPI_API_KEY, model_id, messages, max_tokens,
                temperature=temperature,
            )
            if raw is not None:
                return {
                    "raw_response": raw,
                    "route_used": "linkapi",
                    "key_index": 0,
                    "attempt_count": attempt_count,
                    "success_flag": True,
                    "failure_reason": "",
                    "latency_ms": int((time.monotonic() - t0) * 1000),
                }

    # Secondary: OpenRouter round-robin on the same model
    or_model = _OPENROUTER_MODEL_MAP.get(model_id, model_id)
    _or_idx = 0
    for _ in range(len(OPENROUTER_KEYS)):
        key = OPENROUTER_KEYS[_or_idx % len(OPENROUTER_KEYS)]
        key_index = _or_idx % len(OPENROUTER_KEYS)
        _or_idx += 1
        attempt_count += 1
        raw = _call_openai_compatible(
            OPENROUTER_API_BASE_URL, key, or_model, messages, max_tokens,
            temperature=temperature,
        )
        if raw is not None:
            return {
                "raw_response": raw,
                "route_used": "openrouter",
                "key_index": key_index,
                "attempt_count": attempt_count,
                "success_flag": True,
                "failure_reason": "",
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }

    # Last resort: MegaLLM gateway (single key). Reached only if LinkAPI and
    # OpenRouter both fail; resumes serving automatically if credits are restored.
    if MEGALLM_API_KEY:
        for _ in range(_MEGALLM_ATTEMPTS):
            attempt_count += 1
            raw = _call_openai_compatible(
                MEGALLM_API_BASE_URL, MEGALLM_API_KEY, model_id, messages, max_tokens,
                temperature=temperature,
            )
            if raw is not None:
                return {
                    "raw_response": raw,
                    "route_used": "megallm",
                    "key_index": 0,
                    "attempt_count": attempt_count,
                    "success_flag": True,
                    "failure_reason": "",
                    "latency_ms": int((time.monotonic() - t0) * 1000),
                }

    # All attempts failed
    return {
        "raw_response": "",
        "route_used": "failed",
        "key_index": -1,
        "attempt_count": attempt_count,
        "success_flag": False,
        "failure_reason": "api_error",
        "latency_ms": int((time.monotonic() - t0) * 1000),
    }
