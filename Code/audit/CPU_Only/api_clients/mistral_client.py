"""
File: CPU_Only/api_clients/mistral_client.py
Purpose: Mistral platform client with 2-key round-robin for API-4, plus an
         OpenRouter secondary fallback to the SAME model
         (mistralai/mistral-medium-3-5, a paid model — no :free variant).

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - Mistral AI Python SDK: https://github.com/mistralai/client-python
  - OpenRouter Mistral Medium 3.5: https://openrouter.ai/mistralai/mistral-medium-3-5

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
import time
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    MISTRAL_KEYS,
    MISTRAL_MODEL_NAME,
    OPENROUTER_API_BASE_URL,
    OPENROUTER_KEYS,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 60
_MAX_ATTEMPTS_PER_KEY = 2

# Mistral platform model_id -> OpenRouter model id (same model, paid tier).
_OPENROUTER_MODEL_MAP = {
    "mistral-medium-latest": "mistralai/mistral-medium-3-5",
}


class _MistralRoundRobin:
    def __init__(self) -> None:
        self._idx = 0

    def next(self) -> tuple[str, int]:
        key = MISTRAL_KEYS[self._idx % len(MISTRAL_KEYS)]
        idx = self._idx % len(MISTRAL_KEYS)
        self._idx += 1
        return key, idx


_rr = _MistralRoundRobin()


def _call_mistral(
    key: str,
    model_name: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.0,
) -> str | None:
    """Single Mistral API call. Returns text or None."""
    try:
        from mistralai.client import Mistral  # type: ignore

        client = Mistral(api_key=key)
        call_kwargs: dict = {
            "model": model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "safe_prompt": False,  # do not inject Mistral's safety preamble
        }
        if temperature > 0.0:
            call_kwargs["temperature"] = temperature
        response = client.chat.complete(**call_kwargs)
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("Mistral call failed: %s", exc)
        return None


def call_mistral_with_roundrobin(
    messages: list[dict],
    max_tokens: int = 256,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Call Mistral with round-robin key rotation.
    Skips and flags if all keys exhausted.
    """
    if model_name is None:
        model_name = MISTRAL_MODEL_NAME

    attempt_count = 0
    t0 = time.monotonic()

    for _ in range(len(MISTRAL_KEYS) * _MAX_ATTEMPTS_PER_KEY):
        key, key_index = _rr.next()
        attempt_count += 1
        raw = _call_mistral(key, model_name, messages, max_tokens, temperature=temperature)
        if raw is not None:
            return {
                "raw_response": raw,
                "route_used": "mistral",
                "key_index": key_index,
                "attempt_count": attempt_count,
                "success_flag": True,
                "failure_reason": "",
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }

    return {
        "raw_response": "",
        "route_used": "mistral",
        "key_index": -1,
        "attempt_count": attempt_count,
        "success_flag": False,
        "failure_reason": "api_error",
        "latency_ms": int((time.monotonic() - t0) * 1000),
    }


def _call_openrouter_mistral(
    key: str,
    model_id: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.0,
) -> str | None:
    """Single OpenRouter call for the Mistral fallback. Returns text or None."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=OPENROUTER_API_BASE_URL, timeout=_TIMEOUT)
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
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("OpenRouter (mistral) fallback failed (model=%s): %s", model_id, exc)
        return None


def call_mistral_with_fallback(
    messages: list[dict],
    max_tokens: int = 256,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Mistral platform round-robin, then OpenRouter fallback to the SAME model.

    Used by the API-4 evaluation path. The judge path keeps using
    call_mistral_with_roundrobin directly (no fallback needed).
    """
    result = call_mistral_with_roundrobin(
        messages, max_tokens=max_tokens, model_name=model_name, temperature=temperature
    )
    if result["success_flag"]:
        return result

    or_model = _OPENROUTER_MODEL_MAP.get(model_name or MISTRAL_MODEL_NAME)
    if or_model is None:
        # No OpenRouter mapping for this model — return the Mistral failure as-is.
        return result

    attempt_count = result.get("attempt_count", 0)
    t0 = time.monotonic()
    _idx = 0
    for _ in range(len(OPENROUTER_KEYS)):
        key = OPENROUTER_KEYS[_idx % len(OPENROUTER_KEYS)]
        key_index = _idx % len(OPENROUTER_KEYS)
        _idx += 1
        attempt_count += 1
        raw = _call_openrouter_mistral(key, or_model, messages, max_tokens, temperature=temperature)
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

    return {
        "raw_response": "",
        "route_used": "failed",
        "key_index": -1,
        "attempt_count": attempt_count,
        "success_flag": False,
        "failure_reason": "api_error",
        "latency_ms": int((time.monotonic() - t0) * 1000),
    }
