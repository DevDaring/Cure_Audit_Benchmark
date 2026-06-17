"""
File: CPU_Only/api_clients/openrouter_client.py
Purpose: OpenRouter client with 2-key round-robin (used as fallback for
         Bedrock models, and optionally for standalone evaluation).

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - OpenRouter API: https://openrouter.ai/docs

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
import time
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import OPENROUTER_API_BASE_URL, OPENROUTER_KEYS

logger = logging.getLogger(__name__)

_TIMEOUT = 60
_MAX_ATTEMPTS_PER_KEY = 2


class _OpenRouterRoundRobin:
    def __init__(self) -> None:
        self._idx = 0

    def next(self) -> tuple[str, int]:
        key = OPENROUTER_KEYS[self._idx % len(OPENROUTER_KEYS)]
        idx = self._idx % len(OPENROUTER_KEYS)
        self._idx += 1
        return key, idx


_rr = _OpenRouterRoundRobin()


def _call_openrouter(key: str, model_id: str, messages: list[dict], max_tokens: int) -> str | None:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=OPENROUTER_API_BASE_URL, timeout=_TIMEOUT)
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            timeout=_TIMEOUT,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("OpenRouter call failed (model=%s): %s", model_id, exc)
        return None


def call_openrouter_with_roundrobin(
    model_id: str,
    messages: list[dict],
    max_tokens: int = 256,
) -> dict[str, Any]:
    """
    Call OpenRouter with round-robin key rotation.
    """
    attempt_count = 0
    t0 = time.monotonic()

    for _ in range(len(OPENROUTER_KEYS) * _MAX_ATTEMPTS_PER_KEY):
        key, key_index = _rr.next()
        attempt_count += 1
        raw = _call_openrouter(key, model_id, messages, max_tokens)
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
        "route_used": "openrouter",
        "key_index": -1,
        "attempt_count": attempt_count,
        "success_flag": False,
        "failure_reason": "api_error",
        "latency_ms": int((time.monotonic() - t0) * 1000),
    }
