"""
File: CPU_Only/api_clients/gemini_client.py
Purpose: Gemini API client with 4-key round-robin for API-3.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - Google Generative AI Python SDK: https://github.com/google/generative-ai-python

Part of the audit codebase (diagnosis half of CURE).
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import GEMINI_KEYS, GEMINI_MODEL_NAME, RESEARCH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_TIMEOUT = 60
_MAX_ATTEMPTS_PER_KEY = 2


class _GeminiRoundRobin:
    def __init__(self) -> None:
        self._idx = 0

    def next(self) -> tuple[str, int]:
        key = GEMINI_KEYS[self._idx % len(GEMINI_KEYS)]
        idx = self._idx % len(GEMINI_KEYS)
        self._idx += 1
        return key, idx


_rr = _GeminiRoundRobin()


def _call_gemini(
    key: str,
    model_name: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.0,
) -> str | None:
    """
    Single Gemini API call with safety filters fully disabled.
    Bias-audit prompts contain stereotyped language by design; BLOCK_NONE
    prevents false refusals from corrupting behavioral signal.
    Returns text or None.
    """
    try:
        import google.generativeai as genai  # type: ignore
        from google.generativeai.types import HarmCategory, HarmBlockThreshold  # type: ignore

        genai.configure(api_key=key)

        # Use system message if present, otherwise fall back to research prompt
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        system_instruction = system_parts[0] if system_parts else RESEARCH_SYSTEM_PROMPT

        user_text = "\n".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )

        # Disable all safety filters — required for bias-audit benchmark prompts
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        gen_cfg_kwargs: dict = {
            "response_mime_type": "application/json",
            "max_output_tokens": max_tokens,
        }
        if temperature > 0.0:
            gen_cfg_kwargs["temperature"] = temperature

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
            generation_config=genai.GenerationConfig(**gen_cfg_kwargs),
        )
        response = model.generate_content(user_text, safety_settings=safety_settings)
        return response.text
    except Exception as exc:
        logger.warning("Gemini call failed (key_idx=?): %s", exc)
        return None


def call_gemini_with_roundrobin(
    messages: list[dict],
    max_tokens: int = 256,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Call Gemini with round-robin key rotation.
    Max 2 attempts per key; if all exhausted, skip and flag.

    Returns
    -------
    dict with keys: raw_response, route_used, key_index, attempt_count,
                    success_flag, failure_reason, latency_ms
    """
    if model_name is None:
        model_name = GEMINI_MODEL_NAME

    attempt_count = 0
    t0 = time.monotonic()

    for _ in range(len(GEMINI_KEYS) * _MAX_ATTEMPTS_PER_KEY):
        key, key_index = _rr.next()
        attempt_count += 1
        raw = _call_gemini(key, model_name, messages, max_tokens, temperature=temperature)
        if raw is not None:
            return {
                "raw_response": raw,
                "route_used": "gcp",
                "key_index": key_index,
                "attempt_count": attempt_count,
                "success_flag": True,
                "failure_reason": "",
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }

    return {
        "raw_response": "",
        "route_used": "gcp",
        "key_index": -1,
        "attempt_count": attempt_count,
        "success_flag": False,
        "failure_reason": "api_error",
        "latency_ms": int((time.monotonic() - t0) * 1000),
    }
