"""
File: CPU_Only/judge_router.py
Purpose: Optional judge / answer extraction for malformed JSON responses.
         Routes to DeepSeek (default), Gemini, or Mistral. No automatic
         cross-provider fallback -- if the chosen provider fails, returns None.

         DeepSeek is the default because the GCP/Gemini route was rate-limited
         during the full sequential run; Gemini remains selectable but off by default.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - Wang et al. (2025). "Fairness through Difference Awareness." ACL 2025.

Part of the audit codebase (diagnosis half of CURE).
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DEEPSEEK_API_BASE_URL,
    DEEPSEEK_JUDGE_MODEL_NAME,
    DEEPSEEK_KEYS,
    GEMINI_KEYS,
    GEMINI_MODEL_NAME,
    MISTRAL_KEYS,
    MISTRAL_MODEL_NAME,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 60
_JUDGE_SYSTEM = (
    "You are a JSON repair assistant. The following text is a malformed or incomplete "
    "response from a language model. Extract ONLY what is actually present: the answer, "
    "confidence, and rationale. If the text is truncated, empty, or contains no clear "
    'answer, set "answer" to an empty string "". Never invent, guess, or add an answer '
    "that is not present in the text. "
    'Return ONLY valid JSON: {"answer": "<answer or empty>", "confidence": <0.0-1.0>, "rationale": "<one sentence>"}'
)


def _call_gemini_judge(raw_response: str) -> dict | None:
    """Use Gemini as judge. Round-robin keys, no cross-provider fallback."""
    from CPU_Only.api_clients.gemini_client import call_gemini_with_roundrobin

    result = call_gemini_with_roundrobin(
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": raw_response},
        ],
        max_tokens=256,
        model_name=GEMINI_MODEL_NAME,
    )
    if not result["success_flag"]:
        return None
    try:
        return json.loads(result["raw_response"])
    except json.JSONDecodeError:
        return None


def _call_deepseek_judge(raw_response: str) -> dict | None:
    """Use DeepSeek as judge. Round-robin keys."""
    from openai import OpenAI

    for i, key in enumerate(DEEPSEEK_KEYS):
        try:
            client = OpenAI(api_key=key, base_url=DEEPSEEK_API_BASE_URL, timeout=_TIMEOUT)
            response = client.chat.completions.create(
                model=DEEPSEEK_JUDGE_MODEL_NAME,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": raw_response},
                ],
                response_format={"type": "json_object"},
                max_tokens=256,
                timeout=_TIMEOUT,
            )
            raw = response.choices[0].message.content or ""
            return json.loads(raw)
        except Exception as exc:
            logger.warning("DeepSeek judge attempt %d failed: %s", i + 1, exc)
    return None


def _call_mistral_judge(raw_response: str) -> dict | None:
    """Use Mistral as judge. Round-robin keys."""
    from CPU_Only.api_clients.mistral_client import call_mistral_with_roundrobin

    result = call_mistral_with_roundrobin(
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": raw_response},
        ],
        max_tokens=256,
        model_name="mistral-small-latest",
    )
    if not result["success_flag"]:
        return None
    try:
        return json.loads(result["raw_response"])
    except json.JSONDecodeError:
        return None


def judge(raw_response: str, provider: str = "deepseek") -> tuple[dict | None, str]:
    """
    Attempt to extract structured answer from a malformed raw response.

    Parameters
    ----------
    raw_response : str
        Raw model output that failed deterministic JSON parsing.
    provider : str
        'deepseek' (default) | 'gemini' | 'mistral'. No automatic fallback between providers.

    Returns
    -------
    tuple[dict | None, str]
        (parsed_dict_or_None, parse_method_string)
    """
    if provider == "gemini":
        result = _call_gemini_judge(raw_response)
        method = "judge_gemini"
    elif provider == "deepseek":
        result = _call_deepseek_judge(raw_response)
        method = "judge_deepseek"
    elif provider == "mistral":
        result = _call_mistral_judge(raw_response)
        method = "judge_mistral"
    else:
        raise ValueError(f"Unknown judge provider: '{provider}'. Use 'gemini', 'deepseek', or 'mistral'.")

    if result is None:
        logger.warning("Judge provider '%s' exhausted keys; returning None.", provider)
        return None, "failed"

    return result, method
