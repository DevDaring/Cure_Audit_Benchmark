"""
File: CPU_Only/api_clients/bedrock_client.py
Purpose: AWS Bedrock client with OpenRouter fallback for API-1
         (qwen.qwen3-next-80b-a3b) and API-2 (amazon.nova-2-lite-v1:0).

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - AWS Bedrock documentation: https://docs.aws.amazon.com/bedrock/

Retry/fallback policy (per spec Section 4.2):
  1. Bedrock account 1, retry once (2 attempts).
  2. OpenRouter round-robin (2 keys).
  3. Flag row on failure.

The credential-tier mechanism below also supports leading with a higher-rpm
secondary AWS account for selected models (_MULTI_ACCOUNT_MODELS). No model
uses it at present; it degrades to account 1 only.

Part of the audit codebase (diagnosis half of CURE).
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    AWS_ACCESS_KEY,
    AWS_ACCESS_KEY2,
    AWS_SECRET_KEY,
    AWS_SECRET_KEY2,
    OPENROUTER_API_BASE_URL,
    OPENROUTER_KEYS,
    RESEARCH_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 60
_BEDROCK_REGION = "us-east-1"
_BEDROCK_ATTEMPTS_PER_TIER = 2  # initial call + one retry, per credential tier

_OPENROUTER_MODEL_MAP = {
    "qwen.qwen3-next-80b-a3b": "qwen/qwen3-next-80b-a3b-instruct",
    "us.amazon.nova-2-lite-v1:0": "amazon/nova-lite-v1",
}

# Models whose Bedrock response format is incompatible with the Converse text
# extraction and must go straight to OpenRouter. Currently none — every active
# Bedrock model (qwen3-next, nova) returns text via Converse.
_BEDROCK_SKIP: set[str] = set()

# Models that lead with a higher-rpm secondary AWS account (AWS_*_KEY2) before
# falling back to account 1. Empty at present (the former Llama-3.3-70B entry was
# removed when API-3 moved to DeepSeek). Add a model_id here to re-enable.
_MULTI_ACCOUNT_MODELS: set[str] = set()


def _bedrock_credential_tiers(model_id: str) -> list[tuple[str, str, str]]:
    """Ordered (access_key, secret_key, route_label) Bedrock credential tiers.

    Models in _MULTI_ACCOUNT_MODELS lead with account 2 (higher rpm) then fall
    back to account 1; every other Bedrock model uses account 1 only. Tiers whose
    credentials are empty are dropped, so a missing account 2 never blocks the run.
    """
    if model_id in _MULTI_ACCOUNT_MODELS:
        tiers = [
            (AWS_ACCESS_KEY2, AWS_SECRET_KEY2, "bedrock_acct2"),
            (AWS_ACCESS_KEY, AWS_SECRET_KEY, "bedrock_acct1"),
        ]
    else:
        tiers = [(AWS_ACCESS_KEY, AWS_SECRET_KEY, "bedrock")]
    return [(a, s, label) for (a, s, label) in tiers if a and s]


def _call_bedrock(
    model_id: str,
    messages: list[dict],
    max_tokens: int,
    access_key: str,
    secret_key: str,
    temperature: float = 0.0,
) -> str | None:
    """
    Call AWS Bedrock via the Converse API (model-agnostic) using the supplied
    account credentials.
    Guardrails are intentionally not attached — bias-audit benchmarks contain
    stereotyped language by design and must not be blocked.
    Returns text or None on failure.
    """
    try:
        import boto3  # type: ignore

        client = boto3.client(
            "bedrock-runtime",
            region_name=_BEDROCK_REGION,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        # Separate system message from conversation turns
        system_parts = [m for m in messages if m.get("role") == "system"]
        system_text = system_parts[0]["content"] if system_parts else RESEARCH_SYSTEM_PROMPT
        converse_msgs = [
            {"role": m["role"], "content": [{"text": m.get("content", "")}]}
            for m in messages if m.get("role") != "system"
        ]

        inference_cfg: dict = {"maxTokens": max_tokens}
        if temperature > 0.0:
            inference_cfg["temperature"] = temperature

        response = client.converse(
            modelId=model_id,
            system=[{"text": system_text}],
            messages=converse_msgs,
            inferenceConfig=inference_cfg,
            # No guardrailConfig — guardrails are opt-in; omitting means no filtering
        )
        blocks = response["output"]["message"]["content"]
        for block in blocks:
            if isinstance(block, dict) and "text" in block:
                return block["text"]
        return None
    except Exception as exc:
        logger.warning("Bedrock call failed (model=%s): %s", model_id, exc)
        return None


def _call_openrouter(
    model_id: str,
    messages: list[dict],
    key: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> str | None:
    """Call OpenRouter as fallback. Returns text or None."""
    try:
        from openai import OpenAI

        or_model = _OPENROUTER_MODEL_MAP.get(model_id, model_id)
        client = OpenAI(api_key=key, base_url=OPENROUTER_API_BASE_URL, timeout=_TIMEOUT)
        response = client.chat.completions.create(
            model=or_model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=_TIMEOUT,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("OpenRouter fallback failed (model=%s): %s", model_id, exc)
        return None


def call_bedrock_with_fallback(
    model_id: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Call Bedrock with multi-account + OpenRouter fallback per spec Section 4.2.

    Bedrock models (qwen3-next, nova): account 1 (retry once) -> OpenRouter
    round-robin. Models in _MULTI_ACCOUNT_MODELS lead with account 2 first.

    Returns
    -------
    dict with keys:
        raw_response, route_used, key_index, attempt_count, success_flag, failure_reason
    """
    attempt_count = 0

    # Bedrock credential tiers (skipped only for models listed in _BEDROCK_SKIP).
    if model_id not in _BEDROCK_SKIP:
        for access_key, secret_key, route_label in _bedrock_credential_tiers(model_id):
            for _ in range(_BEDROCK_ATTEMPTS_PER_TIER):
                attempt_count += 1
                t0 = time.monotonic()
                raw = _call_bedrock(
                    model_id, messages, max_tokens, access_key, secret_key,
                    temperature=temperature,
                )
                if raw is not None:
                    return {
                        "raw_response": raw,
                        "route_used": route_label,
                        "key_index": 0,
                        "attempt_count": attempt_count,
                        "success_flag": True,
                        "failure_reason": "",
                        "latency_ms": int((time.monotonic() - t0) * 1000),
                    }

    # Final tier: OpenRouter round-robin (2 keys)
    _or_idx = 0
    for _ in range(len(OPENROUTER_KEYS)):
        attempt_count += 1
        key = OPENROUTER_KEYS[_or_idx % len(OPENROUTER_KEYS)]
        key_index = _or_idx % len(OPENROUTER_KEYS)
        _or_idx += 1
        t0 = time.monotonic()
        raw = _call_openrouter(model_id, messages, key, max_tokens, temperature=temperature)
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

    # All tiers exhausted
    return {
        "raw_response": "",
        "route_used": "failed",
        "key_index": -1,
        "attempt_count": attempt_count,
        "success_flag": False,
        "failure_reason": "api_error",
        "latency_ms": 0,
    }
