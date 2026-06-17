"""
File: Dry_Run/dry_run_apis.py
Purpose: Test all external API keys loaded from .env -- no hardcoded keys.
         Checks DeepSeek (2 keys), OpenRouter (2 keys), Gemini (4 keys),
         Mistral (2 keys). Makes one minimal chat completion call per key.
         All keys read from config.py which reads .env.

Part of the audit codebase (diagnosis half of CURE).
"""

import json
import logging
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DEEPSEEK_API_BASE_URL,
    DEEPSEEK_KEYS,
    DEEPSEEK_PRIMARY_MODEL_NAME,
    GEMINI_KEYS,
    GEMINI_MODEL_NAME,
    MISTRAL_KEYS,
    MISTRAL_MODEL_NAME,
    OPENROUTER_API_BASE_URL,
    OPENROUTER_KEYS,
)
from logger_setup import setup_logging

logger = logging.getLogger(__name__)

_PROBE_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Reply with a single JSON object and nothing else: "
            '{"answer": "ok", "status": "working"}'
        ),
    }
]
_TIMEOUT = 30  # seconds per call

_RESULTS: dict[str, str] = {}


def _mark(key: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    _RESULTS[key] = f"{status}  {detail}".strip()
    (logger.info if passed else logger.error)(
        "  [%s] %-40s %s", status, key, detail
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoint helper
# ---------------------------------------------------------------------------

def _call_openai_compat(
    base_url: str,
    api_key: str,
    model: str,
    extra_headers: dict | None = None,
) -> tuple[bool, str]:
    """
    POST to an OpenAI-compatible /chat/completions endpoint.
    Returns (success, detail_string).
    """
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    payload = {
        "model": model,
        "messages": _PROBE_MESSAGES,
        "max_tokens": 32,
        "temperature": 0.0,
    }
    try:
        t0 = time.monotonic()
        resp = requests.post(url, headers=headers, json=payload, timeout=_TIMEOUT)
        latency_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return True, f"HTTP 200 | {latency_ms}ms | reply={content[:60]!r}"
        return False, f"HTTP {resp.status_code} | {resp.text[:200]}"
    except requests.exceptions.Timeout:
        return False, f"Timeout after {_TIMEOUT}s"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Gemini REST helper
# ---------------------------------------------------------------------------

def _call_gemini(api_key: str, model_name: str) -> tuple[bool, str]:
    """
    POST to the Gemini generateContent REST endpoint.
    Returns (success, detail_string).
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": _PROBE_MESSAGES[0]["content"]}],
            }
        ],
        "generationConfig": {"maxOutputTokens": 32, "temperature": 0.0},
    }
    headers = {"Content-Type": "application/json"}
    try:
        t0 = time.monotonic()
        resp = requests.post(url, headers=headers, json=payload, timeout=_TIMEOUT)
        latency_ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code == 200:
            data = resp.json()
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            return True, f"HTTP 200 | {latency_ms}ms | reply={text[:60]!r}"
        return False, f"HTTP {resp.status_code} | {resp.text[:200]}"
    except requests.exceptions.Timeout:
        return False, f"Timeout after {_TIMEOUT}s"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Individual API tests
# ---------------------------------------------------------------------------

def _test_deepseek() -> bool:
    all_pass = True
    for idx, key in enumerate(DEEPSEEK_KEYS, start=1):
        ok, detail = _call_openai_compat(
            DEEPSEEK_API_BASE_URL, key, DEEPSEEK_PRIMARY_MODEL_NAME
        )
        _mark(f"DEEPSEEK_KEY_{idx}", ok, detail)
        if not ok:
            all_pass = False
    return all_pass


def _test_openrouter() -> bool:
    all_pass = True
    # Test with a small publicly-accessible model on OpenRouter
    model = "openai/gpt-4o-mini"
    extra = {"HTTP-Referer": "https://github.com/MIRAGE-audit", "X-Title": "MIRAGE-DryRun"}
    for idx, key in enumerate(OPENROUTER_KEYS, start=1):
        ok, detail = _call_openai_compat(
            OPENROUTER_API_BASE_URL, key, model, extra_headers=extra
        )
        _mark(f"OPENROUTER_KEY_{idx}", ok, detail)
        if not ok:
            all_pass = False
    return all_pass


def _test_gemini() -> bool:
    all_pass = True
    for idx, key in enumerate(GEMINI_KEYS, start=1):
        ok, detail = _call_gemini(key, GEMINI_MODEL_NAME)
        _mark(f"GEMINI_KEY_{idx}", ok, detail)
        if not ok:
            all_pass = False
    return all_pass


def _test_mistral() -> bool:
    all_pass = True
    for idx, key in enumerate(MISTRAL_KEYS, start=1):
        ok, detail = _call_openai_compat(
            "https://api.mistral.ai/v1", key, MISTRAL_MODEL_NAME
        )
        _mark(f"MISTRAL_KEY_{idx}", ok, detail)
        if not ok:
            all_pass = False
    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> bool:
    run_id = setup_logging()
    logger.info("=== API Keys Dry Run (run_id=%s) ===", run_id)
    logger.info("Testing all external API keys. Keys loaded from .env via config.py.")
    logger.info("")

    suites = [
        ("DeepSeek", _test_deepseek),
        ("OpenRouter", _test_openrouter),
        ("Gemini", _test_gemini),
        ("Mistral", _test_mistral),
    ]

    all_pass = True
    for label, fn in suites:
        logger.info("--- %s ---", label)
        try:
            result = fn()
        except Exception as exc:
            logger.error("Suite %s crashed: %s", label, exc)
            result = False
        all_pass = all_pass and result
        logger.info("")

    logger.info("=== API Dry Run Summary ===")
    max_key_len = max(len(k) for k in _RESULTS) if _RESULTS else 0
    for key, status in _RESULTS.items():
        logger.info("  %-*s  %s", max_key_len, key, status)
    logger.info("")

    total = len(_RESULTS)
    passed = sum(1 for v in _RESULTS.values() if v.startswith("PASS"))
    logger.info("%d / %d API keys PASSED.", passed, total)

    return all_pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run()
    sys.exit(0 if success else 1)
