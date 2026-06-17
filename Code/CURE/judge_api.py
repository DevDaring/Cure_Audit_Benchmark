"""
judge_api.py -- remote judge and answer-extraction client for CURE.

One active tier is chosen by config (default gemini). Keys are round-robined WITHIN
the active tier. There is NO automatic fallback between tiers: if the active tier
fails for an item, the item is recorded as a judge failure and logged. This keeps
every judgement in a run from a single model, which protects reproducibility.

Providers (all keys read from the environment, never hardcoded):
  gemini      gemini-2.5-flash via the Generative Language REST API (GEMINI_API_KEY_1..4)
  deepseek    deepseek-chat, OpenAI-compatible (DEEPSEEK_API_KEY_1..2)
  mistral     mistral-small-latest, OpenAI-compatible (MISTRAL_API_KEY1..2)
  openrouter  OpenAI-compatible gateway (OPENROUTER_API_KEY_1..2)

Used for answer extraction (mapping a free-text model response to one option when
the deterministic JSON parse fails) and for any LLM-as-judge step.
"""

import json
import logging
import re
import threading
import time

import requests

import config_cure as C

log = logging.getLogger("cure.judge")

_TIMEOUT = 60
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def _extract_json_object(text: str) -> dict | None:
    """Robust JSON extraction: direct parse, then first balanced {...} block."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


class RoundRobin:
    """Thread-safe round-robin index over a fixed-size list."""

    def __init__(self, n: int):
        self._n = max(1, n)
        self._i = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            j = self._i
            self._i = (self._i + 1) % self._n
            return j


class JudgeClient:
    """Active-tier judge client with within-tier round-robin and no cross-tier fallback."""

    def __init__(self, provider: str | None = None):
        self.provider = (provider or C.ACTIVE_JUDGE).strip().lower()
        if self.provider not in C.JUDGE_PROVIDERS:
            raise ValueError(f"unknown judge provider {self.provider!r}")
        self.cfg = C.JUDGE_PROVIDERS[self.provider]
        self.keys = self.cfg["keys"]
        if not self.keys:
            raise RuntimeError(f"no API keys present for judge tier {self.provider!r}")
        self.rr = RoundRobin(len(self.keys))

    # ---- low-level single call on one key (no fallback) -----------------
    def _call_once(self, key: str, prompt: str, system: str, max_tokens: int) -> str:
        kind = self.cfg["kind"]
        model = self.cfg["model"]
        if kind == "gemini":
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent?key={key}")
            body = {
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_tokens},
            }
            r = requests.post(url, json=body, timeout=_TIMEOUT,
                              headers={"User-Agent": _UA, "Content-Type": "application/json"})
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        # OpenAI-compatible (deepseek, mistral, openrouter)
        url = self.cfg["base_url"].rstrip("/") + "/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, json=body, timeout=_TIMEOUT,
                          headers={"Authorization": f"Bearer {key}",
                                   "User-Agent": _UA, "Content-Type": "application/json"})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def call(self, prompt: str, system: str = "You are a careful evaluation assistant.",
             max_tokens: int = 256, within_tier_retries: int = 1) -> dict:
        """One judgement. Round-robin within the active tier only. Never crosses tiers."""
        attempts = min(len(self.keys), 1 + max(0, within_tier_retries))
        last_err = None
        for _ in range(attempts):
            ki = self.rr.next()
            try:
                text = self._call_once(self.keys[ki], prompt, system, max_tokens)
                return {"text": text, "status": "ok", "provider": self.provider, "key_index": ki}
            except Exception as exc:  # within-tier rotate only
                last_err = str(exc)[:200]
                time.sleep(0.5)
        return {"text": "", "status": "failed", "provider": self.provider,
                "key_index": -1, "error": last_err}

    # ---- answer extraction ---------------------------------------------
    def extract_answer(self, raw_response: str, options: list[str], gold: str = "") -> dict:
        """Map a model response to one option. Deterministic parse first, judge last.

        Returns {answer, parse_method, judge_status}.
        """
        obj = _extract_json_object(raw_response)
        if obj and isinstance(obj.get("answer"), str) and obj["answer"].strip():
            return {"answer": obj["answer"].strip(), "parse_method": "json", "judge_status": "n/a"}
        # deterministic option-substring repair
        low = (raw_response or "").lower()
        for opt in options:
            if opt and opt.lower() in low:
                return {"answer": opt, "parse_method": "repair", "judge_status": "n/a"}
        # remote judge as the last resort
        sys_p = ("You map a model answer to exactly one of the provided options. "
                 'Return strict JSON: {"answer": "<exact option text>"}.')
        opts = "\n".join(f"- {o}" for o in options)
        prompt = f"Options:\n{opts}\n\nModel answer:\n{raw_response}\n\nReturn the JSON now."
        res = self.call(prompt, system=sys_p, max_tokens=128)
        if res["status"] != "ok":
            return {"answer": "", "parse_method": "judge", "judge_status": "failed"}
        obj = _extract_json_object(res["text"])
        ans = obj.get("answer", "").strip() if obj else ""
        return {"answer": ans, "parse_method": "judge", "judge_status": "ok" if ans else "failed"}


def test_all_keys() -> dict:
    """Dry-run check: test every provider, every key, with its model id.

    Returns a report dict {provider: [{key_index, status, latency_ms, error}]}.
    """
    report = {}
    probe = 'Reply with strict JSON only: {"ok": true}'
    for name, cfg in C.JUDGE_PROVIDERS.items():
        rows = []
        for ki, key in enumerate(cfg["keys"]):
            t0 = time.time()
            try:
                client = JudgeClient.__new__(JudgeClient)
                client.provider, client.cfg, client.keys = name, cfg, [key]
                client.rr = RoundRobin(1)
                text = client._call_once(key, probe, "Return only JSON.", 32)
                ok = _extract_json_object(text) is not None
                rows.append({"key_index": ki, "status": "ok" if ok else "bad_response",
                             "latency_ms": int((time.time() - t0) * 1000), "model": cfg["model"]})
            except Exception as exc:
                rows.append({"key_index": ki, "status": "failed",
                             "latency_ms": int((time.time() - t0) * 1000),
                             "model": cfg["model"], "error": str(exc)[:200]})
        report[name] = rows
    return report
