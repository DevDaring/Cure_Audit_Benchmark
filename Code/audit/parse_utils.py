"""
Shared JSON response parsing for OSM and API behavioral evaluation.

Design (per review): responses are parsed DIRECTLY in Python first; the judge
API (CPU_Only/judge_router.py) is only a last resort when every local method
below fails. Local cascade, cheapest/strictest first:

    1. json.loads on the first balanced {...} object (strips trailing prose)
    2. deterministic repair (smart quotes, trailing commas, unterminated object)
    3. json_repair library (handles single quotes, True/False/None, missing
       commas/quotes, truncation) — optional dependency, skipped if absent
    4. quoted-lines heuristic (Qwen-style brace-less output)
    5. regex field extraction (answer/confidence/rationale)

Best-practice reference: try json.loads first, fall back to json_repair
(mangiucugna/json_repair, PyPI: json-repair).

Part of the audit codebase (diagnosis half of CURE).
"""

from __future__ import annotations

import json
import re

_ANSWER_RE = re.compile(
    r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)
_ANSWER_RE_SQ = re.compile(
    r"'answer'\s*:\s*'((?:[^'\\]|\\.)*)'",
    re.DOTALL,
)
# Key may be single- or double-quoted; value is a bare number.
_CONFIDENCE_RE = re.compile(r"""["']confidence["']\s*:\s*([0-9]+(?:\.[0-9]+)?)""")
_RATIONALE_RE = re.compile(
    r'"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)
_RATIONALE_RE_SQ = re.compile(
    r"'rationale'\s*:\s*'((?:[^'\\]|\\.)*)'",
    re.DOTALL,
)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")

# Curly/smart quotes -> straight quotes (models occasionally emit these).
_SMART_QUOTES = {
    ord("“"): '"', ord("”"): '"',
    ord("‘"): "'", ord("’"): "'",
    ord("〝"): '"', ord("〞"): '"',
}

# Resolve the optional json_repair dependency once.
try:  # pragma: no cover - import guard
    from json_repair import repair_json as _json_repair_fn  # type: ignore
except Exception:  # pragma: no cover
    _json_repair_fn = None


def _json_repair_loads(text: str) -> dict | None:
    """Use the json_repair library if installed; return a dict or None."""
    if _json_repair_fn is None or not text:
        return None
    try:
        result = _json_repair_fn(text, return_objects=True)
    except Exception:
        return None
    if isinstance(result, dict):
        return result
    # json_repair may return a list when several objects are present.
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                return item
    return None


def _strip_code_fence(raw: str) -> str:
    match = _FENCE_RE.search(raw)
    return match.group(1).strip() if match else raw.strip()


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, ignoring braces inside
    strings. Strips any leading or trailing prose around the object. If the
    object is never closed, returns from the first ``{`` to the end so the
    caller can attempt to repair (e.g. append the missing brace)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text[start:]


def repair_json(raw: str) -> dict | None:
    """Deterministic, local repair of near-valid JSON (no external API).

    Handles: code fences, smart quotes, leading/trailing prose around the
    object, unterminated objects, and trailing commas. Falls back to the
    json_repair library for harder cases (single quotes, Python literals,
    missing commas/quotes). Returns a dict or None — never raises.
    """
    if not raw:
        return None
    s = _strip_code_fence(str(raw)).translate(_SMART_QUOTES).strip()

    obj = _extract_first_json_object(s)
    if obj is not None:
        s = obj

    # Close an unterminated object: append as many '}' as are missing.
    open_braces, close_braces = s.count("{"), s.count("}")
    if s.startswith("{") and open_braces > close_braces:
        s = s + "}" * (open_braces - close_braces)

    for candidate in (s, _TRAILING_COMMA_RE.sub(r"\1", s)):
        try:
            val = json.loads(candidate)
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            continue

    return _json_repair_loads(s)


def _extract_fields_regex(raw: str) -> dict | None:
    """Last-resort field extraction when json parsing fails. Handles both
    double- and single-quoted "answer"/"rationale" values."""
    answer_m = _ANSWER_RE.search(raw) or _ANSWER_RE_SQ.search(raw)
    if not answer_m:
        return None
    conf_m = _CONFIDENCE_RE.search(raw)
    rat_m = _RATIONALE_RE.search(raw) or _RATIONALE_RE_SQ.search(raw)
    return {
        "answer": answer_m.group(1).replace('\\"', '"').replace("\\'", "'"),
        "confidence": float(conf_m.group(1)) if conf_m else 0.0,
        "rationale": rat_m.group(1).replace('\\"', '"').replace("\\'", "'") if rat_m else "",
    }


def _parse_quoted_lines(raw: str) -> dict | None:
    """
    Parse Qwen-style multi-line outputs that omit JSON braces, e.g.:

        "C" Can't answer
        "0.7"
        "Insufficient information ..."
    """
    text = raw.strip()
    # Guard: this heuristic is only for genuine brace-less quoted output whose
    # first line begins with a double-quoted token. Refuse JSON-ish blobs and
    # arbitrary prose so we never return a whole line as a bogus answer.
    if not text or text[0] in "{[" or not text.startswith('"'):
        return None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    def _unquote(value: str) -> str:
        value = value.strip()
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            return value[1:-1]
        return value

    if len(lines) == 1:
        only = _unquote(lines[0])
        if only:
            return {"answer": only, "confidence": 0.0, "rationale": ""}
        return None

    if len(lines) < 2:
        return None

    answer_parts: list[str] = []
    first = lines[0]
    match = re.match(r'^"([^"]*)"(.*)$', first)
    if match:
        if match.group(1):
            answer_parts.append(match.group(1))
        trailing = match.group(2).strip().strip('"')
        if trailing:
            answer_parts.append(trailing)
    else:
        answer_parts.append(_unquote(first))

    answer = " ".join(answer_parts).strip()
    if not answer:
        return None

    confidence = 0.0
    try:
        confidence = float(_unquote(lines[1]))
    except (TypeError, ValueError):
        pass

    rationale = _unquote(lines[2]) if len(lines) >= 3 else ""
    return {"answer": answer, "confidence": confidence, "rationale": rationale}


def parse_model_response(raw_response: str) -> tuple[bool, str, float, str, str, str]:
    """
    Parse a raw model response string into result fields, using local methods
    only (no judge API). Run the judge API separately if this returns
    success_flag=False.

    Returns
    -------
    (success_flag, parsed_answer, parsed_confidence, parsed_rationale,
     parse_method, failure_reason)
    """
    if not raw_response or not str(raw_response).strip():
        return False, "", 0.0, "", "failed", "empty_response"

    raw = _strip_code_fence(str(raw_response))
    candidate = _extract_first_json_object(raw)

    parsed: dict | None = None
    parse_method = "json"

    # 1. Strict json.loads on the balanced object.
    if candidate:
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                parsed = loaded
        except json.JSONDecodeError:
            parsed = None

    # 2 + 3. Deterministic repair, then json_repair library.
    if parsed is None:
        parsed = repair_json(raw)
        if parsed is not None:
            parse_method = "json_repaired"

    # 4. Brace-less quoted-line heuristic.
    if parsed is None:
        parsed = _parse_quoted_lines(raw)
        if parsed is not None:
            parse_method = "quoted_lines"

    # 5. Regex field extraction.
    if parsed is None:
        parsed = _extract_fields_regex(raw)
        if parsed is not None:
            parse_method = "regex"

    if not isinstance(parsed, dict):
        return False, "", 0.0, "", "failed", "parse_error"

    answer = str(parsed.get("answer", "")).strip()
    if not answer:
        return False, "", 0.0, "", "failed", "parse_error"

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return (
        True,
        answer,
        confidence,
        str(parsed.get("rationale", "")),
        parse_method,
        "",
    )
