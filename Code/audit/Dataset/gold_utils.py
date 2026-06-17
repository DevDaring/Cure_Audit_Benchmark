"""
Shared gold-answer scorability rules for pentad generation and MIRAGE-B scoring.

BBQ uses labels like "Unknown" and "Can't be determined" as legitimate gold
answers for ambiguous items — these must not be treated as missing gold.
"""

from typing import Any

# Placeholder values that indicate a true construction failure (not BBQ labels).
_INVALID_GOLD = frozenset({"", "nan", "none"})


def is_scorable_gold(gold: Any, source: str = "") -> bool:
    """Return True if gold_answer can be used for MIRAGE-B scoring."""
    if gold is None:
        return False
    g = str(gold).strip()
    if not g or g.lower() in _INVALID_GOLD:
        return False
    # Winobias uses "unknown" as a placeholder when parsing fails.
    src = str(source).strip().lower()
    if src != "bbq" and g.lower() == "unknown":
        return False
    return True
