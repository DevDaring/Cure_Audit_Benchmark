"""Prompt integrity helpers for MIRAGE dataset generation."""


def prompt_embeds_original(api_prompt: str, original: str) -> bool:
    """
    Return True if ``api_prompt`` contains the full original slot-a text.

    DeepSeek must prepend context/preamble without omitting the original prompt.
    """
    if not api_prompt or not original:
        return False
    o = original.strip()
    p = api_prompt.strip()
    if o.lower() in p.lower():
        return True
    # Long BBQ/StereoSet prompts: require a substantial trailing fingerprint.
    if len(o) >= 60:
        tail = o[-120:].strip().lower()
        if tail and tail in p.lower():
            return True
    return False


def validate_api_slot_results(
    result: dict[str, str],
    original: str,
    required_keys: tuple[str, ...],
) -> bool:
    """All required keys must be present and embed the original prompt."""
    if not result:
        return False
    for key in required_keys:
        val = str(result.get(key, "")).strip()
        if not val or not prompt_embeds_original(val, original):
            return False
    return True
