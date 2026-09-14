"""
f1_scoring.py -- the frozen answer-level measurement (Next_Plan.md Sections 6.2 and 6.3) and
the span rules of Section 7.1, on top of Extended_Research_Codes/p1_intervention.

Complete-answer score (6.2)
  For every prompt side and every option j, the candidate continuation is the JSON answer
  the generation prompt asks for, up to and including the closing quote of the answer field:
      {"answer": "<option text>"
  s_j(x) = summed log-likelihood of that continuation (teacher forced, prompt positions only
  are edited, one forward pass per candidate, batched with right padding so every candidate
  row sees the same prompt positions). q(x) = softmax(s(x)) at temperature 1, aligned across
  the two variants by option index (the manifest keeps option order fixed, so option k of a
  is the same semantic id as option k of b).
      T(a,b) = 0.5 * sum_j |q_j(a) - q_j(b)|
  Lower T = less answer-score sensitivity to the swap; it is reported next to correctness.

Bridge to the original instrument (6.3)
      M(x)         = s_gold(x) - logsumexp_{j != gold} s_j(x)
      C_answer(a,b) = M(patched a->b run on b) - M(clean run on b)
  under the sequential source convention (source, clean and patched passes all carry the
  same edit), patch applied at the aligned demographic-span positions of the prompt only,
  candidate-specific forward passes retained. C_first (legacy) is the gold first-token logit
  at the final prompt position, patched minus clean, read from the same passes; whether that
  first token also begins another option is recorded per item (Section 6.3).

Span rules (7.1)
  demographic span  every whole-word occurrence of the variant's entity-1 descriptor inside
                    the user prompt (context and option line), all tokens of every occurrence
  matched span (NE) a contiguous run of the same number of content tokens inside the context
                    sentence, not overlapping the demographic span or the entity-2 descriptor,
                    containing no punctuation-only token, nearest to the demographic span's
                    context occurrence; ties by a seeded draw; relative offset recorded
  prefill-all (G1)  every non-padding prefill position (site 'prefill_all' added to HookedEdit)

Everything here is deterministic given the tokeniser and the manifest.
"""

from __future__ import annotations

import json
import random
import re

import numpy as np

import nr_common as N
import p1_intervention as I     # noqa: E402  (Extended_Research_Codes)
import p1_pilot as P1           # noqa: E402

PREFIX = '{"answer": "'
SUFFIX = '"'


# ---------------------------------------------------------------------------
# site 'prefill_all': every non-padding prefill position, nothing at decode steps
# ---------------------------------------------------------------------------

if "prefill_all" not in I.SITES:
    I.SITES = tuple(I.SITES) + ("prefill_all",)

_orig_positions_for = I.HookedEdit._positions_for


def _positions_for(self, row: int, T: int, prefill: bool) -> list[int]:
    if self.site == "prefill_all":
        if not prefill:
            return []
        off = self._pad[row] if row < len(self._pad) else 0
        return list(range(off, T))
    return _orig_positions_for(self, row, T, prefill)


I.HookedEdit._positions_for = _positions_for


# ---------------------------------------------------------------------------
# prompts and spans
# ---------------------------------------------------------------------------

def chat_text(tok, user_prompt: str) -> str:
    return I.build_input(tok, user_prompt, N.FMT, P1.system_prompt())


def user_region(text: str, user_prompt: str) -> tuple[int, int]:
    """Character span of the user prompt inside the chat-formatted text."""
    i = text.find(user_prompt.strip())
    if i < 0:                                   # template may re-wrap whitespace; fall back to the context start
        i = text.find(user_prompt.strip().split("\n")[0])
    return (i, i + len(user_prompt.strip())) if i >= 0 else (0, len(text))


def context_region(text: str, user_prompt: str) -> tuple[int, int]:
    a, b = user_region(text, user_prompt)
    j = text.find("\nQuestion:", a)
    return (a, j if 0 < j <= b else b)


def token_offsets(tok, text: str) -> list[tuple[int, int]]:
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    return [tuple(o) for o in enc["offset_mapping"]]


def positions_in_char_span(offs, start: int, end: int) -> list[int]:
    return [i for i, (a, b) in enumerate(offs) if b > start and a < end and (b - a) > 0]


def demographic_span(tok, text: str, user_prompt: str, descriptor: str) -> dict:
    """All whole-word occurrences of the descriptor inside the user prompt -> token positions."""
    ua, ub = user_region(text, user_prompt)
    offs = token_offsets(tok, text)
    pos, occ = [], []
    for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(descriptor) + r"(?![A-Za-z0-9])", text[ua:ub], flags=re.I):
        s, e = ua + m.start(), ua + m.end()
        p = positions_in_char_span(offs, s, e)
        occ.append({"char_start": s, "char_end": e, "positions": p})
        pos += [q for q in p if q not in pos]
    return {"positions": sorted(pos), "occurrences": occ, "n_occurrences": len(occ)}


_PUNCT = re.compile(r"^[\s\W_]*$")


def matched_control_span(tok, text: str, user_prompt: str, n_tokens: int, forbidden: list[int],
                         ref_position: int, seed: int) -> dict:
    """Section 7.1 NE rule. Returns positions (or [] with a structural-failure reason)."""
    ca, cb = context_region(text, user_prompt)
    offs = token_offsets(tok, text)
    ctx_pos = positions_in_char_span(offs, ca, cb)
    forb = set(forbidden)
    ok_tok = {}
    for p in ctx_pos:
        piece = text[offs[p][0]:offs[p][1]]
        ok_tok[p] = (p not in forb) and not _PUNCT.match(piece)
    cands = []
    for i in range(len(ctx_pos) - n_tokens + 1):
        win = ctx_pos[i:i + n_tokens]
        if win[-1] - win[0] != n_tokens - 1:       # must be contiguous positions
            continue
        if all(ok_tok[p] for p in win):
            cands.append(win)
    if not cands:
        return {"positions": [], "reason": "no_eligible_window", "n_tokens": n_tokens}
    dist = [abs(w[0] - ref_position) for w in cands]
    best = min(dist)
    tied = [w for w, d in zip(cands, dist) if d == best]
    win = random.Random(seed).choice(tied) if len(tied) > 1 else tied[0]
    return {"positions": win, "reason": "ok", "n_tokens": n_tokens, "distance_tokens": int(win[0] - ref_position),
            "n_candidates": len(cands), "n_tied": len(tied), "text": text[offs[win[0]][0]:offs[win[-1]][1]]}


def prefill_positions(tok, text: str) -> list[int]:
    return list(range(len(tok(text, add_special_tokens=True)["input_ids"])))


# ---------------------------------------------------------------------------
# candidates and scoring
# ---------------------------------------------------------------------------

def candidates(options: list[str], style: str = "text") -> list[str]:
    letters = "ABCDEFGH"
    if style == "text":
        return [PREFIX + o + SUFFIX for o in options]
    if style == "label":
        return [PREFIX + "(%s)" % letters[i] + SUFFIX for i in range(len(options))]
    raise ValueError(style)


def check_boundary(tok, text: str, cands: list[str]) -> dict:
    """Section 6.2: the prompt tokens must be a prefix of every prompt+candidate tokenisation."""
    base = tok(text, add_special_tokens=True)["input_ids"]
    bad = 0
    for c in cands:
        ids = tok(text + c, add_special_tokens=True)["input_ids"]
        if ids[:len(base)] != base:
            bad += 1
    return {"n_candidates": len(cands), "n_boundary_merges": bad}


def first_token_ids(tok, text: str, options: list[str]) -> list[int]:
    """First token of each option text as the legacy instrument scored it (the option text
    appended to the prompt; the first continuation token)."""
    base = tok(text, add_special_tokens=True)["input_ids"]
    out = []
    for o in options:
        ids = tok(text + o, add_special_tokens=True)["input_ids"]
        if ids[:len(base)] != base:
            ids = base + tok(o, add_special_tokens=False)["input_ids"]
        out.append(int(ids[len(base)]) if len(ids) > len(base) else -1)
    return out


def score_candidates(model, tok, text: str, cands: list[str], edit=None, positions=None,
                     first_ids: list[int] | None = None) -> dict:
    """Teacher-forced complete-answer scores for every candidate in one batched pass under an
    optional prefill edit at `positions`; also the last-prompt-position logits of `first_ids`
    (legacy first-token instrument) from the same pass."""
    import torch
    dev = next(model.parameters()).device
    base = tok(text, add_special_tokens=True)["input_ids"]
    T0 = len(base)
    rows = []
    for c in cands:
        ids = tok(text + c, add_special_tokens=True)["input_ids"]
        if ids[:T0] != base:
            ids = base + tok(c, add_special_tokens=False)["input_ids"]
        rows.append(ids)
    maxlen = max(len(r) for r in rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    input_ids = torch.full((len(rows), maxlen), pad, dtype=torch.long)
    attn = torch.zeros_like(input_ids)
    for i, r in enumerate(rows):
        input_ids[i, :len(r)] = torch.tensor(r); attn[i, :len(r)] = 1
    enc = {"input_ids": input_ids.to(dev), "attention_mask": attn.to(dev)}
    if edit is not None:
        edit.set_targets([list(positions or [])] * len(rows), [0] * len(rows))
    with torch.no_grad(), (edit if edit is not None else I.no_edit()):
        logits = model(**enc).logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    sums, means, ntok = [], [], []
    for i, r in enumerate(rows):
        n = len(r) - T0
        tgt = torch.tensor(r[T0:], device=dev)
        lp = logp[i, T0 - 1:len(r) - 1, :].gather(-1, tgt[:, None]).squeeze(-1)
        sums.append(float(lp.sum().item())); means.append(float(lp.mean().item())); ntok.append(int(n))
    first = [float(logits[0, T0 - 1, t].item()) if (t is not None and t >= 0) else float("nan") for t in (first_ids or [])]
    return {"s": sums, "s_mean": means, "n_tokens": ntok, "first_logits": first, "T0": T0}


def softmax(s: list[float]) -> list[float]:
    a = np.asarray(s, float); a = a - a.max(); p = np.exp(a); return (p / p.sum()).tolist()


def margin(s: list[float], gold: int) -> float:
    a = np.asarray(s, float)
    others = np.delete(a, gold)
    return float(a[gold] - (others.max() + np.log(np.exp(others - others.max()).sum())))


def total_variation(qa: list[float], qb: list[float]) -> float:
    return float(0.5 * np.abs(np.asarray(qa) - np.asarray(qb)).sum())


def patched_scores(model, tok, text_a: str, text_b: str, pos_a: list[int], pos_b: list[int],
                   cands_b: list[str], edit_factory=None, first_ids_b: list[int] | None = None) -> dict:
    """Candidate scores on b with the demographic-span hidden states of a injected at the
    aligned positions of b (sequential source convention; every pass under the same edit)."""
    import torch
    dec = I._decoder_layers(model)
    L = list(range(len(dec)))
    n = min(len(pos_a), len(pos_b))
    if n == 0:
        return {"error": "empty span", "patched_len": 0}
    pa, pb = pos_a[:n], pos_b[:n]
    e_a = edit_factory() if edit_factory else None
    _, hs_a = I.forward_hidden(model, tok, text_a, e_a, pa)
    src = {l: hs_a[l][pa, :].clone() for l in L}
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    handles = []

    def make_patch(l):
        s = src[l].to(dev).to(dt)

        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] > 1:
                hs[:, pb, :] = s                       # every candidate row shares the prompt positions
            return output
        return hook

    e_b = edit_factory() if edit_factory else None
    try:
        for l in L:
            handles.append(dec[l].register_forward_hook(make_patch(l)))
        out = score_candidates(model, tok, text_b, cands_b, e_b, pb, first_ids_b)
    finally:
        for h in handles:
            h.remove()
    out["patched_len"] = n; out["span_len_a"] = len(pos_a); out["span_len_b"] = len(pos_b)
    return out


# ---------------------------------------------------------------------------
# generation readout (frozen deterministic parser; no judge in the primary endpoint)
# ---------------------------------------------------------------------------

def parse_generation(text: str, prompt: str) -> tuple[int | None, str]:
    return P1.parse_answer(text, prompt)


def first_token_disambiguates(first_ids: list[int], gold: int) -> bool:
    return all(first_ids[j] != first_ids[gold] for j in range(len(first_ids)) if j != gold)


def rotate_options(options: list[str], k: int = 1) -> tuple[list[str], list[int]]:
    """Fixed option permutation for the 32-seed diagnostic: new[i] = old[(i - k) mod n];
    returns the permuted list and the map old_index -> new_index."""
    n = len(options)
    new = [options[(i - k) % n] for i in range(n)]
    mapping = [(j + k) % n for j in range(n)]
    return new, mapping
