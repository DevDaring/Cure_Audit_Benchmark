"""
p1_intervention.py -- ONE mathematically defined edit, applied identically wherever it is
measured, on a plain Hugging Face CausalLM with explicit hooks.

Next_Plan.md, Section 2.2 and P1:

  The manuscript equation projects h_p at the demographic position. The code has two
  different interventions (demographic-position patching in the audit; last-token projection
  at every generation step in the behavioural evaluation) and two source-state conventions
  (TransformerLens caches the unedited source; NNsight erases in-trace). Define the treatment
  mathematically, implement one source-state convention, and log the exact positions and
  layers changed in both measurements.

The edit
    h'_{l,t} = h_{l,t} - alpha * U_l^T U_l h_{l,t}        for t in S, for every layer l in L
  U_l : k x d orthonormal rows (basis_by_layer[l]); alpha in [0, 1]; S = the edited token
  positions of the sequence, chosen by a SITE policy:
    "span"        every token of the demographic span (all tokens, not only the first) in
                  the prompt; nothing is edited at decode steps, because the span has already
                  been processed and its edited hidden states are what the cache holds.
    "last_token"  the legacy ErasureContext behaviour: the last position of the prefill and
                  every newly generated token. Kept ONLY as a labelled diagnostic condition.
    "all"         every position of the prefill and every generated token (global policy,
                  the only one that can touch an external capability prompt with no span).
  Source-state convention for the commutator (audit):
    "sequential"  the edit hooks are ACTIVE while the source prompt is run, so the cached
                  activation at layer l already reflects edits at layers < l, and the same
                  hooks are active during the patched and clean runs of the target prompt.
                  This is one forward pass per condition on one backend, and it is what the
                  behavioural generation also experiences. It is the canonical convention.
    "frozen"      the source prompt is run WITHOUT hooks, each cached vector is projected
                  numerically, then injected. Kept as the legacy-TL diagnostic.

Prompt format
  The shipped audit patched the RAW prompt (no chat template, TL prepends BOS); the shipped
  behavioural readout generated from the CHAT template with the research system prompt. This
  module takes fmt = "raw" | "chat" and resolves span positions on whatever text is actually
  fed to the model, via the fast tokenizer's offset mapping, so BOS and template offsets are
  handled by construction. Both measurements in the pilot use the SAME fmt.

Every application records what changed: (layer, batch row, position, ||delta||, ||h||), so a
condition's removed activation energy E[||h-h'||^2]/E[||h||^2] is reported per site, and an
alpha=0 or rank-0 run can be ASSERTED to change nothing.

Public API
  resolve_span_positions(tok, text, span_text, occurrence=-1) -> list[int]
  build_input(tok, user_prompt, fmt, system=None) -> str
  HookedEdit(model, basis_by_layer, alpha, site)             context manager
      .set_targets(positions_by_row, pad_offsets)            before each forward/generate
      .changed_log                                           list of dicts
      .energy_by_layer()                                     {layer: ratio}
  commutator(model, tok, text_a, text_b, pos_a, pos_b, target_token_id, edit=None,
             source_state="sequential") -> (C, info)
  option_loglik(model, tok, text, options, edit=None, targets=None) -> dict
  identity_checks(model, tok, texts, basis_by_layer) -> dict
  generate_under_edit(model, tok, texts, edit, positions_by_row, max_new_tokens) -> list[str]

Implements / builds on:
  - Belrose et al. (2023) LEACE, arXiv:2306.03819 -- the projection form only; the
    covariance-aware eraser is compared separately in p2_leace_faithful.py.
  - Vig et al. (2020); Zhang and Nanda (2024) activation patching, arXiv:2309.16042.
  - Pearl (2009) do-calculus intervention semantics for the patch.

Part of the CURE codebase (ICLR 2027, Submission2). GPU required for everything but
resolve_span_positions and build_input.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterable

import numpy as np

log = logging.getLogger("cure.ext.p1.edit")

SITES = ("span", "last_token", "all")
SOURCE_STATES = ("sequential", "frozen")


# ---------------------------------------------------------------------------
# prompt construction and position resolution (CPU)
# ---------------------------------------------------------------------------

def build_input(tok, user_prompt: str, fmt: str = "raw", system: str | None = None) -> str:
    """The exact text fed to the model. fmt='raw' is the shipped audit convention; fmt='chat'
    is the shipped generation convention (osm_behavioral._build_prompt)."""
    if fmt == "raw":
        return user_prompt
    if fmt != "chat":
        raise ValueError(fmt)
    if not hasattr(tok, "apply_chat_template"):
        return f"<|system|>{system or ''}\n<|user|>{user_prompt}\n<|assistant|>"
    msgs = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": user_prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:                                   # Gemma: no system role
        merged = f"{(system or '').strip()}\n\n{user_prompt.strip()}".strip()
        return tok.apply_chat_template([{"role": "user", "content": merged}], tokenize=False,
                                       add_generation_prompt=True)


def resolve_span_positions(tok, text: str, span_text: str, occurrence: int = -1) -> list[int]:
    """Token indices covering `span_text` inside `text`, on the tokenisation the model sees.

    The pentad stores swap tokens with underscores ('the_colombian_boy'); the prompt has
    spaces. Both forms are tried, case-insensitively. occurrence=-1 takes the LAST occurrence
    (the slot-c edit is in the option line near the end; the passage may contain the same
    words earlier). Returns [] if not found. Uses offset_mapping, so BOS / template tokens
    are accounted for by construction. Multi-token spans return every token."""
    cands = [span_text, span_text.replace("_", " ")]
    low = text.lower()
    start = -1
    for c in cands:
        c = c.strip()
        if not c:
            continue
        i = low.rfind(c.lower()) if occurrence == -1 else low.find(c.lower())
        if i >= 0:
            start, end = i, i + len(c)
            break
    if start < 0:
        return []
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    offs = enc["offset_mapping"]
    pos = [i for i, (a, b) in enumerate(offs) if b > start and a < end and (b - a) > 0]
    return pos


# ---------------------------------------------------------------------------
# the hook
# ---------------------------------------------------------------------------

def _decoder_layers(model):
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("cannot find decoder layers on %s" % type(model).__name__)


class HookedEdit:
    """Forward hooks implementing h' = h - alpha U^T U h at the chosen site.

    Usage:
        edit = HookedEdit(model, basis, alpha=1.0, site="span")
        edit.set_targets(positions_by_row=[[12,13,14]], pad_offsets=[0])
        with edit:
            out = model(**inputs)  or  model.generate(**inputs)
        edit.changed_log -> what was edited
    """

    def __init__(self, model, basis_by_layer: dict[int, np.ndarray], alpha: float = 1.0,
                 site: str = "span", record: bool = True):
        import torch
        if site not in SITES:
            raise ValueError(site)
        self.model = model
        self.alpha = float(alpha)
        self.site = site
        self.record = record
        self.layers = _decoder_layers(model)
        dev = next(model.parameters()).device
        dt = next(model.parameters()).dtype
        self.U = {int(l): torch.as_tensor(np.asarray(rows, dtype=np.float32), device=dev).to(dt)
                  for l, rows in basis_by_layer.items() if rows is not None and len(rows)}
        self.handles = []
        self.changed_log: list[dict] = []
        self._targets: list[list[int]] = []
        self._pad: list[int] = []
        self._step = 0
        self.n_decode_edits = 0

    @property
    def rank(self) -> int:
        return max((int(u.shape[0]) for u in self.U.values()), default=0)

    def set_targets(self, positions_by_row: list[list[int]], pad_offsets: list[int] | None = None):
        """positions in UNPADDED token coordinates per batch row; pad_offsets = number of left
        pad tokens per row (0 for right padding or batch size 1)."""
        self._targets = [list(p) for p in positions_by_row]
        self._pad = list(pad_offsets) if pad_offsets is not None else [0] * len(self._targets)
        self._step = 0

    def _positions_for(self, row: int, T: int, prefill: bool) -> list[int]:
        if self.site == "all":
            return list(range(T))
        if self.site == "last_token":
            return [T - 1]
        if not prefill:                       # span policy edits nothing at decode steps
            return []
        if row >= len(self._targets):
            return []
        off = self._pad[row] if row < len(self._pad) else 0
        return [p + off for p in self._targets[row] if 0 <= p + off < T]

    def _make_hook(self, li: int):
        U = self.U[li]

        def hook(module, inputs, output):
            import torch
            hs = output[0] if isinstance(output, tuple) else output
            B, T, _ = hs.shape
            prefill = T > 1
            if not prefill:
                self._step += 1
            for b in range(B):
                pos = self._positions_for(b, T, prefill)
                if not pos:
                    continue
                if not prefill:
                    self.n_decode_edits += 1
                v = hs[b, pos, :]                                  # n x d
                delta = self.alpha * (v @ U.transpose(0, 1)) @ U   # n x d
                if self.record:
                    with torch.no_grad():
                        dn = delta.float().norm(dim=-1); hn = v.float().norm(dim=-1)
                        for p, d_, h_ in zip(pos, dn.tolist(), hn.tolist()):
                            self.changed_log.append({"layer": li, "row": b, "position": int(p),
                                                     "prefill": prefill, "delta_norm": d_,
                                                     "h_norm": h_})
                hs[b, pos, :] = v - delta
            return output
        return hook

    def __enter__(self):
        for li, layer in enumerate(self.layers):
            if li in self.U and self.U[li].shape[0] > 0:
                self.handles.append(layer.register_forward_hook(self._make_hook(li)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False

    def energy_by_layer(self) -> dict[int, float]:
        """E[||h-h'||^2] / E[||h||^2] per layer over everything edited so far."""
        num, den = {}, {}
        for r in self.changed_log:
            num[r["layer"]] = num.get(r["layer"], 0.0) + r["delta_norm"] ** 2
            den[r["layer"]] = den.get(r["layer"], 0.0) + r["h_norm"] ** 2
        return {l: (num[l] / den[l] if den[l] > 0 else float("nan")) for l in num}

    def summary(self) -> dict:
        e = self.energy_by_layer()
        return {"site": self.site, "alpha": self.alpha, "rank": self.rank,
                "n_layers_edited": len(self.U), "n_edits": len(self.changed_log),
                "n_decode_step_edits": self.n_decode_edits,
                "energy_mean_over_layers": float(np.mean(list(e.values()))) if e else float("nan"),
                "energy_by_layer": e}


@contextmanager
def no_edit():
    yield None


# ---------------------------------------------------------------------------
# forward helpers
# ---------------------------------------------------------------------------

def _encode(tok, text: str):
    import torch
    enc = tok(text, return_tensors="pt", add_special_tokens=True)
    dev = None
    return {k: v for k, v in enc.items()}, dev


def _to(model, enc):
    dev = next(model.parameters()).device
    return {k: v.to(dev) for k, v in enc.items()}


def forward_hidden(model, tok, text: str, edit: HookedEdit | None = None,
                   positions: list[int] | None = None):
    """Logits and per-layer post-block hidden states [T, d] for one text, optionally under an
    edit at `positions`. Returns (logits[T,V] float32 cpu, {layer: hidden[T,d] float32 cpu})."""
    import torch
    enc = _to(model, _encode(tok, text)[0])
    if edit is not None:
        edit.set_targets([positions or []], [0])
    ctx = edit if edit is not None else no_edit()
    with torch.no_grad(), ctx:
        out = model(**enc, output_hidden_states=True)
    # hidden_states[0] is the embedding output; hidden_states[l+1] is post-block l
    hs = {l: out.hidden_states[l + 1][0].float().cpu() for l in range(len(out.hidden_states) - 1)}
    return out.logits[0].float().cpu(), hs


def commutator(model, tok, text_a: str, text_b: str, pos_a: list[int], pos_b: list[int],
               target_token_id: int, edit_factory=None, source_state: str = "sequential",
               layers: Iterable[int] | None = None) -> tuple[float, dict]:
    """C(a,b) = logit_g(swap(a->b)) - logit_g(b) at the last position, optionally under an
    edit, with an explicit source-state convention.

    edit_factory: callable returning a fresh HookedEdit (or None for no edit). A fresh edit is
    built per pass so its changed_log is per pass. pos_a / pos_b are full span position lists;
    the patch copies the source span's hidden states into the target span position-by-
    position (spans of unequal token length are truncated to the shorter, and that is logged).
    """
    import torch
    if source_state not in SOURCE_STATES:
        raise ValueError(source_state)
    dec = _decoder_layers(model)
    L = list(layers) if layers is not None else list(range(len(dec)))
    n = min(len(pos_a), len(pos_b))
    info = {"span_len_a": len(pos_a), "span_len_b": len(pos_b), "patched_len": n,
            "source_state": source_state, "n_layers": len(L)}
    if n == 0:
        return float("nan"), {**info, "error": "empty span"}
    pa, pb = pos_a[:n], pos_b[:n]

    # 1. source hidden states
    if source_state == "sequential":
        e_a = edit_factory() if edit_factory else None
        _, hs_a = forward_hidden(model, tok, text_a, e_a, pa)
        src = {l: hs_a[l][pa, :].clone() for l in L}
        info["source_edit"] = e_a.summary() if e_a else None
    else:                                        # frozen: clean run, numeric projection
        _, hs_a = forward_hidden(model, tok, text_a, None, None)
        e_tmp = edit_factory() if edit_factory else None
        src = {}
        for l in L:
            v = hs_a[l][pa, :].clone()
            if e_tmp is not None and l in e_tmp.U:
                U = e_tmp.U[l].float().cpu()
                v = v - e_tmp.alpha * (v @ U.T) @ U
            src[l] = v
        info["source_edit"] = e_tmp.summary() if e_tmp else None

    # 2. patched run on b: inject src at pb (after any edit hook at that layer)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    handles = []

    def make_patch(l):
        s = src[l].to(dev).to(dt)

        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] > 1:
                hs[0, pb, :] = s
            return output
        return hook

    e_b1 = edit_factory() if edit_factory else None
    enc_b = _to(model, _encode(tok, text_b)[0])
    if e_b1 is not None:
        e_b1.set_targets([pb], [0])
    with torch.no_grad(), (e_b1 if e_b1 is not None else no_edit()):
        # register patch hooks AFTER the edit hooks so injection wins at the target positions
        for l in L:
            handles.append(dec[l].register_forward_hook(make_patch(l)))
        try:
            logit_patched = model(**enc_b).logits[0, -1, target_token_id].float().item()
        finally:
            for h in handles:
                h.remove()

    # 3. clean run on b under the same edit
    e_b2 = edit_factory() if edit_factory else None
    if e_b2 is not None:
        e_b2.set_targets([pb], [0])
    with torch.no_grad(), (e_b2 if e_b2 is not None else no_edit()):
        logit_clean = model(**enc_b).logits[0, -1, target_token_id].float().item()

    info["logit_patched"], info["logit_clean"] = logit_patched, logit_clean
    info["target_edit"] = e_b2.summary() if e_b2 else None
    return float(logit_patched - logit_clean), info


def option_loglik(model, tok, text: str, options: list[str], edit: HookedEdit | None = None,
                  positions: list[int] | None = None) -> dict:
    """Canonical option scoring: sum log p(option tokens | text) for each option string, and
    the per-token mean, under an optional edit at `positions` (prefill only). Returns
    argmax by sum-loglik, margin top1-top2, and an entropy over the option distribution."""
    import torch
    dev = next(model.parameters()).device
    base = tok(text, add_special_tokens=True)["input_ids"]
    T0 = len(base)
    rows = []
    for o in options:
        ids = tok(text + o, add_special_tokens=True)["input_ids"]
        if ids[:T0] != base:                                 # tokenizer merged boundary
            ids = base + tok(o, add_special_tokens=False)["input_ids"]
        rows.append(ids)
    maxlen = max(len(r) for r in rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    input_ids = torch.full((len(rows), maxlen), pad, dtype=torch.long)
    attn = torch.zeros_like(input_ids)
    for i, r in enumerate(rows):
        input_ids[i, :len(r)] = torch.tensor(r); attn[i, :len(r)] = 1
    enc = {"input_ids": input_ids.to(dev), "attention_mask": attn.to(dev)}
    if edit is not None:
        edit.set_targets([positions or []] * len(rows), [0] * len(rows))
    with torch.no_grad(), (edit if edit is not None else no_edit()):
        logits = model(**enc).logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    sums, means = [], []
    for i, r in enumerate(rows):
        n = len(r) - T0
        if n <= 0:
            sums.append(float("-inf")); means.append(float("-inf")); continue
        tgt = torch.tensor(r[T0:], device=dev)
        lp = logp[i, T0 - 1:len(r) - 1, :].gather(-1, tgt[:, None]).squeeze(-1)
        sums.append(float(lp.sum().item())); means.append(float(lp.mean().item()))
    s = np.array(sums)
    order = np.argsort(-s)
    p = np.exp(s - s.max()); p = p / p.sum()
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return {"options": options, "loglik_sum": sums, "loglik_mean_per_token": means,
            "argmax": int(order[0]), "margin_top1_top2": float(s[order[0]] - s[order[1]]) if len(s) > 1 else float("nan"),
            "entropy": ent, "probs": p.tolist()}


def generate_under_edit(model, tok, texts: list[str], edit: HookedEdit | None,
                        positions_by_row: list[list[int]], max_new_tokens: int = 48) -> list[str]:
    """Greedy generation with left padding; positions are unpadded coordinates per row."""
    import torch
    dev = next(model.parameters()).device
    side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=True).to(dev)
        pad_off = (enc["attention_mask"].shape[1] - enc["attention_mask"].sum(dim=1)).tolist()
        if edit is not None:
            edit.set_targets(positions_by_row, pad_off)
        with torch.no_grad(), (edit if edit is not None else no_edit()):
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        L = enc["input_ids"].shape[1]
        return [tok.decode(out[i][L:], skip_special_tokens=True) for i in range(out.shape[0])]
    finally:
        tok.padding_side = side


# ---------------------------------------------------------------------------
# identity checks (P1 condition 1 and the robustness list)
# ---------------------------------------------------------------------------

def identity_checks(model, tok, texts: list[str], basis_by_layer: dict, span_positions: list[list[int]],
                    tol: float = 1e-3) -> dict:
    """alpha=0 must equal no-edit; rank-0 must equal no-edit; a span edit must leave every
    non-span position's hidden state unchanged at every layer; decode steps must not fire
    under site='span'. Returns a dict of max deviations and pass flags."""
    import torch
    res = {"tol": tol}
    zero_basis = {l: np.zeros((0, np.asarray(r).shape[1]), np.float32) for l, r in basis_by_layer.items()}
    dev_a0, dev_r0, dev_nonspan, dev_span = 0.0, 0.0, 0.0, 0.0
    for text, pos in zip(texts, span_positions):
        lg0, hs0 = forward_hidden(model, tok, text, None, None)
        e = HookedEdit(model, basis_by_layer, alpha=0.0, site="span")
        lg1, _ = forward_hidden(model, tok, text, e, pos)
        dev_a0 = max(dev_a0, float((lg1 - lg0).abs().max()))
        e = HookedEdit(model, zero_basis, alpha=1.0, site="span")
        lg2, _ = forward_hidden(model, tok, text, e, pos)
        dev_r0 = max(dev_r0, float((lg2 - lg0).abs().max()))
        e = HookedEdit(model, basis_by_layer, alpha=1.0, site="span")
        _, hs3 = forward_hidden(model, tok, text, e, pos)
        T = hs0[0].shape[0]
        non = [t for t in range(T) if t not in set(pos)]
        first_edited = min(basis_by_layer) if basis_by_layer else 0
        # at the first edited layer, non-span positions must be bit-identical; deeper layers
        # legitimately differ because attention reads the edited span
        d_non = float((hs3[first_edited][non] - hs0[first_edited][non]).abs().max()) if non else 0.0
        d_sp = float((hs3[first_edited][pos] - hs0[first_edited][pos]).abs().max()) if pos else 0.0
        dev_nonspan = max(dev_nonspan, d_non); dev_span = max(dev_span, d_sp)
    # decode-step firing under span policy
    e = HookedEdit(model, basis_by_layer, alpha=1.0, site="span")
    _ = generate_under_edit(model, tok, texts[:1], e, span_positions[:1], max_new_tokens=4)
    res.update({
        "max_abs_logit_dev_alpha0": dev_a0, "pass_alpha0": dev_a0 <= tol,
        "max_abs_logit_dev_rank0": dev_r0, "pass_rank0": dev_r0 <= tol,
        "max_abs_hidden_dev_nonspan_first_edited_layer": dev_nonspan,
        "pass_nonspan_untouched": dev_nonspan <= tol,
        "max_abs_hidden_dev_span_first_edited_layer": dev_span,
        "span_actually_edited": dev_span > tol,
        "decode_step_edits_under_span_policy": e.n_decode_edits,
        "pass_no_decode_edit_under_span": e.n_decode_edits == 0,
        "dtype": str(next(model.parameters()).dtype),
    })
    res["pass_all"] = all(res[k] for k in ("pass_alpha0", "pass_rank0", "pass_nonspan_untouched",
                                           "pass_no_decode_edit_under_span")) and res["span_actually_edited"]
    return res


# ---------------------------------------------------------------------------
# basis estimators (kept minimal here; P2 adds LEACE, random and neutral controls)
# ---------------------------------------------------------------------------

def basis_centred_svd(diffs_by_layer: dict[int, list[np.ndarray]], rank: int) -> dict:
    """The shipped CURE estimator: leading right singular vectors of CENTRED differences."""
    out = {}
    for l, ds in diffs_by_layer.items():
        if len(ds) < 2:
            continue
        X = np.stack(ds, 0).astype(np.float64)
        X = X - X.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        out[l] = vt[:min(rank, vt.shape[0])].astype(np.float32)
    return out


def basis_uncentred_svd(diffs_by_layer: dict[int, list[np.ndarray]], rank: int) -> dict:
    out = {}
    for l, ds in diffs_by_layer.items():
        if len(ds) < 2:
            continue
        X = np.stack(ds, 0).astype(np.float64)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        out[l] = vt[:min(rank, vt.shape[0])].astype(np.float32)
    return out


def basis_mean_difference(diffs_by_layer: dict[int, list[np.ndarray]]) -> dict:
    """Rank-1: the normalised mean of the differences (the contrast centring discards)."""
    out = {}
    for l, ds in diffs_by_layer.items():
        if not ds:
            continue
        m = np.mean(np.stack(ds, 0), 0)
        n = float(np.linalg.norm(m))
        if n > 1e-8:
            out[l] = (m / n).reshape(1, -1).astype(np.float32)
    return out


def basis_random(d: int, layers: Iterable[int], rank: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for l in layers:
        q, _ = np.linalg.qr(rng.standard_normal((d, rank)))
        out[l] = q.T.astype(np.float32)
    return out
