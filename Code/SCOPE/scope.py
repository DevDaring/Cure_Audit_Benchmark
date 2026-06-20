"""
scope.py -- the SCOPE method (Surgical COncept-ablation via Patchscope-guided Editing).

Three stages, then a verification:
  Stage 1  LOCALISE. A Few-Shot Token-Identity Patchscope decodes, per layer, how strongly
           the protected attribute is readable from the residual stream at the swapped
           position. The localised layers are those whose decodability is high.
  Stage 2  PROTECT. At the localised layers, estimate the demographic direction and remove
           the part that lies on the model's massive-activation dimensions, so the
           load-bearing structure is left intact.
  Stage 3  EDIT. The result is a per-(localised layer) rank-1 basis. Passing it to the CURE
           evaluators (e3_reaudit, native_accuracy, behavioural_readout, ErasureContext)
           edits ONLY the localised layers with the protected direction -- the whole CURE
           machinery is reused unchanged.
  Verify   Re-decode the edited representation with the same Patchscope; the decodability of
           the protected attribute must drop, which the paper reports against the
           behavioural flip-rate drop.

Why this should beat Faithful-Patchscopes (Gong et al. 2026, arXiv:2602.00300, the
`patchscopes` baseline): SCOPE (a) uses the causal audit signal, (b) localises by
natural-language decodability rather than read-out energy, and (c) edits orthogonal to the
massive-activation dimensions, which Faithful-Patchscopes does not. The code never forces a
win: every number is measured by the same CURE evaluators on the same pairs.

Cites: Patchscopes (Ghandeharioun et al. 2024, arXiv:2401.06102); LEACE (Belrose et al.
2023, arXiv:2306.03819); massive activations (Sun et al. 2024, arXiv:2402.17762).
"""

import logging
from contextlib import nullcontext

import numpy as np

import config_scope as C
import erase
import experiments as E

log = logging.getLogger("scope")


# ---------------------------------------------------------------------------
# Model-shape helpers
# ---------------------------------------------------------------------------

def _n_layers(model) -> int:
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return len(inner.layers)
    if hasattr(model, "layers"):
        return len(model.layers)
    # TransformerLens-wrapped HF models still expose .model.layers; fall back to config.
    return int(getattr(getattr(model, "config", object()), "num_hidden_layers", 32))


def _demo_first_token(tok, term: str):
    """First sub-token id of a demographic term, with a leading space for fluent decoding."""
    if not term:
        return None
    clean = " " + str(term).replace("_", " ").strip()
    ids = tok.encode(clean, add_special_tokens=False)
    return ids[0] if ids else None


# ---------------------------------------------------------------------------
# Patchscope decode: inject a representation at the target prompt's last position and read
# the probability that the model decodes the demographic term.
# ---------------------------------------------------------------------------

def _decode_tl(model, tok, h_by_layer: dict, demo_id: int, layers: list) -> dict:
    import torch
    from utils_attention import _ensure_hooked_transformer
    tl = _ensure_hooked_transformer(model, tok)
    dev = next(tl.parameters()).device
    dt = next(tl.parameters()).dtype
    toks = tl.to_tokens(C.PATCHSCOPE_TARGET)            # [1, seq], BOS prepended
    pos = toks.shape[1] - 1
    out = {}
    for L in layers:
        if L not in h_by_layer:
            continue
        act = torch.tensor(h_by_layer[L], device=dev, dtype=dt)
        key = f"blocks.{L}.hook_resid_post"

        def make(a):
            def fn(value, hook):
                value[0, pos, :] = a
                return value
            return fn
        with torch.no_grad():
            logits = tl.run_with_hooks(toks, fwd_hooks=[(key, make(act))])
        p = torch.softmax(logits[0, -1, :].float(), dim=-1)[demo_id].item()
        out[L] = float(p)
    return out


def _decode_nnsight(model, tok, h_by_layer: dict, demo_id: int, layers: list) -> dict:
    import torch
    from utils_attention import _ensure_nnsight_model
    nn_model = _ensure_nnsight_model(model, tok)
    inner = getattr(model, "model", None)
    shape_b = inner is not None and hasattr(inner, "layers")
    n_layers = len(inner.layers) if shape_b else len(model.layers)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    ids = tok(C.PATCHSCOPE_TARGET, return_tensors="pt")
    pos = ids["input_ids"].shape[1] - 1
    out = {}
    for L in layers:
        if L not in h_by_layer or L >= n_layers:
            continue
        act = torch.tensor(h_by_layer[L], device=dev, dtype=dt)
        with nn_model.trace(C.PATCHSCOPE_TARGET):
            ly = nn_model.model.layers[L] if shape_b else nn_model.layers[L]
            ly.output[0][:, pos, :] = act
            logits = nn_model.lm_head.output.save()
        p = torch.softmax(logits.value[0, -1, :].float(), dim=-1)[demo_id].item()
        out[L] = float(p)
    return out


def _decode(model, tok, h_by_layer, demo_id, layers, lib):
    if lib == "transformer_lens":
        return _decode_tl(model, tok, h_by_layer, demo_id, layers)
    return _decode_nnsight(model, tok, h_by_layer, demo_id, layers)


def _apply_edit(h_by_layer: dict, basis: dict) -> dict:
    """Project the protected direction out of the cached source at the localised layers,
    so the decoded representation is the one SCOPE would expose after editing."""
    if not basis:
        return h_by_layer
    out = {}
    for L, v in h_by_layer.items():
        if L in basis:
            d = basis[L][0]
            out[L] = (v - float(v @ d) * d).astype(np.float32)
        else:
            out[L] = v
    return out


# ---------------------------------------------------------------------------
# Stage 1: decodability map (localisation) and Verify (post-edit decodability)
# ---------------------------------------------------------------------------

def decodability_map(model, tok, cfg, pairs, layers=None, edit_basis=None, n=None) -> dict:
    """Mean Patchscope decodability of the demographic term per layer over `pairs`.

    edit_basis=None  -> the unedited map (Stage 1, localisation).
    edit_basis=basis -> the post-edit map (verification): the cached source is projected
                        onto the complement of the protected direction before decoding.
    """
    from experiments import _positions
    lib = cfg["patching_lib"]
    n = n or C.LOCALIZE_PAIRS
    pairs = pairs[:n]
    if layers is None:
        layers = list(range(_n_layers(model)))
    acc = {L: [] for L in layers}
    for pair in pairs:
        pa, _pb = _positions(tok, pair)
        if pa is None:
            continue
        demo_id = _demo_first_token(tok, pair.get("swap_a") or "")
        if demo_id is None:
            continue
        try:
            h = erase.cache_resid(model, tok, pair["prompt_a"], pa, lib)
            h = _apply_edit(h, edit_basis)
            dec = _decode(model, tok, h, demo_id, layers, lib)
        except Exception as exc:
            log.warning("decode failed seed %s: %s", pair.get("seed_id"), str(exc)[:120])
            continue
        for L, p in dec.items():
            acc[L].append(p)
    return {L: float(np.mean(v)) for L, v in acc.items() if v}


def localise(decod: dict, pctile: float) -> list:
    """The localised layers: those at or above the given decodability percentile."""
    if not decod:
        return []
    thr = float(np.percentile(list(decod.values()), pctile))
    return sorted([L for L, v in decod.items() if v >= thr])


# ---------------------------------------------------------------------------
# Stage 2: protected, massive-activation-orthogonal demographic direction
# ---------------------------------------------------------------------------

def build_scope_basis(model, tok, cfg, train_pairs, localised_layers, protect=True) -> dict:
    """A per-(localised layer) rank-1 basis of the protected demographic direction.

    The demographic direction is the difference-in-means of the counterfactual activation
    differences. When protect=True, its components on the top-`MASSIVE_K` magnitude
    (massive-activation) dimensions are removed, so the edit is orthogonal to the
    load-bearing structure. A layer whose demographic direction lies almost entirely on the
    massive dimensions is dropped as non-removable.
    """
    ctx = E.collect_acts(model, tok, cfg, train_pairs[:C.SUBSPACE_PAIRS])
    diffs, A = ctx["diffs"], ctx["A"]
    basis, dropped = {}, []
    for L in localised_layers:
        if L not in diffs or len(diffs[L]) < 2:
            continue
        d = np.mean(np.stack(diffs[L], 0), axis=0).astype(np.float64)
        nd = np.linalg.norm(d)
        if nd < 1e-8:
            continue
        d = d / nd
        if protect and L in A and len(A[L]) > 0:
            mag = np.abs(np.stack(A[L], 0)).mean(0)
            massive = np.argsort(mag)[::-1][:C.MASSIVE_K]
            dd = d.copy()
            dd[massive] = 0.0
            nn = np.linalg.norm(dd)
            if nn < 0.10:                      # direction is essentially load-bearing -> skip
                dropped.append(int(L))
                continue
            d = dd / nn
        basis[int(L)] = d.reshape(1, -1).astype(np.float32)
    if dropped:
        log.info("scope %s: %d localised layers dropped as load-bearing: %s",
                 cfg["name"], len(dropped), dropped[:10])
    return basis
