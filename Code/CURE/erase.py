"""
erase.py -- E1 bias-subspace extraction and E2 surgical erasure.

E1: from the counterfactual activations the audit already pairs (slot-c swaps),
    collect the protected-position residual-stream difference vectors at every
    layer, and take their top-rank principal subspace. That subspace is the
    direction the answer leans on under a demographic swap.

E2: a forward-pass hook that projects the residual stream at the protected
    position onto the complement of that subspace. No fine-tuning, inference only.

The maths follows difference-in-means concept identification plus an orthonormal
projection eraser, the linear core of LEACE (Belrose et al. 2023, arXiv:2306.03819).

Reuses the audit patching stack (Code/audit/GPU_CPU/utils_attention.py): the same
hook_resid_post sites for TransformerLens and the same layer.output proxy for nnsight.
"""

import logging

import numpy as np

import config_cure as C

log = logging.getLogger("cure.erase")


# ---------------------------------------------------------------------------
# Activation caching at the protected position
# ---------------------------------------------------------------------------

def cache_resid_tl(model, tokenizer, prompt: str, position: int) -> dict:
    """Return {layer: residual vector (numpy) at `position`} for a TL model."""
    import torch
    from utils_attention import _ensure_hooked_transformer
    tl = _ensure_hooked_transformer(model, tokenizer)
    with torch.no_grad():
        _, cache = tl.run_with_cache(prompt, return_type=None)
    out = {}
    for layer in range(tl.cfg.n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key in cache:
            out[layer] = cache[key][0, position, :].float().cpu().numpy()
    return out


def cache_resid_nnsight(model, tokenizer, prompt: str, position: int) -> dict:
    """Return {layer: residual vector (numpy) at `position`} for an nnsight model."""
    from utils_attention import _ensure_nnsight_model
    nn_model = _ensure_nnsight_model(model, tokenizer)
    inner = getattr(model, "model", None)
    shape_b = inner is not None and hasattr(inner, "layers")
    n_layers = len(inner.layers) if shape_b else len(model.layers)
    proxies = {}
    with nn_model.trace(prompt):
        for li in range(n_layers):
            layer = nn_model.model.layers[li] if shape_b else nn_model.layers[li]
            proxies[li] = layer.output[0][:, position, :].save()
    return {li: p.value[0].float().cpu().numpy() for li, p in proxies.items()}


def cache_resid(model, tokenizer, prompt: str, position: int, patching_lib: str) -> dict:
    if patching_lib == "transformer_lens":
        return cache_resid_tl(model, tokenizer, prompt, position)
    return cache_resid_nnsight(model, tokenizer, prompt, position)


# ---------------------------------------------------------------------------
# E1: subspace from counterfactual difference vectors
# ---------------------------------------------------------------------------

def subspace_from_diffs(diffs_by_layer: dict, rank: int) -> dict:
    """Per layer, return an orthonormal basis (rank x d) of the top principal
    directions of the difference vectors. diffs_by_layer: {layer: list[np.ndarray]}.
    """
    basis = {}
    for layer, diffs in diffs_by_layer.items():
        if len(diffs) < 2:
            continue
        X = np.stack(diffs, axis=0)            # n x d
        X = X - X.mean(axis=0, keepdims=True)
        # right singular vectors are the principal directions in feature space
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        k = min(rank, vt.shape[0])
        basis[layer] = vt[:k, :].astype(np.float32)   # k x d, orthonormal rows
    return basis


def project_out(vec, basis_rows):
    """Project `vec` (torch tensor, d) onto the complement of span(basis_rows).

    basis_rows: torch tensor (k x d), orthonormal rows. Returns vec - U^T U vec.
    """
    coeff = basis_rows @ vec          # k
    return vec - basis_rows.transpose(0, 1) @ coeff


# ---------------------------------------------------------------------------
# E2: erasure hooks
# ---------------------------------------------------------------------------

def make_tl_erase_hooks(basis_by_layer: dict, position: int, device, dtype):
    """Build TransformerLens fwd_hooks that erase the subspace at `position`."""
    import torch
    hooks = []
    for layer, rows in basis_by_layer.items():
        key = f"blocks.{layer}.hook_resid_post"
        U = torch.tensor(rows, device=device, dtype=dtype)   # k x d

        def make(U_):
            def hook_fn(value, hook):
                v = value[0, position, :]
                value[0, position, :] = project_out(v, U_)
                return value
            return hook_fn
        hooks.append((key, make(U)))
    return hooks


def _np_project_out(vec_np, rows_np):
    """vec - U^T (U vec) in numpy. rows_np: k x d orthonormal."""
    return vec_np - rows_np.T @ (rows_np @ vec_np)


def erased_commutator_tl(model, tokenizer, prompt_a, prompt_b, pos_a, pos_b,
                         bias_answer, basis_by_layer) -> float:
    """CDVA commutator on the TL model with the subspace erased at the protected position.

    The source activation from prompt_a is erased numerically, then injected into
    prompt_b; the clean pass erases prompt_b at the same position. Both compared
    states therefore lack the bias direction, so a residual commutator means the
    erasure did not remove the dependence.
    """
    import torch
    from utils_attention import _ensure_hooked_transformer
    tl = _ensure_hooked_transformer(model, tokenizer)
    device = next(tl.parameters()).device
    dtype = next(tl.parameters()).dtype

    with torch.no_grad():
        _, cache_a = tl.run_with_cache(prompt_a, return_type=None)

    # erased source activations (numpy projection), injected into prompt_b at pos_b
    patch_hooks = []
    for layer in range(tl.cfg.n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache_a or layer not in basis_by_layer:
            continue
        src = cache_a[key][0, pos_a, :].float().cpu().numpy()
        erased = _np_project_out(src, basis_by_layer[layer])
        act = torch.tensor(erased, device=device, dtype=dtype)

        def make(a):
            def hook_fn(value, hook):
                value[0, pos_b, :] = a
                return value
            return hook_fn
        patch_hooks.append((key, make(act)))

    erase_b = make_tl_erase_hooks(basis_by_layer, pos_b, device, dtype)
    with torch.no_grad():
        logits_patched = tl.run_with_hooks(prompt_b, fwd_hooks=patch_hooks)
        logits_clean = tl.run_with_hooks(prompt_b, fwd_hooks=erase_b)

    bt = tl.to_tokens(bias_answer, prepend_bos=False)[0]
    if len(bt) == 0:
        return 0.0
    tok = bt[0].item()
    return float(logits_patched[0, -1, tok].item() - logits_clean[0, -1, tok].item())


def erased_commutator_nnsight(model, tokenizer, prompt_a, prompt_b, pos_a, pos_b,
                              bias_answer, basis_by_layer) -> float:
    """CDVA commutator on the nnsight model with the subspace erased at the protected position."""
    import torch
    from utils_attention import _ensure_nnsight_model
    nn_model = _ensure_nnsight_model(model, tokenizer)
    inner = getattr(model, "model", None)
    shape_b = inner is not None and hasattr(inner, "layers")
    n_layers = len(inner.layers) if shape_b else len(model.layers)

    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    U = {li: torch.tensor(rows, device=dev, dtype=dt) for li, rows in basis_by_layer.items()}

    def layer_at(li):
        return nn_model.model.layers[li] if shape_b else nn_model.layers[li]

    # Pass 1: cache prompt_a residuals (erased at pos_a)
    cache_a = {}
    with nn_model.trace(prompt_a):
        for li in range(n_layers):
            ly = layer_at(li)
            if li in U:
                v = ly.output[0][:, pos_a, :]
                ly.output[0][:, pos_a, :] = v - (v @ U[li].transpose(0, 1)) @ U[li]
            cache_a[li] = ly.output[0][:, pos_a, :].save()
    cache_a = {li: p.value.clone() for li, p in cache_a.items()}

    # Pass 2: patched + erased on prompt_b
    with nn_model.trace(prompt_b):
        for li in range(n_layers):
            ly = layer_at(li)
            if li in U:
                v = ly.output[0][:, pos_b, :]
                ly.output[0][:, pos_b, :] = v - (v @ U[li].transpose(0, 1)) @ U[li]
            if li in cache_a:
                ly.output[0][:, pos_b, :] = cache_a[li]
        patched = nn_model.lm_head.output.save()

    # Pass 3: clean + erased on prompt_b
    with nn_model.trace(prompt_b):
        for li in range(n_layers):
            ly = layer_at(li)
            if li in U:
                v = ly.output[0][:, pos_b, :]
                ly.output[0][:, pos_b, :] = v - (v @ U[li].transpose(0, 1)) @ U[li]
        clean = nn_model.lm_head.output.save()

    ids = tokenizer.encode(bias_answer, add_special_tokens=False)
    if not ids:
        return 0.0
    tok = ids[0]
    return float(patched.value[0, -1, tok].item() - clean.value[0, -1, tok].item())


def erased_commutator(model, tokenizer, prompt_a, prompt_b, pos_a, pos_b,
                      bias_answer, basis_by_layer, patching_lib) -> float:
    if patching_lib == "transformer_lens":
        return erased_commutator_tl(model, tokenizer, prompt_a, prompt_b, pos_a, pos_b,
                                    bias_answer, basis_by_layer)
    return erased_commutator_nnsight(model, tokenizer, prompt_a, prompt_b, pos_a, pos_b,
                                     bias_answer, basis_by_layer)


# ---------------------------------------------------------------------------
# HF forward-hook erasure context (for generation / utility under erasure, E4)
# ---------------------------------------------------------------------------

class ErasureContext:
    """Project the subspace out of the last-token hidden state at every decoder
    layer, for as long as the context is open. Works on a plain HF CausalLM, so
    the audit's batched generation runs unchanged under erasure.
    """

    def __init__(self, hf_model, basis_by_layer: dict):
        import torch
        self.model = hf_model
        self.handles = []
        inner = getattr(hf_model, "model", None)
        layers = inner.layers if (inner is not None and hasattr(inner, "layers")) else hf_model.layers
        dev = next(hf_model.parameters()).device
        dt = next(hf_model.parameters()).dtype
        self.U = {li: torch.tensor(rows, device=dev, dtype=dt)
                  for li, rows in basis_by_layer.items()}
        self.layers = layers

    def __enter__(self):
        import torch

        def make(li):
            U = self.U[li]

            def hook(module, inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                v = hs[:, -1, :]                       # last token, all batch
                hs[:, -1, :] = v - (v @ U.transpose(0, 1)) @ U
                return output
            return hook

        for li, layer in enumerate(self.layers):
            if li in self.U:
                self.handles.append(layer.register_forward_hook(make(li)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False
