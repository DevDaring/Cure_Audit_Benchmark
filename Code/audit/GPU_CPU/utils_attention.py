"""
File: GPU_CPU/utils_attention.py
Purpose: Uniform interface for causal activation patching across
         TransformerLens (Llama, Gemma) and nnsight (Qwen, Phi-4).

TransformerLens note:
  load_osm.py loads plain HF AutoModelForCausalLM objects.  The patching
  functions here convert them to the appropriate patching-library wrapper
  on-demand and cache the result so conversion happens once per model.

nnsight note (v0.6+):
  After the first trace exits, saved proxies must be accessed via .value
  before being used inside a second trace context.  Failing to do so
  raises a RuntimeError in nnsight 0.6+.

Implements / builds on / cites:
  - Meng et al. (2022). "Locating and Editing Factual Associations in GPT."
    NeurIPS 2022. https://arxiv.org/abs/2202.05262 -- activation patching.
  - Pearl (2009). Causality. Cambridge University Press.
    -- do-calculus / interventional framing.
  - TransformerLens: Nanda & Bloom (2022). https://github.com/neelnanda-io/TransformerLens
  - nnsight: Fiotto-Kaufman et al. (2023). https://github.com/ndif-team/nnsight

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache: HF model id -> HookedTransformer
# Populated lazily so we only convert once per model per process.
# ---------------------------------------------------------------------------
_TL_MODEL_CACHE: dict[str, Any] = {}


def _get_token_position(tokenizer: Any, prompt: str, target_token: str) -> int | None:
    """
    Find the token position of *target_token* inside *prompt*.

    Swap tokens in the pentad dataset are stored with underscores
    (e.g. ``a_girl``, ``middle_aged``, ``a_trailer_park``) but the actual
    prompt text uses spaces.  The original single-pass substring search only
    handled single-word tokens (``african``, ``man``, …); it returned None for
    all multi-word tokens, causing ~53 % position-detection failures and
    trivially-zero delta_logit values for those pairs.

    Search strategy (applied in order, first match wins):

    1. **Single-token match** — ``target_text`` (underscores replaced by spaces)
       is a substring of one decoded token.  Handles single-word tokens and
       tokens that the tokenizer keeps together.

    2. **Full-phrase char-level search** — concatenate all individually-decoded
       token strings into one string and look for ``target_text`` as a
       character-level substring.  Map the match character-offset back to the
       owning token index.  Handles multi-word phrases whose words are split
       across consecutive tokens.

    3. **Last-word fallback** — if the full phrase is not found (rare, e.g. due
       to tokenizer-specific spacing), try each word of the phrase in reverse
       order, skipping very short words (≤ 2 chars).  Returns the first
       (leftmost) match.

    Returns the token index (int), or ``None`` if all strategies fail.
    """
    # Normalise: underscores → spaces, strip
    target_text = target_token.lower().replace("_", " ").strip()

    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    token_strs = [tokenizer.decode([t]) for t in tokens]

    # --- Strategy 1: single-token substring match ---
    for i, tok_str in enumerate(token_strs):
        if target_text in tok_str.lower():
            return i

    # --- Strategy 2: char-level search on concatenated decoded string ---
    # Individually-decoded tokens concatenate to the full prompt text for
    # SentencePiece / tiktoken tokenizers (each word token carries its leading
    # space as a prefix byte).
    concat = "".join(tok_str.lower() for tok_str in token_strs)
    char_pos = concat.find(target_text)
    if char_pos != -1:
        cumlen = 0
        for i, tok_str in enumerate(token_strs):
            cumlen += len(tok_str)
            if cumlen > char_pos:
                return i

    # --- Strategy 3: last-word fallback ---
    words = target_text.split()
    # Try from the most-specific (rightmost) word; skip trivially short words.
    for word in reversed(words):
        if len(word) <= 2:
            continue
        char_pos = concat.find(word)
        if char_pos != -1:
            cumlen = 0
            for i, tok_str in enumerate(token_strs):
                cumlen += len(tok_str)
                if cumlen > char_pos:
                    return i

    return None


# ---------------------------------------------------------------------------
# TransformerLens helpers
# ---------------------------------------------------------------------------

def _ensure_hooked_transformer(model: Any, tokenizer: Any) -> Any:
    """
    Return a HookedTransformer wrapping the given model.

    If `model` is already a HookedTransformer, return it unchanged.
    Otherwise create one from the HF model.  The result is cached by HF ID so
    conversion happens at most once per process.

    fold_ln / center_writing_weights are disabled so logits match the HF model
    exactly (required for valid delta_logit comparisons).

    Device strategy
    ---------------
    Some architectures (Gemma-2, Phi-3/4, …) initialise certain internal
    buffers on CPU inside HookedTransformer.from_pretrained(), even when the
    supplied hf_model lives on GPU.  Any tensor operation inside from_pretrained
    that touches one of these CPU buffers alongside a GPU parameter raises
    "Expected all tensors to be on the same device" — **before** we can call
    .to(device) — so the cache is never populated and every call re-attempts
    the conversion.

    Fix: temporarily move the HF model to CPU so that ALL of TL's initialisation
    tensors are consistently on CPU.  After from_pretrained() returns successfully
    we (a) restore the HF model to the original device, (b) move the TL model to
    the original device, (c) deep-scan all sub-module attributes for any stray CPU
    tensors that .to() misses (non-registered plain-attribute tensors).
    """
    try:
        import transformer_lens  # type: ignore
        if isinstance(model, transformer_lens.HookedTransformer):
            return model
    except ImportError as exc:
        raise ImportError(
            "transformer_lens not installed. "
            "Install with: pip install transformer_lens==2.18.0"
        ) from exc

    import torch

    hf_id = getattr(model.config, "_name_or_path", "") or "unknown_model"

    if hf_id in _TL_MODEL_CACHE:
        return _TL_MODEL_CACHE[hf_id]

    logger.info(
        "Converting HF model '%s' to HookedTransformer for CDVA patching ...", hf_id
    )

    target_device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Step 1 — move HF model to CPU so TL init is all-CPU (no device mismatch).
    logger.debug("Temporarily moving HF model to CPU for TL conversion ...")
    model.cpu()
    try:
        tl_model = transformer_lens.HookedTransformer.from_pretrained(
            hf_id,
            hf_model=model,
            dtype=dtype,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
    finally:
        # Step 2 — always restore HF model to GPU, even if from_pretrained fails.
        logger.debug("Restoring HF model to %s ...", target_device)
        model.to(target_device)

    tl_model.eval()

    # Step 3 — move TL model to target device.
    tl_model = tl_model.to(target_device)

    # Step 4 — deep scan: relocate any non-registered plain-attribute tensors
    # that .to() does not reach (e.g. Gemma-2 RoPE sin/cos tables).
    for _module in tl_model.modules():
        for _attr, _val in list(vars(_module).items()):
            if isinstance(_val, torch.Tensor) and _val.device != target_device:
                logger.debug(
                    "Relocating stray CPU tensor %s.%s to %s",
                    type(_module).__name__, _attr, target_device,
                )
                setattr(_module, _attr, _val.to(target_device))

    _TL_MODEL_CACHE[hf_id] = tl_model
    logger.info("HookedTransformer for '%s' cached (device=%s).", hf_id, target_device)
    return tl_model


# ---------------------------------------------------------------------------
# TransformerLens patching (Llama, Gemma)
# ---------------------------------------------------------------------------

def patch_activation_transformer_lens(
    model: Any,
    tokenizer: Any,
    prompt_a: str,
    prompt_b: str,
    position_a: int,
    position_b: int,
    bias_answer: str,
) -> float:
    """
    Causal activation patch using TransformerLens.

    Accepts either a HookedTransformer or a plain HF AutoModelForCausalLM.
    Plain HF models are converted to HookedTransformer on first call and
    cached (see _ensure_hooked_transformer).

    For each layer, replaces the residual-stream activation at position_b
    in prompt_B with the cached activation at position_a from prompt_A.

    Returns
    -------
    float
        delta_logit = logit_patched(bias_answer) - logit_original(bias_answer)

    Implements:
        Pearl (2009) do-calculus intervention:
            do(activation_{L,b} := activation_{L,a})
    """
    import torch

    tl_model = _ensure_hooked_transformer(model, tokenizer)

    # Tokenise
    tokens_b = tl_model.to_tokens(prompt_b)

    # Forward pass on prompt_A to cache residual activations
    with torch.no_grad():
        _, cache_a = tl_model.run_with_cache(prompt_a, return_type=None)

    n_layers = tl_model.cfg.n_layers

    # Build patching hooks: replace resid_post at position_b with cache_a at position_a
    hooks = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache_a:
            continue
        cached_act = cache_a[key][0, position_a, :].clone()

        def make_hook(act: "torch.Tensor") -> Any:
            def hook_fn(value: "torch.Tensor", hook: Any) -> "torch.Tensor":
                # Move cached activation to the same device as the hook target;
                # guards against residual CPU tensors in any architecture.
                value[0, position_b, :] = act.to(value.device)
                return value
            return hook_fn

        hooks.append((key, make_hook(cached_act)))

    # Forward pass on prompt_B with hooks (patched)
    with torch.no_grad():
        logits_patched = tl_model.run_with_hooks(prompt_b, fwd_hooks=hooks)

    # Forward pass on prompt_B without hooks (original)
    with torch.no_grad():
        logits_original = tl_model(prompt_b)

    # Find logit for bias_answer token
    bias_token_ids = tl_model.to_tokens(bias_answer, prepend_bos=False)[0]
    if len(bias_token_ids) == 0:
        logger.warning("Could not tokenise bias_answer '%s'.", bias_answer)
        return 0.0

    bias_tok = bias_token_ids[0].item()
    last_pos = -1
    logit_patched = logits_patched[0, last_pos, bias_tok].item()
    logit_original = logits_original[0, last_pos, bias_tok].item()
    return float(logit_patched - logit_original)


# ---------------------------------------------------------------------------
# nnsight patching (Qwen, Phi-4)
# ---------------------------------------------------------------------------

# Module-level LanguageModel cache for nnsight — prevents re-wrapping the same
# HF model on every seed call (which leaks GPU memory and slows CDVA by ~2×).
_NNSIGHT_MODEL_CACHE: dict[str, Any] = {}


def _ensure_nnsight_model(model: Any, tokenizer: Any) -> Any:
    """Return a cached nnsight LanguageModel wrapper for the given HF model."""
    try:
        from nnsight import LanguageModel  # type: ignore
    except ImportError as exc:
        raise ImportError("nnsight not installed.") from exc

    hf_id = getattr(model.config, "_name_or_path", "") or id(model)
    key = str(hf_id)
    if key not in _NNSIGHT_MODEL_CACHE:
        logger.info("Creating nnsight LanguageModel wrapper for '%s' (cached).", hf_id)
        _NNSIGHT_MODEL_CACHE[key] = LanguageModel(model, tokenizer=tokenizer)
    return _NNSIGHT_MODEL_CACHE[key]


def _nnsight_layer_proxies(nn_model: Any, hf_model: Any) -> tuple[Any, Any]:
    """
    Return (layers_proxy, lm_head_proxy) for the given nnsight LanguageModel.

    HuggingFace CausalLM classes come in two shapes:

      Shape A — layers directly on top-level class (rare):
          hf_model.layers        → decoder stack
          hf_model.lm_head       → vocab projection

      Shape B — layers nested inside an inner .model attribute (Qwen2, Phi3/4,
                LlamaForCausalLM, MistralForCausalLM, GemmaForCausalLM, …):
          hf_model.model.layers  → decoder stack
          hf_model.lm_head       → vocab projection (always at top level)

    We detect the shape from the *real* (non-proxy) HF model and return the
    correct nnsight proxy paths so the trace assignment works for all models.
    """
    inner = getattr(hf_model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        # Shape B: Qwen2ForCausalLM, Phi3ForCausalLM, LlamaForCausalLM, …
        # Use actual HF module references (not nnsight proxy chains).
        # Inside a trace(), nnsight intercepts .output on real nn.Module objects
        # that are part of the traced graph; proxy chains like nn_model.model.model.layers
        # cause AttributeError because nnsight's .model property resolves to the inner
        # model (e.g. Qwen2Model) which has no further .model attribute.
        layers_proxy = inner.layers       # hf_model.model.layers  (nn.ModuleList)
        lm_head_proxy = hf_model.lm_head  # hf_model.lm_head       (nn.Linear)
        logger.debug(
            "nnsight layer path: hf_model.model.layers (inner model) "
            "for %s", type(hf_model).__name__,
        )
    elif hasattr(hf_model, "layers"):
        # Shape A: direct .layers (some GPT-NeoX style models)
        layers_proxy = hf_model.layers
        lm_head_proxy = hf_model.lm_head
        logger.debug(
            "nnsight layer path: hf_model.layers (top-level) "
            "for %s", type(hf_model).__name__,
        )
    else:
        raise AttributeError(
            f"Cannot locate decoder layers in {type(hf_model).__name__}. "
            "Expected .model.layers or .layers on the CausalLM object."
        )
    return layers_proxy, lm_head_proxy


def patch_activation_nnsight(
    model: Any,
    tokenizer: Any,
    prompt_a: str,
    prompt_b: str,
    position_a: int,
    position_b: int,
    bias_answer: str,
) -> float:
    """
    Causal activation patch using nnsight (v0.3.7+).

    Replaces residual stream at (layer, position_b) in prompt_B with the
    cached residual at (layer, position_a) from prompt_A, for every layer.

    Architecture note
    -----------------
    nnsight's `LanguageModel.model` property resolves to the *inner* transformer
    (e.g. `Qwen2Model` for `Qwen2ForCausalLM`), so the correct layer access path
    inside a trace is `nn_model.model.layers[i]`.  Accessing layers via raw HF
    module references (`hf_model.model.layers[i]`) gives a plain `nn.Module`
    object; plain modules do NOT have a `.output` attribute — that only exists on
    nnsight proxy objects returned inside a trace context.

    Practical rule: every `layer.output` or `lm_head.output` access MUST happen
    inside a `with nn_model.trace(...)` block, accessed through `nn_model.*`.

    nnsight proxy note
    ------------------
    After a trace context exits, saved proxies are resolved.  Their tensor value
    is accessible via `.value`.  When assigning a saved activation inside a
    *second* trace context, pass `.value` explicitly; passing the proxy object
    itself raises `RuntimeError`.

    Returns
    -------
    float
        delta_logit = logit_patched(bias_answer) − logit_original(bias_answer)
    """
    import torch

    nn_model = _ensure_nnsight_model(model, tokenizer)

    # Determine layer count from the real HF model (not the nnsight proxy).
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        n_layers = len(inner.layers)       # Shape B: Qwen2ForCausalLM, Phi3ForCausalLM
        _shape_b = True
    elif hasattr(model, "layers"):
        n_layers = len(model.layers)       # Shape A: top-level decoder (rare)
        _shape_b = False
    else:
        raise AttributeError(
            f"Cannot determine n_layers for {type(model).__name__}. "
            "Expected .model.layers or .layers."
        )

    # ------------------------------------------------------------------
    # Pass 1: collect residual activations from prompt_A
    # Layer accesses happen INSIDE the trace through nnsight proxy chain.
    # ------------------------------------------------------------------
    cache_a_proxies: dict[int, Any] = {}
    with nn_model.trace(prompt_a):
        for layer_idx in range(n_layers):
            # Proxy chain inside trace: nn_model.model = inner transformer proxy
            layer = nn_model.model.layers[layer_idx] if _shape_b else nn_model.layers[layer_idx]
            cache_a_proxies[layer_idx] = layer.output[0][:, position_a, :].save()

    # Resolve .value OUTSIDE the trace so we have concrete tensors.
    cache_a_vals: dict[int, "torch.Tensor"] = {
        idx: proxy.value.clone() for idx, proxy in cache_a_proxies.items()
    }

    # ------------------------------------------------------------------
    # Pass 2: patched forward on prompt_B (inject cache_a_vals)
    # lm_head lives on the outer CausalLM; nn_model.lm_head proxies it.
    # ------------------------------------------------------------------
    with nn_model.trace(prompt_b):
        for layer_idx in range(n_layers):
            layer = nn_model.model.layers[layer_idx] if _shape_b else nn_model.layers[layer_idx]
            if layer_idx in cache_a_vals:
                layer.output[0][:, position_b, :] = cache_a_vals[layer_idx]
        patched_logits = nn_model.lm_head.output.save()

    # ------------------------------------------------------------------
    # Pass 3: unpatched forward on prompt_B (baseline)
    # ------------------------------------------------------------------
    with nn_model.trace(prompt_b):
        original_logits = nn_model.lm_head.output.save()

    bias_token_ids = tokenizer.encode(bias_answer, add_special_tokens=False)
    if not bias_token_ids:
        logger.warning("Could not tokenise bias_answer '%s'.", bias_answer)
        return 0.0

    bias_tok = bias_token_ids[0]
    logit_patched = patched_logits.value[0, -1, bias_tok].item()
    logit_original = original_logits.value[0, -1, bias_tok].item()
    return float(logit_patched - logit_original)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def patch_activation(
    model: Any,
    tokenizer: Any,
    prompt_a: str,
    prompt_b: str,
    position_a: int,
    position_b: int,
    bias_answer: str,
    patching_lib: str,
) -> float:
    """
    Dispatch to the appropriate patching library.

    Parameters
    ----------
    model : Any
        Loaded model object (plain HF AutoModelForCausalLM or HookedTransformer).
        For transformer_lens path, the model is auto-converted if needed.
    tokenizer : Any
        Corresponding tokenizer.
    prompt_a : str
        Source prompt (activation source).
    prompt_b : str
        Target prompt (to be patched).
    position_a : int
        Demographic-token position in prompt_A.
    position_b : int
        Demographic-token position in prompt_B.
    bias_answer : str
        The answer token whose logit shift is measured.
    patching_lib : str
        'transformer_lens' or 'nnsight'.

    Returns
    -------
    float
        delta_logit (patched - original).
    """
    if patching_lib == "transformer_lens":
        return patch_activation_transformer_lens(
            model, tokenizer, prompt_a, prompt_b, position_a, position_b, bias_answer
        )
    elif patching_lib == "nnsight":
        return patch_activation_nnsight(
            model, tokenizer, prompt_a, prompt_b, position_a, position_b, bias_answer
        )
    else:
        raise ValueError(
            f"Unknown patching_lib: '{patching_lib}'. "
            "Use 'transformer_lens' or 'nnsight'."
        )
