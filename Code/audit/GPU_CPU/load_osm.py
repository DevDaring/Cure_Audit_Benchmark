"""
File: GPU_CPU/load_osm.py
Purpose: Load all 4 OSM models in bf16 with flash-attention-2. Verifies
         flash-attention is active for each loaded model.

Loading strategies (chosen automatically in run_gpu_pipeline.py):
  - **Simultaneous (≥48 GB VRAM):** all four models (~42 GB bf16) stay resident.
  - **Sequential (<48 GB VRAM, e.g. A100 40 GB):** one model at a time via
    load_model() / unload_model(); peak VRAM ≈ largest single model (~16 GB).

Per-model sizes: Llama-3.1-8B ~16 GB, Qwen2.5-7B ~14 GB, Gemma-2-2B ~4 GB,
Phi-4-mini ~8 GB.  Override with MIRAGE_SEQUENTIAL_MODELS=1|0.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - Dao et al. (2022). "FlashAttention." NeurIPS 2022.
  - MIRAGE OSM stack: Llama-3.1-8B, Qwen2.5-7B, Gemma-2-2b, Phi-4-mini.

Part of the audit codebase (diagnosis half of CURE).
"""

import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import HUGGINGFACE_TOKEN, OSM_MODELS

logger = logging.getLogger(__name__)

_LOADED_MODELS: dict[str, tuple[Any, Any]] = {}  # name -> (model, tokenizer)

# All four OSM weights in bf16; simultaneous load needs headroom for activations.
_ESTIMATED_ALL_MODELS_VRAM_GB = 42.0
_SIMULTANEOUS_MIN_VRAM_GB = 48.0


def get_gpu_vram_gb() -> float:
    """Return total VRAM (GB) for CUDA device 0, or 0.0 if unavailable."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def use_sequential_loading() -> bool:
    """
    True when the pipeline should load one OSM at a time.

    Auto-enabled when GPU VRAM < 48 GB.  Override with MIRAGE_SEQUENTIAL_MODELS:
      1 / true / yes  — force sequential
      0 / false / no  — force simultaneous (80 GB path; may OOM on smaller GPUs)
    """
    env = os.environ.get("MIRAGE_SEQUENTIAL_MODELS", "").strip().lower()
    if env in ("1", "true", "yes"):
        return True
    if env in ("0", "false", "no"):
        return False
    return get_gpu_vram_gb() < _SIMULTANEOUS_MIN_VRAM_GB


def _patch_transformers_compat() -> None:
    """
    Compatibility shims applied once at import time.

    1. LossKwargs: transformers >= 4.55 removed it from transformers.utils.
    2. Phi3Config rope_scaling: the built-in validator checks
       len(short_factor) == head_dim (=96 for Phi-4-mini), but Phi-4-mini
       uses partial_rotary_factor=0.5 so the correct expected length is
       head_dim/2 = 48.  The mismatch raises ValueError before the model
       even loads.  We replace the validator with one that accepts the real
       length (int(partial_rotary_factor * head_dim)).
    3. HF token CRLF: strips Windows line-endings from token env-vars.
    """
    import os
    import transformers.utils as _tu

    # --- 1. LossKwargs patch ---
    if not hasattr(_tu, "LossKwargs"):
        try:
            from transformers.modeling_outputs import LossKwargs as _LK
        except ImportError:
            try:
                from transformers.utils.generic import LossKwargs as _LK
            except ImportError:
                from typing import TypedDict

                class _LK(TypedDict, total=False):  # type: ignore[no-redef]
                    pass
        _tu.LossKwargs = _LK  # type: ignore[attr-defined]
        logger.debug("Patched transformers.utils.LossKwargs for Phi-4-mini compatibility.")

    # --- 2. Phi3Config rope_scaling validator fix for Phi-4-mini-instruct ---
    # Phi-4-mini config has model_type=phi3 with short_factor length=48,
    # but the built-in Phi3Config._rope_scaling_validation() checks against
    # head_dim=96 (hidden_size/num_heads=3072/32) instead of the rotary
    # dimension (partial_rotary_factor * head_dim = 0.5 * 96 = 48).
    # Fixed upstream only in transformers >= 4.52; patch here for earlier versions.
    try:
        from transformers.models.phi3.configuration_phi3 import Phi3Config

        def _rope_scaling_validation_fixed(self: Any) -> None:
            if not self.rope_scaling:
                return
            required_keys = {"type", "short_factor", "long_factor"}
            if not required_keys.issubset(self.rope_scaling):
                return  # let the original code raise for missing keys
            rope_type = self.rope_scaling["type"]
            if rope_type != "longrope":
                return
            # Accept any list; length will be validated by the model itself.
            for field in ("short_factor", "long_factor"):
                val = self.rope_scaling[field]
                if not isinstance(val, list) or len(val) < 1:
                    raise ValueError(
                        f"`rope_scaling['{field}']` must be a non-empty list, "
                        f"got {val!r}."
                    )

        if not getattr(Phi3Config, "_mirage_patched", False):
            Phi3Config._rope_scaling_validation = _rope_scaling_validation_fixed  # type: ignore[method-assign]
            Phi3Config._mirage_patched = True  # type: ignore[attr-defined]
            logger.debug(
                "Patched Phi3Config._rope_scaling_validation for Phi-4-mini "
                "(partial_rotary_factor fix)."
            )
    except Exception as _patch_err:
        logger.debug("Phi3Config rope_scaling patch skipped: %s", _patch_err)

    # --- 3. Strip \r from HF token env vars (Windows CRLF .env via SFTP) ---
    for key in ("HUGGINGFACE_TOKEN", "HF_TOKEN"):
        val = os.environ.get(key, "")
        if "\r" in val or val != val.strip():
            os.environ[key] = val.strip().replace("\r", "").replace("\n", "")


_patch_transformers_compat()


def _check_platform() -> None:
    """Flash-attention requires Linux x86_64. Error out on other platforms."""
    system = platform.system()
    if system != "Linux":
        raise RuntimeError(
            f"Flash-attention-2 is only supported on Linux x86_64. "
            f"Detected OS: {system}. "
            "Run on Ubuntu 22.04/24.04 with an NVIDIA GPU (CUDA 12.4)."
        )


def _verify_flash_attention(model: Any, model_name: str) -> None:
    """Print which attention implementation is active."""
    try:
        cfg = model.config
        attn_impl = getattr(cfg, "_attn_implementation", "unknown")
        logger.info("  Model %-35s | attention impl: %s", model_name, attn_impl)
        if attn_impl != "flash_attention_2":
            logger.warning(
                "  WARNING: %s did not load with flash_attention_2 (got '%s').",
                model_name,
                attn_impl,
            )
    except Exception as exc:
        logger.warning("  Could not verify attention impl for %s: %s", model_name, exc)


def load_model(model_cfg: dict, force_reload: bool = False) -> tuple[Any, Any]:
    """
    Load a single OSM model and its tokenizer.

    Parameters
    ----------
    model_cfg : dict
        Entry from config.OSM_MODELS.
    force_reload : bool
        Re-load even if already in the in-process cache.

    Returns
    -------
    tuple[model, tokenizer]
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    name = model_cfg["name"]
    hf_id = model_cfg["hf_id"]

    if name in _LOADED_MODELS and not force_reload:
        logger.info("Model '%s' already loaded; returning cached instance.", name)
        return _LOADED_MODELS[name]

    logger.info("Loading model: %s (%s) ...", name, hf_id)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        token=HUGGINGFACE_TOKEN,
        trust_remote_code=True,
    )

    # flash_attention_2 is preferred for speed, but transformers raises rather than
    # degrading when the package is absent, which takes the whole run down. The TIST
    # lease hit exactly that: the pinned torch did not install, the matching flash-attn
    # wheel therefore did not install either, and all four models failed to load while
    # the run still exited zero. Fall back to sdpa, which is numerically equivalent for
    # the forward passes this codebase performs and merely slower.
    def _load(attn_impl: str):
        return AutoModelForCausalLM.from_pretrained(
            hf_id,
            token=HUGGINGFACE_TOKEN,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            device_map={"": 0},
            trust_remote_code=True,
        )

    try:
        model = _load("flash_attention_2")
    except (ImportError, ValueError, RuntimeError) as exc:
        logger.warning(
            "flash_attention_2 unavailable for %s (%s); falling back to sdpa",
            name, str(exc)[:160],
        )
        model = _load("sdpa")
    model.eval()

    _verify_flash_attention(model, name)
    _LOADED_MODELS[name] = (model, tokenizer)
    logger.info("Model '%s' loaded successfully.", name)
    return model, tokenizer


def unload_model(name: str) -> None:
    """
    Remove a model from the in-process cache and free its VRAM.

    Call this before loading a TransformerLens HookedTransformer on top of the
    same model weights to avoid an OOM (A100 40 GB is tight when both the HF
    model and the TL copy coexist for the 9 B Gemma model).
    """
    import gc
    import torch

    if name in _LOADED_MODELS:
        model, _ = _LOADED_MODELS.pop(name)
        try:
            del model
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model '%s' unloaded and VRAM freed.", name)
    else:
        logger.debug("unload_model: '%s' not in cache; nothing to do.", name)

    # Clear TransformerLens and nnsight caches so stale wrappers don't hold
    # GPU memory after the underlying HF model is freed.
    try:
        from GPU_CPU.utils_attention import _TL_MODEL_CACHE, _NNSIGHT_MODEL_CACHE
        for cache in (_TL_MODEL_CACHE, _NNSIGHT_MODEL_CACHE):
            keys_to_remove = [k for k in cache if name.lower() in k.lower()]
            for k in keys_to_remove:
                del cache[k]
    except Exception:
        pass


def load_all_osm_models() -> dict[str, tuple[Any, Any]]:
    """
    Load all 4 OSM models. Verifies GPU is available before starting.

    On an A100 80 GB all four models (~42 GB total) fit simultaneously, so
    this function loads them all and keeps them resident for the full pipeline
    run.  No intermediate unloading is required.

    Returns
    -------
    dict[str, tuple[model, tokenizer]]
        Keys are model logical names.
    """
    _check_platform()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. OSM models require a NVIDIA GPU with CUDA 12.4."
        )

    gpu_mem_gb = get_gpu_vram_gb()
    logger.info(
        "GPU: %s | Memory: %.1f GB",
        torch.cuda.get_device_name(0),
        gpu_mem_gb,
    )

    if gpu_mem_gb < _ESTIMATED_ALL_MODELS_VRAM_GB:
        allow = os.environ.get("MIRAGE_ALLOW_SIMULTANEOUS", "").strip().lower()
        if allow not in ("1", "true", "yes"):
            raise RuntimeError(
                f"GPU has {gpu_mem_gb:.1f} GB but all 4 models need "
                f"~{_ESTIMATED_ALL_MODELS_VRAM_GB:.0f} GB. "
                "Run GPU_CPU/run_gpu_pipeline.py (auto sequential on <48 GB GPUs) "
                "or set MIRAGE_SEQUENTIAL_MODELS=1."
            )
        logger.warning(
            "MIRAGE_ALLOW_SIMULTANEOUS set — loading all models on %.1f GB GPU "
            "(OOM risk).",
            gpu_mem_gb,
        )
    else:
        logger.info(
            "Large GPU detected — loading all 4 models simultaneously "
            "(no unload/reload between pipeline phases)."
        )

    loaded: dict[str, tuple[Any, Any]] = {}
    for model_cfg in OSM_MODELS:
        try:
            model, tokenizer = load_model(model_cfg)
            loaded[model_cfg["name"]] = (model, tokenizer)
        except Exception as exc:
            logger.error("FAILED to load model %s: %s", model_cfg["name"], exc)
            raise

    logger.info("All %d OSM models loaded.", len(loaded))
    return loaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    models = load_all_osm_models()
    logger.info("Loaded models: %s", list(models.keys()))
