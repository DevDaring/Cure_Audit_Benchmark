"""
File: GPU_CPU/cdva_patching.py
Purpose: Causal Discriminative Validity Audit (CDVA) -- activation patching
         for all 4 OSM models across the counterfactual (c) probe variants.

Three fixes from the first review (finding A5):
  (a) bias_answer is now taken per-seed from the gold_answer column of the
      pentad dataset (was hard-coded "Yes", which is never a valid answer for
      most items).
  (b) Pairs where prompt_a == prompt_b are detected and skipped rather than
      yielding a meaningless delta_logit ≈ 0.  This prevents the 583/668
      identical-prompt pairs from inflating CDVA scores.
  (c) Normalisation uses a global calibrated constant (CDVA_GLOBAL_SCALE)
      instead of the within-seed max, which was forcing the largest-delta pair
      of every seed to score exactly 0 regardless of true effect size.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
    -- CDVA discriminative validity test, FM2.
  - Meng et al. (2022). "Locating and Editing Factual Associations in GPT."
    NeurIPS 2022. https://arxiv.org/abs/2202.05262 -- activation patching.
  - Pearl (2009). Causality. Cambridge University Press.
    -- do-calculus / interventional framing.
  - Liu et al. (2026). "DIFFHEADS." AAAI 2026.

Part of the audit codebase (diagnosis half of CURE).
"""

import gc
import itertools
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RESULTS_DIR, ensure_dirs
from GPU_CPU.utils_attention import _get_token_position, patch_activation
from results_utils import dedup_cdva

logger = logging.getLogger(__name__)

_CDVA_PATH = RESULTS_DIR / "cdva_results.parquet"

# Global normalisation constant for delta_logit (replaces within-seed max).
# A delta_logit of this magnitude or greater saturates the [0,1] scale.
# Chosen to cover typical residual-stream scale (≈ 3–15 std-dev units).
CDVA_GLOBAL_SCALE: float = 5.0

# Fallback when swap_token is absent: use position 1 (after BOS) as before,
# but flag the row so downstream analysis can filter these out.
_POSITION_FALLBACK = 1


def _cdva_pair_score(delta_logit: float) -> float:
    """
    CDVA pairwise score: 1 - min(|delta_logit| / CDVA_GLOBAL_SCALE, 1.0)
    Higher = more causally invariant (better).

    Using a global scale rather than within-seed max ensures:
    - Small |delta| relative to the global scale → score ≈ 1.0 (invariant).
    - Large |delta| relative to the global scale → score ≈ 0.0 (sensitive).
    The previous within-seed normalisation always produced score = 0 for the
    pair with the largest delta, regardless of whether that delta was actually
    meaningful (review finding A5-c).
    """
    return float(1.0 - min(abs(delta_logit) / CDVA_GLOBAL_SCALE, 1.0))


def _get_bias_answer(c_variants: pd.DataFrame) -> str:
    """
    Token whose logit delta is measured during CDVA patching.

    Priority:
    1. First word of gold_answer (when not BBQ-style Unknown).
    2. swap_token from slot-c (demographic token under test).
    """
    if "gold_answer" in c_variants.columns:
        gold_vals = c_variants["gold_answer"].dropna().unique()
        if len(gold_vals) > 0:
            gold = str(gold_vals[0]).strip()
            if gold and gold.lower() != "unknown":
                return gold.split()[0] if gold.split() else gold

    if "swap_token" in c_variants.columns:
        for tok in c_variants["swap_token"].dropna().astype(str):
            tok = tok.strip()
            if tok and tok.lower() not in {"none", "nan", ""}:
                return tok.split()[0] if tok.split() else tok
    return ""


def run_cdva_for_seed(
    seed_id: str,
    c_variants: pd.DataFrame,
    model_cfg: dict,
    model: Any,
    tokenizer: Any,
    run_id: str,
) -> list[dict]:
    """
    Run CDVA for all C(5,2)=10 pairwise comparisons of slot-c variants
    for a single seed and model.

    Pairs where prompt_a == prompt_b are skipped (tagged
    skipped_reason="identical_prompts") to avoid spurious delta_logit ≈ 0
    that previously inflated CDVA scores for 87% of seeds.

    Returns
    -------
    list[dict]
        One dict per pair (including skipped pairs with success_flag=False).
    """
    rows: list[dict] = []
    model_name = model_cfg["name"]
    patching_lib = model_cfg["patching_lib"]
    try:
        model_version = model.config._name_or_path
    except Exception:
        model_version = model_cfg["hf_id"]

    variant_list = c_variants.to_dict("records")
    if len(variant_list) < 2:
        logger.warning("Seed %s has fewer than 2 slot-c variants; skipping CDVA.", seed_id)
        return rows

    # Determine bias_answer from gold label (per-seed)
    bias_answer = _get_bias_answer(c_variants)
    if not bias_answer:
        logger.debug("Seed %s: no gold_answer; using empty bias_answer token.", seed_id)

    for va, vb in itertools.combinations(variant_list, 2):
        prompt_a = str(va.get("prompt_text", ""))
        prompt_b = str(vb.get("prompt_text", ""))
        subvariant_a = str(va.get("subvariant", ""))
        subvariant_b = str(vb.get("subvariant", ""))
        swap_a = str(va.get("swap_token", ""))
        swap_b = str(vb.get("swap_token", ""))

        # Skip pairs with identical prompt texts -- they yield delta ≈ 0
        # trivially, which would falsely inflate CDVA "causal invariance".
        if prompt_a.strip() == prompt_b.strip():
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "seed_id": seed_id,
                    "model_name": model_name,
                    "model_version": model_version,
                    "pair_A_subvariant": subvariant_a,
                    "pair_B_subvariant": subvariant_b,
                    "delta_logit": 0.0,
                    "cdva_pair_score": float("nan"),
                    "success_flag": False,
                    "failure_reason": "identical_prompts",
                    "position_fallback_used": False,
                }
            )
            continue

        # Find demographic-token position in each prompt
        pos_a = _get_token_position(tokenizer, prompt_a, swap_a) if swap_a else None
        pos_b = _get_token_position(tokenizer, prompt_b, swap_b) if swap_b else None
        position_fallback = False
        if pos_a is None or pos_b is None:
            pos_a = _POSITION_FALLBACK
            pos_b = _POSITION_FALLBACK
            position_fallback = True
            logger.debug(
                "Seed %s: swap token not found in prompt; falling back to position %d.",
                seed_id, _POSITION_FALLBACK,
            )

        success_flag = True
        failure_reason = ""
        delta_logit = 0.0

        if bias_answer:
            try:
                delta_logit = patch_activation(
                    model, tokenizer,
                    prompt_a, prompt_b,
                    pos_a, pos_b,
                    bias_answer,
                    patching_lib,
                )
            except Exception as exc:
                logger.warning(
                    "Patching failed for seed %s, pair (%s, %s): %s",
                    seed_id, subvariant_a, subvariant_b, exc,
                )
                success_flag = False
                failure_reason = str(exc)
        else:
            # No bias_answer token available; record the pair but mark as
            # skipped so downstream code can report coverage.
            success_flag = False
            failure_reason = "no_bias_answer_token"

        rows.append(
            {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seed_id": seed_id,
                "model_name": model_name,
                "model_version": model_version,
                "pair_A_subvariant": subvariant_a,
                "pair_B_subvariant": subvariant_b,
                "delta_logit": delta_logit,
                "cdva_pair_score": _cdva_pair_score(delta_logit) if success_flag else float("nan"),
                "success_flag": success_flag,
                "failure_reason": failure_reason,
                "position_fallback_used": position_fallback,
            }
        )

    return rows


def compute_cdva_seed_score(pair_rows: list[dict]) -> float:
    """Mean cdva_pair_score across successful, non-identical-prompt pairs."""
    scores = [
        r["cdva_pair_score"]
        for r in pair_rows
        if r["success_flag"] and r.get("failure_reason", "") == ""
        and not (isinstance(r["cdva_pair_score"], float) and r["cdva_pair_score"] != r["cdva_pair_score"])
    ]
    return float(sum(scores) / len(scores)) if scores else float("nan")


def run_cdva(
    pentad_df: pd.DataFrame,
    models: dict[str, tuple[Any, Any]],
    run_id: str,
) -> pd.DataFrame:
    """
    Run CDVA for all seeds and all OSM models.
    Writes incremental results to cdva_results.parquet.

    Returns
    -------
    pd.DataFrame
    """
    ensure_dirs()

    if _CDVA_PATH.exists():
        existing = dedup_cdva(pd.read_parquet(_CDVA_PATH))
        if len(existing) > 0:
            existing.to_parquet(_CDVA_PATH, index=False)
        logger.info("Loaded %d existing CDVA results (deduped).", len(existing))
    else:
        existing = pd.DataFrame()

    completed: set[tuple] = set()
    if len(existing) > 0 and "success_flag" in existing.columns:
        for _, row in existing[existing["success_flag"] == True].iterrows():  # noqa: E712
            completed.add((row["seed_id"], row["model_name"]))

    c_variants = pentad_df[pentad_df["slot"] == "c"].copy()
    seed_ids = c_variants["seed_id"].unique().tolist()

    all_rows: list[dict] = []
    if len(existing) > 0:
        all_rows.extend(existing.to_dict("records"))

    for model_cfg in OSM_MODELS:
        model_name = model_cfg["name"]
        patching_lib = model_cfg["patching_lib"]

        if model_name not in models:
            logger.warning("Model '%s' not loaded, skipping CDVA.", model_name)
            continue

        model, tokenizer = models[model_name]

        # For TransformerLens patching (Llama, Gemma) the HF model is converted
        # to a HookedTransformer inside utils_attention.py.  The TL copy adds
        # roughly the same amount of VRAM as the HF model itself.
        #
        # On an A100 80 GB (all 4 models loaded, ~42 GB used, ~38 GB free):
        #   - Llama 8 B  (~16 GB): TL copy fits in headroom → no unload needed.
        #   - Gemma 2 B  (~4 GB):  TL copy fits easily.
        #
        # On a 40 GB A100 (single model loaded, ~16 GB used, ~24 GB free):
        #   TL coexistence usually fits; unload HF first if free_gb < model_param_gb.
        #
        # The threshold was previously 1.5× which triggered unnecessarily on 80 GB.
        tl_unloaded = False
        if patching_lib == "transformer_lens":
            from GPU_CPU.load_osm import unload_model, load_model as _reload
            free_gb = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()
            ) / (1024 ** 3)
            model_param_gb = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / (1024 ** 3)
            # Unload only when free VRAM is genuinely insufficient for TL coexistence.
            # Use 1.0× margin (down from 1.5×) so 80 GB GPUs never trigger unload.
            if free_gb < model_param_gb * 1.0:
                logger.info(
                    "CDVA: freeing HF model '%s' (%.1f GB) before TL conversion "
                    "(only %.1f GB VRAM free).",
                    model_name, model_param_gb, free_gb,
                )
                unload_model(model_name)
                del model
                gc.collect()
                torch.cuda.empty_cache()
                tl_unloaded = True
                # Load a fresh reference for the TL converter (uses local HF cache)
                model, tokenizer = _reload(model_cfg)

        logger.info("CDVA: model=%s, %d seeds ...", model_name, len(seed_ids))

        for i, seed_id in enumerate(seed_ids):
            if (seed_id, model_name) in completed:
                continue

            seed_c = c_variants[c_variants["seed_id"] == seed_id]
            try:
                pair_rows = run_cdva_for_seed(seed_id, seed_c, model_cfg, model, tokenizer, run_id)
                all_rows.extend(pair_rows)
            except Exception as exc:
                logger.error("CDVA failed for seed %s, model %s: %s", seed_id, model_name, exc)

            if (i + 1) % 10 == 0:
                df_partial = pd.DataFrame(all_rows)
                df_partial.to_parquet(_CDVA_PATH, index=False)
                logger.info("  CDVA checkpoint: %d seeds done.", i + 1)

        # Restore the HF model in the shared dict if we unloaded it for TL
        if tl_unloaded:
            from GPU_CPU.load_osm import load_model as _reload
            try:
                restored_model, restored_tok = _reload(model_cfg)
                models[model_name] = (restored_model, restored_tok)
                logger.info("HF model '%s' reloaded after CDVA.", model_name)
            except Exception as exc:
                logger.warning(
                    "Could not reload HF model '%s' after CDVA: %s", model_name, exc
                )

    final = pd.DataFrame(all_rows)
    if len(final) > 0:
        final = dedup_cdva(final)
        final.to_parquet(_CDVA_PATH, index=False)

    # Report coverage
    if len(final) > 0:
        total = len(final)
        skipped_identical = (final["failure_reason"] == "identical_prompts").sum()
        successful = final["success_flag"].sum()
        logger.info(
            "CDVA complete. Total pair rows: %d | successful: %d | "
            "skipped (identical prompts): %d",
            total, successful, skipped_identical,
        )

    return final
