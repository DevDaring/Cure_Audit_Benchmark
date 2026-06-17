"""
baselines.py -- E5 comparative debiasing baselines.

Every baseline runs on the same four open models and three datasets and is scored
with the same metrics as CURE: causal residual removed (re-audit), behavioural gap
closed, native utility, and cross-category spillover. This keeps the head-to-head fair.

Two kinds of baseline are provided:

  In-stack (implemented here, runnable now):
    prompt_debias     a fairness system prompt (Self-Debias style, Schick et al. 2021).
    generic_erase     non-audit-guided LEACE: subspace from generic slot-a activation
                      variance, not from the counterfactual swap. Isolates the value of
                      audit guidance.
    meandiff_steer    contrastive activation steering: subtract the mean-difference
                      vector (ITI / CAA family) instead of projecting it out.

  Adapters (plug the official code or a faithful re-implementation; citations given):
    fairsteer         FairSteer, ACL 2025 Findings, arXiv:2504.14492.
    biasgym           BiasGym BiasScope, arXiv:2508.08855.
    sae_debias        SAE feature ablation, arXiv:2511.00177 / 2505.15038 (needs a public SAE).
    hsal              H-SAL latent concept erasure, arXiv:2606.12088.
    patchscopes       Faithful-Patchscopes, arXiv:2602.00300.
    nofreelunch       Logit Steering / Activation Patching suite, arXiv:2511.18635.

Each adapter raises NotImplementedError with a clear message until its official code
is dropped in; the dry run reports which adapters are wired and which are pending.
"""

import logging

import numpy as np

import config_cure as C
import erase
import experiments as E

log = logging.getLogger("cure.baselines")


# ---------------------------------------------------------------------------
# In-stack baselines
# ---------------------------------------------------------------------------

FAIR_PROMPT = ("Answer only from the evidence in the passage. Do not let the group "
               "identity of any person influence the answer. If the passage does not "
               "say, choose the unknown option.")


def prompt_debias(model, tokenizer, cfg, pairs):
    """Returns a marker; the runner prepends FAIR_PROMPT to the system prompt.
    Prompt-only methods do not change activations, so their erased subspace is empty
    and their causal residual removed is expected to be near zero by construction.
    """
    return {"kind": "prompt", "system_suffix": FAIR_PROMPT, "basis": {}}


def generic_erase(model, tokenizer, cfg, pairs, rank=2):
    """Non-audit-guided LEACE: subspace from the variance of slot-a activations
    (no counterfactual swap), to show that audit guidance matters."""
    import pandas as pd
    pentad = pd.read_parquet(C.PENTAD_PATH)
    sa = pentad[(pentad["slot"] == "a") & (pentad["subvariant"] == "surface")].head(len(pairs) * 2)
    acts = {}
    for _, r in sa.iterrows():
        toks = tokenizer.encode(str(r["prompt_text"]))
        pos = max(0, len(toks) - 1)
        try:
            c = erase.cache_resid(model, tokenizer, str(r["prompt_text"]), pos, cfg["patching_lib"])
        except Exception:
            continue
        for layer, v in c.items():
            acts.setdefault(layer, []).append(v)
    diffs = {layer: [v - np.mean(vs, axis=0) for v in vs] for layer, vs in acts.items() if len(vs) >= 2}
    return {"kind": "erase", "basis": erase.subspace_from_diffs(diffs, rank)}


def meandiff_steer(model, tokenizer, cfg, pairs, rank=1):
    """Contrastive steering (ITI/CAA family): use the rank-1 mean-difference as a
    steering direction. Here represented as a rank-1 erasure subspace for scoring
    parity; a true steering variant subtracts alpha*direction at inference."""
    basis = E.build_subspace(model, tokenizer, cfg, pairs, rank=rank)
    return {"kind": "steer", "basis": basis}


# ---------------------------------------------------------------------------
# Adapters for the six recent published methods
# ---------------------------------------------------------------------------

def _adapter(name, citation):
    def fn(model, tokenizer, cfg, pairs):
        raise NotImplementedError(
            f"{name} baseline not wired. Drop the official implementation here. "
            f"Reference: {citation}. Score it with the same re-audit + utility path.")
    fn.citation = citation
    return fn


fairsteer = _adapter("FairSteer", "arXiv:2504.14492 (ACL 2025 Findings)")
biasgym = _adapter("BiasGym BiasScope", "arXiv:2508.08855")
sae_debias = _adapter("SAE-Debias", "arXiv:2511.00177 / 2505.15038 (requires a public SAE)")
hsal = _adapter("H-SAL", "arXiv:2606.12088")
patchscopes = _adapter("Faithful-Patchscopes", "arXiv:2602.00300")
nofreelunch = _adapter("No Free Lunch suite", "arXiv:2511.18635")


REGISTRY = {
    "prompt_debias": prompt_debias,
    "generic_erase": generic_erase,
    "meandiff_steer": meandiff_steer,
    "fairsteer": fairsteer,
    "biasgym": biasgym,
    "sae_debias": sae_debias,
    "hsal": hsal,
    "patchscopes": patchscopes,
    "nofreelunch": nofreelunch,
}

IN_STACK = ["prompt_debias", "generic_erase", "meandiff_steer"]


def score_baseline(method: str, model, tokenizer, cfg, pairs) -> dict:
    """Run one baseline and score it on the causal axis (residual removed) the same
    way CURE is scored. Adapters that are not wired return status='pending'."""
    fn = REGISTRY[method]
    try:
        built = fn(model, tokenizer, cfg, pairs)
    except NotImplementedError as exc:
        return {"method": method, "model_name": cfg["name"], "status": "pending",
                "note": str(exc)[:160]}
    basis = built.get("basis", {})
    if not basis:
        # prompt-only: no activation change -> causal residual unchanged by construction
        return {"method": method, "model_name": cfg["name"], "status": "ok",
                "causal_residual_removed": 0.0, "kind": built.get("kind")}
    re = E.e3_reaudit(model, tokenizer, cfg, pairs, basis)
    if re.empty:
        return {"method": method, "model_name": cfg["name"], "status": "ok",
                "causal_residual_removed": float("nan"), "kind": built.get("kind")}
    removed = float((re["erased_commutator"] < re["orig_commutator"]).mean())
    return {"method": method, "model_name": cfg["name"], "status": "ok",
            "causal_residual_removed": removed, "kind": built.get("kind"),
            "mean_erased_commutator": float(re["erased_commutator"].mean())}
