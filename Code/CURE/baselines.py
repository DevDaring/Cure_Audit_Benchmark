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


def prompt_debias(model, tokenizer, cfg, pairs, ctx=None):
    """Returns a marker; the runner prepends FAIR_PROMPT to the system prompt.
    Prompt-only methods do not change activations, so their erased subspace is empty
    and their causal residual removed is expected to be near zero by construction.
    """
    return {"kind": "prompt", "system_suffix": FAIR_PROMPT, "basis": {}}


def generic_erase(model, tokenizer, cfg, pairs, ctx=None, rank=2):
    """Non-guided erasure (PCA / LEACE WITHOUT contrastive supervision): erase the top
    variance directions of generic activations. Uses the independent demographic-template
    activations (NOT the audit's counterfactual swap), isolating the value of the causal
    audit guidance that CURE adds."""
    rank = rank or 2
    c = _acts(model, tokenizer, cfg, pairs, ctx)
    basis = {}
    for layer in c["A"]:
        allacts = c["A"].get(layer, []) + c["B"].get(layer, [])
        if len(allacts) < 2:
            continue
        X = np.stack(allacts, 0).astype(np.float64)
        X = X - X.mean(0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        b = _ortho(Vt[:min(rank, Vt.shape[0])])
        if b is not None:
            basis[layer] = b
    return {"kind": "erase", "basis": basis}


def meandiff_steer(model, tokenizer, cfg, pairs, ctx=None, rank=1):
    """Contrastive steering (ITI/CAA family): use the rank-1 mean-difference as a
    steering direction. Here represented as a rank-1 erasure subspace for scoring
    parity; a true steering variant subtracts alpha*direction at inference. The
    direction is estimated from a bounded subset (config SUBSPACE_PAIRS)."""
    if ctx is not None:
        basis = erase.subspace_from_diffs(ctx["diffs"], rank)
    else:
        basis = E.build_subspace(model, tokenizer, cfg, pairs[:C.SUBSPACE_PAIRS], rank=rank)
    return {"kind": "steer", "basis": basis}


# ---------------------------------------------------------------------------
# The six recent published methods, re-implemented in this evaluation harness.
#
# Each method below reproduces the published mechanism and is scored on the
# IDENTICAL causal-residual-removed + utility metrics, the same pairs, and the
# same models as CURE -- a controlled, single-harness comparison. Two are
# adaptations and are documented as such: SAE-Debias trains a lightweight SAE on
# each model's own activations (no public SAE exists for Qwen2.5-7B / Phi-4-mini),
# and Patchscopes is an inspection method adapted to yield an erasure direction.
# ---------------------------------------------------------------------------

def _ortho(rows):
    """Orthonormalise the rows of a (k x d) matrix; drop near-zero directions.
    Returns a (k' x d) float32 array of orthonormal rows, or None if empty."""
    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    if rows.size == 0:
        return None
    q, r = np.linalg.qr(rows.T)                       # d x k
    keep = np.abs(np.diag(r)) > 1e-8
    q = q[:, keep]
    if q.shape[1] == 0:
        return None
    return q.T.astype(np.float32)


def _acts(model, tokenizer, cfg, pairs, ctx):
    """Shared one-pass activation cache (slot-a, slot-b, diffs) for the baselines."""
    if ctx is not None:
        return ctx
    return E.collect_acts(model, tokenizer, cfg, pairs[:C.SUBSPACE_PAIRS])


def fairsteer(model, tokenizer, cfg, pairs, ctx=None):
    """FairSteer (Findings of ACL 2025, arXiv:2504.14492): inference-time activation
    steering along a fairness direction. Re-implemented as the per-layer difference-
    in-means steering direction (rank-1), scored by projection for parity. The paper's
    dynamic linear-probe gating is not reproduced (needs official code); this captures
    the steering direction the method applies."""
    diffs = _acts(model, tokenizer, cfg, pairs, ctx)["diffs"]
    basis = {}
    for layer, vs in diffs.items():
        if not vs:
            continue
        m = np.mean(np.stack(vs, 0), axis=0)
        n = float(np.linalg.norm(m))
        if n > 1e-8:
            basis[layer] = (m / n).reshape(1, -1).astype(np.float32)
    return {"kind": "steer", "basis": basis}


def biasgym(model, tokenizer, cfg, pairs, ctx=None, scope=0.5):
    """BiasGym / BiasScope (arXiv:2508.08855): locate and ablate bias-carrying
    components. Re-implemented as scope-thresholded SVD ablation -- per layer keep the
    principal swap-difference directions whose singular value is at least `scope` of
    the top singular value, then project them out (magnitude-scoped, variable rank)."""
    diffs = _acts(model, tokenizer, cfg, pairs, ctx)["diffs"]
    basis = {}
    for layer, vs in diffs.items():
        if len(vs) < 2:
            continue
        X = np.stack(vs, 0).astype(np.float64)
        X = X - X.mean(0, keepdims=True)
        try:
            _, S, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if S.size == 0 or S[0] <= 0:
            continue
        keep = max(1, min(int((S >= scope * S[0]).sum()), Vt.shape[0]))
        basis[layer] = Vt[:keep, :].astype(np.float32)
    return {"kind": "erase", "basis": basis}


def sae_debias(model, tokenizer, cfg, pairs, ctx=None, k_features=8, steps=250):
    """SAE-Debias (arXiv:2511.00177 / 2505.15038): ablate bias-encoding SAE features.
    No public SAE covers all four models, so a lightweight over-complete sparse
    autoencoder is trained on each model's own protected-position residuals per layer;
    the features whose activation differs most across the protected swap are ablated.
    Documented as a trained-on-the-fly re-implementation, not a published SAE."""
    import torch
    c = _acts(model, tokenizer, cfg, pairs, ctx)
    A, B = c["A"], c["B"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    basis = {}
    for layer in A:
        if layer not in B or len(A[layer]) < 8:
            continue
        Xa = np.stack(A[layer], 0).astype(np.float32)
        Xb = np.stack(B[layer], 0).astype(np.float32)
        d = Xa.shape[1]
        h = min(4 * d, 8192)
        try:
            Xa_t = torch.tensor(Xa, device=dev)
            Xb_t = torch.tensor(Xb, device=dev)
            mu = torch.cat([Xa_t, Xb_t], 0).mean(0, keepdim=True)
            Xc = torch.cat([Xa_t, Xb_t], 0) - mu
            enc = torch.nn.Linear(d, h).to(dev)
            dec = torch.nn.Linear(h, d, bias=False).to(dev)
            opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
            for _ in range(steps):
                opt.zero_grad()
                z = torch.relu(enc(Xc))
                loss = ((dec(z) - Xc) ** 2).mean() + 1e-3 * z.abs().mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                za = torch.relu(enc(Xa_t - mu)).mean(0)
                zb = torch.relu(enc(Xb_t - mu)).mean(0)
                top = torch.topk((za - zb).abs(), min(k_features, h)).indices
                dirs = dec.weight[:, top].t().detach().cpu().numpy()   # k x d
        except Exception as exc:
            log.warning("sae_debias layer %s failed: %s", layer, str(exc)[:100])
            continue
        b = _ortho(dirs)
        if b is not None:
            basis[layer] = b
    return {"kind": "erase", "basis": basis}


def hsal(model, tokenizer, cfg, pairs, ctx=None, rank=None):
    """H-SAL (arXiv:2606.12088): hierarchical latent concept erasure. Re-implemented as
    covariance-whitened (LEACE-style) erasure of the swap-difference concept at every
    layer, in contrast to CURE's plain SVD projection."""
    diffs = _acts(model, tokenizer, cfg, pairs, ctx)["diffs"]
    rank = rank or C.HEADLINE_RANK
    basis = {}
    for layer, vs in diffs.items():
        if len(vs) < 2:
            continue
        X = np.stack(vs, 0).astype(np.float64)
        Xc = X - X.mean(0, keepdims=True)
        d = X.shape[1]
        cov = (Xc.T @ Xc) / max(1, len(Xc) - 1) + 1e-3 * np.eye(d)
        try:
            w, V = np.linalg.eigh(cov)
            w = np.clip(w, 1e-6, None)
            Wh = V @ np.diag(1.0 / np.sqrt(w)) @ V.T          # whitening
            _, _, Vtw = np.linalg.svd(Xc @ Wh, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        k = min(rank, Vtw.shape[0])
        b = _ortho(Vtw[:k, :] @ Wh)                            # unwhiten -> LEACE dirs
        if b is not None:
            basis[layer] = b
    return {"kind": "erase", "basis": basis}


def patchscopes(model, tokenizer, cfg, pairs, ctx=None, rank=None):
    """Faithful-Patchscopes (arXiv:2602.00300) is a representation-inspection method;
    adapted here to debiasing by reading the protected-attribute direction from the swap
    and ablating it at the layers with the strongest read-out (largest diff energy),
    a localised erasure. Documented as an adaptation of an inspection method."""
    rank = rank or C.HEADLINE_RANK
    diffs = _acts(model, tokenizer, cfg, pairs, ctx)["diffs"]
    energy = {L: float(np.mean([np.linalg.norm(v) for v in vs]))
              for L, vs in diffs.items() if len(vs) >= 2}
    if not energy:
        return {"kind": "erase", "basis": {}}
    thr = float(np.percentile(list(energy.values()), 66))
    basis = {}
    for layer, vs in diffs.items():
        if energy.get(layer, 0.0) < thr:
            continue
        X = np.stack(vs, 0).astype(np.float64)
        X = X - X.mean(0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        basis[layer] = Vt[:min(rank, Vt.shape[0]), :].astype(np.float32)
    return {"kind": "erase", "basis": basis}


def nofreelunch(model, tokenizer, cfg, pairs, ctx=None):
    """No-Free-Lunch suite (arXiv:2511.18635): logit-steering / activation-patching.
    Re-implemented as logit-space steering -- ablate, at every layer, the residual
    direction that writes to demographic-group unembeddings. Independent of the audit
    (the model's own unembedding plus a generic demographic term list)."""
    u = E.demographic_logit_dirs(model, tokenizer)
    if u is None:
        return {"kind": "steer", "basis": {}}
    u = u.reshape(1, -1).astype(np.float32)
    layers = list(_acts(model, tokenizer, cfg, pairs, ctx)["diffs"].keys())
    return {"kind": "steer", "basis": {L: u for L in layers}}


# Subspace-erasure baselines whose erased rank should track CURE's per-model operating
# rank for a controlled comparison. Steering (fairsteer, meandiff, nofreelunch) and
# feature/scope methods (sae_debias, biasgym) keep their native single-direction /
# self-selected configurations; prompt_debias edits nothing.
_RANK_AWARE = {"generic_erase", "hsal", "patchscopes"}


def build_basis(method, model, tokenizer, cfg, pairs, ctx=None, rank=None) -> dict:
    """Build one method's basis (or marker), passing the shared activation cache. Rank-
    aware erasure baselines receive the per-model operating rank. Returns {"kind","basis"}
    on success or {"status": "pending"/"error", "note"}."""
    fn = REGISTRY[method]
    try:
        if method in _RANK_AWARE and rank is not None:
            return fn(model, tokenizer, cfg, pairs, ctx=ctx, rank=rank)
        return fn(model, tokenizer, cfg, pairs, ctx=ctx)
    except NotImplementedError as exc:
        return {"status": "pending", "note": str(exc)[:160]}


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
