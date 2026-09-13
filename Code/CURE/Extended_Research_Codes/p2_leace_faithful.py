"""
p2_leace_faithful.py -- LEACE as the authors published it, fitted on the same concept labels
and representation sites as the in-house eraser, applied as its exact affine map.

Purpose (Next_Plan.md, P2 "Minimum conditions", row "Actual LEACE")
  The shipped CURE eraser is an orthogonal projection h' = h - U^T U h onto the complement
  of a centred-difference SVD subspace (Code/CURE/erase.py). LEACE is not that: it is the
  least-squares affine map that makes every linear classifier of the concept chance-level,
  and it is covariance-aware (whitening before the concept direction is found). The plan
  forbids approximating it by orthonormalising its output into the existing basis interface.
  This module therefore provides two things:
    fit_leace / fit_leace_erasers   fit the official EleutherAI eraser per layer on span-
                                    position activations with label = swap side (A=0, B=1)
    LeaceAffineEdit                 a HookedEdit subclass that applies eraser(x) at the span
                                    positions, so the faithful condition uses LEACE's own
                                    affine treatment inside the canonical edit machinery
  and a third for through-depth fidelity:
    fit_leace_sequential            layer l's eraser is fitted on activations that were ALREADY
                                    edited by the erasers of layers < l (concept scrubbing,
                                    Belrose et al. 2023, Section 5). Costs one forward pass per
                                    prompt per layer instead of one per prompt; see the note
                                    on the function.

Concept label and rank
  The concept is binary (which side of the counterfactual swap the activation came from), so
  Z is one real column in {0, 1} and the LEACE cross-covariance has rank at most one. LEACE
  fixes its own erasure rank from the concept; the `rank` argument can only truncate and is
  otherwise recorded, never imposed. Random and SVD controls of "matched rank" therefore match
  the fitted LEACE rank, which p2_controlled_erasure records from `actual_rank`.

What the eraser does (concept_erasure.LeaceEraser.__call__)
  delta = x - bias
  x'    = x - (delta @ proj_right^H) @ proj_left^H
  LeaceAffineEdit scales the removed component by alpha, so alpha = 1 is LEACE exactly and
  alpha in (0, 1) is the same partial-strength convention the projection edits use. The map is
  evaluated in float32 and cast back to the model dtype. The erased subspace reported for the
  basis interface is the column space of proj_left (the null space of the LEACE projection P =
  I - proj_left proj_right), orthonormalised by QR only for reporting, energy bookkeeping and
  the `rank` property; the hook never uses that orthonormal basis.

Requires
  pip install concept-erasure      (https://github.com/EleutherAI/concept-erasure)
  The version actually imported is recorded by leace_version() into every fit record.

Implements
  - Belrose et al. (2023) LEACE: Perfect linear concept erasure in closed form, NeurIPS 2023,
    arXiv:2306.03819. Fitting through LeaceFitter; application through LeaceEraser.__call__;
    sequential fitting is the concept-scrubbing protocol of their Section 5.
  - Builds on p1_intervention.HookedEdit (site and source-state conventions).

Usage (from p2_controlled_erasure.py)
  from p2_leace_faithful import fit_leace, fit_leace_erasers, fit_leace_sequential, \
      LeaceAffineEdit, leace_version
  erasers = fit_leace_erasers(acts_by_layer, labels)           # frozen fitting
  edit = LeaceAffineEdit(model, erasers, alpha=1.0, site="span")

Part of the CURE codebase (ICLR 2027, Submission2). GPU needed only for fit_leace_sequential.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

import p1_intervention as I

log = logging.getLogger("cure.ext.p2.leace")

PIP_HINT = ("concept_erasure is not installed. Run:  pip install concept-erasure  "
            "(official EleutherAI implementation of Belrose et al. 2023, arXiv:2306.03819, "
            "https://github.com/EleutherAI/concept-erasure)")
COLUMN_TOL = 1e-4     # relative column-norm cutoff when reading the erased subspace off proj_left


# ---------------------------------------------------------------------------
# package access
# ---------------------------------------------------------------------------

def _import_ce():
    try:
        import concept_erasure as ce
    except ImportError as exc:
        raise ImportError(PIP_HINT) from exc
    return ce


def leace_available() -> bool:
    try:
        _import_ce()
        return True
    except ImportError:
        return False


def leace_version() -> str:
    """The installed concept-erasure version, for the fit record."""
    ce = _import_ce()
    v = getattr(ce, "__version__", None)
    if v is None:
        try:
            from importlib.metadata import version
            v = version("concept-erasure")
        except Exception:
            v = "unknown"
    return str(v)


# ---------------------------------------------------------------------------
# eraser <-> numpy
# ---------------------------------------------------------------------------

def eraser_to_numpy(eraser, rank: int | None = None) -> dict:
    """{'proj_left': d x k, 'proj_right': k x d, 'bias': d} as float32 numpy. `rank` truncates
    the leading directions if smaller than what LEACE fitted; it never adds any."""
    pl = eraser.proj_left.detach().float().cpu().numpy()
    pr = eraser.proj_right.detach().float().cpu().numpy()
    b = eraser.bias
    bias = b.detach().float().cpu().numpy() if b is not None else np.zeros(pl.shape[0], np.float32)
    fitted = _effective_rank(pl)
    if rank is not None and rank < fitted:
        log.warning("truncating LEACE eraser from rank %d to %d", fitted, rank)
        pl, pr = pl[:, :rank], pr[:rank, :]
    elif rank is not None and rank > fitted:
        log.warning("requested rank %d but LEACE fitted rank %d for this concept; keeping %d",
                    rank, fitted, fitted)
    return {"proj_left": pl.astype(np.float32), "proj_right": pr.astype(np.float32),
            "bias": bias.astype(np.float32)}


def _effective_rank(proj_left: np.ndarray) -> int:
    norms = np.linalg.norm(proj_left, axis=0)
    if norms.size == 0 or norms.max() <= 0:
        return 0
    return int((norms > COLUMN_TOL * norms.max()).sum())


def erased_subspace_rows(proj_left: np.ndarray) -> np.ndarray:
    """Orthonormal rows spanning the column space of proj_left (the null space of P). Used
    only for reporting and for HookedEdit's rank bookkeeping, never inside the hook."""
    norms = np.linalg.norm(proj_left, axis=0)
    keep = norms > COLUMN_TOL * (norms.max() if norms.size else 0.0)
    if not keep.any():
        return np.zeros((0, proj_left.shape[0]), np.float32)
    q, _ = np.linalg.qr(proj_left[:, keep].astype(np.float64))
    return q.T.astype(np.float32)


def erasers_to_npz_dict(erasers: dict[int, dict]) -> dict[str, np.ndarray]:
    out = {}
    for l, e in erasers.items():
        for k in ("proj_left", "proj_right", "bias"):
            out[f"L{int(l)}_{k}"] = e[k]
    return out


def erasers_from_npz(npz) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for key in npz.files:
        if not key.startswith("L") or "_" not in key:
            continue
        l, name = key[1:].split("_", 1)
        out.setdefault(int(l), {})[name] = npz[key]
    return {l: e for l, e in out.items() if {"proj_left", "proj_right", "bias"} <= set(e)}


# ---------------------------------------------------------------------------
# fitting on given activations (frozen protocol)
# ---------------------------------------------------------------------------

def _fitter(ce, d: int, device, shrinkage: bool):
    """Construct LeaceFitter with the documented keyword set; older package versions lack
    `shrinkage`, in which case the accepted keyword set is reduced and logged."""
    import torch
    kw = {"method": "leace", "affine": True, "constrain_cov_trace": True,
          "device": device, "dtype": torch.float32, "shrinkage": shrinkage}
    try:
        return ce.LeaceFitter(d, 1, **kw), kw
    except TypeError:
        kw.pop("shrinkage")
        log.warning("LeaceFitter does not accept shrinkage=; using package defaults")
        return ce.LeaceFitter(d, 1, **kw), kw


def fit_leace_erasers(activations_by_layer: dict[int, np.ndarray], labels: np.ndarray,
                      rank: int | None = None, shrinkage: bool = True,
                      device: str = "cpu") -> dict[int, dict]:
    """Per layer, the LEACE eraser fitted on X (n x d) with binary labels z (n,). Returns
    {layer: {'proj_left', 'proj_right', 'bias', 'actual_rank', 'n', 'fit_kwargs'}}."""
    ce = _import_ce()
    import torch
    z = torch.as_tensor(np.asarray(labels, dtype=np.float32)).reshape(-1, 1)
    if set(np.unique(labels).tolist()) - {0, 1}:
        raise ValueError("labels must be 0 (side A) / 1 (side B)")
    out = {}
    for l, X in activations_by_layer.items():
        X = np.asarray(X, dtype=np.float32)
        if X.shape[0] != z.shape[0]:
            raise ValueError("layer %s: %d rows but %d labels" % (l, X.shape[0], z.shape[0]))
        x = torch.as_tensor(X, device=device)
        fitter, kw = _fitter(ce, x.shape[1], device, shrinkage)
        fitter.update(x, z.to(device))
        rec = eraser_to_numpy(fitter.eraser, rank)
        rec.update({"actual_rank": _effective_rank(rec["proj_left"]), "n": int(X.shape[0]),
                    "fit_kwargs": {k: (str(v) if k in ("device", "dtype") else v) for k, v in kw.items()}})
        out[int(l)] = rec
        del fitter, x
    return out


def fit_leace(activations_by_layer: dict[int, np.ndarray], labels: np.ndarray,
              rank: int | None = None, shrinkage: bool = True,
              device: str = "cpu") -> dict[int, np.ndarray]:
    """Basis-interface view: per layer, orthonormal rows spanning the LEACE erasure subspace
    (the null space of its projection). For reporting and rank matching ONLY; the faithful
    condition applies LeaceAffineEdit, never a projection onto these rows."""
    er = fit_leace_erasers(activations_by_layer, labels, rank, shrinkage, device)
    return {l: erased_subspace_rows(e["proj_left"]) for l, e in er.items()}


# ---------------------------------------------------------------------------
# the affine edit
# ---------------------------------------------------------------------------

class LeaceAffineEdit(I.HookedEdit):
    """HookedEdit that applies the LEACE affine map at the chosen site.

        h' = h - alpha * ((h - bias) @ proj_right^T) @ proj_left^T       for t in S

    alpha = 1 is eraser(h) exactly. Site and target conventions are inherited unchanged
    (span: prefill-only at the demographic span; all: every position). Only the sequential
    source-state convention of p1_intervention.commutator is meaningful for this edit; the
    frozen path projects numerically onto the reported orthonormal rows and would NOT be the
    affine map, so p2 never calls it with source_state='frozen'.

    `erasers_by_layer` values are dicts with numpy 'proj_left' (d x k), 'proj_right' (k x d),
    'bias' (d), as produced by fit_leace_erasers or erasers_from_npz.
    """

    affine = True

    def __init__(self, model, erasers_by_layer: dict[int, dict], alpha: float = 1.0,
                 site: str = "span", record: bool = True):
        import torch
        rows = {l: erased_subspace_rows(e["proj_left"]) for l, e in erasers_by_layer.items()}
        super().__init__(model, rows, alpha=alpha, site=site, record=record)
        dev = next(model.parameters()).device
        self.PL, self.PR, self.bias = {}, {}, {}
        for l, e in erasers_by_layer.items():
            l = int(l)
            if l not in self.U:
                continue
            self.PL[l] = torch.as_tensor(np.asarray(e["proj_left"], np.float32), device=dev)
            self.PR[l] = torch.as_tensor(np.asarray(e["proj_right"], np.float32), device=dev)
            self.bias[l] = torch.as_tensor(np.asarray(e["bias"], np.float32), device=dev)

    def _make_hook(self, li: int):
        PL, PR, bias = self.PL[li], self.PR[li], self.bias[li]

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
                v = hs[b, pos, :].float()                                   # n x d
                delta = self.alpha * ((v - bias) @ PR.transpose(0, 1)) @ PL.transpose(0, 1)
                if self.record:
                    with torch.no_grad():
                        dn = delta.norm(dim=-1); hn = v.norm(dim=-1)
                        for p, d_, h_ in zip(pos, dn.tolist(), hn.tolist()):
                            self.changed_log.append({"layer": li, "row": b, "position": int(p),
                                                     "prefill": prefill, "delta_norm": d_,
                                                     "h_norm": h_})
                hs[b, pos, :] = (v - delta).to(hs.dtype)
            return output
        return hook

    def summary(self) -> dict:
        s = super().summary()
        s["edit_kind"] = "leace_affine"
        return s


# ---------------------------------------------------------------------------
# sequential (concept-scrubbing) fitting
# ---------------------------------------------------------------------------

def resolve_fit_items(tok, fmt: str, pairs: list[dict], c, system: str | None) -> list[dict]:
    """Both prompts of every fit pair with their span positions, truncated to the shorter
    span so the sites are exactly those of the difference-vector estimators. label 0 = A,
    1 = B."""
    var = {(r["seed_id"], r["subvariant"]): r for _, r in c.iterrows()}
    items, n_skipped = [], 0
    for p in pairs:
        ka, kb = (p["seed_id"], p["subvariant_A"]), (p["seed_id"], p["subvariant_B"])
        if ka not in var or kb not in var:
            n_skipped += 1; continue
        ta = I.build_input(tok, str(var[ka]["prompt_text"]), fmt, system)
        tb = I.build_input(tok, str(var[kb]["prompt_text"]), fmt, system)
        pa = I.resolve_span_positions(tok, ta, str(var[ka].get("swap_token", "")))
        pb = I.resolve_span_positions(tok, tb, str(var[kb].get("swap_token", "")))
        n = min(len(pa), len(pb))
        if n == 0:
            n_skipped += 1; continue
        items.append({"text": ta, "pos": pa[:n], "label": 0, "seed_id": p["seed_id"]})
        items.append({"text": tb, "pos": pb[:n], "label": 1, "seed_id": p["seed_id"]})
    if n_skipped:
        log.info("resolve_fit_items: %d pairs skipped (missing variant or unresolved span)", n_skipped)
    return items


def fit_leace_sequential(model, tok, fmt: str, pairs: list[dict], c, layers: Iterable[int] | None = None,
                         rank: int | None = None, shrinkage: bool = True, system: str | None = None,
                         time_budget_s: float | None = None) -> tuple[dict[int, dict], dict]:
    """Concept scrubbing: fit layer l's eraser on activations ALREADY edited by the erasers of
    layers < l, by running every fit prompt with those erasers active and reading layer l at
    the span positions.

    Cost: len(items) forward passes PER LAYER, i.e. L times the frozen protocol (for 300 pairs
    and 32 layers that is 19,200 prompt forwards instead of 600). `time_budget_s` stops the
    loop cleanly; the returned info records how many layers were fitted, and the caller must
    treat a partial fit as a partial condition.
    """
    import time
    dec = I._decoder_layers(model)
    L = list(layers) if layers is not None else list(range(len(dec)))
    items = resolve_fit_items(tok, fmt, pairs, c, system)
    erasers: dict[int, dict] = {}
    t0, n_forward = time.time(), 0
    for l in L:
        if time_budget_s is not None and time.time() - t0 > time_budget_s:
            log.warning("sequential LEACE: time budget hit before layer %d", l); break
        prev = {k: v for k, v in erasers.items() if k < l}
        edit = LeaceAffineEdit(model, prev, alpha=1.0, site="span", record=False) if prev else None
        acts, labels = [], []
        for it in items:
            _, hs = I.forward_hidden(model, tok, it["text"], edit, it["pos"])
            n_forward += 1
            acts.append(hs[l][it["pos"], :].numpy()); labels += [it["label"]] * len(it["pos"])
        X = np.concatenate(acts, 0)
        dev = str(next(model.parameters()).device)
        erasers[l] = fit_leace_erasers({l: X}, np.asarray(labels), rank, shrinkage, device=dev)[l]
        log.info("sequential LEACE layer %d: n=%d rank=%d (%.0fs, %d forwards so far)",
                 l, X.shape[0], erasers[l]["actual_rank"], time.time() - t0, n_forward)
    info = {"protocol": "sequential", "n_items": len(items), "n_layers_requested": len(L),
            "n_layers_fitted": len(erasers), "complete": len(erasers) == len(L),
            "n_forward_passes": n_forward, "sec": time.time() - t0,
            "concept_erasure_version": leace_version(), "shrinkage": shrinkage}
    return erasers, info


if __name__ == "__main__":
    print(__doc__)
    print("concept_erasure available:", leace_available())
    if leace_available():
        print("version:", leace_version())
