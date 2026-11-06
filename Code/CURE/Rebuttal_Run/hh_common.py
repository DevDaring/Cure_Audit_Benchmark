"""
hh_common.py -- shared code of the rebuttal cycle of Submission2 (ICLR 2027 author response,
discussion period 5-18 November 2026). Builds on Code/CURE/Next_Run (nr_common, f1_scoring,
f2_runner), Code/CURE/Next2Next_Run (g1_invariant, nn_common) and
Code/CURE/Extended_Research_Codes (p1_intervention hooks, p2 bases, official LEACE). Never
modifies anything under results/v2, results/final_20260914 or results/feedback_20260915.
Every output goes to results/rebuttal_20261107/ with a model suffix in the file name, so two VMs
never write one file.

The reviewers' four empirical objections and the stage that answers each
  site effect confounded with the amount of editing (dfbY W1/Q1, JwYi Q3)
      -> h1_dose_site.py: every site x operator at energies matched PER PROMPT SIDE to the span
         erasure's own energy on that side (levels 0.03, 0.1, 0.3, 1.0 of it); T, accuracy,
         control-set accuracy and the answer-level patching effect at every level
  LEACE / mean difference compared at similar strength (JwYi, uG1a W2)
      -> h2_gender_dose.py: on the gender-flip set, each baseline at its own strength and the
         erasure scaled DOWN per side to that baseline's energy
  results without the largest-|C| pair selection (JwYi)
      -> h3_all_pairs.py: every eligible demographic pair of every pooled test seed, four models
  a prompt rewrite instead of an activation edit (uG1a W3)
      -> h4_rewrite.py: the demographic term neutralised in the text, same readouts
  the behavioural benefit rests on one of two models (JwYi, dfbY W2, uG1a W1)
      -> h5_replication_prep.py + h1 on Qwen2.5-7B and Phi-4-mini: the fresh items are unseen by
         these two models, so their T results are a true out-of-sample replication
  the benefit reverses under length-normalised scoring (JwYi Q2, PAT)
      -> h7_token_scores.py: token-level option scores attribute the change in T

Energy. One number per edited pass: the sum over decoder layers and edited positions of the
squared displacement ||h - h'||^2 recorded by the hook (f2_runner.energy_of, 'num'). The
reference of a side is E_S1(side) = that number for the rank-one span erasure at alpha = 1 on
that side. A cell at level f is solved for alpha such that its energy equals f * E_S1(side)
within MATCH_TOL (5 per cent). A projection cannot exceed its alpha = 1 energy; such a cell is
evaluated at alpha = 1 and flagged feasible = False with its achieved fraction.

Hook ordering in the patching bridge. The edit hooks are registered BEFORE the patch hooks, so
the injected source states win at the patched positions (p1_intervention.commutator's order, and
what Section 3.1 of the manuscript describes). The identity reference then equals the edited
run exactly, so the same-input null is exactly zero for every operator and strength.

Part of the CURE codebase (ICLR 2027, Submission2).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent                      # Code/CURE/Rebuttal_Run
CURE = HERE.parent
NR = CURE / "Next_Run"
NN = CURE / "Next2Next_Run"
EXT = CURE / "Extended_Research_Codes"
for p in (str(NR), str(NN), str(EXT), str(CURE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import nr_common as N          # noqa: E402  (Next_Run)
import common as K             # noqa: E402  (Extended_Research_Codes)

OUT = K.RESULTS / os.environ.get("HH_OUT_DIR", "rebuttal_20261107")
FINAL = N.FINAL                # results/final_20260914 (read only)
V2 = N.V2                      # results/v2 (read only)
FB = K.RESULTS / "feedback_20260915"   # results/feedback_20260915 (read only)

FRESH_MODELS = ["gemma-2-2b-it", "llama-3.1-8b-instruct"]      # have the final cycle's spans, g3 erasers and g5 vectors
REPLICATION_MODELS = ["qwen2.5-7b-instruct", "phi-4-mini-instruct"]   # fresh items unseen; prepared by h5
POOLED_MODELS = list(K.MODELS)
ALL_MODELS = POOLED_MODELS

RANDOM_SEED = 20261107
N_BOOT = N.N_BOOT_FINAL
MATCH_TOL = N.MATCH_TOL        # 0.05
SITES = ("span", "ne", "last", "all")
ROW_KEY = ["model_name", "seed_id", "side", "cond", "variant", "kind"]

log = N.setup_logging("cure.rebuttal")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def utc_now() -> str:
    return N.utc_now()


def write_json(obj, path: Path) -> Path:
    return N.write_json(obj, path)


def read_json(path: Path, default=None):
    return N.read_json(path, default)


def append_rows(path: Path, rows: list[dict]) -> None:
    N.append_rows(path, rows)


def done_keys(path: Path, cols: list[str] = ROW_KEY) -> set:
    return N.done_keys(path, cols)


def read_rows(path: Path) -> pd.DataFrame:
    return N.read_parquet_retry(path) if path.exists() else pd.DataFrame()


def load_model(model_name: str):
    import f2_runner as F2
    return F2.load_for_run(model_name)


def is_fresh(model: str) -> bool:
    return model in FRESH_MODELS


# ---------------------------------------------------------------------------
# data: manifests, spans, bases, vectors
# ---------------------------------------------------------------------------

def fresh_manifest() -> pd.DataFrame:
    return pd.read_csv(FINAL / "source_manifest.csv")


def gender_manifest(model: str) -> pd.DataFrame | None:
    p = FB / ("g3_gender_manifest_%s.csv" % model)
    return pd.read_csv(p) if p.exists() else None


def spans_path(model: str) -> Path:
    p = FINAL / ("spans_%s.json" % model)
    return p if p.exists() else OUT / ("h5_spans_%s.json" % model)


def load_spans(model: str) -> dict:
    p = spans_path(model)
    sp = read_json(p)
    if sp is None:
        raise SystemExit("no spans for %s (run h5_replication_prep.py first)" % model)
    return sp


def target_basis(model: str) -> tuple[dict, dict]:
    """The pooled audit's frozen rank-one contrast-derived basis, reused unchanged."""
    import f2_runner as F2
    return F2.load_target_basis(model)


def random_direction(basis_like: dict) -> dict:
    """The final cycle's fixed random rank-one direction (same seed), one per layer."""
    import f2_runner as F2
    return F2.random_basis(basis_like, N.RANDOM_SEED_FINAL)


def random_subspace(basis_like: dict, rank: int = 64) -> dict:
    """A fixed random orthonormal rank-k subspace per layer. A rank-one random direction removes
    about 1/d of the activation and so cannot reach the erasure's energy at the span; rank 64
    can, which gives the direction control at full strength."""
    out = {}
    for l, rows in basis_like.items():
        d = rows.shape[1]
        g = np.random.default_rng(RANDOM_SEED * 1000 + int(l)).standard_normal((d, rank))
        q, _ = np.linalg.qr(g)
        out[int(l)] = q.T.astype(np.float32)
    return out


def cv_vector(model: str) -> dict | None:
    """The oriented contrast (RepE / ActAdd-style) steering vector: the feedback cycle's for the
    two fresh models, h5's for the two replication models (same rule)."""
    p = FB / ("g5_vectors_%s.npz" % model)
    if not p.exists():
        p = OUT / ("h5_vectors_%s.npz" % model)
    if not p.exists():
        return None
    z = np.load(p)
    return {int(k[3:]): np.asarray(z[k], np.float32) for k in z.files if k.startswith("CV_")}


def leace_erasers(model: str, tag: str) -> dict | None:
    import p2_leace_faithful as LF
    if tag == "f3":
        p = FINAL / ("f3_baseline_%s.npz" % model)
    else:
        p = FB / ("g3_leace_%s_%s.npz" % (model, tag))
    return LF.erasers_from_npz(np.load(p)) if p.exists() else None


def md_basis(model: str, tag: str) -> dict | None:
    """Mean-difference projection bases: 'f3' (man/woman, 45 pairs) or 'gender' (large pool)."""
    p = FINAL / ("f3_baseline_%s.npz" % model) if tag == "f3" else FB / ("g3_md_%s_gender.npz" % model)
    if not p.exists():
        return None
    z = np.load(p)
    out = {int(k[3:]): np.asarray(z[k], np.float32).reshape(1, -1) for k in z.files if k.startswith("md_")}
    return out or None


def positions_for_site(sp_side: dict, site: str) -> list[int]:
    if site == "span":
        return list(sp_side["demo"])
    if site == "ne":
        return list(sp_side.get("ne") or [])
    if site == "last":
        return [int(sp_side["n_prefill"]) - 1]
    if site == "all":
        return list(range(int(sp_side["n_prefill"])))
    raise ValueError(site)


# ---------------------------------------------------------------------------
# the edit: one hook class for both operators and every site
# ---------------------------------------------------------------------------

def _hooked_edit_class():
    import p1_intervention as I
    return I.HookedEdit


class DoseEdit(_hooked_edit_class()):
    """h' = h - alpha U^T U h (mode 'proj') or h' = h + alpha v (mode 'steer') at explicit
    target positions of the prefill; with decode_edit = True (the last-token policy of a
    generation wrapper) every generated token is edited as well. Energy bookkeeping is the
    base class's, so f2_runner.energy_of applies unchanged."""

    def __init__(self, model, rows_by_layer: dict[int, np.ndarray], alpha: float, mode: str = "proj",
                 decode_edit: bool = False, record: bool = True):
        import torch
        rows = {int(l): np.asarray(v, np.float32).reshape(-1, np.asarray(v).shape[-1]) for l, v in rows_by_layer.items()}
        super().__init__(model, rows, alpha=alpha, site="span", record=record)
        if mode not in ("proj", "steer"):
            raise ValueError(mode)
        self.mode = mode
        self.decode_edit = bool(decode_edit)
        if mode == "steer":
            dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
            self.V = {l: torch.as_tensor(rows[l][0], device=dev).to(dt) for l in rows}

    def _positions_for(self, row: int, T: int, prefill: bool) -> list[int]:
        if not prefill:
            return [T - 1] if self.decode_edit else []
        if row >= len(self._targets):
            return []
        off = self._pad[row] if row < len(self._pad) else 0
        return [p + off for p in self._targets[row] if 0 <= p + off < T]

    def _make_hook(self, li: int):
        if self.mode == "proj":
            return super()._make_hook(li)
        v = self.V[li]

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
                cur = hs[b, pos, :]
                delta = (-self.alpha) * v.unsqueeze(0).expand(len(pos), -1)      # h' = h - delta = h + alpha v
                if self.record:
                    with torch.no_grad():
                        dn = delta.float().norm(dim=-1); hn = cur.float().norm(dim=-1)
                        for p, d_, h_ in zip(pos, dn.tolist(), hn.tolist()):
                            self.changed_log.append({"layer": li, "row": b, "position": int(p), "prefill": prefill,
                                                     "delta_norm": d_, "h_norm": h_})
                hs[b, pos, :] = cur - delta
            return output
        return hook


def make_factory(model, op: str, rows: dict, alpha: float, site: str):
    """op: 'proj' (projection of the given rows), 'steer' (additive vector), 'leace' (official
    affine map; rows is the erasers dict). Returns a zero-argument factory of fresh edits."""
    if op == "leace":
        import p2_leace_faithful as LF
        return lambda: LF.LeaceAffineEdit(model, rows, alpha=alpha, site="span")
    # every site is prefill-only, so no site spends energy at decode steps that another does not
    # (the manuscript's last-token wrapper, which also edits generated tokens, is not rerun here)
    return lambda: DoseEdit(model, rows, alpha, mode=("steer" if op == "steer" else "proj"), decode_edit=False)


# ---------------------------------------------------------------------------
# energy and per-side calibration
# ---------------------------------------------------------------------------

def energy_num(edit) -> float:
    import f2_runner as F2
    return float(F2.energy_of(edit)["num"]) if edit is not None else 0.0


def measure_energy(model, tok, text: str, factory, positions: list[int]) -> float:
    """One cheap pass (a single one-character candidate) under the edit; returns its energy."""
    import f1_scoring as S
    e = factory()
    S.score_candidates(model, tok, text, [S.PREFIX + "x" + S.SUFFIX], e, positions, None)
    return energy_num(e)


def calibrate_alpha(model, tok, text: str, positions: list[int], make_edit_at, target: float,
                    e1: float | None = None, max_iter: int = 4, tol: float = MATCH_TOL, capped: bool = True) -> dict:
    """alpha such that the edit's energy on THIS prompt equals `target`. Energy is quadratic in
    alpha for an additive vector and nearly so for a projection through depth, so a square-root
    update converges in one to three evaluations. capped = True for a projection, which cannot
    remove more than its component: alpha stays at or below 1 and a cell that cannot reach the
    target is flagged infeasible with its achieved energy. capped = False for an additive vector,
    which has no such limit (its alpha is of the order of 10). make_edit_at(alpha) -> factory."""
    evals = []
    if e1 is None:
        e1 = measure_energy(model, tok, text, make_edit_at(1.0), positions)
    evals.append({"alpha": 1.0, "energy": e1})
    if target <= 0 or e1 <= 0:
        return {"alpha": 1.0, "achieved": e1, "target": target, "e1": e1, "rel_err": float("nan"), "feasible": False,
                "n_eval": len(evals), "evaluations": evals, "reason": "zero target or zero energy"}
    if capped and e1 <= target * (1.0 + tol):
        feasible = abs(e1 - target) / target <= tol
        return {"alpha": 1.0, "achieved": e1, "target": target, "e1": e1, "rel_err": (e1 - target) / target, "feasible": feasible,
                "n_eval": len(evals), "evaluations": evals, "reason": "" if feasible else "alpha capped at 1"}
    a = float(np.sqrt(target / e1)); best = None
    for _ in range(max_iter):
        en = measure_energy(model, tok, text, make_edit_at(a), positions)
        evals.append({"alpha": a, "energy": en})
        if best is None or abs(en - target) < abs(best[1] - target):
            best = (a, en)
        if abs(en - target) / target <= tol:
            break
        lim = 1.0 if capped else float("inf")
        if en <= 0:
            a = min(lim, a * 2.0)
        else:
            a = float(min(lim, a * np.sqrt(target / en)))
    a, en = best
    return {"alpha": a, "achieved": en, "target": target, "e1": e1, "rel_err": (en - target) / target,
            "feasible": abs(en - target) / target <= tol, "n_eval": len(evals), "evaluations": evals, "reason": ""}


# ---------------------------------------------------------------------------
# readouts: one (seed, side, condition) row; the patching bridge with decoupled sites
# ---------------------------------------------------------------------------

def eval_cell(model, tok, mname: str, r, side: str, sp_side: dict, factory, positions: list[int], meta: dict,
              generate: bool, options: list[str] | None = None, text: str | None = None, prompt: str | None = None) -> dict:
    """f2_runner.eval_side generalised to any edit factory and any position set. `options`,
    `text` and `prompt` override the manifest's (the rewrite baseline passes rewritten ones)."""
    import f1_scoring as S
    import p1_intervention as I
    t0 = time.time()
    up = prompt if prompt is not None else getattr(r, "prompt_%s" % side)
    options = options if options is not None else json.loads(getattr(r, "options_%s" % side))
    gold = int(r.gold_index)
    text = text if text is not None else sp_side["text"]
    cands = S.candidates(options, "text")
    first_ids = S.first_token_ids(tok, text, options)
    row = {"model_name": mname, "seed_id": r.seed_id, "group": r.group, "side": side, "variant": "std", "kind": "score",
           "category": r.category, "template": r.template, "positions": json.dumps(positions), "n_edited_positions": len(positions),
           "gold_index": gold, "options": json.dumps(options), "status": "ok", "utc": utc_now()}
    row.update(meta)
    try:
        e = factory() if factory else None
        sc = S.score_candidates(model, tok, text, cands, e, positions, first_ids)
        q = S.softmax(sc["s"])
        row.update({"s": json.dumps(sc["s"]), "s_mean": json.dumps(sc["s_mean"]), "n_cand_tokens": json.dumps(sc["n_tokens"]),
                    "q": json.dumps(q), "argmax": int(np.argmax(sc["s"])), "score_correct": bool(int(np.argmax(sc["s"])) == gold),
                    "M": S.margin(sc["s"], gold), "first_logits": json.dumps(sc["first_logits"]),
                    "first_disambiguates": S.first_token_disambiguates(first_ids, gold), "T0": sc["T0"]})
        if e is not None:
            import f2_runner as F2
            en = F2.energy_of(e)
            # energy_num sums over the candidate rows of the batch (the final cycle's convention);
            # energy_per_pass is one prompt's energy, the quantity calibrate_alpha matches
            row.update({"energy_num": en["num"], "energy_per_pass": en["num"] / max(1, len(cands)), "energy_den_edited": en["den_edited"],
                        "energy_by_layer": json.dumps(en["by_layer"]), "n_decode_edits_score": en["n_decode_edits"]})
        else:
            row.update({"energy_num": 0.0, "energy_per_pass": 0.0, "energy_den_edited": float("nan"), "energy_by_layer": "{}", "n_decode_edits_score": 0})
        if generate:
            e2 = factory() if factory else None
            gen = I.generate_under_edit(model, tok, [text], e2, [positions], max_new_tokens=N.MAX_NEW_TOKENS)[0]
            gi, why = S.parse_generation(gen, _prompt_with_options(up, options))
            row.update({"gen_raw": gen, "gen_idx": (gi if gi is not None else -1), "gen_parse": why, "gen_valid": gi is not None,
                        "gen_correct": (gi == gold) if gi is not None else False, "n_decode_edits_gen": (e2.n_decode_edits if e2 is not None else 0),
                        "energy_num_gen": energy_num(e2)})
        else:
            row.update({"gen_raw": None, "gen_idx": None, "gen_parse": "not_generated", "gen_valid": None, "gen_correct": None})
    except Exception as ex:                           # a failure is an outcome, never a dropped row
        row.update({"status": "error", "error": str(ex)[:300]})
        log.exception("eval_cell failed %s %s %s", r.seed_id, side, meta.get("cond"))
    row["sec"] = time.time() - t0
    return row


def _prompt_with_options(up: str, options: list[str]) -> str:
    """The parser reads the option list off the prompt; when the options were rewritten the
    prompt must carry the rewritten list."""
    letters = "ABCDEFGH"
    if all(("(%s) %s" % (letters[i], o)) in up for i, o in enumerate(options)):
        return up
    head = up.split("\n(A)")[0]
    tail = up.strip().split("\n")[-1]
    return head + "\n" + "\n".join("(%s) %s" % (letters[i], o) for i, o in enumerate(options)) + "\n" + tail


def score_under_patch(model, tok, text_a: str, text_b: str, patch_a: list[int], patch_b: list[int], cands_b: list[str],
                      factory_a, factory_b, edit_pos_a: list[int], edit_pos_b: list[int], first_ids_b: list[int] | None) -> dict:
    """Candidate scores on b with a's demographic-span states injected at b's aligned span
    positions (sequential source convention), the edit applied at ITS OWN site on each pass
    (edit_pos_a on the source run, edit_pos_b on the b run). Edit hooks are registered first,
    patch hooks second, so the injection wins at the patched positions."""
    import torch
    import p1_intervention as I
    dec = I._decoder_layers(model)
    L = list(range(len(dec)))
    n = min(len(patch_a), len(patch_b))
    if n == 0:
        return {"error": "empty span", "patched_len": 0}
    pa, pb = patch_a[:n], patch_b[:n]
    e_a = factory_a() if factory_a else None
    _, hs_a = I.forward_hidden(model, tok, text_a, e_a, edit_pos_a)
    src = {l: hs_a[l][pa, :].clone() for l in L}
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    base = tok(text_b, add_special_tokens=True)["input_ids"]
    T0 = len(base)
    rows = []
    for c in cands_b:
        ids = tok(text_b + c, add_special_tokens=True)["input_ids"]
        if ids[:T0] != base:
            ids = base + tok(c, add_special_tokens=False)["input_ids"]
        rows.append(ids)
    maxlen = max(len(x) for x in rows)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    input_ids = torch.full((len(rows), maxlen), pad, dtype=torch.long)
    attn = torch.zeros_like(input_ids)
    for i, x in enumerate(rows):
        input_ids[i, :len(x)] = torch.tensor(x); attn[i, :len(x)] = 1
    enc = {"input_ids": input_ids.to(dev), "attention_mask": attn.to(dev)}

    def make_patch(l):
        s = src[l].to(dev).to(dt)

        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] > 1:
                hs[:, pb, :] = s
            return output
        return hook

    e_b = factory_b() if factory_b else None
    if e_b is not None:
        e_b.set_targets([list(edit_pos_b)] * len(rows), [0] * len(rows))
    handles = []
    with torch.no_grad(), (e_b if e_b is not None else I.no_edit()):
        try:
            for l in L:
                handles.append(dec[l].register_forward_hook(make_patch(l)))
            logits = model(**enc).logits.float()
        finally:
            for h in handles:
                h.remove()
    logp = torch.log_softmax(logits, dim=-1)
    sums, means, ntok = [], [], []
    for i, x in enumerate(rows):
        tgt = torch.tensor(x[T0:], device=dev)
        lp = logp[i, T0 - 1:len(x) - 1, :].gather(-1, tgt[:, None]).squeeze(-1)
        sums.append(float(lp.sum().item())); means.append(float(lp.mean().item())); ntok.append(int(len(x) - T0))
    first = [float(logits[0, T0 - 1, t].item()) if (t is not None and t >= 0) else float("nan") for t in (first_ids_b or [])]
    return {"s": sums, "s_mean": means, "n_tokens": ntok, "first_logits": first, "T0": T0, "patched_len": n,
            "span_len_a": len(patch_a), "span_len_b": len(patch_b), "energy_b": energy_num(e_b), "energy_a": energy_num(e_a)}


def bridge_cell(model, tok, mname: str, r, sp: dict, factory_a, factory_b, edit_pos_a: list[int], edit_pos_b: list[int], meta: dict) -> dict:
    """C_answer(a -> b) = M(patched) - M(identity-patched reference), both under the edit."""
    import f1_scoring as S
    t0 = time.time()
    options = json.loads(r.options_B); gold = int(r.gold_index)
    pa, pb = sp["A"]["demo"], sp["B"]["demo"]
    cands = S.candidates(options, "text")
    first_ids = S.first_token_ids(tok, sp["B"]["text"], options)
    row = {"model_name": mname, "seed_id": r.seed_id, "group": r.group, "side": "B", "variant": "std", "kind": "bridge",
           "category": r.category, "template": r.template, "gold_index": gold, "status": "ok", "utc": utc_now()}
    row.update(meta)
    try:
        ref = score_under_patch(model, tok, sp["B"]["text"], sp["B"]["text"], pb, pb, cands, factory_b, factory_b, edit_pos_b, edit_pos_b, first_ids)
        pat = score_under_patch(model, tok, sp["A"]["text"], sp["B"]["text"], pa, pb, cands, factory_a, factory_b, edit_pos_a, edit_pos_b, first_ids)
        if "error" in ref or "error" in pat:
            row.update({"status": "error", "error": pat.get("error") or ref.get("error")})
        else:
            Mc, Mp = S.margin(ref["s"], gold), S.margin(pat["s"], gold)
            row.update({"M_clean": Mc, "M_patched": Mp, "C_answer": Mp - Mc, "C_first": pat["first_logits"][gold] - ref["first_logits"][gold],
                        "first_disambiguates": S.first_token_disambiguates(first_ids, gold), "patched_len": pat["patched_len"],
                        "span_len_a": pat["span_len_a"], "span_len_b": pat["span_len_b"], "s_clean": json.dumps(ref["s"]), "s_patched": json.dumps(pat["s"]),
                        "energy_num": pat["energy_b"], "energy_num_source": pat["energy_a"]})
    except Exception as ex:
        row.update({"status": "error", "error": str(ex)[:300]}); log.exception("bridge_cell failed %s %s", r.seed_id, meta.get("cond"))
    row["sec"] = time.time() - t0
    return row


# ---------------------------------------------------------------------------
# pair-level quantities and statistics (CPU; used by h6_analysis and the checker)
# ---------------------------------------------------------------------------

def tv_from_rows(a, b, col: str = "q") -> float:
    qa = np.asarray(json.loads(a[col]) if isinstance(a[col], str) else a[col], float)
    qb = np.asarray(json.loads(b[col]) if isinstance(b[col], str) else b[col], float)
    return float(0.5 * np.abs(qa - qb).sum())


def softmax_np(s) -> np.ndarray:
    a = np.asarray(s, float); a = a - a.max(); p = np.exp(a); return p / p.sum()


def pair_table(score_rows: pd.DataFrame, extra_cols: tuple = ()) -> pd.DataFrame:
    """One row per (group, seed, cond): T (raw-sum scores), T_mean (length-normalised scores),
    accuracy readouts, energy; both sides must be present and ok."""
    recs = []
    cols = ["group", "seed_id", "cond"]
    for key, g in score_rows[score_rows.status == "ok"].groupby(cols):
        if set(g.side) != {"A", "B"}:
            continue
        a, b = g[g.side == "A"].iloc[0], g[g.side == "B"].iloc[0]
        qa_m, qb_m = softmax_np(json.loads(a.s_mean)), softmax_np(json.loads(b.s_mean))
        rec = {"group": key[0], "seed_id": key[1], "cond": key[2], "template": a.template, "category": a.category,
               "T": tv_from_rows(a, b), "T_mean": float(0.5 * np.abs(qa_m - qb_m).sum()),
               "score_acc": 0.5 * (float(a.score_correct) + float(b.score_correct)),
               "acc": (0.5 * (float(a.gen_correct) + float(b.gen_correct))) if pd.notna(a.get("gen_correct")) and pd.notna(b.get("gen_correct")) else np.nan,
               "both": (float(bool(a.gen_correct) and bool(b.gen_correct))) if pd.notna(a.get("gen_correct")) and pd.notna(b.get("gen_correct")) else np.nan,
               "validity": (0.5 * (float(a.gen_valid) + float(b.gen_valid))) if pd.notna(a.get("gen_valid")) and pd.notna(b.get("gen_valid")) else np.nan,
               "energy": float(a.energy_num) + float(b.energy_num), "energy_A": float(a.energy_num), "energy_B": float(b.energy_num),
               "alpha_A": float(a.get("alpha", np.nan)), "alpha_B": float(b.get("alpha", np.nan)),
               "feasible": bool(a.get("feasible", True)) and bool(b.get("feasible", True))}
        for c in extra_cols:
            rec[c] = a.get(c)
        recs.append(rec)
    return pd.DataFrame(recs)


def paired(diff: np.ndarray, clusters: np.ndarray, alternative: str = "two-sided") -> dict:
    df = pd.DataFrame({"d": np.asarray(diff, float), "c": np.asarray(clusters)}).dropna()
    g = df.groupby("c")["d"].mean().to_numpy()
    t = N.studentized_paired_test(g, n_boot=N_BOOT, seed=RANDOM_SEED, alternative=alternative)
    return {"n_pairs": int(len(df)), "n_clusters": int(len(g)), "estimate": t["mean"], "lo": t["lo"], "hi": t["hi"], "p": t["p"], "degenerate": t["degenerate"]}


def level(vals: np.ndarray, clusters: np.ndarray) -> dict:
    df = pd.DataFrame({"v": np.asarray(vals, float), "c": np.asarray(clusters)}).dropna()
    g = df.groupby("c")["v"].mean().to_numpy()
    est, lo, hi = N.paired_interval(g, n_boot=N_BOOT, seed=RANDOM_SEED)
    return {"n": int(len(df)), "n_clusters": int(len(g)), "estimate": est, "lo": lo, "hi": hi}


# ---------------------------------------------------------------------------
# the prompt-rewrite rule (h4)
# ---------------------------------------------------------------------------

_WITH = re.compile(r"\b(with\s+no|with|dressed\s+in|wearing|in|uses)\s*$", re.I)


def rewrite_rule(option_with_desc: str, desc: str) -> tuple[str, str]:
    """Decide how the descriptor is neutralised, from the option that names the entity.
    Returns (rule, replacement):
      name       the option IS the descriptor (a proper name)              -> 'the person'
      head       the descriptor is the noun phrase after a determiner      -> 'person'
      attribute  the descriptor follows 'with' / 'dressed in' / 'uses'     -> 'an unspecified feature'
      modifier   an adjective, attributive or predicative ('The Jewish man', 'who is slim') -> 'unspecified'
    Every replacement leaves the entity identifiable from the context, so a disambiguated item
    stays answerable (the prompts were audited by reading them)."""
    opt = option_with_desc.strip()
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(desc) + r"(?![A-Za-z0-9])", re.I)
    m = pat.search(opt)
    if m is None:
        return "modifier", "unspecified"
    before, after = opt[:m.start()], opt[m.end():]
    if not before.strip() and not after.strip():
        return "name", "the person"
    if before.strip().lower() in ("the", "a", "an", "my", "our", "their", "his", "her") and not after.strip():
        return "head", "person"
    if _WITH.search(before.strip()):
        return "attribute", "an unspecified feature"
    return "modifier", "unspecified"


def apply_rewrite(text: str, desc: str, rule: str, repl: str, protect: str | None = None) -> str:
    """Neutralise every whole-word occurrence of `desc`. `protect` is the other entity's
    descriptor: a term that contains `desc` ('African American' for 'African') is left intact.
    A replacement is capitalised only at the start of the text, a sentence or an option."""
    word = r"(?<![A-Za-z0-9])%s(?![A-Za-z0-9])"
    keep = {}
    if protect and desc.lower() != protect.lower() and re.search(word % re.escape(desc), protect, flags=re.I):
        def hide(m):
            key = "\x00%d\x00" % len(keep)
            keep[key] = m.group(0)
            return key
        text = re.sub(word % re.escape(protect), hide, text, flags=re.I)

    def rep(m):
        prefix = m.string[:m.start()]
        start = (not prefix.strip()) or bool(re.search(r"(?:[.!?]\s+|\n|\([A-H]\)\s+)$", prefix))
        return repl[0].upper() + repl[1:] if start else repl
    text = re.sub(word % re.escape(desc), rep, text, flags=re.I)
    for key, val in keep.items():
        text = text.replace(key, val)
    text = re.sub(r"\b([Aa]) (unspecified)\b", lambda m: m.group(1) + "n " + m.group(2), text)
    text = re.sub(r"\b([Aa])n (person)\b", lambda m: m.group(1) + " " + m.group(2), text)
    text = re.sub(r"\b[Aa]n? an unspecified feature\b", "an unspecified feature", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()
