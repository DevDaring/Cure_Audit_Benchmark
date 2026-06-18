"""
experiments.py -- CURE experiment drivers (E1, E3, E4, E6).

  E1  build the per-layer bias subspace from the audit's counterfactual pairs.
  E3  re-audit: erased commutator vs the original commutator (cdva_results.delta_logit).
  E4  utility: native accuracy under erasure across erasure rank.
  E6  prognosis: audit score (pre-erasure severity) vs repair effort (min rank to cure).

Reuses the audit pair definitions (Code/audit/results/cdva_results.parquet and the
slot-c pentad prompts), and the audit patching stack via erase.py.
"""

import logging

import numpy as np
import pandas as pd

import config_cure as C
import erase

log = logging.getLogger("cure.exp")


def _load_pairs(cfg: dict, limit: int | None = None) -> list[dict]:
    """Build the per-pair inputs (prompts, positions, bias answer, original commutator)."""
    from utils_attention import _get_token_position
    from cdva_patching import _get_bias_answer

    name = cfg["name"]
    cdva = pd.read_parquet(C.CDVA_PATH)
    cdva = cdva[(cdva["model_name"] == name)
                & (cdva["position_fallback_used"] == False)        # noqa: E712
                & (cdva["success_flag"] == True)].copy()           # noqa: E712
    if limit:
        cdva = cdva.head(limit)

    pentad = pd.read_parquet(C.PENTAD_PATH)
    pc = pentad[pentad["slot"] == "c"]
    look = {(r["seed_id"], r["subvariant"]): (str(r["prompt_text"]), str(r.get("swap_token", "")))
            for _, r in pc.iterrows()}
    bias_by_seed = {sid: _get_bias_answer(g) for sid, g in pc.groupby("seed_id")}

    pairs = []
    for _, pr in cdva.iterrows():
        sid = pr["seed_id"]
        ka = (sid, pr["pair_A_subvariant"]); kb = (sid, pr["pair_B_subvariant"])
        if ka not in look or kb not in look:
            continue
        pa, swa = look[ka]; pb, swb = look[kb]
        bias = bias_by_seed.get(sid, "")
        if not bias or not pa.strip() or not pb.strip() or pa.strip() == pb.strip():
            continue
        # token positions are resolved later (they need the tokenizer) in _positions()
        pairs.append({
            "seed_id": sid, "prompt_a": pa, "prompt_b": pb,
            "swap_a": swa, "swap_b": swb, "bias": bias,
            "subvariant_A": pr["pair_A_subvariant"], "subvariant_B": pr["pair_B_subvariant"],
            "orig_delta": float(pr.get("delta_logit", float("nan"))),
        })
    return pairs


def _positions(tokenizer, pair: dict):
    from utils_attention import _get_token_position
    pa = _get_token_position(tokenizer, pair["prompt_a"], pair["swap_a"]) if pair["swap_a"] else None
    pb = _get_token_position(tokenizer, pair["prompt_b"], pair["swap_b"]) if pair["swap_b"] else None
    return pa, pb


# ---------------------------------------------------------------------------
# E1: subspace
# ---------------------------------------------------------------------------

def collect_diffs(model, tokenizer, cfg, pairs) -> dict:
    """One caching pass over the pairs; returns {layer: list of (a-b) diff vectors}.
    The bias subspace is estimated from these diffs ONCE; all ranks are then slices
    of the same SVD (erase.bases_at_ranks), so activations are not re-extracted per
    rank. A low-rank direction is robust from a few hundred pairs, so this caching
    runs on a bounded subspace-estimation subset (config SUBSPACE_PAIRS)."""
    lib = cfg["patching_lib"]
    diffs = {}
    for pair in pairs:
        pa, pb = _positions(tokenizer, pair)
        if pa is None or pb is None:
            continue
        try:
            ca = erase.cache_resid(model, tokenizer, pair["prompt_a"], pa, lib)
            cb = erase.cache_resid(model, tokenizer, pair["prompt_b"], pb, lib)
        except Exception as exc:
            log.warning("cache failed seed %s: %s", pair["seed_id"], str(exc)[:120])
            continue
        for layer in ca:
            if layer in cb:
                diffs.setdefault(layer, []).append(ca[layer] - cb[layer])
    return diffs


def build_subspace(model, tokenizer, cfg, pairs, rank: int) -> dict:
    """Single-rank basis (used by the dry run and the baselines)."""
    return erase.subspace_from_diffs(collect_diffs(model, tokenizer, cfg, pairs), rank)


# ---------------------------------------------------------------------------
# Independent baseline supervision.
#
# The comparative baselines (E5) must derive their debiasing direction from their OWN
# signal, NEVER from the causal audit (cdva_results). Only CURE uses the audit's
# counterfactual pairs. The baselines instead receive a generic protected-attribute
# contrast built from neutral templates and demographic term pairs -- the standard
# steering-vector / concept-erasure supervision, independent of this audit. This makes
# the head-to-head a fair test: audit-guided (CURE) vs each method's own supervision.
# ---------------------------------------------------------------------------

_DEMOG_TEMPLATES = [
    "{T} walked into the room.",
    "I had a conversation with {T} yesterday.",
    "The committee decided to hire {T}.",
    "Everyone at the party noticed {T}.",
    "{T} gave a presentation at work.",
    "We invited {T} to the meeting.",
    "The doctor examined {T}.",
    "{T} applied for the position.",
    "My neighbour introduced me to {T}.",
    "The article was written about {T}.",
]

# Protected-attribute term pairs spanning the audited categories (gender, race,
# religion, age, disability, sexual orientation, socioeconomic, nationality, appearance).
_DEMOG_PAIRS = [
    ("a man", "a woman"), ("he", "she"), ("the father", "the mother"),
    ("the boy", "the girl"), ("the gentleman", "the lady"),
    ("a White person", "a Black person"), ("a European person", "an African person"),
    ("an Asian person", "a Hispanic person"), ("a Caucasian man", "an African American man"),
    ("a Christian", "a Muslim"), ("a Christian person", "a Jewish person"),
    ("a Hindu person", "a Muslim person"), ("a religious person", "an atheist"),
    ("a Catholic", "a Buddhist"),
    ("a young person", "an elderly person"), ("a teenager", "a senior citizen"),
    ("a young man", "an old man"),
    ("an able-bodied person", "a disabled person"),
    ("a person without a disability", "a person with a disability"),
    ("a healthy person", "a sick person"),
    ("a straight person", "a gay person"), ("a heterosexual man", "a homosexual man"),
    ("a straight woman", "a lesbian woman"),
    ("a rich person", "a poor person"), ("a wealthy person", "a low-income person"),
    ("an upper-class person", "a working-class person"),
    ("an American", "an immigrant"), ("a local", "a foreigner"),
    ("a thin person", "an overweight person"), ("an attractive person", "an unattractive person"),
    ("a tall person", "a short person"),
]


def collect_acts_demographic(model, tokenizer, cfg) -> dict:
    """Independent demographic-contrast cache (templates x term pairs), used to supervise
    the comparative baselines WITHOUT touching the causal audit. Returns the same shape as
    collect_acts: {"A","B","diffs"} at the protected (last) position."""
    lib = cfg["patching_lib"]
    A, B, diffs = {}, {}, {}
    for ta, tb in _DEMOG_PAIRS:
        for tmpl in _DEMOG_TEMPLATES:
            pa = tmpl.format(T=ta); pb = tmpl.format(T=tb)
            posa = max(0, len(tokenizer.encode(pa)) - 1)
            posb = max(0, len(tokenizer.encode(pb)) - 1)
            try:
                ca = erase.cache_resid(model, tokenizer, pa, posa, lib)
                cb = erase.cache_resid(model, tokenizer, pb, posb, lib)
            except Exception as exc:
                log.warning("demographic cache failed (%s|%s): %s", ta, tb, str(exc)[:80])
                continue
            for layer in ca:
                if layer in cb:
                    A.setdefault(layer, []).append(ca[layer])
                    B.setdefault(layer, []).append(cb[layer])
                    diffs.setdefault(layer, []).append(ca[layer] - cb[layer])
    return {"A": A, "B": B, "diffs": diffs}


def demographic_logit_dirs(model, tokenizer):
    """Unembedding-space steering direction from the demographic term pairs (independent
    of the audit). Mean of (group-A token - group-B token) unembedding rows, normalised."""
    head = getattr(model, "lm_head", None)
    W = head.weight if head is not None else model.get_output_embeddings().weight
    W = W.detach().float().cpu().numpy()
    rows = []
    for ta, tb in _DEMOG_PAIRS:
        ia = tokenizer.encode(" " + ta.split()[-1], add_special_tokens=False)
        ib = tokenizer.encode(" " + tb.split()[-1], add_special_tokens=False)
        if ia and ib and ia[0] < W.shape[0] and ib[0] < W.shape[0]:
            rows.append(W[ia[0]] - W[ib[0]])
    if not rows:
        return None
    u = np.mean(np.stack(rows, 0), axis=0)
    n = float(np.linalg.norm(u))
    return (u / n).astype(np.float32) if n > 1e-8 else None


def collect_acts(model, tokenizer, cfg, pairs) -> dict:
    """One caching pass returning slot-a and slot-b activations AND their diffs.

    Returns {"A": {layer: [vec]}, "B": {layer: [vec]}, "diffs": {layer: [a-b]}}.
    The eight comparative baselines (E5) share this single pass so the protected-
    position activations are extracted ONCE per model, not once per baseline.
    """
    lib = cfg["patching_lib"]
    A, B, diffs = {}, {}, {}
    for pair in pairs:
        pa, pb = _positions(tokenizer, pair)
        if pa is None or pb is None:
            continue
        try:
            ca = erase.cache_resid(model, tokenizer, pair["prompt_a"], pa, lib)
            cb = erase.cache_resid(model, tokenizer, pair["prompt_b"], pb, lib)
        except Exception as exc:
            log.warning("collect_acts cache failed seed %s: %s", pair["seed_id"], str(exc)[:120])
            continue
        for layer in ca:
            if layer in cb:
                A.setdefault(layer, []).append(ca[layer])
                B.setdefault(layer, []).append(cb[layer])
                diffs.setdefault(layer, []).append(ca[layer] - cb[layer])
    return {"A": A, "B": B, "diffs": diffs}


def stratified_subset(pairs, n, seed):
    """Fixed-seed subset stratified by benchmark (the seed_id prefix bbq/crows/stereo).
    Keeps the same benchmark mix as the full set, so a sweep on the subset stays
    representative and statistically sound."""
    import random
    if not n or n >= len(pairs):
        return list(pairs)
    rng = random.Random(seed)
    by_src = {}
    for p in pairs:
        src = str(p["seed_id"]).split("_", 1)[0]
        by_src.setdefault(src, []).append(p)
    out = []
    total = len(pairs)
    for grp in by_src.values():
        k = max(1, round(n * len(grp) / total))
        rng.shuffle(grp)
        out.extend(grp[:k])
    rng.shuffle(out)
    return out[:n]


def head_per_dataset(pairs, k):
    """Up to k pairs from each benchmark (bbq, crows_pairs, stereoset). Used by the dry
    run so it exercises two instances of EVERY dataset, not just the first two pairs."""
    seen, out = {}, []
    for p in pairs:
        src = str(p["seed_id"]).split("_", 1)[0]
        if seen.get(src, 0) < k:
            out.append(p)
            seen[src] = seen.get(src, 0) + 1
    return out


# ---------------------------------------------------------------------------
# E3: re-audit (erased commutator)
# ---------------------------------------------------------------------------

def e3_reaudit(model, tokenizer, cfg, pairs, basis_by_layer) -> pd.DataFrame:
    lib = cfg["patching_lib"]
    rows = []
    for pair in pairs:
        pa, pb = _positions(tokenizer, pair)
        if pa is None or pb is None:
            continue
        try:
            erased = erase.erased_commutator(model, tokenizer, pair["prompt_a"], pair["prompt_b"],
                                             pa, pb, pair["bias"], basis_by_layer, lib)
        except Exception as exc:
            log.warning("erased commutator failed seed %s: %s", pair["seed_id"], str(exc)[:120])
            continue
        orig = abs(pair["orig_delta"]) if np.isfinite(pair["orig_delta"]) else float("nan")
        rows.append({
            "model_name": cfg["name"], "seed_id": pair["seed_id"],
            "subvariant_A": pair["subvariant_A"], "subvariant_B": pair["subvariant_B"],
            "condition": "erased", "orig_commutator": orig,
            "erased_commutator": abs(erased),
            "cured": bool(abs(erased) <= C.TAU < orig) if np.isfinite(orig) else False,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E4: utility under erasure
# ---------------------------------------------------------------------------

def native_accuracy(model, tokenizer, cfg, basis_by_layer=None,
                    limit: int | None = None, max_tokens: int = 64) -> float:
    """Native slot-a accuracy, optionally with erasure active (via the audit path).
    Short generations (max_tokens default 64) suffice for the option answer, which is
    the safe expedite here. Passing basis_by_layer=None gives the un-erased baseline,
    computed once per model and reused across ranks by the caller."""
    from osm_behavioral import evaluate_osm_model
    import uuid

    pentad = pd.read_parquet(C.PENTAD_PATH)
    slot_a = pentad[(pentad["slot"] == "a") & (pentad["subvariant"] == "surface")].copy()
    slot_a = slot_a[slot_a["prompt_text"].astype(str).str.strip() != ""]
    if limit:
        slot_a = slot_a.head(limit)
    run_id = f"e4-{uuid.uuid4().hex[:8]}"

    def _acc(df):
        ok = df[df["success_flag"] == True]                  # noqa: E712
        if ok.empty:
            return float("nan")
        m = ok.apply(lambda r: str(r["parsed_answer"]).strip().lower() in
                     str(r["gold_answer"]).strip().lower() or
                     str(r["gold_answer"]).strip().lower() in str(r["parsed_answer"]).strip().lower(),
                     axis=1)
        return float(m.mean())

    if basis_by_layer:
        with erase.ErasureContext(model, basis_by_layer):
            df = evaluate_osm_model(cfg, model, tokenizer, slot_a, run_id,
                                    temperature=0.0, sample_index=0, max_tokens=max_tokens)
    else:
        df = evaluate_osm_model(cfg, model, tokenizer, slot_a, run_id,
                                temperature=0.0, sample_index=0, max_tokens=max_tokens)
    return _acc(df)


# ---------------------------------------------------------------------------
# E6: prognosis (audit score vs repair effort)
# ---------------------------------------------------------------------------

def e6_prognosis(per_rank_reaudit: dict, utility_by_rank: dict, model_name: str) -> tuple[pd.DataFrame, dict]:
    """For each item, the audit score is the original severity; the repair effort is
    the smallest rank at which the erased commutator drops to tau or below, with the
    utility cost at that rank. Fit the relation.

    per_rank_reaudit: {rank: e3 DataFrame}. utility_by_rank: {rank: utility_cost}.
    """
    ranks = sorted(per_rank_reaudit)
    base = per_rank_reaudit[ranks[0]][["seed_id", "orig_commutator"]].drop_duplicates("seed_id")
    eff = {}
    for sid in base["seed_id"]:
        min_rank = None
        for r in ranks:
            d = per_rank_reaudit[r]
            row = d[d["seed_id"] == sid]
            if len(row) and float(row["erased_commutator"].iloc[0]) <= C.TAU:
                min_rank = r
                break
        eff[sid] = min_rank if min_rank is not None else (ranks[-1] + 1)
    rows = []
    for _, b in base.iterrows():
        sid = b["seed_id"]; score = float(b["orig_commutator"])
        if not np.isfinite(score):
            continue
        rows.append({"model_name": model_name, "seed_id": sid, "audit_score": score,
                     "repair_rank": eff[sid],
                     "utility_cost": float(utility_by_rank.get(eff[sid], float("nan")))})
    df = pd.DataFrame(rows)
    fit = {"model_name": model_name, "n": int(len(df))}
    if len(df) >= 5 and df["repair_rank"].nunique() > 1:
        try:
            from scipy.stats import spearmanr, pearsonr
            sp = spearmanr(df["audit_score"], df["repair_rank"])
            pe = pearsonr(df["audit_score"], df["repair_rank"])
            slope, intercept = np.polyfit(df["audit_score"], df["repair_rank"], 1)
            yhat = slope * df["audit_score"] + intercept
            ss_res = float(((df["repair_rank"] - yhat) ** 2).sum())
            ss_tot = float(((df["repair_rank"] - df["repair_rank"].mean()) ** 2).sum())
            fit.update({"spearman": float(sp.correlation), "spearman_p": float(sp.pvalue),
                        "pearson": float(pe[0]), "pearson_p": float(pe[1]),
                        "slope": float(slope), "intercept": float(intercept),
                        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")})
        except Exception as exc:
            fit["error"] = str(exc)[:160]
    return df, fit
