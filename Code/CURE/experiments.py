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

def build_subspace(model, tokenizer, cfg, pairs, rank: int) -> dict:
    """Collect per-layer difference vectors from the pairs, return rank-r basis."""
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
    return erase.subspace_from_diffs(diffs, rank)


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

def e4_utility(model, tokenizer, cfg, basis_by_layer, limit: int | None = None) -> pd.DataFrame:
    """Native slot-a accuracy with erasure active, via the audit generation path."""
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

    base = evaluate_osm_model(cfg, model, tokenizer, slot_a, run_id, temperature=0.0, sample_index=0)
    base_acc = _acc(base)
    with erase.ErasureContext(model, basis_by_layer):
        er = evaluate_osm_model(cfg, model, tokenizer, slot_a, run_id, temperature=0.0, sample_index=0)
    er_acc = _acc(er)
    return pd.DataFrame([
        {"model_name": cfg["name"], "erase_rank": len(next(iter(basis_by_layer.values()))) if basis_by_layer else 0,
         "metric": "native_accuracy", "baseline": base_acc, "erased": er_acc,
         "utility_cost": (base_acc - er_acc) if np.isfinite(base_acc) and np.isfinite(er_acc) else float("nan")},
    ])


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
