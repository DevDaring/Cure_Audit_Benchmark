"""
run_scope.py -- single entry point for the whole SCOPE study. One launch produces every
SCOPE artifact; the audit and the eight baselines are reused from the CURE stack.

  python3 run_scope.py --mode dry    validate the full SCOPE path on two pairs per model.
  python3 run_scope.py --mode main   localise + edit + verify, then head-to-head vs the
                                      nine baselines (incl. Faithful-Patchscopes), held-out
                                      + behavioural, and the design ablations, for all four
                                      open models, with 15-minute GitHub checkpoints.

Outputs (Code/CURE/results/, so the user takes audit + SCOPE + study from one folder):
  scope_localization_<model>.json   decodability map, localised layers, verification drop
  scope_final_<model>.parquet       SCOPE vs nine baselines on the shared eval set
  scope_extra_<model>.parquet       held-out bias removed + behavioural (acc, flip)
  scope_ablation_<model>.parquet    localised vs all-layer, protect vs no-protect, random
  scope_prognosis_<model>.{parquet,json}
  SCOPE_DONE

Nothing is hard-coded to win. Every number is produced by the same CURE evaluators
(e3_reaudit, native_accuracy, behavioural_readout) on the same pairs as CURE used.
"""

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

import config_scope as C
import scope as S

# Reused verbatim from the CURE stack.
import integrity
import erase
import experiments as E
import baselines as B
from run_tacl_extra import split_by_seed, behavioural_readout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(C.LOGS / "run_scope.log")],
)
log = logging.getLogger("scope.run")

# The baselines worth a held-out + behavioural head-to-head (the two strongest utility
# keepers plus Faithful-Patchscopes, the closest prior method SCOPE must beat).
BEHAV_BASELINES = ["generic_erase", "sae_debias", "patchscopes"]


def _save(df, name):
    if df is not None and len(df):
        df.to_parquet(C.RESULTS / name, index=False)


def _op_rank(name: str) -> int:
    """Baseline operating rank from the CURE run, so the baseline numbers match cure_final."""
    p = C.RESULTS / f"cure_rankcurve_{name}.json"
    if p.exists():
        try:
            return int(json.loads(p.read_text())["operating_rank"])
        except Exception:
            pass
    return 4


def _crr(re_df) -> tuple:
    if not len(re_df):
        return float("nan"), float("nan"), 0
    crr = float((re_df["erased_commutator"] < re_df["orig_commutator"]).mean())
    return crr, float(re_df["erased_commutator"].mean()), int(len(re_df))


def select_scope_basis(model, tok, cfg, train_pairs, decod, base_acc, dry=False):
    """Utility-aware localisation. Build the protected SCOPE edit at each percentile on the
    ladder (lower percentile = more layers = more removal). Pick the most aggressive edit
    whose utility cost stays within the budget; otherwise the least-damaging one. Returns
    (basis, chosen_pctile, localised_layers, utility_cost)."""
    ladder = [C.DECODE_PCTILE] if dry else C.PCTILE_LADDER
    trials = []
    for pctile in ladder:
        layers = S.localise(decod, pctile)
        if not layers:
            continue
        basis = S.build_scope_basis(model, tok, cfg, train_pairs, layers, protect=C.PROTECT)
        if not basis:
            continue
        acc = E.native_accuracy(model, tok, cfg, basis, C.BASELINE_E4_LIMIT, C.E4_MAX_TOKENS)
        uc = (base_acc - acc) if (np.isfinite(base_acc) and np.isfinite(acc)) else float("inf")
        trials.append((pctile, sorted(basis.keys()), basis, float(uc)))
        if dry:
            break
    if not trials:
        return {}, None, [], float("nan")
    safe = [t for t in trials if np.isfinite(t[3]) and t[3] <= C.MAX_UTILITY_COST]
    chosen = (max(safe, key=lambda t: len(t[1])) if safe
              else min(trials, key=lambda t: t[3]))
    return chosen[2], chosen[0], chosen[1], chosen[3]


def run_model(cfg, dry, push):
    from load_osm import load_model, unload_model
    name = cfg["name"]
    fin_name = f"scope_final_{name}.parquet"
    if not dry and integrity.parquet_nonempty(C.RESULTS / fin_name):
        try:
            if "scope" in set(pd.read_parquet(C.RESULTS / fin_name)["method"].astype(str)):
                log.info("scope for %s already complete; skipping", name); return
        except Exception:
            pass

    pairs = E._load_pairs(cfg, limit=None)
    if not pairs:
        log.error("no pairs for %s", name); return
    sub_pairs = E.stratified_subset(pairs, C.SUBSPACE_PAIRS, C.RANDOM_SEED)
    eval_pairs = E.stratified_subset(pairs, C.SWEEP_SUBSET, C.RANDOM_SEED + 7)
    tr_pairs, te_pairs, tr_seeds, te_seeds = split_by_seed(pairs)
    op_rank = _op_rank(name)
    if dry:
        sub_pairs = E.head_per_dataset(sub_pairs, C.DRY_LIMIT)
        eval_pairs = E.head_per_dataset(eval_pairs, C.DRY_LIMIT)
        tr_pairs = E.head_per_dataset(tr_pairs, C.DRY_LIMIT)
        te_pairs = E.head_per_dataset(te_pairs, C.DRY_LIMIT)
        te_seeds = {p["seed_id"] for p in te_pairs}
    te_eval = te_pairs[: (C.TEST_PAIR_CAP if not dry else len(te_pairs))]

    model, tok = load_model(cfg)
    try:
        base_acc = E.native_accuracy(model, tok, cfg, None, C.BASELINE_E4_LIMIT, C.E4_MAX_TOKENS)

        # ---- Stage 1: localise (decodability map) ----
        decod = S.decodability_map(model, tok, cfg, sub_pairs, n=(C.DRY_LIMIT if dry else None))
        # ---- Stage 2+3: protected, utility-aware SCOPE edit ----
        scope_basis, pctile, layers, scope_uc = select_scope_basis(
            model, tok, cfg, sub_pairs, decod, base_acc, dry=dry)
        # ---- Verify: decodability before vs after the edit ----
        verify_layers = layers or list(decod.keys())
        post = S.decodability_map(model, tok, cfg, sub_pairs, layers=verify_layers,
                                  edit_basis=scope_basis, n=(C.DRY_LIMIT if dry else None))
        base_dec = float(np.mean([decod[L] for L in verify_layers if L in decod])) if verify_layers else float("nan")
        post_dec = float(np.mean(list(post.values()))) if post else float("nan")
        drop = (1 - post_dec / base_dec) if (np.isfinite(base_dec) and base_dec > 1e-9) else float("nan")
        loc_json = {"model_name": name, "decodability_by_layer": {int(k): round(v, 5) for k, v in decod.items()},
                    "localised_layers": layers, "chosen_pctile": pctile, "utility_cost": round(scope_uc, 4) if np.isfinite(scope_uc) else None,
                    "decodability_before": round(base_dec, 5) if np.isfinite(base_dec) else None,
                    "decodability_after": round(post_dec, 5) if np.isfinite(post_dec) else None,
                    "decodability_drop": round(drop, 4) if np.isfinite(drop) else None,
                    "protect_massive": C.PROTECT, "massive_k": C.MASSIVE_K}
        (C.RESULTS / f"scope_localization_{name}.json").write_text(json.dumps(loc_json, indent=2))
        log.info("scope %s: localised layers=%s pctile=%s util=%.3f decod %.4f->%.4f (drop %.2f)",
                 name, layers, pctile, scope_uc, base_dec, post_dec, drop)
        push(f"scope: {name} localised {len(layers)} layers, decod drop {drop:.2f}")

        if not scope_basis:
            log.error("empty SCOPE basis for %s; skipping model", name); unload_model(name); return

        # ---- (A) Head-to-head: SCOPE vs the nine baselines on the shared eval set ----
        ctx_indep = E.collect_acts_demographic(model, tok, cfg)
        rows = []
        sc_re = E.e3_reaudit(model, tok, cfg, eval_pairs, scope_basis)
        crr, mer, npair = _crr(sc_re)
        sc_acc = E.native_accuracy(model, tok, cfg, scope_basis, C.BASELINE_E4_LIMIT, C.E4_MAX_TOKENS)
        rows.append({"method": "scope", "model_name": name, "status": "ok", "signal": "audit_causal",
                     "causal_residual_removed": crr, "mean_erased_commutator": mer, "n_pairs": npair,
                     "utility_cost": (base_acc - sc_acc), "baseline_acc": base_acc, "erased_acc": sc_acc,
                     "n_localised_layers": len(layers)})
        for m in B.REGISTRY:
            try:
                built = B.build_basis(m, model, tok, cfg, sub_pairs, ctx=ctx_indep, rank=op_rank)
                basis = built.get("basis", {})
                row = {"method": m, "model_name": name, "status": "ok",
                       "signal": "independent_demographic"}
                if not basis:
                    row.update({"causal_residual_removed": 0.0, "utility_cost": 0.0, "n_pairs": 0})
                else:
                    re = E.e3_reaudit(model, tok, cfg, eval_pairs, basis)
                    c2, m2, n2 = _crr(re)
                    er = E.native_accuracy(model, tok, cfg, basis, C.BASELINE_E4_LIMIT, C.E4_MAX_TOKENS)
                    row.update({"causal_residual_removed": c2, "mean_erased_commutator": m2, "n_pairs": n2,
                                "utility_cost": (base_acc - er), "baseline_acc": base_acc, "erased_acc": er})
                rows.append(row)
            except Exception as exc:
                log.error("baseline %s/%s raised: %s", name, m, str(exc)[:160])
                rows.append({"method": m, "model_name": name, "status": "error", "note": str(exc)[:160]})
            _save(pd.DataFrame(rows), fin_name)
        _save(pd.DataFrame(rows), fin_name)
        push(f"scope: {name} head-to-head ({len(rows)} methods)")

        # ---- (B) Held-out + behavioural (SCOPE vs unedited + 3 baselines) ----
        scope_basis_tr = S.build_scope_basis(model, tok, cfg, tr_pairs, layers, protect=C.PROTECT)
        bbasis = {"scope": scope_basis_tr}
        for m in BEHAV_BASELINES:
            try:
                bbasis[m] = B.build_basis(m, model, tok, cfg, tr_pairs, ctx=ctx_indep, rank=op_rank).get("basis", {})
            except Exception as exc:
                log.warning("held-out baseline %s failed: %s", m, str(exc)[:100]); bbasis[m] = {}
        orig_mean = float(np.nanmean([abs(p["orig_delta"]) for p in te_eval if np.isfinite(p["orig_delta"])])) if te_eval else float("nan")
        erows = []
        for method in ["unedited", "scope"] + BEHAV_BASELINES:
            basis = None if method == "unedited" else bbasis.get(method, {})
            hc, hmer, hn = 0.0, orig_mean, 0
            if basis:
                re = E.e3_reaudit(model, tok, cfg, te_eval, basis)
                hc, hmer, hn = _crr(re)
            acc, flip, na, nf = behavioural_readout(model, tok, cfg, basis, te_seeds)
            erows.append({"method": method, "model_name": name,
                          "heldout_residual_removed": round(hc, 4) if np.isfinite(hc) else None,
                          "behav_accuracy": round(acc, 4) if np.isfinite(acc) else None,
                          "behav_flip_rate": round(flip, 4) if np.isfinite(flip) else None,
                          "n_test_pairs": hn, "n_flip_seeds": nf})
            _save(pd.DataFrame(erows), f"scope_extra_{name}.parquet")
            push(f"scope: {name} held-out {method} flip={erows[-1]['behav_flip_rate']}")

        # ---- (C) Ablations: localised vs all-layer, protect vs no-protect, random ----
        all_layers = sorted(decod.keys())
        rng = np.random.default_rng(C.RANDOM_SEED)
        rand_layers = sorted(rng.choice(all_layers, size=min(len(layers) or 1, len(all_layers)), replace=False).tolist())
        ab = {
            "scope": scope_basis,
            "scope_alllayers": S.build_scope_basis(model, tok, cfg, sub_pairs, all_layers, protect=C.PROTECT),
            "scope_noprotect": S.build_scope_basis(model, tok, cfg, sub_pairs, layers, protect=False),
            "scope_random": S.build_scope_basis(model, tok, cfg, sub_pairs, rand_layers, protect=C.PROTECT),
        }
        abrows = []
        for variant, basis in ab.items():
            if not basis:
                continue
            re = E.e3_reaudit(model, tok, cfg, eval_pairs, basis)
            c3, m3, n3 = _crr(re)
            er = E.native_accuracy(model, tok, cfg, basis, C.BASELINE_E4_LIMIT, C.E4_MAX_TOKENS)
            abrows.append({"variant": variant, "model_name": name, "n_layers": len(basis),
                           "causal_residual_removed": c3, "mean_erased_commutator": m3,
                           "utility_cost": (base_acc - er), "erased_acc": er})
            _save(pd.DataFrame(abrows), f"scope_ablation_{name}.parquet")
        push(f"scope: {name} ablations done")

        # ---- (D) Prognosis: does the audit score predict whether SCOPE cures the pair? ----
        if len(sc_re):
            pr = sc_re[["seed_id", "orig_commutator", "erased_commutator"]].copy()
            pr["model_name"] = name
            pr["cured"] = (pr["erased_commutator"] <= C.TAU).astype(int)
            _save(pr, f"scope_prognosis_{name}.parquet")
            try:
                from scipy import stats
                m = pr["orig_commutator"].notna()
                sp = stats.spearmanr(pr.loc[m, "orig_commutator"], pr.loc[m, "erased_commutator"])
                (C.RESULTS / f"scope_prognosis_{name}.json").write_text(json.dumps(
                    {"model_name": name, "n": int(m.sum()),
                     "spearman_audit_vs_residual": float(sp.correlation),
                     "spearman_p": float(sp.pvalue),
                     "cured_fraction": float(pr["cured"].mean())}, indent=2))
            except Exception as exc:
                log.warning("scope prognosis stats failed: %s", str(exc)[:100])
    except Exception as exc:
        log.error("model %s raised: %s", name, str(exc)[:300])
        push(f"scope: error on {name}")
    finally:
        unload_model(name)
    log.info("scope %s complete", name)


def cmd_main(dry=False):
    from checkpoint import CheckpointPusher, push_checkpoint
    integrity.run()
    pusher, push = None, (lambda *_: None)
    if not dry:
        pusher = CheckpointPusher(); pusher.start(); push = push_checkpoint
    for cfg in C.OSM_MODELS:
        run_model(cfg, dry, push)
    if not dry:
        (C.RESULTS / "SCOPE_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        pusher.stop_and_flush("scope: ALL DONE")
    log.info("SCOPE COMPLETE (dry=%s)", dry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "main"], required=True)
    args = ap.parse_args()
    import transformer_lens  # noqa: F401
    import nnsight  # noqa: F401
    cmd_main(dry=(args.mode == "dry"))


if __name__ == "__main__":
    main()
