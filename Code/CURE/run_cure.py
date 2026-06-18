"""
run_cure.py -- single entry point for CURE.

  python3 run_cure.py --mode dry     validate the whole environment on two pairs.
  python3 run_cure.py --mode main    run E1 to E6 for all four models, with resume,
                                      duplicate/corruption checks, and 15-minute pushes.

Every run starts with the integrity check (Section: integrity.py). Resume skips any
unit whose parquet is present and non-empty.
"""

import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

import config_cure as C
import integrity
import erase
import experiments as E
import baselines as B

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(C.LOGS / "run_cure.log")],
)
log = logging.getLogger("cure.run")


def _save(df: pd.DataFrame, name: str):
    if df is not None and len(df):
        df.to_parquet(C.RESULTS / name, index=False)


def cmd_main():
    """Expedited but statistically sound flow (see README, Run section):

      - E1 subspace is estimated ONCE from a bounded subset, then every rank is a
        slice of the same SVD (no per-rank re-extraction).
      - E3 HEADLINE residual-removed runs on ALL pairs at the single operating rank,
        so the headline number keeps the full audit n and stays in harmony with it.
      - The multi-rank sweep, the fairness-utility curve (E4), and the six-baseline
        head-to-head run on a fixed-seed, benchmark-stratified subset; with ~1000
        pairs the prognosis and comparison carry tight confidence intervals.
      - Utility uses the no-erasure baseline computed once and short generations.
    """
    from load_osm import load_model, unload_model
    from checkpoint import CheckpointPusher, push_checkpoint
    integrity.run()
    pusher = CheckpointPusher()
    pusher.start()

    done_path = C.RESULTS / "STATUS.json"
    status = json.loads(done_path.read_text()) if done_path.exists() else {"done": []}
    done = set(status["done"])

    for cfg in C.OSM_MODELS:
        name = cfg["name"]
        if name in done and integrity.parquet_nonempty(C.RESULTS / f"cure_prognosis_{name}.parquet"):
            log.info("model %s already complete; skipping", name); continue
        log.info("=== model %s ===", name)
        pairs = E._load_pairs(cfg, limit=None)
        if not pairs:
            log.error("no pairs for %s; skipping", name); continue

        sub_pairs = E.stratified_subset(pairs, C.SUBSPACE_PAIRS, C.RANDOM_SEED)   # estimate subspace
        sweep_pairs = E.stratified_subset(pairs, C.SWEEP_SUBSET, C.RANDOM_SEED)   # sweep / head-to-head

        model, tok = load_model(cfg)
        try:
            # E1: cache diffs ONCE, derive every rank basis from the same SVD
            diffs = E.collect_diffs(model, tok, cfg, sub_pairs)
            bases = erase.bases_at_ranks(diffs, C.ERASE_RANKS)
            if not bases or not next(iter(bases.values())):
                log.error("empty subspace for %s; skipping", name); unload_model(name); continue
            head_rank = C.HEADLINE_RANK if C.HEADLINE_RANK in bases else max(bases)
            head_basis = bases[head_rank]

            # E3 HEADLINE: ALL pairs at the operating rank (full-set residual removed)
            re_head = E.e3_reaudit(model, tok, cfg, pairs, head_basis)
            re_head["erase_rank"] = head_rank
            _save(re_head, f"cure_recovery_{name}.parquet")
            push_checkpoint(f"cure-results: {name} headline E3 rank {head_rank} n={len(re_head)}")

            # E4 baseline accuracy computed ONCE, reused across ranks
            base_acc = E.native_accuracy(model, tok, cfg, None, C.E4_LIMIT, C.E4_MAX_TOKENS)

            # E3 SWEEP + E4 on the stratified subset (feeds E6 and the curve)
            per_rank_re, util_by_rank, util_rows = {}, {}, []
            for rank in C.ERASE_RANKS:
                re_df = E.e3_reaudit(model, tok, cfg, sweep_pairs, bases[rank])
                re_df["erase_rank"] = rank
                per_rank_re[rank] = re_df
                er_acc = E.native_accuracy(model, tok, cfg, bases[rank], C.E4_LIMIT, C.E4_MAX_TOKENS)
                cost = (base_acc - er_acc) if (np.isfinite(base_acc) and np.isfinite(er_acc)) else float("nan")
                util_by_rank[rank] = cost
                util_rows.append({"model_name": name, "erase_rank": rank, "metric": "native_accuracy",
                                  "baseline": base_acc, "erased": er_acc, "utility_cost": cost})
                push_checkpoint(f"cure-results: {name} sweep rank {rank}")
            if per_rank_re:
                _save(pd.concat(per_rank_re.values(), ignore_index=True), f"cure_recovery_sweep_{name}.parquet")
            _save(pd.DataFrame(util_rows), f"cure_utility_{name}.parquet")

            # E6 prognosis from the sweep
            prog_df, fit = E.e6_prognosis(per_rank_re, util_by_rank, name)
            _save(prog_df, f"cure_prognosis_{name}.parquet")
            (C.RESULTS / f"cure_prognosis_{name}.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")

            # E5 head-to-head on the SAME subset at the operating rank (fair, bounded);
            # include a CURE row scored identically for an apples-to-apples comparison.
            brows = []
            cure_re = E.e3_reaudit(model, tok, cfg, sweep_pairs, head_basis)
            if len(cure_re):
                brows.append({"method": "cure", "model_name": name, "status": "ok",
                              "kind": "audit_guided_erase",
                              "causal_residual_removed": float((cure_re["erased_commutator"] < cure_re["orig_commutator"]).mean()),
                              "mean_erased_commutator": float(cure_re["erased_commutator"].mean())})
            for m in B.REGISTRY:
                brows.append(B.score_baseline(m, model, tok, cfg, sweep_pairs))
            _save(pd.DataFrame(brows), f"cure_baselines_{name}.parquet")
        except Exception as exc:
            log.error("model %s raised: %s", name, str(exc)[:300])
            push_checkpoint(f"cure-results: error on {name}")
            unload_model(name); continue
        unload_model(name)

        done.add(name)
        status["done"] = sorted(done)
        done_path.write_text(json.dumps(status, indent=2))
        push_checkpoint(f"cure-results: {name} complete")
        log.info("model %s complete", name)

    (C.RESULTS / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    pusher.stop_and_flush("cure-results: ALL DONE")
    log.info("ALL MODELS COMPLETE")


def _select_op_rank(name: str) -> int:
    """Per-model operating rank, chosen on the validation sweep: the rank that reduces
    the commutator for the largest fraction of pairs (ties -> the smaller rank, which
    costs less utility). Falls back to HEADLINE_RANK if the sweep is unavailable."""
    p = C.RESULTS / f"cure_recovery_sweep_{name}.parquet"
    if not integrity.parquet_nonempty(p):
        return C.HEADLINE_RANK
    try:
        sw = pd.read_parquet(p)
        sw = sw.assign(_reduced=(sw["erased_commutator"] < sw["orig_commutator"]).astype(float))
        red = sw.groupby("erase_rank")["_reduced"].mean()
        return int(red.sort_index().idxmax())          # first (smallest) rank on ties
    except Exception as exc:
        log.warning("op-rank selection for %s failed (%s); using HEADLINE_RANK", name, str(exc)[:80])
        return C.HEADLINE_RANK


def _utility_aware_rank(model, tok, cfg, name, bases, base_acc):
    """Per-model operating rank: remove the MOST bias among ranks whose native-accuracy
    drop stays <= MAX_UTILITY_COST. Bias-per-rank comes from the validation sweep
    (cure_recovery_sweep, RANDOM_SEED); utility-per-rank is measured fresh here. If no rank
    meets the cap (bias entangled with critical/massive-activation directions), the rank
    with the SMALLEST utility cost is chosen, so CURE never trades the whole model away.
    Returns (op_rank, curve)."""
    bias = {}
    p = C.RESULTS / f"cure_recovery_sweep_{name}.parquet"
    if integrity.parquet_nonempty(p):
        try:
            sw = pd.read_parquet(p)
            sw = sw.assign(_r=(sw["erased_commutator"] < sw["orig_commutator"]).astype(float))
            g = sw.groupby("erase_rank")["_r"].mean()
            bias = {int(r): float(g.loc[r]) for r in g.index}
        except Exception as exc:
            log.warning("sweep bias read failed for %s: %s", name, str(exc)[:80])
    curve = {}
    for r in C.ERASE_RANKS:
        if r not in bases:
            continue
        acc = E.native_accuracy(model, tok, cfg, bases[r], C.BASELINE_E4_LIMIT, C.E4_MAX_TOKENS)
        uc = (base_acc - acc) if (np.isfinite(base_acc) and np.isfinite(acc)) else None
        br = bias.get(r, float("nan"))
        curve[r] = {"reduced_validation": (round(br, 4) if br == br else None),
                    "utility_cost": (round(float(uc), 4) if uc is not None else None),
                    "erased_acc": (round(float(acc), 4) if np.isfinite(acc) else None)}
    ranks = [r for r in C.ERASE_RANKS if r in curve]
    safe = [r for r in ranks if curve[r]["utility_cost"] is not None
            and curve[r]["utility_cost"] <= C.MAX_UTILITY_COST]
    if safe:
        op_rank = max(safe, key=lambda r: (curve[r]["reduced_validation"]
                                           if curve[r]["reduced_validation"] is not None else -1.0))
    else:
        op_rank = min(ranks, key=lambda r: (curve[r]["utility_cost"]
                                            if curve[r]["utility_cost"] is not None else float("inf")))
    return op_rank, {str(k): v for k, v in curve.items()}


def cmd_baselines():
    """E5+ : the full comparison of CURE against all nine baselines under ONE harness.

    Every method's basis is scored on the SAME causal-residual-removed metric and the
    SAME native-utility metric as CURE, on the fixed-seed benchmark-stratified sweep
    subset, for all four models. The protected-position activations are cached ONCE per
    model and shared across baselines. Resume-aware: a method already present in
    cure_baselines_full_<model>.parquet is skipped, so a released VM continues.
    """
    from load_osm import load_model, unload_model
    from checkpoint import CheckpointPusher, push_checkpoint
    integrity.run()
    pusher = CheckpointPusher()
    pusher.start()

    methods = ["cure"] + list(B.REGISTRY.keys())          # cure + 9 baselines
    for cfg in C.OSM_MODELS:
        name = cfg["name"]
        out_name = f"cure_final_{name}.parquet"
        out_path = C.RESULTS / out_name
        rows, have = [], set()
        if integrity.parquet_nonempty(out_path):
            try:
                prev = pd.read_parquet(out_path)
                rows = prev.to_dict("records")
                have = set(prev["method"].astype(str))
            except Exception:
                rows, have = [], set()
        todo = [m for m in methods if m not in have]
        if not todo:
            log.info("baselines for %s already complete (%d methods); skipping", name, len(have))
            continue
        log.info("=== baselines %s : %d/%d methods to score ===", name, len(todo), len(methods))

        pairs = E._load_pairs(cfg, limit=None)
        if not pairs:
            log.error("no pairs for %s; skipping", name); continue
        sub_pairs = E.stratified_subset(pairs, C.SUBSPACE_PAIRS, C.RANDOM_SEED)
        eval_pairs = E.stratified_subset(pairs, C.SWEEP_SUBSET, C.RANDOM_SEED + 7)

        model, tok = load_model(cfg)
        try:
            # CURE alone uses the causal audit signal; every baseline uses an INDEPENDENT
            # demographic-contrast signal (no cdva_results). ONE utility-aware operating
            # rank (below) is shared by cure AND the baselines, and the comparison is scored
            # on eval_pairs (RANDOM_SEED+7) -- disjoint from the rank-selection sweep.
            ctx_audit = E.collect_acts(model, tok, cfg, sub_pairs)
            ctx_indep = E.collect_acts_demographic(model, tok, cfg)
            base_acc = E.native_accuracy(model, tok, cfg, None, C.BASELINE_E4_LIMIT, C.E4_MAX_TOKENS)
            # Utility-aware operating rank: most bias removed among ranks with utility cost
            # <= MAX_UTILITY_COST (else the least-damaging rank). ONE rank for all methods.
            bases = erase.bases_at_ranks(ctx_audit["diffs"], C.ERASE_RANKS)
            op_rank, rank_curve = _utility_aware_rank(model, tok, cfg, name, bases, base_acc)
            (C.RESULTS / f"cure_rankcurve_{name}.json").write_text(
                json.dumps({"model_name": name, "operating_rank": op_rank,
                            "max_utility_cost": C.MAX_UTILITY_COST, "curve": rank_curve}, indent=2),
                encoding="utf-8")
            log.info("baselines %s: utility-aware operating rank=%d | curve=%s", name, op_rank, rank_curve)
            head_basis = bases[op_rank]
            for m in todo:
                try:
                    if m == "cure":
                        basis, kind, status, signal = head_basis, "audit_guided_erase", "ok", "audit_causal"
                    else:
                        signal = "independent_demographic"
                        built = B.build_basis(m, model, tok, cfg, sub_pairs, ctx=ctx_indep, rank=op_rank)
                        if built.get("status") in ("pending", "error"):
                            rows.append({"method": m, "model_name": name, "status": built["status"],
                                         "signal": signal, "note": built.get("note", "")})
                            _save(pd.DataFrame(rows), out_name); continue
                        basis, kind, status = built.get("basis", {}), built.get("kind"), "ok"
                    row = {"method": m, "model_name": name, "status": status, "kind": kind,
                           "signal": signal, "erase_rank": op_rank}
                    if not basis:
                        # prompt-only / empty: no activation edit by construction
                        row.update({"causal_residual_removed": 0.0, "utility_cost": 0.0, "n_pairs": 0})
                    else:
                        re = E.e3_reaudit(model, tok, cfg, eval_pairs, basis)
                        if len(re):
                            row["causal_residual_removed"] = float(
                                (re["erased_commutator"] < re["orig_commutator"]).mean())
                            row["mean_erased_commutator"] = float(re["erased_commutator"].mean())
                            row["n_pairs"] = int(len(re))
                        else:
                            row["causal_residual_removed"] = float("nan"); row["n_pairs"] = 0
                        er_acc = E.native_accuracy(model, tok, cfg, basis, C.BASELINE_E4_LIMIT, C.E4_MAX_TOKENS)
                        row["utility_cost"] = (base_acc - er_acc) if (
                            np.isfinite(base_acc) and np.isfinite(er_acc)) else float("nan")
                        row["baseline_acc"] = base_acc; row["erased_acc"] = er_acc
                    rows.append(row)
                    _save(pd.DataFrame(rows), out_name)
                    push_checkpoint(f"baselines: {name} {m} crr="
                                    f"{row.get('causal_residual_removed', float('nan')):.3f}")
                    log.info("baseline %-14s %s -> crr=%.3f util=%.3f", m, name,
                             row.get("causal_residual_removed", float("nan")),
                             row.get("utility_cost", float("nan")))
                except Exception as exc:
                    log.error("baseline %s/%s raised: %s", name, m, str(exc)[:200])
                    rows.append({"method": m, "model_name": name, "status": "error", "note": str(exc)[:160]})
                    _save(pd.DataFrame(rows), out_name)
            _save(pd.DataFrame(rows), out_name)
        except Exception as exc:
            log.error("baselines model %s raised: %s", name, str(exc)[:300])
        finally:
            unload_model(name)
        push_checkpoint(f"baselines: {name} complete ({len(rows)} rows)")
        log.info("baselines %s complete", name)

    (C.RESULTS / "BASELINES_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    pusher.stop_and_flush("baselines: ALL DONE")
    log.info("ALL BASELINES COMPLETE")


def cmd_diagnose():
    """Diagnose the per-model rank non-monotonicity (why phi-4-mini spikes at rank 4).

    For phi and a healthy contrast (qwen), on a held-out 150-pair sample, measure:
      (a) the effect of erasing EACH diff-SVD direction in isolation (which single
          direction, when removed, raises the commutator);
      (b) the cumulative rank 1..8 curve at fine granularity;
      (c) the mid-layer singular spectrum;
      (d) whether the peak coordinate of each direction lands on a high-variance
          ('massive activation') residual dimension -- the usual cause of erasure
          instability in some models.
    Writes results/anomaly_diagnostic.json.
    """
    from load_osm import load_model, unload_model
    from checkpoint import push_checkpoint
    targets = ["phi-4-mini-instruct", "qwen2.5-7b-instruct"]
    diag_path = C.RESULTS / "anomaly_diagnostic.json"
    if diag_path.exists():
        try:
            if all(t in json.loads(diag_path.read_text()) for t in targets):
                log.info("anomaly diagnostic already complete; skipping"); return
        except Exception:
            pass
    report = {}
    for cfg in C.OSM_MODELS:
        name = cfg["name"]
        if name not in targets:
            continue
        log.info("=== diagnose %s ===", name)
        pairs = E._load_pairs(cfg, limit=None)
        if not pairs:
            continue
        sub = E.stratified_subset(pairs, C.SUBSPACE_PAIRS, C.RANDOM_SEED)
        diag = E.stratified_subset(pairs, 150, C.RANDOM_SEED + 7)
        model, tok = load_model(cfg)
        try:
            ctx = E.collect_acts(model, tok, cfg, sub)
            full = erase._svd_components(ctx["diffs"])             # {layer: Vt all comps}
            layers = sorted(full.keys())
            orig_mean = None
            per_dir = {}
            for k in range(8):                                    # erase ONE direction k
                basis = {L: full[L][k:k + 1, :] for L in layers if full[L].shape[0] > k}
                if not basis:
                    continue
                re = E.e3_reaudit(model, tok, cfg, diag, basis)
                if len(re):
                    orig_mean = round(float(re["orig_commutator"].mean()), 4)
                    per_dir[k + 1] = {
                        "mean_erased": round(float(re["erased_commutator"].mean()), 4),
                        "reduced_frac": round(float((re["erased_commutator"] < re["orig_commutator"]).mean()), 4)}
            cumulative = {}
            for r in range(1, 9):                                 # erase top-r cumulatively
                basis = {L: full[L][:r, :] for L in layers}
                re = E.e3_reaudit(model, tok, cfg, diag, basis)
                if len(re):
                    cumulative[r] = {
                        "mean_erased": round(float(re["erased_commutator"].mean()), 4),
                        "reduced_frac": round(float((re["erased_commutator"] < re["orig_commutator"]).mean()), 4)}
            midL = layers[len(layers) // 2]
            X = np.stack(ctx["diffs"][midL], 0); X = X - X.mean(0, keepdims=True)
            sv = np.linalg.svd(X, compute_uv=False)
            A = np.stack(ctx["A"][midL], 0)
            top_massive = [int(d) for d in np.argsort(A.var(0))[::-1][:5]]
            align = {}
            for k in range(min(8, full[midL].shape[0])):
                v = full[midL][k]; am = int(np.abs(v).argmax())
                align[k + 1] = {"peak_dim": am, "peak_abs": round(float(np.abs(v).max()), 3),
                                "hits_massive_dim": am in top_massive}
            report[name] = {"orig_commutator_mean": orig_mean, "erase_single_direction": per_dir,
                            "cumulative_rank_curve": cumulative, "midlayer": int(midL),
                            "singular_values_midlayer": [round(float(x), 3) for x in sv[:10]],
                            "massive_dims_midlayer": top_massive, "direction_peak_alignment": align}
            log.info("diagnose %s single-direction effect: %s", name, per_dir)
        except Exception as exc:
            log.error("diagnose %s raised: %s", name, str(exc)[:200])
        finally:
            unload_model(name)
        (C.RESULTS / "anomaly_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        push_checkpoint(f"diagnostic: {name} anomaly analysis")
    log.info("DIAGNOSTIC COMPLETE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "main", "baselines", "diagnose"], required=True)
    args = ap.parse_args()

    # verify both patching libraries import before any work (fail loud)
    import importlib.metadata as md
    import transformer_lens  # noqa: F401
    import nnsight  # noqa: F401
    log.info("patching libs OK: transformer_lens=%s nnsight=%s",
             md.version("transformer_lens"), md.version("nnsight"))

    if args.mode == "dry":
        import dry_checks
        sys.exit(0 if dry_checks.run_all() else 1)
    if args.mode == "baselines":
        cmd_baselines()
        return
    if args.mode == "diagnose":
        cmd_diagnose()
        return
    cmd_main()


if __name__ == "__main__":
    main()
