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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "main"], required=True)
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
    cmd_main()


if __name__ == "__main__":
    main()
