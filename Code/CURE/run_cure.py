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

import pandas as pd

import config_cure as C
import integrity
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
    from load_osm import load_model, unload_model
    integrity.run()
    from checkpoint import CheckpointPusher
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

        model, tok = load_model(cfg)
        try:
            per_rank_re = {}
            util_by_rank = {}
            util_rows = []
            top_basis = None
            for rank in C.ERASE_RANKS:
                basis = E.build_subspace(model, tok, cfg, pairs, rank=rank)
                if not basis:
                    log.warning("empty subspace at rank %d for %s", rank, name); continue
                top_basis = basis
                re_df = E.e3_reaudit(model, tok, cfg, pairs, basis)
                per_rank_re[rank] = re_df
                _save(re_df, f"cure_recovery_{name}_rank{rank}.parquet")
                util = E.e4_utility(model, tok, cfg, basis, limit=None)
                util_rows.append(util)
                if len(util):
                    util_by_rank[rank] = float(util["utility_cost"].iloc[0])
                from checkpoint import push_checkpoint
                push_checkpoint(f"cure-results: {name} rank {rank}")

            if util_rows:
                _save(pd.concat(util_rows, ignore_index=True), f"cure_utility_{name}.parquet")
            if per_rank_re:
                prog_df, fit = E.e6_prognosis(per_rank_re, util_by_rank, name)
                _save(prog_df, f"cure_prognosis_{name}.parquet")
                (C.RESULTS / f"cure_prognosis_{name}.json").write_text(json.dumps(fit, indent=2), encoding="utf-8")

            # E5 baselines (in-stack now; adapters report pending until wired)
            brows = []
            for m in B.REGISTRY:
                brows.append(B.score_baseline(m, model, tok, cfg, pairs))
            _save(pd.DataFrame(brows), f"cure_baselines_{name}.parquet")
        except Exception as exc:
            log.error("model %s raised: %s", name, str(exc)[:300])
            from checkpoint import push_checkpoint
            push_checkpoint(f"cure-results: error on {name}")
            unload_model(name); continue
        unload_model(name)

        done.add(name)
        status["done"] = sorted(done)
        done_path.write_text(json.dumps(status, indent=2))
        from checkpoint import push_checkpoint
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
