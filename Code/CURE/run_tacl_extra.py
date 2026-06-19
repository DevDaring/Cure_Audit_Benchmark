"""
run_tacl_extra.py -- the TACL rebuttal experiments (no re-fit of the headline run).

Closes the two open reviewer objections with held-out, output-level evidence:

  T1b  HELD-OUT bias removal. The CURE subspace is fitted on a TRAIN split of seeds
       and the causal residual removed is measured on a DISJOINT TEST split, so the
       number is no longer in-distribution with the subspace-estimation seeds.

  T1a  INDEPENDENT behavioural readout. On the same held-out TEST seeds, the model is
       generated under erasure and scored by two output-level metrics that do NOT use
       the activation commutator CURE optimises:
         - behav_accuracy : slot-a (disambiguated) answer accuracy (utility).
         - behav_flip_rate: fraction of seeds whose generated answer CHANGES across the
                            demographic sub-variants (slot c). Lower is more invariant.
       CURE is compared with the two strongest baselines (GenericErase, SAE-Debias),
       each on the independent 310-template demographic signal, and with the unedited
       model. If CURE lowers behav_flip_rate out of sample, the bias removal is not an
       artefact of the metric it targets.

Modes:
  python3 run_tacl_extra.py --mode dry    two seeds per split, every model, fail-loud.
  python3 run_tacl_extra.py --mode main   full held-out run, 15-min GitHub checkpoints.

Reuses the audited stack (experiments.py / erase.py / baselines.py); nothing is re-fit
that the headline run already produced. Operating rank per model is read from the
existing cure_rankcurve_<model>.json.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
import uuid
from collections import defaultdict
from contextlib import nullcontext

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
              logging.FileHandler(C.LOGS / "run_tacl_extra.log")],
)
log = logging.getLogger("cure.tacl")

# The two baselines worth a head-to-head on the independent signal (strongest, utility-keeping).
BASELINES = ["generic_erase", "sae_debias"]
FRAC_TRAIN = float(os.environ.get("TACL_FRAC_TRAIN", "0.55"))
TEST_PAIR_CAP = int(os.environ.get("TACL_TEST_PAIR_CAP", "600"))
BEHAV_SEED_CAP = int(os.environ.get("TACL_BEHAV_SEED_CAP", "160"))
BEHAV_MAXTOK = int(os.environ.get("TACL_BEHAV_MAXTOK", "48"))


def _save(df, name):
    if df is not None and len(df):
        df.to_parquet(C.RESULTS / name, index=False)


def _op_rank(name: str) -> int:
    """Operating rank per model from the headline run; fall back to HEADLINE_RANK."""
    p = C.RESULTS / f"cure_rankcurve_{name}.json"
    if p.exists():
        try:
            return int(json.loads(p.read_text())["operating_rank"])
        except Exception as exc:
            log.warning("rankcurve read failed for %s: %s", name, str(exc)[:80])
    return C.HEADLINE_RANK


def split_by_seed(pairs, frac_train=FRAC_TRAIN, seed=C.RANDOM_SEED):
    """Disjoint TRAIN/TEST partition of seeds, stratified by source benchmark."""
    seeds = sorted({p["seed_id"] for p in pairs})
    by_src = defaultdict(list)
    for s in seeds:
        by_src[s.split("_", 1)[0]].append(s)
    rng = random.Random(seed)
    train, test = set(), set()
    for src, ss in sorted(by_src.items()):
        ss = sorted(ss)
        rng.shuffle(ss)
        k = max(1, int(round(len(ss) * frac_train)))
        train |= set(ss[:k])
        test |= set(ss[k:])
    tr = [p for p in pairs if p["seed_id"] in train]
    te = [p for p in pairs if p["seed_id"] in test]
    return tr, te, train, test


def _acc(df):
    ok = df[df["success_flag"] == True]                              # noqa: E712
    if ok.empty:
        return float("nan")
    m = ok.apply(lambda r: str(r["parsed_answer"]).strip().lower() in str(r["gold_answer"]).strip().lower()
                 or str(r["gold_answer"]).strip().lower() in str(r["parsed_answer"]).strip().lower(), axis=1)
    return float(m.mean())


def behavioural_readout(model, tok, cfg, basis, test_seeds, max_tokens=BEHAV_MAXTOK, cap=BEHAV_SEED_CAP):
    """Output-level metrics on held-out TEST seeds under (optional) erasure.

    Returns (behav_accuracy, behav_flip_rate, n_acc_seeds, n_flip_seeds).
      accuracy : slot-a surface answer accuracy (utility).
      flip_rate: fraction of seeds whose generated answer differs across slot-c
                 demographic sub-variants (output-level swap sensitivity).
    """
    from osm_behavioral import evaluate_osm_model
    pentad = pd.read_parquet(C.PENTAD_PATH)
    test_seeds = set(test_seeds)
    acc_src = pentad[(pentad["slot"] == "a") & (pentad["subvariant"] == "surface")
                     & (pentad["seed_id"].isin(test_seeds))].copy()
    flip_src = pentad[(pentad["slot"] == "c") & (pentad["seed_id"].isin(test_seeds))].copy()
    for d in (acc_src, flip_src):
        d.drop(d[d["prompt_text"].astype(str).str.strip() == ""].index, inplace=True)
    if cap:
        keep = list(dict.fromkeys(flip_src["seed_id"].tolist()))[:cap]
        flip_src = flip_src[flip_src["seed_id"].isin(keep)]
        acc_src = acc_src[acc_src["seed_id"].isin(set(keep))]
    rid = f"tacl-{uuid.uuid4().hex[:8]}"
    ctx = erase.ErasureContext(model, basis) if basis else nullcontext()
    with ctx:
        acc_df = evaluate_osm_model(cfg, model, tok, acc_src, rid + "a", temperature=0.0,
                                    sample_index=0, max_tokens=max_tokens) if len(acc_src) else pd.DataFrame()
        flip_df = evaluate_osm_model(cfg, model, tok, flip_src, rid + "c", temperature=0.0,
                                     sample_index=0, max_tokens=max_tokens) if len(flip_src) else pd.DataFrame()
    acc = _acc(acc_df) if len(acc_df) else float("nan")
    flip_rate, n_flip = float("nan"), 0
    if len(flip_df):
        ok = flip_df[flip_df["success_flag"] == True]                # noqa: E712
        if len(ok):
            g = ok.groupby("seed_id")["parsed_answer"].apply(
                lambda s: s.astype(str).str.strip().str.lower().nunique())
            g = g[g.index.map(lambda sid: (ok["seed_id"] == sid).sum() >= 2)]  # need >=2 variants
            if len(g):
                flip_rate = float((g > 1).mean())
                n_flip = int(len(g))
    n_acc = int(acc_df["seed_id"].nunique()) if len(acc_df) else 0
    return acc, flip_rate, n_acc, n_flip


def run_model(cfg, dry: bool, push):
    from load_osm import load_model, unload_model
    name = cfg["name"]
    out_dir = (C.RESULTS / "dryrun") if dry else C.RESULTS   # dry never pollutes real results
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"tacl_extra_{name}.parquet"
    out_path = out_dir / out_name
    want = ["unedited", "cure"] + BASELINES
    have = set()
    rows = []
    if not dry and integrity.parquet_nonempty(out_path):
        try:
            prev = pd.read_parquet(out_path)
            rows = prev.to_dict("records")
            have = set(prev["method"].astype(str))
        except Exception:
            rows, have = [], set()
    if all(m in have for m in want) and not dry:
        log.info("tacl_extra for %s already complete; skipping", name)
        return
    op_rank = _op_rank(name)
    pairs = E._load_pairs(cfg, limit=None)
    if not pairs:
        log.error("no pairs for %s", name)
        return
    tr_pairs, te_pairs, tr_seeds, te_seeds = split_by_seed(pairs)
    if dry:
        tr_pairs = E.head_per_dataset(tr_pairs, C.DRY_LIMIT)
        te_pairs = E.head_per_dataset(te_pairs, C.DRY_LIMIT)
        te_seeds = {p["seed_id"] for p in te_pairs}
    te_eval = te_pairs[:TEST_PAIR_CAP] if (TEST_PAIR_CAP and not dry) else te_pairs
    log.info("=== %s | op_rank=%d | train_seeds=%d test_seeds=%d test_pairs(eval)=%d ===",
             name, op_rank, len(tr_seeds), len(te_seeds), len(te_eval))

    model, tok = load_model(cfg)
    try:
        # Fit CURE on TRAIN only; baselines on the independent demographic signal.
        ctx_audit = E.collect_acts(model, tok, cfg, tr_pairs[:C.SUBSPACE_PAIRS])
        cure_bases = erase.bases_at_ranks(ctx_audit["diffs"], C.ERASE_RANKS)
        if not cure_bases:
            log.error("empty CURE subspace for %s", name); unload_model(name); return
        cure_basis = cure_bases.get(op_rank) or cure_bases[max(cure_bases)]
        ctx_indep = E.collect_acts_demographic(model, tok, cfg)
        bases = {"cure": cure_basis}
        for m in BASELINES:
            try:
                built = B.build_basis(m, model, tok, cfg, tr_pairs[:C.SUBSPACE_PAIRS],
                                      ctx=ctx_indep, rank=op_rank)
                bases[m] = built.get("basis", {})
            except Exception as exc:
                log.warning("baseline %s basis failed on %s: %s", m, name, str(exc)[:120])
                bases[m] = {}

        # Baseline orig commutator on the held-out test set (for reference).
        orig_mean = float(np.nanmean([abs(p["orig_delta"]) for p in te_eval
                                      if np.isfinite(p["orig_delta"])])) if te_eval else float("nan")

        for method in want:
            if method in have and not dry:
                continue
            basis = None if method == "unedited" else bases.get(method, {})
            # held-out causal residual removed
            crr, mean_er, n_pairs = 0.0, orig_mean, 0
            if basis:
                re = E.e3_reaudit(model, tok, cfg, te_eval, basis)
                if len(re):
                    crr = float((re["erased_commutator"] < re["orig_commutator"]).mean())
                    mean_er = float(re["erased_commutator"].mean())
                    n_pairs = int(len(re))
            # independent behavioural readout
            acc, flip, n_acc, n_flip = behavioural_readout(model, tok, cfg, basis, te_seeds)
            row = {"method": method, "model_name": name, "op_rank": op_rank,
                   "heldout_residual_removed": round(crr, 4),
                   "mean_erased_commutator": round(mean_er, 4) if np.isfinite(mean_er) else None,
                   "orig_commutator_test": round(orig_mean, 4) if np.isfinite(orig_mean) else None,
                   "behav_accuracy": round(acc, 4) if np.isfinite(acc) else None,
                   "behav_flip_rate": round(flip, 4) if np.isfinite(flip) else None,
                   "n_test_pairs": n_pairs, "n_acc_seeds": n_acc, "n_flip_seeds": n_flip,
                   "signal": ("audit_causal" if method == "cure"
                              else ("none" if method == "unedited" else "independent_demographic"))}
            rows = [r for r in rows if r.get("method") != method] + [row]
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            push(f"tacl-extra: {name} {method} flip={row['behav_flip_rate']} crr={row['heldout_residual_removed']}")
            log.info("%-12s %s | heldout_crr=%.3f behav_acc=%s flip=%s",
                     method, name, crr, row["behav_accuracy"], row["behav_flip_rate"])
        if len(rows):
            pd.DataFrame(rows).to_parquet(out_path, index=False)
    except Exception as exc:
        log.error("model %s raised: %s", name, str(exc)[:300])
    finally:
        unload_model(name)
    log.info("tacl_extra %s complete", name)


def cmd_main(dry=False):
    from checkpoint import CheckpointPusher, push_checkpoint
    integrity.run()
    pusher = None
    push = (lambda *_: None)
    if not dry:
        pusher = CheckpointPusher()
        pusher.start()
        push = push_checkpoint
    for cfg in C.OSM_MODELS:
        run_model(cfg, dry, push)
    if not dry:
        (C.RESULTS / "TACL_EXTRA_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        pusher.stop_and_flush("tacl-extra: ALL DONE")
    log.info("TACL EXTRA COMPLETE (dry=%s)", dry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry", "main"], required=True)
    args = ap.parse_args()
    import transformer_lens  # noqa: F401
    import nnsight  # noqa: F401
    cmd_main(dry=(args.mode == "dry"))


if __name__ == "__main__":
    main()
