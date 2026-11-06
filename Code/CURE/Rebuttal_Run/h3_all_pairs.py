"""
h3_all_pairs.py -- the pooled audit without the largest-|C| pair selection (rebuttal cycle,
answers JwYi 'show results without this selection'). GPU, one model at a time, all four models.

The pooled audit evaluated, per test seed, the demographic pair with the largest unedited |C|
among the seed's mask-clean pairs (p1_pilot.pairs_for_seed). This stage evaluates EVERY eligible
demographic pair of every one of the 160 test seeds (about 6.5 pairs per seed) under

  unedited_r0_a0           no edit
  cure_centred_svd_r1_a1   the rank-one span erasure (the manuscript's cell)
  random_ortho_1_r1_a1     the first rank-matched random subspace

with the full final-position logit vector kept, so the raw-logit effect C_raw, the shift-
invariant log-probability effect C_logp and the option margin C_margin are all available
(g1_invariant.full_logit_patch / stats). Each row records the pair's rank by the stored
unedited |C| within its seed (1 = the pair the pooled audit used) and whether it was that pair.

Rows: results/rebuttal_20261107/h3_pairs_<model>.parquet (resumable).
Usage: python h3_all_pairs.py --model qwen2.5-7b-instruct [--smoke]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import hh_common as H

log = H.log
KEY = ["model_name", "seed_id", "subvariant_A", "subvariant_B", "cond_id"]
CONDS = ("unedited_r0_a0", "cure_centred_svd_r1_a1", "random_ortho_1_r1_a1")


def all_demographic_pairs(model: str, seed_id: str, cdva: pd.DataFrame, c: pd.DataFrame, eligible: set) -> list[dict]:
    """Every eligible mask-clean demographic pair of the seed (p1_pilot.pairs_for_seed without
    the argmax), ranked by the stored unedited |C| of the audit."""
    import p1_pilot as P1
    rows = c[c["seed_id"] == seed_id]
    var = {r["subvariant"]: r for _, r in rows.iterrows()}
    cd = cdva[(cdva["model_name"] == model) & (cdva["seed_id"] == seed_id) & (cdva["success_flag"] == True) & (cdva["position_fallback_used"] == False)]   # noqa: E712
    cd = cd[[(seed_id, a, b) in eligible for a, b in zip(cd["pair_A_subvariant"], cd["pair_B_subvariant"])]]

    def is_neutral(sv):
        return str(var[sv].get("swap_token", "")).strip().lower().replace("_", " ") in P1.NEUTRAL_TOKENS

    demo = cd[[not is_neutral(a) and not is_neutral(b) for a, b in zip(cd["pair_A_subvariant"], cd["pair_B_subvariant"])]].copy()
    demo["absC"] = demo["delta_logit"].abs()
    demo = demo.sort_values("absC", ascending=False).drop_duplicates(["pair_A_subvariant", "pair_B_subvariant"])
    out = []
    for k, r in enumerate(demo.itertuples(index=False)):
        o = {"seed_id": seed_id, "A": r.pair_A_subvariant, "B": r.pair_B_subvariant, "C_pre_shipped": float(abs(r.delta_logit)), "rank_by_Cpre": k + 1,
             "n_pairs_in_seed": len(demo)}
        for side, sv in (("A", o["A"]), ("B", o["B"])):
            v = var[sv]
            o["prompt_%s" % side] = str(v["prompt_text"]); o["swap_%s" % side] = str(v.get("swap_token", "")); o["gold_%s" % side] = str(v.get("gold_answer", ""))
        out.append(o)
    return out


def run(mname: str, smoke: bool) -> None:
    import g1_invariant as G1
    import p1_intervention as I
    import p1_pilot as P1
    import p2_controlled_erasure as P2
    H.ensure_out()
    path = H.OUT / ("h3_pairs_%s.parquet" % mname)
    c, _ = P1._pentad_c()
    cdva = pd.read_parquet(H.K.CDVA_PATH)
    elig = pd.read_csv(H.K.OUT_P0 / "pair_sets" / ("%s_eligible.csv" % mname))
    eligible = set(zip(elig.seed_id, elig.subvariant_A, elig.subvariant_B))
    test = pd.read_csv(H.V2 / "p2_test_seeds.csv")
    stored = G1.stored_pairs(mname)[["seed_id", "subvariant_A", "subvariant_B"]]
    selected = set(zip(stored.seed_id, stored.subvariant_A, stored.subvariant_B))
    seeds = list(test.seed_id)
    if smoke:
        seeds = seeds[:2]
    done = H.done_keys(path, KEY)
    cfg, model, tok = H.load_model(mname)
    store = P2.BasisStore.load(mname, "chat")
    if store is None:
        raise SystemExit("no P2 basis store for %s" % mname)
    fac = {"unedited_r0_a0": None,
           "cure_centred_svd_r1_a1": (lambda: I.HookedEdit(model, store.rows("cure_centred_svd", 1), alpha=1.0, site="span")),
           "random_ortho_1_r1_a1": (lambda: I.HookedEdit(model, store.rows("random_ortho_1", 1), alpha=1.0, site="span"))}
    H.write_json({"model": mname, "conditions": list(CONDS), "n_test_seeds": len(seeds), "selection_rule_of_the_audit": "largest unedited |C| among mask-clean demographic pairs",
                  "this_stage": "every eligible mask-clean demographic pair of every test seed", "utc": H.utc_now()}, H.OUT / ("h3_protocol_%s.json" % mname))
    sysm = P1.system_prompt()
    t0 = time.time(); n = 0
    for sid in seeds:
        pairs = all_demographic_pairs(mname, sid, cdva, c, eligible)
        rows = []
        for pair in pairs:
            ta = I.build_input(tok, pair["prompt_A"], "chat", sysm); tb = I.build_input(tok, pair["prompt_B"], "chat", sysm)
            pa = I.resolve_span_positions(tok, ta, pair["swap_A"]); pb = I.resolve_span_positions(tok, tb, pair["swap_B"])
            tid = P2.target_token_id(tok, pair)
            others = G1.option_first_ids(tok, pair["prompt_B"])
            for cond_id in CONDS:
                if (mname, sid, pair["A"], pair["B"], cond_id) in done:
                    continue
                row = {"model_name": mname, "seed_id": sid, "benchmark": H.K.benchmark_of(sid), "subvariant_A": pair["A"], "subvariant_B": pair["B"], "cond_id": cond_id,
                       "rank_by_Cpre": pair["rank_by_Cpre"], "n_pairs_in_seed": pair["n_pairs_in_seed"], "C_pre_shipped": pair["C_pre_shipped"],
                       "is_selected_pair": (sid, pair["A"], pair["B"]) in selected, "gold_token_id": tid, "status": "ok", "utc": H.utc_now()}
                try:
                    if not pa or not pb or tid is None:
                        row.update({"status": "unresolved", "error": "span or target unresolved"})
                    else:
                        res = G1.full_logit_patch(model, tok, ta, tb, pa, pb, fac[cond_id])
                        if "error" in res:
                            row.update({"status": "error", "error": res["error"]})
                        else:
                            row.update(G1.stats(res["patched"], res["clean"], tid, others))
                            row.update({k: res[k] for k in ("patched_len", "span_len_a", "span_len_b", "energy_target_edit")})
                except Exception as ex:
                    row.update({"status": "error", "error": str(ex)[:300]}); log.exception("h3 %s %s %s", mname, sid, cond_id)
                rows.append(row)
        if rows:
            H.append_rows(path, rows); n += len(rows)
    log.info("h3 %s: %d new rows in %.1f min", mname, n, (time.time() - t0) / 60)
    (H.OUT / ("DONE_h3_%s.txt" % mname)).write_text(H.utc_now(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=H.POOLED_MODELS)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.model, a.smoke)


if __name__ == "__main__":
    main()
