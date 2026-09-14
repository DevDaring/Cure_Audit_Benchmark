"""
g1_invariant.py -- A1 of Submission2/Next_Plan.md (ICLR-feedback cycle): re-read the pooled
audit's patching effect with shift-invariant statistics. GPU, one model at a time.

The pooled audit (results/v2, P2) recorded C(a,b) = logit_g(swap) - logit_g(clean) on the RAW
gold-token logit at the final prompt position. An orthogonal projection lowers the norm of
the hidden state; depending on the final normalisation that can shift every logit by the same
amount and move C without moving the softmax. This stage repeats exactly the same passes (same
test seeds, same demographic pair per seed, same bases, sequential source convention, same
truncation rule) and keeps the FULL final-position logit vector, from which it computes

  C_raw     logit_g(patched) - logit_g(clean)                (must equal the stored C)
  C_logp    log p_g(patched) - log p_g(clean)                (full-vocabulary log-softmax)
  C_margin  [logit_g - LSE_{j != g} logit_j](patched) - same(clean), over the first tokens
            of the other options of prompt b (the pooled-audit analogue of the margin M)
  shift     mean over the vocabulary of logit(patched) - logit(clean); and the difference of
            the two log-partition functions LSE_V(patched) - LSE_V(clean), i.e. the confound

for the conditions the confirmatory claims rest on: unedited, the rank-one span erasure, the
three rank-matched random subspaces, and the two LEACE fits (swap-side, frozen and
sequential). Optionally (--extra-leace) the conditioned man/woman LEACE erasers of g3.

Rows: results/feedback_20260915/g1_invariant_<model>.parquet, one per (seed, condition);
resumable. Usage: python g1_invariant.py --model llama-3.1-8b-instruct [--smoke]
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import nn_common as C
import p1_intervention as I     # noqa: E402
import p1_pilot as P1           # noqa: E402
import p2_controlled_erasure as P2   # noqa: E402

log = C.log
ROW_KEY = ["model_name", "seed_id", "cond_id"]
CONDS = ["unedited_r0_a0", "cure_centred_svd_r1_a1", "random_ortho_1_r1_a1", "random_ortho_2_r1_a1", "random_ortho_3_r1_a1",
         "leace_faithful_r1_a1", "leace_sequential_r1_a1"]


def stored_pairs(model: str) -> pd.DataFrame:
    """The exact (seed, demographic pair) rows the pooled audit evaluated on the test split."""
    p = pd.read_parquet(C.V2 / "p2_per_item.parquet",
                        columns=["model_name", "phase", "pair_type", "cond_id", "seed_id", "subvariant_A", "subvariant_B", "C", "absC",
                                 "patched_len", "span_len_A", "span_len_B", "benchmark"])
    p = p[(p.model_name == model) & (p.phase == "test") & (p.pair_type == "demographic")]
    keep = p[p.cond_id.isin(CONDS)].pivot_table(index=["seed_id", "subvariant_A", "subvariant_B", "benchmark"], columns="cond_id", values="C").reset_index()
    return keep


def build_pair(c: pd.DataFrame, seed_id: str, sa: str, sb: str) -> dict:
    var = {r["subvariant"]: r for _, r in c[c["seed_id"] == seed_id].iterrows()}
    o = {"seed_id": seed_id, "A": sa, "B": sb}
    for side, sv in (("A", sa), ("B", sb)):
        v = var[sv]
        o["prompt_%s" % side] = str(v["prompt_text"]); o["swap_%s" % side] = str(v.get("swap_token", "")); o["gold_%s" % side] = str(v.get("gold_answer", ""))
    return o


def option_first_ids(tok, prompt_b: str) -> list[int]:
    """First token of each option text of prompt b, with the same ' ' + first word rule the
    pooled audit used for the gold token (p2_controlled_erasure.target_token_id)."""
    out = []
    for _, text in P1.options_of(prompt_b):
        t = text.strip()
        ids = tok(" " + t.split()[0] if t.split() else t, add_special_tokens=False)["input_ids"]
        if not ids:
            ids = tok(t, add_special_tokens=False)["input_ids"]
        out.append(int(ids[0]) if ids else -1)
    return out


def full_logit_patch(model, tok, text_a: str, text_b: str, pos_a: list[int], pos_b: list[int], edit_factory) -> dict:
    """The sequential-convention patch of p1_intervention.commutator, returning the full
    final-position logit vectors of the patched and the clean run (float32, cpu)."""
    import torch
    dec = I._decoder_layers(model)
    L = list(range(len(dec)))
    n = min(len(pos_a), len(pos_b))
    if n == 0:
        return {"error": "empty span"}
    pa, pb = pos_a[:n], pos_b[:n]
    e_a = edit_factory() if edit_factory else None
    _, hs_a = I.forward_hidden(model, tok, text_a, e_a, pa)
    src = {l: hs_a[l][pa, :].clone() for l in L}
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    handles = []

    def make_patch(l):
        s = src[l].to(dev).to(dt)

        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] > 1:
                hs[0, pb, :] = s
            return output
        return hook

    enc_b = I._to(model, I._encode(tok, text_b)[0])
    e1 = edit_factory() if edit_factory else None
    if e1 is not None:
        e1.set_targets([pb], [0])
    with torch.no_grad(), (e1 if e1 is not None else I.no_edit()):
        for l in L:
            handles.append(dec[l].register_forward_hook(make_patch(l)))
        try:
            lp = model(**enc_b).logits[0, -1, :].float().cpu()
        finally:
            for h in handles:
                h.remove()
    e2 = edit_factory() if edit_factory else None
    if e2 is not None:
        e2.set_targets([pb], [0])
    with torch.no_grad(), (e2 if e2 is not None else I.no_edit()):
        lc = model(**enc_b).logits[0, -1, :].float().cpu()
    return {"patched": lp.numpy(), "clean": lc.numpy(), "patched_len": n, "span_len_a": len(pos_a), "span_len_b": len(pos_b),
            "energy_target_edit": (e2.summary()["energy_mean_over_layers"] if e2 is not None else 0.0)}


def stats(lp: np.ndarray, lc: np.ndarray, g: int, others: list[int]) -> dict:
    def lse(x):
        m = x.max(); return float(m + np.log(np.exp(x - m).sum()))
    others = [o for o in others if o >= 0 and o != g]
    out = {"C_raw": float(lp[g] - lc[g]),
           "C_logp": float((lp[g] - lse(lp)) - (lc[g] - lse(lc))),
           "logZ_shift": lse(lp) - lse(lc),
           "mean_logit_shift": float((lp - lc).mean())}
    if others:
        mo_p = lse(lp[others]); mo_c = lse(lc[others])
        out["C_margin"] = float((lp[g] - mo_p) - (lc[g] - mo_c))
    else:
        out["C_margin"] = float("nan")
    out["n_other_options"] = len(others)
    return out


def factories(model, store: P2.BasisStore) -> dict:
    fac = {"unedited_r0_a0": None,
           "cure_centred_svd_r1_a1": (lambda: I.HookedEdit(model, store.rows("cure_centred_svd", 1), alpha=1.0, site="span"))}
    for k in (1, 2, 3):
        name = "random_ortho_%d" % k
        if name in store.proj:
            fac["random_ortho_%d_r1_a1" % k] = (lambda n=name: I.HookedEdit(model, store.rows(n, 1), alpha=1.0, site="span"))
    import p2_leace_faithful as LF
    if store.leace:
        fac["leace_faithful_r1_a1"] = (lambda: LF.LeaceAffineEdit(model, store.leace, alpha=1.0, site="span"))
    if store.leace_seq:
        fac["leace_sequential_r1_a1"] = (lambda: LF.LeaceAffineEdit(model, store.leace_seq, alpha=1.0, site="span"))
    return fac


def extra_leace_factories(model, mname: str) -> dict:
    """Conditioned man/woman erasers of g3 (if present) and the F3 coherent eraser (final cycle)."""
    import p2_leace_faithful as LF
    out = {}
    for tag in ("swap_small", "swap_300", "swap_max", "swap_max_noshrink", "gender_large", "gender_small"):
        p3 = C.OUT / ("g3_leace_%s_%s.npz" % (mname, tag))
        if p3.exists():
            er = LF.erasers_from_npz(np.load(p3))
            out["leace_%s_r1_a1" % tag] = (lambda e=er: LF.LeaceAffineEdit(model, e, alpha=1.0, site="span"))
    pf = C.FINAL / ("f3_baseline_%s.npz" % mname)
    if pf.exists():
        er = LF.erasers_from_npz(np.load(pf))
        out["leace_mw_f3_r1_a1"] = (lambda e=er: LF.LeaceAffineEdit(model, e, alpha=1.0, site="span"))
    return out


def run(mname: str, smoke: bool, extra_leace: bool) -> None:
    C.ensure_out()
    path = C.OUT / ("g1_invariant_%s.parquet" % mname)
    pairs = stored_pairs(mname)
    if smoke:
        pairs = pairs.head(2)
    done = C.done_keys(path, ROW_KEY)
    c, _ = P1._pentad_c()
    cfg, model, tok = C.load_model(mname)
    store = P2.BasisStore.load(mname, "chat")
    if store is None:
        raise SystemExit("no P2 basis store for %s" % mname)
    fac = factories(model, store)
    if extra_leace:
        fac.update(extra_leace_factories(model, mname))
    sysm = P1.system_prompt()
    t0 = time.time(); n = 0
    for r in pairs.itertuples(index=False):
        pair = build_pair(c, r.seed_id, r.subvariant_A, r.subvariant_B)
        ta = I.build_input(tok, pair["prompt_A"], "chat", sysm); tb = I.build_input(tok, pair["prompt_B"], "chat", sysm)
        pa = I.resolve_span_positions(tok, ta, pair["swap_A"]); pb = I.resolve_span_positions(tok, tb, pair["swap_B"])
        tid = P2.target_token_id(tok, pair)
        others = option_first_ids(tok, pair["prompt_B"])
        rows = []
        for cond_id, f in fac.items():
            if (mname, r.seed_id, cond_id) in done:
                continue
            row = {"model_name": mname, "seed_id": r.seed_id, "benchmark": r.benchmark, "cond_id": cond_id,
                   "subvariant_A": r.subvariant_A, "subvariant_B": r.subvariant_B, "gold_token_id": tid,
                   "C_stored": float(getattr(r, cond_id)) if cond_id in pairs.columns and pd.notna(getattr(r, cond_id, np.nan)) else float("nan"),
                   "status": "ok", "utc": C.utc_now()}
            try:
                if not pa or not pb or tid is None:
                    row.update({"status": "unresolved", "error": "span or target unresolved"})
                else:
                    res = full_logit_patch(model, tok, ta, tb, pa, pb, f)
                    if "error" in res:
                        row.update({"status": "error", "error": res["error"]})
                    else:
                        row.update(stats(res["patched"], res["clean"], tid, others))
                        row.update({k: res[k] for k in ("patched_len", "span_len_a", "span_len_b", "energy_target_edit")})
                        row["equal_span_len"] = bool(res["span_len_a"] == res["span_len_b"])
                        row["C_raw_matches_stored"] = bool(np.isfinite(row["C_stored"]) and abs(row["C_raw"] - row["C_stored"]) < 0.1)
            except Exception as ex:
                row.update({"status": "error", "error": str(ex)[:300]}); log.exception("g1 %s %s %s", mname, r.seed_id, cond_id)
            rows.append(row)
        if rows:
            C.append_rows(path, rows); n += len(rows)
    log.info("g1 %s: %d new rows in %.1f min", mname, n, (time.time() - t0) / 60)
    (C.OUT / ("DONE_g1_%s.txt" % mname)).write_text(C.utc_now(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=C.ALL_MODELS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--extra-leace", action="store_true", help="also the conditioned man/woman LEACE erasers of g3 and the F3 eraser")
    a = ap.parse_args()
    run(a.model, a.smoke, a.extra_leace)


if __name__ == "__main__":
    main()
