"""
p1_pilot.py -- the small protocol-validation pilot (Next_Plan.md, P1). GPU.

Purpose: decide whether the reported phenomenon survives ONE consistent treatment and honest
scoring, before any confirmation budget is spent. Not final-test evidence.

Design (Next_Plan.md P1, item by item)
  Models     Qwen2.5-7B (severe reported failure) and Llama-3.1-8B (contrast). --smoke uses
             gemma-2-2b-it on 4 seeds. Final models are NOT chosen by effect size.
  Data       24 seeds from the DEV split of reanalysis_v2/split_manifest.csv, 8 per benchmark,
             never from TEST. Written to results/v2/pilot_seeds.csv for manual checking; pass
             --seeds-file to run a manually checked list instead. Per seed, up to three pair
             types from mask-clean slot-c variants: demographic (largest |C_pre|), identity
             (a variant against itself), neutral (the 'person'-style non-demographic
             substitution when the seed has one). A separate fit set of 200 mask-clean pairs
             from the FIT split estimates the rank-1 basis (never dev or test).
  Conditions
    unedited            no hooks
    identity_alpha0     the canonical hook with alpha = 0; asserted identical to unedited
    span_seq_r1         rank-1 projection at every token of the demographic span, prefill
                        only, sequential source state, SAME edit for audit and behaviour
    span_frozen_r1      same basis, frozen (legacy-TL) source state; audit only
    last_token_r1       legacy ErasureContext site (last prefill token and every generated
                        token), labelled as transfer across positions
  Prompt format         --fmt chat (default) or raw; both the audit and the behavioural
                        readout use the chosen format. unedited is additionally audited in
                        the other format so the template effect on C_pre is on record.
  Readouts per item     raw |C|; canonical option log-likelihoods on both prompts of the pair
                        (argmax, margin, entropy) so option-level preference change is read
                        without generation; greedy generation with the research system prompt,
                        raw text saved, JSON-parsed answer mapped to an option INDEX
                        (identity mapped through the swap), validity flag; per-layer removed
                        activation energy; seconds.
  Checks                p1_intervention.identity_checks before anything else; multi-token
                        spans; BOS / chat offsets by offset mapping; left padding in
                        generation; decode-step policy asserted.
  Budget                --gpu-hours-cap (default 8.0) is a wall-clock cap including loading;
                        the runner stops cleanly at the cap and everything done is on disk.

Outputs (Code/CURE/results/v2/)
  pilot_seeds.csv            the 24 seeds with prompts, for manual review
  pilot_per_item.parquet     one row per (model, fmt, condition, seed, pair_type); resumable
  pilot_protocol.json        exact definitions, identity-check results, timings, throughput
  pilot_report.md            decision-gate summary (see Next_Plan.md P1 "Decision gate")

Usage
  python p1_pilot.py --models qwen2.5-7b-instruct llama-3.1-8b-instruct --gpu-hours-cap 8
  python p1_pilot.py --smoke
  python p1_pilot.py --report-only          (rebuild the report from the parquet)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time

import numpy as np
import pandas as pd

import common as K
import p1_intervention as I

log = K.setup_logging("cure.ext.p1.pilot")

OUT = K.OUT_V2
N_PILOT_PER_BENCH = 8
N_FIT_PAIRS = 200
RANK = 1
MAX_NEW_TOKENS = 48
CONDITIONS = ["unedited", "identity_alpha0", "span_seq_r1", "span_frozen_r1", "last_token_r1"]
NEUTRAL_TOKENS = {"person", "person a", "person b", "someone", "the person", "a person"}
OPT_RE = re.compile(r"^\(([A-Z])\)\s*(.+?)\s*$", re.M)


# ---------------------------------------------------------------------------
# data selection (CPU)
# ---------------------------------------------------------------------------

def _pentad_c():
    pen = pd.read_parquet(K.PENTAD_CLEAN)
    return pen[pen["slot"] == "c"].copy(), pen[(pen["slot"] == "a") & (pen["subvariant"] == "surface")].set_index("seed_id")


def options_of(prompt: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in OPT_RE.finditer(prompt)]


def gold_index(prompt: str, gold: str) -> int | None:
    g = str(gold).strip().lower()
    for i, (letter, text) in enumerate(options_of(prompt)):
        if text.strip().lower() == g or letter.lower() == g:
            return i
    return None


def select_seeds(n_per_bench: int, seed: int) -> pd.DataFrame:
    man = pd.read_csv(K.OUT_P0 / "split_manifest.csv")
    dev = man[man["split"] == "dev"]
    rng = random.Random(seed)
    take = []
    for src, g in dev.groupby("seed_source"):
        ids = sorted(g["seed_id"]); rng.shuffle(ids); take += ids[:n_per_bench]
    return dev[dev["seed_id"].isin(take)].sort_values(["seed_source", "seed_id"]).reset_index(drop=True)


def pairs_for_seed(model: str, seed_id: str, cdva: pd.DataFrame, c: pd.DataFrame,
                   eligible_keys: set) -> list[dict]:
    """demographic / identity / neutral pairs for one seed from mask-clean variants."""
    rows = c[c["seed_id"] == seed_id]
    var = {r["subvariant"]: r for _, r in rows.iterrows()}
    cd = cdva[(cdva["model_name"] == model) & (cdva["seed_id"] == seed_id)
              & (cdva["success_flag"] == True) & (cdva["position_fallback_used"] == False)]   # noqa: E712
    cd = cd[[(seed_id, a, b) in eligible_keys for a, b in zip(cd["pair_A_subvariant"], cd["pair_B_subvariant"])]]
    out = []

    def is_neutral(sv):
        t = str(var[sv].get("swap_token", "")).strip().lower().replace("_", " ")
        return t in NEUTRAL_TOKENS

    demo = cd[[not is_neutral(a) and not is_neutral(b) for a, b in zip(cd["pair_A_subvariant"], cd["pair_B_subvariant"])]]
    if len(demo):
        r = demo.iloc[demo["delta_logit"].abs().argmax()]
        out.append({"pair_type": "demographic", "A": r["pair_A_subvariant"], "B": r["pair_B_subvariant"],
                    "C_pre_shipped": float(abs(r["delta_logit"]))})
    neu = cd[[is_neutral(a) != is_neutral(b) for a, b in zip(cd["pair_A_subvariant"], cd["pair_B_subvariant"])]]
    if len(neu):
        r = neu.iloc[0]
        out.append({"pair_type": "neutral", "A": r["pair_A_subvariant"], "B": r["pair_B_subvariant"],
                    "C_pre_shipped": float(abs(r["delta_logit"]))})
    if out:
        a = out[0]["A"]
        out.append({"pair_type": "identity", "A": a, "B": a, "C_pre_shipped": 0.0})
    for o in out:
        for side in ("A", "B"):
            v = var[o[side]]
            o[f"prompt_{side}"] = str(v["prompt_text"]); o[f"swap_{side}"] = str(v.get("swap_token", ""))
            o[f"gold_{side}"] = str(v.get("gold_answer", ""))
    return out


def fit_pairs(model: str, n: int, seed: int) -> list[dict]:
    man = pd.read_csv(K.OUT_P0 / "split_manifest.csv")
    fit_seeds = set(man[man["split"] == "fit"]["seed_id"])
    elig = pd.read_csv(K.OUT_P0 / "pair_sets" / f"{model}_eligible.csv")
    elig = elig[elig["seed_id"].isin(fit_seeds)]
    rng = random.Random(seed)
    take = []
    for bm, g in elig.groupby("benchmark"):
        recs = g.to_dict("records"); rng.shuffle(recs)
        take += recs[:max(1, round(n * len(g) / len(elig)))]
    rng.shuffle(take)
    return take[:n]


# ---------------------------------------------------------------------------
# GPU pieces
# ---------------------------------------------------------------------------

def load(model_name: str):
    import config_cure as C
    from load_osm import load_model
    cfg = dict(C.model_cfg(model_name))
    model, tok = load_model(cfg)
    model.eval()
    cfg["attn_implementation"] = getattr(model.config, "_attn_implementation", None)
    cfg["model_class"] = type(model).__name__
    cfg["dtype"] = str(next(model.parameters()).dtype)
    return cfg, model, tok


def system_prompt() -> str:
    import config_cure as C
    return C.RESEARCH_SYSTEM_PROMPT


def estimate_basis(model, tok, fmt: str, pairs: list[dict], c: pd.DataFrame, rank: int) -> tuple[dict, dict]:
    """Span-position difference vectors over the fit pairs -> centred-SVD basis (the shipped
    estimator) at every layer. Every token of the span contributes a difference."""
    var = {(r["seed_id"], r["subvariant"]): r for _, r in c.iterrows()}
    diffs, n_used, n_skipped = {}, 0, 0
    sysm = system_prompt() if fmt == "chat" else None
    for p in pairs:
        ka, kb = (p["seed_id"], p["subvariant_A"]), (p["seed_id"], p["subvariant_B"])
        if ka not in var or kb not in var:
            n_skipped += 1; continue
        ta = I.build_input(tok, str(var[ka]["prompt_text"]), fmt, sysm)
        tb = I.build_input(tok, str(var[kb]["prompt_text"]), fmt, sysm)
        pa = I.resolve_span_positions(tok, ta, str(var[ka].get("swap_token", "")))
        pb = I.resolve_span_positions(tok, tb, str(var[kb].get("swap_token", "")))
        n = min(len(pa), len(pb))
        if n == 0:
            n_skipped += 1; continue
        _, ha = I.forward_hidden(model, tok, ta, None, None)
        _, hb = I.forward_hidden(model, tok, tb, None, None)
        for l in ha:
            for i in range(n):
                diffs.setdefault(l, []).append((ha[l][pa[i]] - hb[l][pb[i]]).numpy())
        n_used += 1
    basis = I.basis_centred_svd(diffs, rank)
    return basis, {"n_fit_pairs_used": n_used, "n_fit_pairs_skipped": n_skipped,
                   "n_diff_vectors": sum(len(v) for v in diffs.values()) // max(1, len(diffs)),
                   "estimator": "centred_svd", "rank": rank, "site": "span (all tokens)"}


def parse_answer(text: str, prompt: str) -> tuple[int | None, str]:
    """Robust JSON extraction then canonical option mapping. Returns (option index or None,
    reason)."""
    opts = options_of(prompt)
    cand = None
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            j = json.loads(m.group(0)); cand = str(j.get("answer", "")).strip()
        except Exception:
            mm = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
            cand = mm.group(1).strip() if mm else None
    if cand is None:
        cand = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not cand:
        return None, "empty_output"
    cl = cand.lower().strip().strip(".")
    for i, (letter, otext) in enumerate(opts):
        if cl == otext.lower().strip() or cl == f"({letter.lower()})" or cl == letter.lower() \
           or cl.startswith(f"({letter.lower()})"):
            return i, "exact"
    for i, (letter, otext) in enumerate(opts):
        if otext.lower().strip() in cl or cl in otext.lower().strip():
            return i, "substring"
    return None, "unmapped"


def run_item(model, tok, fmt: str, cond: str, basis: dict, pair: dict, target_id_fn) -> dict:
    sysm = system_prompt() if fmt == "chat" else None
    ta = I.build_input(tok, pair["prompt_A"], fmt, sysm)
    tb = I.build_input(tok, pair["prompt_B"], fmt, sysm)
    pa = I.resolve_span_positions(tok, ta, pair["swap_A"])
    pb = I.resolve_span_positions(tok, tb, pair["swap_B"])
    row = {"span_len_A": len(pa), "span_len_B": len(pb), "fmt": fmt, "condition": cond}
    site, alpha, src = {"unedited": (None, 0.0, "sequential"), "identity_alpha0": ("span", 0.0, "sequential"),
                        "span_seq_r1": ("span", 1.0, "sequential"), "span_frozen_r1": ("span", 1.0, "frozen"),
                        "last_token_r1": ("last_token", 1.0, "sequential")}[cond]
    factory = (lambda: I.HookedEdit(model, basis, alpha=alpha, site=site)) if site else None

    # 1. audit: commutator on the target token (first token of gold answer of B, else swap)
    tgt_text = pair["gold_B"] if pair["gold_B"].strip().lower() not in ("unknown", "", "nan") else pair["swap_B"].replace("_", " ")
    tgt_ids = tok(" " + tgt_text.split()[0] if tgt_text.split() else tgt_text, add_special_tokens=False)["input_ids"]
    if not tgt_ids:
        tgt_ids = tok(tgt_text, add_special_tokens=False)["input_ids"]
    t0 = time.time()
    if pa and pb and tgt_ids:
        C, info = I.commutator(model, tok, ta, tb, pa, pb, tgt_ids[0], factory, src)
        row.update({"C": C, "absC": abs(C) if C == C else np.nan, "patched_len": info.get("patched_len"),
                    "logit_patched": info.get("logit_patched"), "logit_clean": info.get("logit_clean"),
                    "energy_audit": (info.get("target_edit") or {}).get("energy_mean_over_layers")})
    else:
        row.update({"C": np.nan, "absC": np.nan, "audit_error": "span or target unresolved"})
    row["sec_audit"] = time.time() - t0

    # 2. option-level preference on both prompts under the same edit (prefill only)
    t0 = time.time()
    for side, text, pos, prompt in (("A", ta, pa, pair["prompt_A"]), ("B", tb, pb, pair["prompt_B"])):
        opts = [f"({l}) {t}" for l, t in options_of(prompt)]
        if not opts:
            continue
        e = factory() if factory else None
        r = I.option_loglik(model, tok, text, opts, e, pos if site == "span" else ([len(tok(text)["input_ids"]) - 1] if site == "last_token" else None))
        gi = gold_index(prompt, pair[f"gold_{side}"])
        row.update({f"opt_argmax_{side}": r["argmax"], f"opt_margin_{side}": r["margin_top1_top2"],
                    f"opt_entropy_{side}": r["entropy"], f"gold_idx_{side}": gi,
                    f"opt_correct_{side}": (r["argmax"] == gi) if gi is not None else None,
                    f"opt_probs_{side}": json.dumps([round(x, 6) for x in r["probs"]])})
    row["opt_pref_changed_AB"] = (row.get("opt_argmax_A") != row.get("opt_argmax_B")) \
        if ("opt_argmax_A" in row and "opt_argmax_B" in row) else None
    row["sec_options"] = time.time() - t0

    # 3. generation under the same edit; raw text saved
    t0 = time.time()
    e = factory() if factory else None
    texts = I.generate_under_edit(model, tok, [ta, tb], e, [pa, pb], MAX_NEW_TOKENS)
    for side, txt, prompt in (("A", texts[0], pair["prompt_A"]), ("B", texts[1], pair["prompt_B"])):
        idx, why = parse_answer(txt, prompt)
        gi = gold_index(prompt, pair[f"gold_{side}"])
        row.update({f"gen_raw_{side}": txt, f"gen_idx_{side}": idx, f"gen_parse_{side}": why,
                    f"gen_valid_{side}": idx is not None,
                    f"gen_correct_{side}": (idx == gi) if (idx is not None and gi is not None) else None})
    row["gen_flip_AB"] = (row["gen_idx_A"] != row["gen_idx_B"]) if (row["gen_valid_A"] and row["gen_valid_B"]) else None
    if e is not None:
        s = e.summary()
        row.update({"energy_gen": s["energy_mean_over_layers"], "decode_step_edits": s["n_decode_step_edits"],
                    "n_edits_gen": s["n_edits"]})
    row["sec_generation"] = time.time() - t0
    return row


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def done_keys(path) -> set:
    if not path.exists():
        return set()
    d = pd.read_parquet(path, columns=["model_name", "fmt", "condition", "seed_id", "pair_type"])
    return set(map(tuple, d.to_numpy().tolist()))


def append_rows(path, rows: list[dict]):
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old, new], ignore_index=True)
    new.to_parquet(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap_defaults_models = ["qwen2.5-7b-instruct", "llama-3.1-8b-instruct"]
    ap.add_argument("--models", nargs="*", default=ap_defaults_models)
    ap.add_argument("--fmt", choices=["chat", "raw"], default="chat")
    ap.add_argument("--gpu-hours-cap", type=float, default=8.0)
    ap.add_argument("--seeds-file", default=None, help="manually checked seed list (csv with seed_id)")
    ap.add_argument("--smoke", action="store_true", help="2 seeds per benchmark (gemma-2-2b-it unless --models is given)")
    ap.add_argument("--n-per-bench", type=int, default=None, help="seeds per benchmark (default %d)" % N_PILOT_PER_BENCH)
    ap.add_argument("--n-fit-pairs", type=int, default=N_FIT_PAIRS)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    per_item = OUT / "pilot_per_item.parquet"
    proto_path = OUT / "pilot_protocol.json"

    if args.report_only:
        write_report(per_item, proto_path); return

    t_start = time.time()
    cap_s = args.gpu_hours_cap * 3600.0
    models = args.models if (args.models and args.models != ap_defaults_models) or not args.smoke else ["gemma-2-2b-it"]
    n_per = args.n_per_bench if args.n_per_bench else (2 if args.smoke else N_PILOT_PER_BENCH)
    seeds_df = pd.read_csv(args.seeds_file) if args.seeds_file else select_seeds(n_per, K.RANDOM_SEED_V2)
    c, surface = _pentad_c()
    seeds_df = seeds_df.merge(surface[["prompt_text"]].rename(columns={"prompt_text": "surface_prompt"}),
                              left_on="seed_id", right_index=True, how="left")
    seeds_df.to_csv(OUT / "pilot_seeds.csv", index=False)
    cdva = K.read_cdva()

    proto = json.loads(proto_path.read_text()) if proto_path.exists() else {}
    proto.update({"generated_utc": K.utc_now(), "fmt": args.fmt, "rank": RANK, "conditions": CONDITIONS,
                  "n_seeds": int(len(seeds_df)), "seeds_manually_checked": bool(args.seeds_file),
                  "max_new_tokens": MAX_NEW_TOKENS, "gpu_hours_cap": args.gpu_hours_cap,
                  "edit_definition": I.__doc__.split("The edit")[1].split("Prompt format")[0].strip(),
                  "models": {}})
    done = done_keys(per_item)

    for mname in models:
        if time.time() - t_start > cap_s:
            log.warning("cap reached before %s", mname); break
        log.info("=== pilot: %s ===", mname)
        t_load = time.time()
        cfg, model, tok = load(mname)
        mrec = proto["models"].setdefault(mname, {})

        mrec.update({k: cfg.get(k) for k in ("hf_id", "attn_implementation", "model_class", "dtype")})
        mrec["sec_load"] = time.time() - t_load
        try:
            elig = pd.read_csv(K.OUT_P0 / "pair_sets" / f"{mname}_eligible.csv")
            elig_keys = set(zip(elig["seed_id"], elig["subvariant_A"], elig["subvariant_B"]))

            # basis on the FIT split
            t0 = time.time()
            fp = fit_pairs(mname, 8 if args.smoke else args.n_fit_pairs, K.RANDOM_SEED_V2)
            basis, binfo = estimate_basis(model, tok, args.fmt, fp, c, RANK)
            binfo["sec_fit"] = time.time() - t0
            mrec["basis"] = binfo
            d_model = next(iter(basis.values())).shape[1] if basis else None
            np.savez_compressed(OUT / f"pilot_basis_{mname}_{args.fmt}_r{RANK}.npz",
                                **{str(l): b for l, b in basis.items()})

            # identity checks on the first two pilot prompts
            sysm = system_prompt() if args.fmt == "chat" else None
            texts, spans = [], []
            for _, s in seeds_df.head(2).iterrows():
                pr = pairs_for_seed(mname, s["seed_id"], cdva, c, elig_keys)
                if pr:
                    t = I.build_input(tok, pr[0]["prompt_A"], args.fmt, sysm)
                    texts.append(t); spans.append(I.resolve_span_positions(tok, t, pr[0]["swap_A"]))
            if texts:
                chk = I.identity_checks(model, tok, texts, basis, spans)
                mrec["identity_checks"] = chk
                log.info("identity checks %s: pass_all=%s %s", mname, chk.get("pass_all"), chk)
                if not chk.get("pass_all"):
                    log.error("IDENTITY CHECKS FAILED on %s: stop and fix the implementation (P1 gate)", mname)
                    proto_path.write_text(json.dumps(proto, indent=2, default=K._json_default))
                    continue

            # items
            rows, n_done_here = [], 0
            for _, s in seeds_df.iterrows():
                if time.time() - t_start > cap_s:
                    log.warning("cap reached at seed %s", s["seed_id"]); break
                pr = pairs_for_seed(mname, s["seed_id"], cdva, c, elig_keys)
                if not pr:
                    log.warning("no clean pairs for %s on %s", s["seed_id"], mname); continue
                for pair in pr:
                    for cond in CONDITIONS:
                        key = (mname, args.fmt, cond, s["seed_id"], pair["pair_type"])
                        if key in done:
                            continue
                        t0 = time.time()
                        try:
                            r = run_item(model, tok, args.fmt, cond, basis, pair, None)
                        except Exception as exc:
                            r = {"error": str(exc)[:300], "condition": cond, "fmt": args.fmt}
                            log.error("item failed %s: %s", key, str(exc)[:200])
                        r.update({"model_name": mname, "seed_id": s["seed_id"], "benchmark": s["seed_source"],
                                  "pair_type": pair["pair_type"], "subvariant_A": pair["A"], "subvariant_B": pair["B"],
                                  "C_pre_shipped": pair["C_pre_shipped"], "sec_total": time.time() - t0,
                                  "rank": RANK, "ts": K.utc_now()})
                        rows.append(r); done.add(key); n_done_here += 1
                        if len(rows) >= 10:
                            append_rows(per_item, rows); rows = []
                # template effect on C_pre: unedited in the OTHER fmt, demographic pair only
                other = "raw" if args.fmt == "chat" else "chat"
                key = (mname, other, "unedited", s["seed_id"], "demographic")
                dp = [p for p in pr if p["pair_type"] == "demographic"]
                if dp and key not in done:
                    try:
                        r = run_item(model, tok, other, "unedited", basis, dp[0], None)
                        r.update({"model_name": mname, "seed_id": s["seed_id"], "benchmark": s["seed_source"],
                                  "pair_type": "demographic", "subvariant_A": dp[0]["A"], "subvariant_B": dp[0]["B"],
                                  "C_pre_shipped": dp[0]["C_pre_shipped"], "rank": RANK, "ts": K.utc_now()})
                        rows.append(r); done.add(key)
                    except Exception as exc:
                        log.error("template-effect item failed: %s", str(exc)[:160])
            append_rows(per_item, rows)
            mrec["n_items_this_run"] = n_done_here
            mrec["sec_model_total"] = time.time() - t_load
        finally:
            from load_osm import unload_model
            unload_model(mname)
        proto_path.write_text(json.dumps(proto, indent=2, default=K._json_default))

    proto["sec_wall_total"] = time.time() - t_start
    proto_path.write_text(json.dumps(proto, indent=2, default=K._json_default))
    write_report(per_item, proto_path)


def write_report(per_item, proto_path) -> None:
    if not per_item.exists():
        log.warning("no per-item results yet"); return
    d = pd.read_parquet(per_item)
    proto = json.loads(proto_path.read_text()) if proto_path.exists() else {}
    lines = ["# P1 pilot report", "", "Generated %s. Development data only; never final-test evidence." % K.utc_now(), ""]
    for m, mrec in proto.get("models", {}).items():
        ic = mrec.get("identity_checks", {})
        lines += ["## %s" % K.DISPLAY.get(m, m), "",
                  "- load %.0fs; basis fit %s; identity checks pass_all = **%s**" % (
                      mrec.get("sec_load", 0), mrec.get("basis", {}), ic.get("pass_all"))]
        if ic:
            lines += ["  - alpha=0 max |dlogit| = %.2e; rank-0 = %.2e; non-span hidden dev = %.2e; "
                      "decode-step edits under span = %s" % (ic.get("max_abs_logit_dev_alpha0", np.nan),
                                                              ic.get("max_abs_logit_dev_rank0", np.nan),
                                                              ic.get("max_abs_hidden_dev_nonspan_first_edited_layer", np.nan),
                                                              ic.get("decode_step_edits_under_span_policy"))]
        lines.append("")
    d = d[d.get("error").isna()] if "error" in d else d
    g = d.groupby(["model_name", "fmt", "pair_type", "condition"])
    summ = g.agg(n=("seed_id", "size"), absC_mean=("absC", "mean"),
                 opt_correct_A=("opt_correct_A", "mean"), opt_correct_B=("opt_correct_B", "mean"),
                 opt_pref_changed=("opt_pref_changed_AB", "mean"),
                 gen_valid_A=("gen_valid_A", "mean"), gen_valid_B=("gen_valid_B", "mean"),
                 gen_correct_A=("gen_correct_A", "mean"), gen_flip=("gen_flip_AB", "mean"),
                 energy_gen=("energy_gen", "mean"), sec=("sec_total", "mean")).reset_index()
    lines += ["## Per-condition means (n = pilot items)", "", summ.round(3).to_markdown(index=False), ""]
    # decision gate
    # identity pairs (a == b): every convention that is a clean null gives |C| == 0 here. The
    # frozen source state (legacy TransformerLens) injects the CLEAN run's projected activation
    # into the SEQUENTIALLY edited run, so it is not null even on identity pairs: its |C|
    # measures the edit's own through-depth effect, not the demographic contrast.
    idp = d[d["pair_type"] == "identity"].groupby(["model_name", "condition"])["absC"].agg(["max", "mean", "size"]).reset_index()
    if len(idp):
        lines += ["## Identity pairs (a == b): |C| by condition", "",
                  "A convention that is a clean null must give 0 here. span_frozen_r1 (legacy TransformerLens "
                  "source state) injects the clean run's projected activation into the sequentially edited run, "
                  "so a non-zero value there is the edit's own through-depth effect, not a demographic effect; "
                  "the sequential convention used by every other edited condition is null by construction.", "",
                  idp.round(4).to_markdown(index=False), ""]
    lines += ["## Decision gate (Next_Plan.md P1)", ""]
    for m in d["model_name"].unique():
        dm = d[(d["model_name"] == m) & (d["pair_type"] == "demographic")]
        # the edited conditions run in ONE prompt format; the unedited reference is restricted to
        # that format (unedited is additionally audited in the other format for the record)
        fmts = dm[dm["condition"] != "unedited"]["fmt"].dropna().unique() if "fmt" in dm else []
        ref_fmt = fmts[0] if len(fmts) else None
        def mean_of(cond, col):
            x = dm[dm["condition"] == cond]
            if cond == "unedited" and ref_fmt is not None and "fmt" in x:
                x = x[x["fmt"] == ref_fmt]
            x = x[col]
            return float(x.mean()) if len(x) else float("nan")
        un_c, sp_c, lt_c = (mean_of(c_, "absC") for c_ in ("unedited", "span_seq_r1", "last_token_r1"))
        un_a, sp_a, lt_a = (mean_of(c_, "gen_correct_A") for c_ in ("unedited", "span_seq_r1", "last_token_r1"))
        un_v, sp_v, lt_v = (mean_of(c_, "gen_valid_A") for c_ in ("unedited", "span_seq_r1", "last_token_r1"))
        fr_c = mean_of("span_frozen_r1", "absC")
        lines += ["- **%s** mean |C|: unedited %.3f, span-sequential %.3f, span-frozen %.3f, last-token %.3f" % (
                      K.DISPLAY.get(m, m), un_c, sp_c, fr_c, lt_c),
                  "  gen accuracy (A): unedited %.3f, span %.3f, last-token %.3f; validity: %.2f / %.2f / %.2f" % (
                      un_a, sp_a, lt_a, un_v, sp_v, lt_v)]
        if np.isfinite(sp_a) and np.isfinite(lt_a) and np.isfinite(un_a):
            if (un_a - lt_a) > 0.10 and (un_a - sp_a) < 0.05:
                lines.append("  -> damage appears under the last-token transfer edit but NOT under the matched "
                             "span edit: the old explanation is unsupported (position transfer / implementation).")
            elif (un_a - sp_a) > 0.10:
                lines.append("  -> damage persists under the matched span edit: proceed to P2 with this treatment.")
            else:
                lines.append("  -> no meaningful damage under either edit on these pilot items.")
        if np.isfinite(un_v) and np.isfinite(lt_v) and (un_v - lt_v) > 0.2:
            lines.append("  -> a large share of the old 'collapse' is output INVALIDITY (parse failure), which must "
                         "be reported as failure, not as accuracy.")
    (OUT / "pilot_report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", K.rel(OUT / "pilot_report.md"))


if __name__ == "__main__":
    main()
