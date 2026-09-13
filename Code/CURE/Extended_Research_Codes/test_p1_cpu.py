"""
test_p1_cpu.py -- functional tests of p1_intervention on CPU with a tiny random Llama, plus
CPU checks of the pilot's selection logic with the real Qwen tokenizer.

These do not test scientific claims; they test that the edit does what its definition says:
  1. alpha=0 and rank-0 are identities (logits bit-for-bit within tolerance)
  2. a span edit changes ONLY the span positions at the first edited layer
  3. the span policy never fires at decode steps; last_token fires at every decode step
  4. the projection removes the component along U: (h' . u) == 0 for each basis row
  5. commutator: identity pair (a == b) gives C == 0 under every source state
  6. option_loglik returns finite scores whose probabilities sum to 1
  7. resolve_span_positions finds multi-token spans under raw and chat formats on Qwen
  8. pilot pair selection yields demographic / identity / neutral pairs from the manifest

Run:  python test_p1_cpu.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import common as K
import p1_intervention as I

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print("%-58s %s %s" % (name, "PASS" if cond else "FAIL", detail))


def load_cached_tokenizer(repo_id: str):
    """Load a tokenizer from the local HF cache even when the snapshot holds tokenizer files
    only (no config.json), which transformers>=5 otherwise refuses offline."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
    except Exception:
        from huggingface_hub import scan_cache_dir
        for r in scan_cache_dir().repos:
            if r.repo_id == repo_id:
                snap = sorted(r.revisions, key=lambda x: x.last_modified)[-1].snapshot_path
                return AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
        raise


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY)
    model = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32).eval()
    layers = I._decoder_layers(model)
    d = model.config.hidden_size
    L = len(layers)
    print("tiny model: %d layers, d=%d" % (L, d))

    text = "The committee decided to hire the Colombian boy for the position. Answer: (A) yes (B) no"
    pos = I.resolve_span_positions(tok, text, "the_colombian_boy")
    check("span resolved (multi-token)", len(pos) >= 2, str(pos))

    basis = I.basis_random(d, range(L), rank=2, seed=1)

    # 1-3. identity checks
    chk = I.identity_checks(model, tok, [text], basis, [pos], tol=1e-5)
    check("alpha=0 identity", chk["pass_alpha0"], "%.2e" % chk["max_abs_logit_dev_alpha0"])
    check("rank-0 identity", chk["pass_rank0"], "%.2e" % chk["max_abs_logit_dev_rank0"])
    check("non-span untouched at first edited layer", chk["pass_nonspan_untouched"],
          "%.2e" % chk["max_abs_hidden_dev_nonspan_first_edited_layer"])
    check("span actually edited", chk["span_actually_edited"],
          "%.2e" % chk["max_abs_hidden_dev_span_first_edited_layer"])
    check("no decode-step edits under span policy", chk["pass_no_decode_edit_under_span"],
          str(chk["decode_step_edits_under_span_policy"]))

    # 3b. last_token fires at decode steps
    e = I.HookedEdit(model, basis, alpha=1.0, site="last_token")
    _ = I.generate_under_edit(model, tok, [text], e, [pos], max_new_tokens=5)
    check("last_token fires at decode steps", e.n_decode_edits > 0, str(e.n_decode_edits))

    # 4. projection removes the U component at the span (first edited layer, alpha=1)
    e = I.HookedEdit(model, basis, alpha=1.0, site="span")
    _, hs = I.forward_hidden(model, tok, text, e, pos)
    l0 = min(basis)
    U = basis[l0]
    resid = np.abs(hs[l0][pos].numpy() @ U.T).max()
    check("projected component along U is zero at span", resid < 1e-4, "%.2e" % resid)
    en = e.energy_by_layer()
    check("energy ratio in (0,1)", all(0 < v < 1 for v in en.values()), "%.3f mean" % np.mean(list(en.values())))

    # 4b. batched generation with LEFT padding: rows of different length, span positions in
    # unpadded coordinates must land on the span tokens after the per-row pad offset
    text_short = "Hire the Colombian boy. Answer: (A) yes (B) no"
    pos_short = I.resolve_span_positions(tok, text_short, "the_colombian_boy")
    e = I.HookedEdit(model, basis, alpha=1.0, site="span")
    side = tok.padding_side; tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    enc = tok([text, text_short], return_tensors="pt", padding=True, add_special_tokens=True)
    tok.padding_side = side
    pad_off = (enc["attention_mask"].shape[1] - enc["attention_mask"].sum(dim=1)).tolist()
    _ = I.generate_under_edit(model, tok, [text, text_short], e, [pos, pos_short], max_new_tokens=2)
    first_layer = min(basis)
    edited = {(r["row"], r["position"]) for r in e.changed_log if r["layer"] == first_layer and r["prefill"]}
    expected = {(0, p + pad_off[0]) for p in pos} | {(1, p + pad_off[1]) for p in pos_short}
    check("left-padded batch: edited positions == span + pad offset", edited == expected,
          "pad_off=%s edited=%d expected=%d" % (pad_off, len(edited), len(expected)))
    ids = enc["input_ids"]
    decoded = tok.decode([int(ids[1, p + pad_off[1]]) for p in pos_short])
    check("left-padded batch: edited tokens decode to the span", "colombian" in decoded.lower(), repr(decoded))

    # 5. commutator on an identity pair is zero under both source states
    tgt = tok(" yes", add_special_tokens=False)["input_ids"][0]
    for src in ("sequential", "frozen"):
        C, info = I.commutator(model, tok, text, text, pos, pos, tgt,
                               lambda: I.HookedEdit(model, basis, 1.0, "span"), src)
        check("commutator identity pair == 0 (%s)" % src, abs(C) < 1e-4, "%.2e" % C)
    # unedited commutator on a genuinely different pair is finite
    text_b = text.replace("Colombian boy", "American")
    pos_b = I.resolve_span_positions(tok, text_b, "American")
    C, info = I.commutator(model, tok, text, text_b, pos, pos_b, tgt, None, "sequential")
    check("commutator finite on a real pair", np.isfinite(C), "C=%.4f patched_len=%s" % (C, info["patched_len"]))

    # 6. option loglik
    r = I.option_loglik(model, tok, text, ["(A) yes", "(B) no"], I.HookedEdit(model, basis, 1.0, "span"), pos)
    check("option probs sum to 1", abs(sum(r["probs"]) - 1) < 1e-6, str(np.round(r["probs"], 3)))
    check("option logliks finite", all(np.isfinite(r["loglik_sum"])), "")

    # 7. real Qwen tokenizer, raw and chat formats, on a real pentad prompt
    import pandas as pd
    qtok = load_cached_tokenizer("Qwen/Qwen2.5-7B-Instruct")
    pen = pd.read_parquet(K.PENTAD_CLEAN)
    row = pen[(pen["seed_id"] == "bbq_b6c251ec") & (pen["subvariant"] == "the_colombian_boy")].iloc[0]
    for fmt in ("raw", "chat"):
        t = I.build_input(qtok, str(row["prompt_text"]), fmt, "You are a test system." if fmt == "chat" else None)
        p = I.resolve_span_positions(qtok, t, str(row["swap_token"]))
        ids = qtok(t, add_special_tokens=True)["input_ids"]
        back = qtok.decode([ids[i] for i in p])
        check("Qwen %s: span resolved to '%s'" % (fmt, back.strip()), "colombian" in back.lower(), str(p))
    check("Qwen chat offset > raw offset",
          I.resolve_span_positions(qtok, I.build_input(qtok, str(row["prompt_text"]), "chat", "sys"), str(row["swap_token"]))[0]
          > I.resolve_span_positions(qtok, str(row["prompt_text"]), str(row["swap_token"]))[0], "")

    # 8. pilot selection
    import p1_pilot as P
    seeds = P.select_seeds(8, K.RANDOM_SEED_V2)
    check("24 pilot seeds from dev split", len(seeds) == 24 and set(seeds["split"]) == {"dev"},
          str(seeds["seed_source"].value_counts().to_dict()))
    c, _ = P._pentad_c(); cdva = K.read_cdva()
    elig = pd.read_csv(K.OUT_P0 / "pair_sets" / "qwen2.5-7b-instruct_eligible.csv")
    keys = set(zip(elig["seed_id"], elig["subvariant_A"], elig["subvariant_B"]))
    types = {}
    for sid in seeds["seed_id"]:
        for pr in P.pairs_for_seed("qwen2.5-7b-instruct", sid, cdva, c, keys):
            types[pr["pair_type"]] = types.get(pr["pair_type"], 0) + 1
    check("pair types present", types.get("demographic", 0) >= 20 and types.get("identity", 0) >= 20, str(types))
    fp = P.fit_pairs("qwen2.5-7b-instruct", 200, K.RANDOM_SEED_V2)
    man = pd.read_csv(K.OUT_P0 / "split_manifest.csv")
    fit_seeds = set(man[man["split"] == "fit"]["seed_id"])
    check("fit pairs only from fit split", len(fp) == 200 and all(p["seed_id"] in fit_seeds for p in fp), str(len(fp)))
    gi = P.gold_index(str(row["prompt_text"]), "Unknown")
    check("gold index maps 'Unknown' to option C", gi == 2, str(gi))
    idx, why = P.parse_answer('{"answer": "The Nigerian boy", "confidence": 0.9, "rationale": "x"}', str(row["prompt_text"]))
    check("parse_answer maps JSON answer to option index", idx == 1 and why == "exact", "%s %s" % (idx, why))
    idx, why = P.parse_answer("", str(row["prompt_text"]))
    check("parse_answer flags empty output", idx is None and why == "empty_output", why)

    print("\nALL PASS" if ok_all else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
