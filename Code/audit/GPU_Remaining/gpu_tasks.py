"""
gpu_tasks.py -- the three GPU-only review tasks (T0.2, T0.1, T1.2).

All three reuse the production code paths:
  - GPU_CPU.utils_attention.patch_activation / _ensure_*   (T0.2)
  - GPU_CPU.osm_behavioral.evaluate_osm_model              (T0.1, T1.2)
Each task accepts `limit` so the dry run can exercise the real code on 2 instances.
"""

import logging
import random
import re
import uuid

import numpy as np
import pandas as pd

from GPU_CPU.utils_attention import patch_activation, _get_token_position
from GPU_CPU.cdva_patching import _get_bias_answer
from GPU_CPU.osm_behavioral import evaluate_osm_model

log = logging.getLogger("gpu_tasks")

MIN_EFFECT = 0.5   # |logit_A - logit_B| below this carries no swap effect to recover


def _is_nnsight(model_cfg: dict) -> bool:
    return "nnsight" in str(model_cfg.get("patching_lib", "")).lower()


def get_bias_logit(model, tokenizer, prompt: str, bias_answer: str, patching_lib: str):
    """Unpatched bias-answer logit at the last position. Mirrors patch_activation's read."""
    import torch
    if "nnsight" in str(patching_lib).lower():
        from GPU_CPU.utils_attention import _ensure_nnsight_model
        nn_model = _ensure_nnsight_model(model, tokenizer)
        ids = tokenizer.encode(bias_answer, add_special_tokens=False)
        if not ids:
            return None
        with nn_model.trace(prompt):
            logits = nn_model.lm_head.output.save()
        return float(logits.value[0, -1, ids[0]].item())
    else:
        from GPU_CPU.utils_attention import _ensure_hooked_transformer
        tl = _ensure_hooked_transformer(model, tokenizer)
        with torch.no_grad():
            logits = tl(prompt)
        bt = tl.to_tokens(bias_answer, prepend_bos=False)[0]
        if len(bt) == 0:
            return None
        return float(logits[0, -1, bt[0].item()].item())


# ------------------------------- T0.2 recovery -------------------------------

def run_t02_recovery(model_cfg, model, tokenizer, cdva_pairs, pentad_df, limit=None):
    """
    Patch-site recovery: for each CDVA pair (A,B), recovery = delta / (logit_A - logit_B).
    delta = patch_activation(A->B). A recovery near 1 proves the patched protected
    position carries the demographic effect (the site is not inert).
    """
    name = model_cfg["name"]
    lib = model_cfg["patching_lib"]
    # Fail loud and early if the patching library cannot be imported. Without
    # this, a missing transformer_lens raises the SAME ImportError on every one
    # of the ~6000 pairs (all swallowed below), yielding a silent empty frame
    # that gets marked "done" -- the bug that produced empty llama/gemma T0.2.
    if "nnsight" in str(lib).lower():
        import nnsight  # noqa: F401
    else:
        import transformer_lens  # noqa: F401
    sub = cdva_pairs[cdva_pairs["model_name"] == name]
    if limit:
        sub = sub.head(limit)

    # slot-c prompt + swap-token lookup, and per-seed bias_answer
    pc = pentad_df[pentad_df["slot"] == "c"]
    look = {(r["seed_id"], r["subvariant"]): (str(r["prompt_text"]), str(r.get("swap_token", "")))
            for _, r in pc.iterrows()}
    bias_by_seed = {}
    for sid, grp in pc.groupby("seed_id"):
        bias_by_seed[sid] = _get_bias_answer(grp)

    rows = []
    for _, pr in sub.iterrows():
        sid = pr["seed_id"]
        ka = (sid, pr["pair_A_subvariant"]); kb = (sid, pr["pair_B_subvariant"])
        if ka not in look or kb not in look:
            continue
        pa, swa = look[ka]; pb, swb = look[kb]
        bias = bias_by_seed.get(sid, "")
        if not bias or not pa.strip() or not pb.strip() or pa.strip() == pb.strip():
            continue
        try:
            pos_a = _get_token_position(tokenizer, pa, swa) if swa else None
            pos_b = _get_token_position(tokenizer, pb, swb) if swb else None
            if pos_a is None or pos_b is None:
                continue
            la = get_bias_logit(model, tokenizer, pa, bias, lib)
            lb = get_bias_logit(model, tokenizer, pb, bias, lib)
            if la is None or lb is None:
                continue
            delta = patch_activation(model, tokenizer, pa, pb, pos_a, pos_b, bias, lib)
            denom = la - lb
            used = abs(denom) >= MIN_EFFECT
            recovery = float(delta / denom) if used else float("nan")
            rows.append({
                "model_name": name, "seed_id": sid,
                "subvariant_A": pr["pair_A_subvariant"], "subvariant_B": pr["pair_B_subvariant"],
                "logit_A": la, "logit_B": lb, "swap_effect": denom,
                "patch_delta": float(delta), "recovery_fraction": recovery, "used": bool(used),
            })
        except Exception as exc:
            log.warning("recovery pair failed (seed %s): %s", sid, str(exc)[:120])
    return pd.DataFrame(rows)


def summarize_recovery(df: pd.DataFrame) -> dict:
    out = {}
    # An empty recovery frame (0 rows -> no columns) must not crash the run:
    # df.groupby("model_name") on a column-less frame raises KeyError. Guard it.
    if df is None or len(df) == 0 or "model_name" not in df.columns:
        return out
    for name, g in df.groupby("model_name"):
        u = g[g["used"]]
        out[name] = {
            "n_pairs": int(len(g)), "n_used": int(len(u)),
            "mean_recovery": float(u["recovery_fraction"].mean()) if len(u) else float("nan"),
            "median_recovery": float(u["recovery_fraction"].median()) if len(u) else float("nan"),
            "frac_recovery_gt_0.5": float((u["recovery_fraction"] > 0.5).mean()) if len(u) else float("nan"),
        }
    return out


# ------------------------------- T0.1 temperature -------------------------------

def run_t01_temperature(model_cfg, model, tokenizer, pentad_df, temperature=1.0, limit=None):
    """Re-run slot-a at the given temperature, 6 samples (0..5), to recompute FM4 at T."""
    name = model_cfg["name"]
    slot_a = pentad_df[(pentad_df["slot"] == "a") & (pentad_df["subvariant"] == "surface")].copy()
    slot_a = slot_a[slot_a["prompt_text"].astype(str).str.strip() != ""]
    if limit:
        slot_a = slot_a.head(limit)
    run_id = f"t01-{uuid.uuid4().hex[:8]}"
    frames = []
    for si in range(6):
        df = evaluate_osm_model(model_cfg, model, tokenizer, slot_a, run_id,
                                temperature=temperature, sample_index=si)
        df["sweep_temperature"] = temperature
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fm4_from_samples(df: pd.DataFrame) -> dict:
    """FM4 incidence: fraction of seeds whose parsed_answer varies across the samples."""
    ok = df[df["success_flag"] == True]  # noqa: E712
    inc = []
    for sid, g in ok.groupby("seed_id"):
        if g["sample_index"].nunique() >= 3:
            inc.append(int(g["parsed_answer"].nunique() > 1))
    return {"n_seeds": len(inc), "fm4_incidence": float(np.mean(inc)) if inc else float("nan")}


# ------------------------------- T1.2 option order -------------------------------

_OPT_RE = re.compile(r"^\(([A-Z])\)\s*(.+?)\s*$")


def _permute_options(prompt: str, rng: random.Random):
    """Shuffle the (A)/(B)/(C) option lines in an MCQ prompt; returns (new_prompt, ok)."""
    lines = prompt.split("\n")
    idx = [i for i, ln in enumerate(lines) if _OPT_RE.match(ln.strip())]
    if len(idx) < 2 or idx != list(range(idx[0], idx[0] + len(idx))):
        return prompt, False  # options not contiguous / not found
    texts = [_OPT_RE.match(lines[i].strip()).group(2) for i in idx]
    order = list(range(len(texts)))
    rng.shuffle(order)
    letters = [chr(ord("A") + k) for k in range(len(texts))]
    for pos, i in enumerate(idx):
        lines[i] = f"({letters[pos]}) {texts[order[pos]]}"
    return "\n".join(lines), True


def run_t12_optionorder(model_cfg, model, tokenizer, pentad_df, limit=None, seed=20260101):
    """
    Re-run slot-a for MCQ seeds (BBQ, StereoSet) with a DIFFERENT random option order
    per sample. If the answer (option text) stays stable, FM4 is content, not position.
    """
    name = model_cfg["name"]
    slot_a = pentad_df[(pentad_df["slot"] == "a") & (pentad_df["subvariant"] == "surface")
                       & (pentad_df["seed_source"].isin(["bbq", "stereoset"]))].copy()
    slot_a = slot_a[slot_a["prompt_text"].astype(str).str.strip() != ""]
    if limit:
        slot_a = slot_a.head(limit)
    run_id = f"t12-{uuid.uuid4().hex[:8]}"
    frames = []
    for si in range(5):
        rng = random.Random(seed + si)
        perm = slot_a.copy()
        flags = []
        new_prompts = []
        for _, r in perm.iterrows():
            np_, ok = _permute_options(str(r["prompt_text"]), rng)
            new_prompts.append(np_); flags.append(ok)
        perm["prompt_text"] = new_prompts
        perm["order_permuted"] = flags
        df = evaluate_osm_model(model_cfg, model, tokenizer, perm, run_id,
                                temperature=0.7, sample_index=si + 1)
        df = df.merge(perm[["prompt_id", "order_permuted"]], on="prompt_id", how="left")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
