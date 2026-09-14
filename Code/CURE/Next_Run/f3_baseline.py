"""
f3_baseline.py -- F3 (Next_Plan.md Section 8): finish the supervised-baseline validity check
with one bounded experiment.

The existing LEACE run labelled activations by swap side (A = 0, B = 1). Across heterogeneous
swaps (hindu/sikh, black/white, man/woman, person/wealthy, ...) that label has no consistent
demographic meaning, so the existing run is a swap-side eraser, not a validated demographic
eraser. This module:

  1. audits the A/B label semantics of the existing fit pairs (which contrasts, how many);
  2. picks the one coherent binary contrast with the most fit pairs: man (0) vs woman (1),
     already present in the fit split (45 eligible pairs on every model; the choice is made
     from the pair table before any activation is collected);
  3. collects unedited span-position activations for those pairs on the two final models
     (chat format, the same span rule as the fitted basis), split BY SEED into fit (2/3) and
     held-out (1/3) with the frozen random seed;
  4. fits the official LEACE eraser (concept_erasure, affine, shrinkage) and the simple
     mean-difference direction on exactly the same fit rows and labels;
  5. checks class counts, pre-edit held-out linear predictability (logistic probe, seed-split),
     post-edit held-out probe accuracy with bootstrap intervals, residual feature-label
     covariance on the fit rows (the algebraic LEACE guarantee: Cov(eraser(x), z) = 0 on the
     fitted distribution), and that the hook's affine operation equals the published operator
     on identical vectors;
  6. saves the erasers for the LC / MC conditions of f2_runner (applied at the demographic span
     of the eligible gender subset of the final set).

Outputs: results/final_20260914/f3_baseline_<model>.npz, baseline_validation_<model>.json
CPU is enough for the fits and probes; the activation collection needs the loaded model.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

import nr_common as N
import f1_scoring as S
import p1_intervention as I     # noqa: E402
import p1_pilot as P1           # noqa: E402
import p2_leace_faithful as LF  # noqa: E402

log = N.setup_logging("cure.final.f3")
CONTRAST = ("man", "woman")           # label 0 = man, 1 = woman
OUT = N.FINAL


def label_audit(model: str) -> dict:
    """What the existing A/B labels meant on this model's fit pairs."""
    man = pd.read_csv(N.P0 / "split_manifest.csv"); fit = set(man[man.split == "fit"].seed_id)
    e = pd.read_csv(N.P0 / "pair_sets" / ("%s_eligible.csv" % model)); e = e[e.seed_id.isin(fit)]
    pen = pd.read_parquet(N.K.PENTAD_CLEAN); c = pen[pen.slot == "c"]
    tokmap = {(r.seed_id, r.subvariant): str(r.swap_token).lower().replace("_", " ") for r in c.itertuples(index=False)}
    from collections import Counter
    cnt = Counter()
    rows = []
    for r in e.itertuples(index=False):
        a, b = tokmap.get((r.seed_id, r.subvariant_A)), tokmap.get((r.seed_id, r.subvariant_B))
        cnt[tuple(sorted([str(a), str(b)]))] += 1
        rows.append((r.seed_id, r.subvariant_A, r.subvariant_B, a, b))
    coherent = [(r, sa, sb, a, b) for r, sa, sb, a, b in rows if {a, b} == set(CONTRAST)]
    return {"n_fit_pairs_eligible": len(rows), "n_distinct_contrasts": len(cnt),
            "top_contrasts": [{"pair": list(k), "n": v} for k, v in cnt.most_common(10)],
            "chosen_contrast": list(CONTRAST), "n_chosen_pairs": len(coherent),
            "verdict": "swap side A/B is not a coherent demographic label across the fit set (%d distinct contrasts)" % len(cnt),
            "_coherent_pairs": coherent}


def collect(model_obj, tok, pairs: list[tuple]) -> tuple[dict, np.ndarray, np.ndarray, list]:
    """Span activations (unedited, chat format) for every side of every coherent pair."""
    pen = pd.read_parquet(N.K.PENTAD_CLEAN); c = pen[pen.slot == "c"]
    var = {(r.seed_id, r.subvariant): r for r in c.itertuples(index=False)}
    X, z, seeds = {}, [], []
    sysm = P1.system_prompt()
    for seed_id, sa, sb, a, b in pairs:
        for sv, term in ((sa, a), (sb, b)):
            v = var[(seed_id, sv)]
            text = I.build_input(tok, str(v.prompt_text), N.FMT, sysm)
            pos = I.resolve_span_positions(tok, text, str(v.swap_token))
            if not pos:
                continue
            _, hs = I.forward_hidden(model_obj, tok, text, None, None)
            lab = 0 if term == CONTRAST[0] else 1
            for l in hs:
                X.setdefault(l, []).append(hs[l][pos, :].numpy())
            z += [lab] * len(pos); seeds += [seed_id] * len(pos)
    X = {l: np.concatenate(v, axis=0).astype(np.float32) for l, v in X.items()}
    return X, np.asarray(z), np.asarray(seeds), []


def probe_accuracy(Xtr, ztr, Xte, zte, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(sc.transform(Xtr), ztr)
    return float((clf.predict(sc.transform(Xte)) == zte).mean())


def apply_leace_np(X: np.ndarray, e: dict) -> np.ndarray:
    return X - ((X - e["bias"]) @ e["proj_right"].T) @ e["proj_left"].T


def fit_and_validate(model: str, model_obj, tok, bases) -> None:
    rec = label_audit(model)
    pairs = rec.pop("_coherent_pairs")
    rng = random.Random(N.RANDOM_SEED_FINAL)
    seeds = sorted({p[0] for p in pairs}); rng.shuffle(seeds)
    n_fit = int(round(2 * len(seeds) / 3))
    fit_seeds, held_seeds = set(seeds[:n_fit]), set(seeds[n_fit:])
    X, z, sid, _ = collect(model_obj, tok, pairs)
    tr = np.isin(sid, list(fit_seeds)); te = ~tr
    rec.update({"n_rows": int(len(z)), "n_fit_rows": int(tr.sum()), "n_heldout_rows": int(te.sum()),
                "class_counts_fit": {"man": int((z[tr] == 0).sum()), "woman": int((z[tr] == 1).sum())},
                "n_fit_seeds": len(fit_seeds), "n_heldout_seeds": len(held_seeds)})
    layers = sorted(X)
    # fit on the model's device: the CPU LAPACK eigh failed on Llama's float32 covariance
    # ("linalg.eigh: Argument 8 has illegal value"), the GPU path is what results/v2 used
    dev = str(next(model_obj.parameters()).device)
    erasers = LF.fit_leace_erasers({l: X[l][tr] for l in layers}, z[tr], device=dev)
    md = {}
    for l in layers:
        m = X[l][tr][z[tr] == 1].mean(0) - X[l][tr][z[tr] == 0].mean(0)
        md[l] = (m / np.linalg.norm(m))[None, :].astype(np.float32)
    per_layer = []
    probe_layers = [layers[len(layers) // 4], layers[len(layers) // 2], layers[3 * len(layers) // 4]]
    for l in probe_layers:
        e = erasers[l]
        pre = probe_accuracy(X[l][tr], z[tr], X[l][te], z[te], N.RANDOM_SEED_FINAL)
        Xl_tr, Xl_te = apply_leace_np(X[l][tr], e), apply_leace_np(X[l][te], e)
        post = probe_accuracy(Xl_tr, z[tr], Xl_te, z[te], N.RANDOM_SEED_FINAL)
        Um = md[l][0]
        Xm_tr, Xm_te = X[l][tr] - np.outer(X[l][tr] @ Um, Um), X[l][te] - np.outer(X[l][te] @ Um, Um)
        post_md = probe_accuracy(Xm_tr, z[tr], Xm_te, z[te], N.RANDOM_SEED_FINAL)
        # bootstrap over held-out rows clustered by seed for the post-LEACE probe accuracy
        boots = []
        rs = np.random.default_rng(N.RANDOM_SEED_FINAL)
        held = sorted(held_seeds); sid_te = sid[te]
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xl_tr); clf = LogisticRegression(max_iter=2000, random_state=N.RANDOM_SEED_FINAL).fit(sc.transform(Xl_tr), z[tr])
        pred = clf.predict(sc.transform(Xl_te)) == z[te]
        for _ in range(500):
            take = rs.choice(held, size=len(held), replace=True)
            mask = np.concatenate([np.where(sid_te == s)[0] for s in take]) if len(take) else np.array([], int)
            if len(mask):
                boots.append(float(pred[mask].mean()))
        cov = float(np.abs((Xl_tr - Xl_tr.mean(0)).T @ (z[tr] - z[tr].mean()) / len(z[tr])).max())
        cov_pre = float(np.abs((X[l][tr] - X[l][tr].mean(0)).T @ (z[tr] - z[tr].mean()) / len(z[tr])).max())
        per_layer.append({"layer": int(l), "probe_pre": pre, "probe_post_leace": post, "probe_post_leace_lo": float(np.percentile(boots, 2.5)) if boots else None,
                          "probe_post_leace_hi": float(np.percentile(boots, 97.5)) if boots else None, "probe_post_mean_difference": post_md,
                          "max_abs_cov_pre": cov_pre, "max_abs_cov_post_leace": cov, "leace_rank": int(e["actual_rank"])})
    # algebraic hook check: the affine map inside the hook equals the published operator on identical vectors
    import torch
    l0 = probe_layers[1]; e0 = erasers[l0]
    x = torch.as_tensor(X[l0][te][:8])
    hook_val = x.float() - ((x.float() - torch.as_tensor(e0["bias"])) @ torch.as_tensor(e0["proj_right"]).T) @ torch.as_tensor(e0["proj_left"]).T
    ref = apply_leace_np(X[l0][te][:8], e0)
    rec["hook_vs_published_max_abs_dev"] = float(np.abs(hook_val.numpy() - ref).max())
    rec["per_layer_checks"] = per_layer
    rec["probe_layers"] = probe_layers
    rec["labels"] = "0 = man, 1 = woman (swap token of the fit-split variant)"
    rec["utc"] = N.utc_now()
    npz = LF.erasers_to_npz_dict(erasers)
    npz.update({"md_%d" % l: md[l] for l in layers})
    np.savez(OUT / ("f3_baseline_%s.npz" % model), **npz)
    N.write_json(rec, OUT / ("baseline_validation_%s.json" % model))
    log.info("F3 %s: %d coherent pairs, probe pre/post at mid layer %.2f/%.2f", model, rec["n_chosen_pairs"],
             per_layer[1]["probe_pre"], per_layer[1]["probe_post_leace"])

