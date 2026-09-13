"""
p3_explanation.py -- ONE focused explanation of the erasure damage, run only after P2. GPU.

Purpose: Next_Plan.md P3 asks for one explanation chosen on development evidence and confirmed
once on the P2 test seeds, not a new method. Three candidate accounts are implemented as
separate, resumable runs of the same script. Every account uses P2's FROZEN targeted basis,
P2's operating strength and rank, the same span-site sequential-source edit as P1 and P2
(p1_intervention.HookedEdit, h' = h - alpha U^T U h at every token of the demographic span,
prefill only) and the same per-item readouts (raw |C| by activation patching, canonical option
log-likelihoods). Nothing here refits the targeted basis on dev or test items.

Design (Next_Plan.md P3, bullet by bullet)
  depth      Depth / distribution-shift account. Conditions: unedited; the P2 basis at every
             layer (full_frozen); the same basis restricted to a fixed early / middle / late
             third of the layers (early_frozen, middle_frozen, late_frozen); and a SEQUENTIALLY
             fitted basis (full_sequential): layer l's basis is refitted on FIT-split
             activations that were already edited at layers < l, obtained by forward passes
             with the earlier-layer HookedEdits active (block size --seq-block; block 1 is
             exact per-layer refitting). For every item and condition the script records, per
             layer, the relative deviation from the unedited run of (i) the residual norm at
             the span positions (the direct, attention-free residual edit at the first edited
             layer, plus propagation after it), (ii) the residual norm at the last prefill
             position and at the positions after the span (deviation that can only arrive
             through attention, since the span policy never edits those positions), (iii) the
             positions before the span (must be exactly zero under causal attention; a
             sanity check), and the cumulative-depth gold-option margin (the edit applied at
             layers < c for cutoffs c). The first layer index at which each quantity exceeds
             a STATED tolerance (--dev-tol relative norm, --margin-tol nats) is stored.
             LAYER CHOICES ARE DIAGNOSTIC CONTROLS, NOT LOCALISATION: a block that carries
             most of the damage is not a locus of the demographic computation and this script
             makes no Patchscope-style claim. If sequential fitting removes the harm, the
             conclusion concerns stale representations under repeated frozen edits.
  magnitude  Magnitude account. At MATCHED removed energy E[||h-h'||^2]/E[||h||^2] (pooled
             over span positions and edited layers), the targeted basis is compared with
             random orthonormal bases (P2's draws when present, else --n-random-draws seeded
             draws) and P2's neutral-contrast basis (when present; never invented) on the same
             items. Matched strengths are read from p2_matched_alphas.json when present, else
             found on DEV by the analytic guess alpha = sqrt(E_target / E(alpha=1)) refined on
             a five-point grid, then frozen to p3_matched_alphas_<model>.json; the test phase
             refuses to match. Matching runs in both directions: control strength raised to
             the targeted energy (infeasible when alpha <= 1 cannot reach it, which is
             recorded), and targeted strength lowered to each control's alpha = 1 energy
             (always feasible). Per-item removed energy is related to per-item loss of
             correct-option probability, accuracy and |C| change with seed-cluster bootstrap
             Spearman correlations.
  massive    Massive-activation account. High-magnitude residual coordinates are identified
             on INDEPENDENT neutral calibration prompts (NEUTRAL_CALIBRATION below, no
             demographic terms, raw format) using the mean absolute activation AND its input
             stability (coefficient of variation across prompts), not variance alone; the old
             top-variance criterion is computed only to report the overlap. A controlled
             RESTORATION then adds back alpha * M (U^T U h) on those coordinates (M a
             coordinate mask, h the pre-erasure state) through a second hook registered after
             the erasure hook (RestoredEdit). Controls: random coordinates of the same count,
             equal-energy restoration (random coordinates rescaled per token to the energy the
             massive restoration puts back), restoration of the complement (erase only the
             identified coordinates), and restore-all (a numerical identity check). A
             DIAGNOSTIC RESTORATION IS NOT A PROTECTED-ERASURE ALGORITHM. Without a
             restoration effect the account stays a hypothesis; coordinate overlap alone is
             not evidence. The word "bias" in the cited massive-activation papers denotes a
             constant attention offset, not social bias.

  Items      the largest-|C_pre| mask-clean demographic pair of each seed (p1_pilot
             .pairs_for_seed). --phase dev draws a benchmark-stratified subset of the manifest
             DEV seeds (--n-seeds); --phase test uses P2's test seed list once, for
             confirmation, and never chooses anything on it.
  Budget     --gpu-hours-cap is a wall-clock cap including loading; the runner stops cleanly
             and everything done is on disk (resumable parquet appends keyed by
             model, phase, account, condition, seed).

Inputs (the script exits with a message if a required file is absent)
  results/v2/p2_test_seeds.csv                       seed_id [, seed_source]   (test phase)
  results/v2/p2_basis_<model>_<fmt>_<family>.npz     {str(layer): [max_rank, d]} frozen basis, sliced to --rank
  results/v2/p2_matched_alphas.json                  optional. {model: {condition: {"alpha":
                                                     a, "rank": r, "energy_dev": e}}}; a bare
                                                     number is read as alpha.
  results/v2/p2_basis_<model>_<fmt>_random_ortho_*.npz  control bases from P2
  results/v2/p2_basis_<model>_<fmt>_neutral_contrast.npz
  reanalysis_v2/split_manifest.csv, pair_sets/<model>_eligible.csv  (P0)

Outputs (results/v2/)
  p3_<account>_per_item.parquet    one row per (model, phase, condition, seed); resumable
  p3_<account>_summary.csv         per (model, phase, condition) with seed-cluster bootstrap CIs
  p3_report.md                     one section per account x model x phase, replaced on rerun,
                                   ending with the plan's decision text
  p3_basis_<model>_sequential_b<block>_r<rank>.npz   (depth; fitted on the FIT split only)
  p3_matched_alphas_<model>.json                     (magnitude; written in the dev phase)
  p3_magnitude_energy_relationships.csv              (magnitude)
  p3_massive_coords_<model>.csv, p3_massive_basis_mass_<model>.csv   (massive; coords frozen once written)

Usage
  python p3_explanation.py --account depth     --models qwen2.5-7b-instruct --phase dev
  python p3_explanation.py --account magnitude --models qwen2.5-7b-instruct --phase dev
  python p3_explanation.py --account massive   --models qwen2.5-7b-instruct --phase dev
  python p3_explanation.py --account massive   --models qwen2.5-7b-instruct --phase test
  python p3_explanation.py --account depth --report-only

Builds on
  - Sun, Chen, Kolter and Xie (2024) Massive Activations in Large Language Models,
    arXiv:2402.17762 (fixed high-magnitude, input-independent residual coordinates).
  - Yu, Ergen, Vishwanath et al. (2024) The Super Weight in Large Language Models,
    arXiv:2411.07191 (a few coordinates dominate the residual norm).
  - Belrose et al. (2023) LEACE, arXiv:2306.03819, Section 6 and Appendix H (sequential
    concept scrubbing: later erasers fitted on representations already affected by earlier
    erasers) -- the frozen versus sequential comparison in the depth account.
  - Efron and Tibshirani (1993) An Introduction to the Bootstrap -- seed-cluster percentile
    intervals (common.ratio_bootstrap and spearman_cluster_ci below).
  - p1_intervention.HookedEdit is the only edit implementation used; RestoredEdit wraps it.

Part of the CURE codebase (ICLR 2027, Submission2). GPU required for everything but the
summaries and the report.
"""

from __future__ import annotations

import argparse
import glob
import re
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import common as K
import p1_intervention as I
import p1_pilot as P

log = K.setup_logging("cure.ext.p3")

ACCOUNTS = ("depth", "magnitude", "massive")
PHASES = ("dev", "test")
KEY_COLS = ["model_name", "phase", "account", "condition", "seed_id", "pair_type"]
PAIR_KEY = ["seed_id", "subvariant_A", "subvariant_B"]
SIDE_KEY = PAIR_KEY + ["side"]

# 40 short neutral English sentences for the massive-activation calibration. No demographic
# terms (no names, pronouns, nationalities, religions, ages, group-linked occupations).
NEUTRAL_CALIBRATION = [
    "The train leaves the station at nine in the morning.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The library closes early on public holidays.",
    "A square has four equal sides and four right angles.",
    "The recipe calls for two cups of flour and one egg.",
    "The museum added a new wing for modern sculpture.",
    "Copper conducts electricity better than iron.",
    "The meeting was moved from Tuesday to Thursday.",
    "Rain is expected across the valley later this week.",
    "The printer on the second floor needs more toner.",
    "Most of the apples in the crate were still green.",
    "The bridge was repainted during the summer closure.",
    "A kilometre is one thousand metres.",
    "The bakery sells bread, rolls and small cakes.",
    "The software update fixes three known errors.",
    "The garden wall needs a new coat of paint.",
    "Sound travels faster through water than through air.",
    "The bus route was changed because of road works.",
    "The report summarises the sales figures for March.",
    "A spider has eight legs and most insects have six.",
    "The kettle switches off when the water boils.",
    "The path along the river is closed after heavy rain.",
    "The invoice lists the parts and the labour separately.",
    "Glass is made mainly from sand, soda and lime.",
    "The concert hall seats about two thousand people.",
    "The thermostat keeps the room at twenty degrees.",
    "Oak trees can live for several hundred years.",
    "The parcel arrived two days after it was posted.",
    "The map shows the footpaths and the cycle lanes.",
    "The moon orbits the earth roughly once a month.",
    "The shop on the corner repairs bicycles and kettles.",
    "The spreadsheet totals each column automatically.",
    "The lighthouse was built on the rocks in 1854.",
    "Fresh snow covered the car park overnight.",
    "The engine needs an oil change every ten thousand kilometres.",
    "The orchestra rehearses on Wednesday evenings.",
    "The river floods the lower fields most springs.",
    "A prime number has exactly two divisors.",
    "The lift is out of service until further notice.",
    "The clock in the hall runs about five minutes fast.",
]


# ---------------------------------------------------------------------------
# io helpers (tolerate an --out-dir outside the repository)
# ---------------------------------------------------------------------------

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(K.REPO))
    except ValueError:
        return str(path)


def wcsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", _rel(path), len(df))


def wjson(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=K._json_default), encoding="utf-8")
    log.info("wrote %s", _rel(path))


def require(path: Path, hint: str) -> None:
    if not path.exists():
        log.error("missing %s. %s", path, hint)
        sys.exit(2)


# ---------------------------------------------------------------------------
# P2 artifacts (read only)
# ---------------------------------------------------------------------------

_FMT = "chat"   # set from --fmt in main(); P2 names its basis files by prompt format

# P3's short control names -> P2's basis family names
P2_FAMILY = {"neutral": "neutral_contrast", "cure_centred_svd": "cure_centred_svd"}
for _i in range(1, 10):
    P2_FAMILY[f"random_{_i}"] = f"random_ortho_{_i}"


def p2_family(condition: str) -> str:
    return P2_FAMILY.get(condition, condition)


def basis_path(out: Path, model: str, condition: str, rank: int) -> Path:
    """P2 (BasisStore.path) writes p2_basis_<model>_<fmt>_<name>.npz holding every basis at
    its MAXIMUM fitted rank; `rank` is applied by load_basis, not by the file name."""
    return out / f"p2_basis_{model}_{_FMT}_{p2_family(condition)}.npz"


def load_basis(path: Path, rank: int | None = None) -> dict[int, np.ndarray]:
    z = np.load(path)
    b = {int(k): np.asarray(z[k], dtype=np.float32) for k in z.files}
    return {l: r[:rank] for l, r in b.items()} if rank else b


def save_basis(path: Path, basis: dict) -> None:
    np.savez_compressed(path, **{str(l): b for l, b in basis.items()})


def read_matched(out: Path) -> dict:
    p = out / "p2_matched_alphas.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _parse_cond_id(cid: str) -> tuple[float | None, int | None]:
    m = re.search(r"_r(\d+)_a([0-9.]+)", str(cid))
    return (float(m.group(2)), int(m.group(1))) if m else (None, None)


def matched_entry(matched: dict, model: str, condition: str) -> dict | None:
    """{alpha, rank, energy_dev} for (model, condition) from P2's p2_matched_alphas.json:
    the target family is the reference at its own strength; controls sit under "controls"
    keyed by P2 family name. A bare number is still read as alpha."""
    v = matched.get(model)
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return {"alpha": float(v)}
    if not isinstance(v, dict):
        return None
    fam = p2_family(condition)
    tgt = str(v.get("target_condition", ""))
    if tgt.startswith(fam):
        a, r = _parse_cond_id(tgt)
        return {"alpha": 1.0 if a is None else a, "rank": 1 if r is None else r,
                "energy_dev": v.get("target_energy")}
    c = (v.get("controls") or {}).get(fam)
    if isinstance(c, dict) and "alpha" in c:
        return {"alpha": float(c["alpha"]), "rank": int(c.get("rank", 1)),
                "energy_dev": c.get("energy_achieved"), "matchable": c.get("matchable")}
    legacy = v.get(condition)                      # older flat layout {model: {condition: ...}}
    if isinstance(legacy, (int, float)):
        return {"alpha": float(legacy)}
    if isinstance(legacy, dict):
        return {k: legacy[k] for k in ("alpha", "rank", "energy_dev") if k in legacy}
    return None


def operating_alpha(matched: dict, model: str, condition: str, cli_alpha: float | None) -> tuple[float, str]:
    if cli_alpha is not None:
        return float(cli_alpha), "cli"
    ent = matched_entry(matched, model, condition)
    if ent and "alpha" in ent:
        return float(ent["alpha"]), "p2_matched_alphas.json"
    return 1.0, "default (no p2_matched_alphas.json entry, no --alpha)"


def control_basis_files(out: Path, model: str, rank: int, kind: str) -> list[str]:
    """P2 control bases: random_ortho_<i> and neutral_contrast, one file each at max rank."""
    pat = {"random": "random_ortho_*", "neutral": "neutral_contrast"}.get(kind, kind + "*")
    return sorted(glob.glob(str(out / f"p2_basis_{model}_{_FMT}_{pat}.npz")))


# ---------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------

def select_seeds(out: Path, phase: str, n_seeds: int, seed: int) -> pd.DataFrame:
    man = pd.read_csv(K.OUT_P0 / "split_manifest.csv")
    if phase == "test":
        p = out / "p2_test_seeds.csv"
        require(p, "P3 runs only after P2: run p2 first so its test seed list exists.")
        ts = pd.read_csv(p)
        bad = set(ts["seed_id"]) - set(man[man["split"] == "test"]["seed_id"])
        if bad:
            log.error("%d P2 test seeds are not in the manifest TEST split, e.g. %s", len(bad), sorted(bad)[:5])
            sys.exit(2)
        return man[man["seed_id"].isin(ts["seed_id"])].sort_values(["seed_source", "seed_id"]).reset_index(drop=True)
    dev = man[man["split"] == "dev"]
    if n_seeds <= 0 or n_seeds >= len(dev):
        return dev.sort_values(["seed_source", "seed_id"]).reset_index(drop=True)
    rng = random.Random(seed)
    take = []
    for _, g in dev.groupby("seed_source"):
        ids = sorted(g["seed_id"]); rng.shuffle(ids)
        take += ids[:max(1, round(n_seeds * len(g) / len(dev)))]
    return dev[dev["seed_id"].isin(take)].sort_values(["seed_source", "seed_id"]).reset_index(drop=True)


def select_items(model: str, seeds: pd.DataFrame, cdva: pd.DataFrame, c: pd.DataFrame, elig_keys: set) -> list[dict]:
    items = []
    for _, s in seeds.iterrows():
        pr = [p for p in P.pairs_for_seed(model, s["seed_id"], cdva, c, elig_keys) if p["pair_type"] == "demographic"]
        if not pr:
            log.warning("no clean demographic pair for %s on %s", s["seed_id"], model); continue
        it = dict(pr[0]); it.update({"seed_id": s["seed_id"], "benchmark": s["seed_source"]})
        items.append(it)
    return items


def target_token_id(tok, pair: dict) -> int | None:
    """First token of the gold answer of B (else the swap token), as in p1_pilot.run_item."""
    gold = pair["gold_B"]
    txt = gold if gold.strip().lower() not in ("unknown", "", "nan") else pair["swap_B"].replace("_", " ")
    ids = tok((" " + txt.split()[0]) if txt.split() else txt, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = tok(txt, add_special_tokens=False)["input_ids"]
    return int(ids[0]) if ids else None


def texts_and_spans(tok, fmt: str, pair: dict) -> tuple[str, str, list[int], list[int]]:
    sysm = P.system_prompt() if fmt == "chat" else None
    ta = I.build_input(tok, pair["prompt_A"], fmt, sysm)
    tb = I.build_input(tok, pair["prompt_B"], fmt, sysm)
    return ta, tb, I.resolve_span_positions(tok, ta, pair["swap_A"]), I.resolve_span_positions(tok, tb, pair["swap_B"])


def option_strings(prompt: str) -> list[str]:
    return [f"({l}) {t}" for l, t in P.options_of(prompt)]


# ---------------------------------------------------------------------------
# energy accounting shared by HookedEdit and RestoredEdit
# ---------------------------------------------------------------------------

def pooled_energy_of_log(changed_log: list[dict], num_key: str = "delta_norm") -> float:
    num = sum(r[num_key] ** 2 for r in changed_log)
    den = sum(r["h_norm"] ** 2 for r in changed_log)
    return float(num / den) if den > 0 else float("nan")


def energy_fields(edit, side: str) -> dict:
    s = edit.summary()
    out = {f"energy_mean_{side}": s.get("energy_mean_over_layers"),
           f"energy_total_{side}": pooled_energy_of_log(edit.changed_log),
           f"n_edits_{side}": s.get("n_edits")}
    if isinstance(edit, RestoredEdit):
        out[f"removed_energy_{side}"] = pooled_energy_of_log(edit.changed_log, "removed_norm")
        out[f"restored_energy_{side}"] = pooled_energy_of_log(edit.changed_log, "restored_norm")
    return out


# ---------------------------------------------------------------------------
# the shared per-item readout: audit |C| and option scores on both prompts
# ---------------------------------------------------------------------------

def readout(model, tok, fmt: str, pair: dict, factory, site: str = "span") -> dict:
    ta, tb, pa, pb = texts_and_spans(tok, fmt, pair)
    row = {"span_len_A": len(pa), "span_len_B": len(pb), "fmt": fmt, "site": site}
    tgt = target_token_id(tok, pair)
    t0 = time.time()
    if pa and pb and tgt is not None:
        C, info = I.commutator(model, tok, ta, tb, pa, pb, tgt, factory, "sequential")
        row.update({"C": C, "absC": abs(C) if C == C else np.nan, "patched_len": info.get("patched_len"),
                    "logit_patched": info.get("logit_patched"), "logit_clean": info.get("logit_clean"),
                    "energy_audit": (info.get("target_edit") or {}).get("energy_mean_over_layers")})
    else:
        row.update({"C": np.nan, "absC": np.nan, "audit_error": "span or target unresolved"})
    row["sec_audit"] = time.time() - t0
    t0 = time.time()
    for side, text, pos, prompt in (("A", ta, pa, pair["prompt_A"]), ("B", tb, pb, pair["prompt_B"])):
        opts = option_strings(prompt)
        if not opts:
            continue
        e = factory() if factory else None
        edit_pos = pos if site == "span" else ([len(tok(text)["input_ids"]) - 1] if site == "last_token" else None)
        r = I.option_loglik(model, tok, text, opts, e, edit_pos)
        gi = P.gold_index(prompt, pair[f"gold_{side}"])
        row.update({f"opt_argmax_{side}": r["argmax"], f"opt_margin_{side}": r["margin_top1_top2"],
                    f"opt_entropy_{side}": r["entropy"], f"gold_idx_{side}": gi,
                    f"opt_correct_{side}": (r["argmax"] == gi) if gi is not None else None,
                    f"p_gold_{side}": float(r["probs"][gi]) if gi is not None else np.nan,
                    f"opt_probs_{side}": json.dumps([round(x, 6) for x in r["probs"]])})
        if e is not None:
            row.update(energy_fields(e, side))
    row["sec_options"] = time.time() - t0
    return row


def common_fields(mname: str, args, account: str, it: dict, cond: dict, alpha: float, t0: float) -> dict:
    return {"model_name": mname, "phase": args.phase, "account": account, "seed_id": it["seed_id"],
            "benchmark": it["benchmark"], "pair_type": "demographic", "subvariant_A": it["A"],
            "subvariant_B": it["B"], "C_pre_shipped": it["C_pre_shipped"], "rank": args.rank,
            "alpha": (alpha if cond["factory"] is not None else 0.0), "target_condition": args.target_condition,
            "sec_total": time.time() - t0, "ts": K.utc_now()}


def run_conditions(model, tok, args, mname: str, account: str, items: list[dict], conds: list[dict],
                   per_item: Path, done: set, clock, extra_cols: list[str]) -> None:
    """Generic resumable loop for the magnitude and massive accounts."""
    rows = []
    for it in items:
        if clock():
            break
        for cond in conds:
            key = (mname, args.phase, account, cond["condition"], it["seed_id"], "demographic")
            if key in done:
                continue
            if clock():
                break
            t0 = time.time()
            row = {"condition": cond["condition"]}
            try:
                row.update(readout(model, tok, args.fmt, it, cond["factory"]))
            except Exception as exc:
                row["error"] = str(exc)[:300]; log.error("%s item failed %s: %s", account, key, str(exc)[:200])
            row.update(common_fields(mname, args, account, it, cond, cond.get("alpha_used", 0.0), t0))
            row.update({k: cond.get(k) for k in extra_cols})
            rows.append(row); done.add(key)
            if len(rows) >= 10:
                P.append_rows(per_item, rows); rows = []
    P.append_rows(per_item, rows)


# ---------------------------------------------------------------------------
# depth account
# ---------------------------------------------------------------------------

def layer_thirds(L: int) -> dict[str, list[int]]:
    parts = np.array_split(np.arange(L), 3)
    return {"early": [int(x) for x in parts[0]], "middle": [int(x) for x in parts[1]], "late": [int(x) for x in parts[2]]}


def restrict_layers(basis: dict, layers: list[int]) -> dict:
    return {l: basis[l] for l in layers if l in basis}


def fit_sequential_basis(model, tok, fmt: str, pairs: list[dict], c: pd.DataFrame, rank: int,
                         alpha: float, block: int) -> tuple[dict, dict]:
    """Sequential (concept-scrubbing style) fitting: the basis of layer l is estimated on span
    differences of FIT-split pairs whose activations were already edited at layers < l, via
    forward passes with the earlier-layer HookedEdits active. With block > 1, layers inside a
    block are fitted on activations edited at layers before the block only."""
    var = {(r["seed_id"], r["subvariant"]): r for _, r in c.iterrows()}
    prepared = []
    for p in pairs:
        ka, kb = (p["seed_id"], p["subvariant_A"]), (p["seed_id"], p["subvariant_B"])
        if ka not in var or kb not in var:
            continue
        pr = {"prompt_A": str(var[ka]["prompt_text"]), "prompt_B": str(var[kb]["prompt_text"]),
              "swap_A": str(var[ka].get("swap_token", "")), "swap_B": str(var[kb].get("swap_token", ""))}
        ta, tb, pa, pb = texts_and_spans(tok, fmt, pr)
        if min(len(pa), len(pb)) > 0:
            prepared.append((ta, tb, pa, pb))
    L = len(I._decoder_layers(model))
    basis: dict[int, np.ndarray] = {}
    n_pass = 0
    for start in range(0, L, block):
        blk = list(range(start, min(start + block, L)))
        edit = I.HookedEdit(model, basis, alpha=alpha, site="span", record=False) if basis else None
        diffs: dict[int, list] = {l: [] for l in blk}
        for ta, tb, pa, pb in prepared:
            _, ha = I.forward_hidden(model, tok, ta, edit, pa)
            _, hb = I.forward_hidden(model, tok, tb, edit, pb)
            n_pass += 2
            n = min(len(pa), len(pb))
            for l in blk:
                for i in range(n):
                    diffs[l].append((ha[l][pa[i]] - hb[l][pb[i]]).numpy())
        basis.update(I.basis_centred_svd(diffs, rank))
        log.info("sequential fit: layers %s done (%d passes so far)", blk, n_pass)
    return basis, {"n_fit_pairs_used": len(prepared), "n_fit_pairs_given": len(pairs), "block": block,
                   "rank": rank, "alpha": alpha, "n_forward_passes": n_pass, "estimator": "centred_svd, sequential",
                   "fit_split": "fit (split_manifest.csv)"}


def _rel_dev(h1, h0, idx: list[int]) -> float:
    if not idx:
        return float("nan")
    d = (h1[idx] - h0[idx]).norm(dim=-1)
    n = h0[idx].norm(dim=-1).clamp_min(1e-6)
    return float((d / n).mean())


def first_exceeding(profile: list[float], tol: float) -> int:
    for i, v in enumerate(profile):
        if v == v and v > tol:
            return i
    return -1


def deviation_profiles(hs0: dict, model, tok, text: str, pos: list[int], factory, tol: float) -> dict:
    """Per-layer relative deviation from the unedited run at the span, the last prefill position,
    the positions after the span and the positions before the span, plus residual norms."""
    e = factory()
    _, hs1 = I.forward_hidden(model, tok, text, e, pos)
    L = len(hs0); T = hs0[0].shape[0]
    span = sorted(pos)
    pre = [t for t in range(T) if t < span[0]]
    post = [t for t in range(span[-1] + 1, T - 1)]
    last = [T - 1] if (T - 1) not in span else []
    prof = {"span": [_rel_dev(hs1[l], hs0[l], span) for l in range(L)],
            "last": [_rel_dev(hs1[l], hs0[l], last) for l in range(L)],
            "postspan": [_rel_dev(hs1[l], hs0[l], post) for l in range(L)],
            "prespan": [_rel_dev(hs1[l], hs0[l], pre) for l in range(L)]}
    norms = {"last_unedited": [float(hs0[l][T - 1].norm()) for l in range(L)],
             "last_edited": [float(hs1[l][T - 1].norm()) for l in range(L)],
             "span_unedited": [float(hs0[l][span].norm(dim=-1).mean()) for l in range(L)],
             "span_edited": [float(hs1[l][span].norm(dim=-1).mean()) for l in range(L)]}
    pre_max = float(np.nanmax(prof["prespan"])) if pre else 0.0
    return {"dev_profile_span": json.dumps([round(v, 5) for v in prof["span"]]),
            "dev_profile_last": json.dumps([round(v, 5) for v in prof["last"]]),
            "dev_profile_postspan": json.dumps([round(v, 5) for v in prof["postspan"]]),
            "norm_profile_last_unedited": json.dumps([round(v, 3) for v in norms["last_unedited"]]),
            "norm_profile_last_edited": json.dumps([round(v, 3) for v in norms["last_edited"]]),
            "norm_profile_span_unedited": json.dumps([round(v, 3) for v in norms["span_unedited"]]),
            "norm_profile_span_edited": json.dumps([round(v, 3) for v in norms["span_edited"]]),
            "first_dev_layer_span": first_exceeding(prof["span"], tol),
            "first_dev_layer_last": first_exceeding(prof["last"], tol),
            "first_dev_layer_postspan": first_exceeding(prof["postspan"], tol),
            "max_dev_prespan": pre_max, "prespan_untouched": bool(pre_max <= 1e-4),
            "dev_tol": tol, "energy_profile_pass": pooled_energy_of_log(e.changed_log)}


def margin_scan(model, tok, text: str, positions: list[int], prompt: str, gold: str, basis: dict,
                alpha: float, stride: int, margin0: float, tol: float) -> dict:
    """Gold-option margin with the edit applied at layers < c, for cutoffs c = stride, 2 stride,
    ..., L. The first cutoff whose margin moves by more than tol nats from the unedited margin
    is recorded; the whole profile is kept so the tolerance can be revisited from the parquet."""
    L = max(basis) + 1 if basis else 0
    cutoffs = sorted(set(list(range(stride, L, stride)) + [L])) if L else []
    opts = option_strings(prompt)
    if not opts or not cutoffs:
        return {"margin_scan": json.dumps({}), "first_margin_dev_cutoff": -1, "margin_tol": tol,
                "margin_scan_stride": stride}
    gi = P.gold_index(prompt, gold)
    scan = {}
    for cut in cutoffs:
        b = {l: U for l, U in basis.items() if l < cut}
        e = I.HookedEdit(model, b, alpha=alpha, site="span", record=False)
        r = I.option_loglik(model, tok, text, opts, e, positions)
        scan[cut] = {"margin": r["margin_top1_top2"], "argmax": r["argmax"],
                     "p_gold": float(r["probs"][gi]) if gi is not None else None,
                     "correct": (r["argmax"] == gi) if gi is not None else None}
    first = -1
    for cut in cutoffs:
        m = scan[cut]["margin"]
        if m == m and margin0 == margin0 and abs(m - margin0) > tol:
            first = cut; break
    return {"margin_scan": json.dumps(scan), "first_margin_dev_cutoff": first, "margin_tol": tol,
            "margin_scan_stride": stride}


def unedited_margin(model, tok, text: str, prompt: str) -> float:
    opts = option_strings(prompt)
    if not opts:
        return float("nan")
    return float(I.option_loglik(model, tok, text, opts, None, None)["margin_top1_top2"])


def depth_conditions(model, basis_frozen: dict, basis_seq: dict, alpha: float, L: int) -> list[dict]:
    def mk(b):
        return lambda: I.HookedEdit(model, b, alpha=alpha, site="span")
    conds = [{"condition": "unedited", "factory": None, "layers": [], "fitting": "none", "scan": False, "profile": False},
             {"condition": "full_frozen", "factory": mk(basis_frozen), "layers": sorted(basis_frozen),
              "fitting": "frozen", "scan": True, "profile": True, "basis": basis_frozen}]
    for name, layers in layer_thirds(L).items():
        b = restrict_layers(basis_frozen, layers)
        conds.append({"condition": f"{name}_frozen", "factory": mk(b), "layers": sorted(b), "fitting": "frozen",
                      "scan": False, "profile": True, "basis": b})
    conds.append({"condition": "full_sequential", "factory": mk(basis_seq), "layers": sorted(basis_seq),
                  "fitting": "sequential", "scan": True, "profile": True, "basis": basis_seq})
    return conds


def run_depth(model, tok, args, mname: str, items: list[dict], basis_frozen: dict, basis_seq: dict,
              alpha: float, per_item: Path, done: set, clock) -> None:
    L = len(I._decoder_layers(model))
    conds = depth_conditions(model, basis_frozen, basis_seq, alpha, L)
    rows = []
    for it in items:
        if clock():
            break
        cache: dict = {}
        for cond in conds:
            key = (mname, args.phase, "depth", cond["condition"], it["seed_id"], "demographic")
            if key in done:
                continue
            if clock():
                break
            t0 = time.time()
            row = {"condition": cond["condition"]}
            try:
                row.update(readout(model, tok, args.fmt, it, cond["factory"]))
                if cond["profile"] or cond["scan"]:
                    if "hs0" not in cache:                       # one unedited pass per item
                        ta, _, pa, _ = texts_and_spans(tok, args.fmt, it)
                        _, hs0 = I.forward_hidden(model, tok, ta, None, None)
                        cache.update({"ta": ta, "pa": pa, "hs0": hs0,
                                      "margin0": unedited_margin(model, tok, ta, it["prompt_A"])})
                    if cache["pa"]:
                        if cond["profile"]:
                            row.update(deviation_profiles(cache["hs0"], model, tok, cache["ta"], cache["pa"],
                                                          cond["factory"], args.dev_tol))
                        if cond["scan"]:
                            row.update(margin_scan(model, tok, cache["ta"], cache["pa"], it["prompt_A"], it["gold_A"],
                                                   cond["basis"], alpha, args.margin_scan_stride, cache["margin0"],
                                                   args.margin_tol))
            except Exception as exc:
                row["error"] = str(exc)[:300]; log.error("depth item failed %s: %s", key, str(exc)[:200])
            row.update(common_fields(mname, args, "depth", it, cond, alpha, t0))
            row.update({"layers_edited": json.dumps(cond["layers"]), "n_layers_edited": len(cond["layers"]),
                        "fitting": cond["fitting"], "n_layers_model": L})
            rows.append(row); done.add(key)
            if len(rows) >= 10:
                P.append_rows(per_item, rows); rows = []
    P.append_rows(per_item, rows)


# ---------------------------------------------------------------------------
# magnitude account
# ---------------------------------------------------------------------------

def pooled_energy(model, tok, fmt: str, items: list[dict], basis: dict, alpha: float) -> float:
    """E[||h-h'||^2]/E[||h||^2] pooled over span positions and edited layers of prompt A of the
    given items (one prefill forward each), under the sequential application."""
    e = I.HookedEdit(model, basis, alpha=alpha, site="span", record=True)
    for it in items:
        ta, _, pa, _ = texts_and_spans(tok, fmt, it)
        if pa:
            I.forward_hidden(model, tok, ta, e, pa)
    return pooled_energy_of_log(e.changed_log)


def match_alpha(model, tok, fmt: str, items: list[dict], basis: dict, target: float) -> dict:
    """Strength alpha in (0, 1] whose pooled removed energy is closest to `target`. Analytic guess
    alpha = sqrt(target / E(alpha=1)) (exact for a frozen application, approximate under the
    sequential one), refined on a five-point grid. Infeasible when E(alpha=1) < target."""
    e1 = pooled_energy(model, tok, fmt, items, basis, 1.0)
    if not np.isfinite(e1) or e1 <= 0:
        return {"alpha": 1.0, "energy": e1, "feasible": False, "trace": [], "target": target, "energy_at_alpha1": e1}
    if e1 <= target:
        return {"alpha": 1.0, "energy": e1, "feasible": False, "trace": [[1.0, e1]], "target": target, "energy_at_alpha1": e1}
    guess = float(np.sqrt(target / e1))
    grid = sorted({float(min(1.0, max(0.01, guess * f))) for f in (0.7, 0.85, 1.0, 1.15, 1.3)})
    trace = [[a, pooled_energy(model, tok, fmt, items, basis, a)] for a in grid]
    best = min(trace, key=lambda t: abs(t[1] - target))
    return {"alpha": best[0], "energy": best[1], "feasible": True, "trace": trace, "target": target,
            "energy_at_alpha1": e1}


def control_bases(out: Path, mname: str, rank: int, layers: list[int], d: int, n_draws: int) -> list[dict]:
    """P2's random and neutral bases when present; otherwise seeded random draws (recorded). A
    neutral basis is never invented here."""
    ctrls = []
    files = control_basis_files(out, mname, rank, "random")
    if files:
        for i, f in enumerate(files[:n_draws]):
            ctrls.append({"name": f"random_{i + 1}", "kind": "random", "draw": i + 1, "basis": load_basis(Path(f), rank), "source": f})
    else:
        for i in range(n_draws):
            ctrls.append({"name": f"random_{i + 1}", "kind": "random", "draw": i + 1,
                          "basis": I.basis_random(d, layers, rank, K.RANDOM_SEED_V2 + i),
                          "source": f"p1_intervention.basis_random(seed={K.RANDOM_SEED_V2 + i}); no P2 random basis file"})
    files = control_basis_files(out, mname, rank, "neutral")
    if files:
        ctrls.append({"name": "neutral", "kind": "neutral", "draw": 0, "basis": load_basis(Path(files[0]), rank), "source": files[0]})
    else:
        log.warning("no P2 neutral basis for %s at rank %d: the neutral control is skipped, not invented", mname, rank)
    return ctrls


def magnitude_plan(model, tok, args, mname: str, items: list[dict], basis_t: dict, alpha_t: float,
                   controls: list[dict], out: Path, matched_p2: dict) -> dict:
    """Frozen strengths for every magnitude condition. Dev phase: match and write
    p3_matched_alphas_<model>.json. Test phase: read p3's file (else P2's entries), never match."""
    p3_path = out / f"p3_matched_alphas_{mname}.json"
    if args.phase == "test":
        if p3_path.exists():
            log.info("test phase: matched strengths read from %s", p3_path.name)
            return json.loads(p3_path.read_text(encoding="utf-8"))
        ent = {c["name"]: matched_entry(matched_p2, mname, c["name"]) for c in controls}
        if controls and all(v and "alpha" in v for v in ent.values()):
            log.info("test phase: matched strengths read from p2_matched_alphas.json (no reverse match available)")
            return {"target_energy": (matched_entry(matched_p2, mname, args.target_condition) or {}).get("energy_dev"),
                    "target_alpha": alpha_t, "matched_on": "P2 dev matching",
                    "controls": {k: {"alpha": v["alpha"], "energy": v.get("energy_dev"), "feasible": True,
                                     "source": "p2_matched_alphas.json"} for k, v in ent.items()},
                    "target_at_control_energy": {}}
        log.error("test phase needs frozen strengths: run --phase dev first (or provide p2_matched_alphas.json "
                  "entries for every control)")
        sys.exit(2)
    t_energy = pooled_energy(model, tok, args.fmt, items, basis_t, alpha_t)
    plan = {"target_energy": t_energy, "target_alpha": alpha_t, "controls": {}, "target_at_control_energy": {},
            "matched_on": "dev items, prompt A, span positions, all edited layers", "n_items": len(items),
            "control_sources": {c["name"]: c["source"] for c in controls}}
    for c in controls:
        ent = matched_entry(matched_p2, mname, c["name"])
        if ent and "alpha" in ent:
            plan["controls"][c["name"]] = {"alpha": ent["alpha"], "energy": ent.get("energy_dev"), "feasible": True,
                                           "source": "p2_matched_alphas.json"}
        else:
            m = match_alpha(model, tok, args.fmt, items, c["basis"], t_energy)
            m["source"] = "p3 grid on dev"
            plan["controls"][c["name"]] = m
        e_ctrl1 = pooled_energy(model, tok, args.fmt, items, c["basis"], 1.0)
        m2 = match_alpha(model, tok, args.fmt, items, basis_t, e_ctrl1)
        m2["control_energy_at_alpha1"] = e_ctrl1
        plan["target_at_control_energy"][c["name"]] = m2
    wjson(plan, p3_path)
    return plan


def magnitude_conditions(model, basis_t: dict, alpha_t: float, controls: list[dict], plan: dict) -> list[dict]:
    def mk(b, a):
        return lambda: I.HookedEdit(model, b, alpha=a, site="span")
    conds = [{"condition": "unedited", "factory": None, "kind": "none", "draw": 0, "alpha_used": 0.0,
              "matched_to": "", "energy_target": np.nan, "feasible": True},
             {"condition": "targeted", "factory": mk(basis_t, alpha_t), "kind": "targeted", "draw": 0,
              "alpha_used": alpha_t, "matched_to": "operating", "energy_target": plan.get("target_energy", np.nan),
              "feasible": True}]
    for c in controls:
        pc = plan["controls"].get(c["name"], {})
        a = float(pc.get("alpha", 1.0))
        conds.append({"condition": f"{c['name']}_matched", "factory": mk(c["basis"], a), "kind": c["kind"],
                      "draw": c["draw"], "alpha_used": a, "matched_to": "targeted energy",
                      "energy_target": plan.get("target_energy", np.nan), "feasible": bool(pc.get("feasible", True))})
        if a < 1.0:
            conds.append({"condition": f"{c['name']}_alpha1", "factory": mk(c["basis"], 1.0), "kind": c["kind"],
                          "draw": c["draw"], "alpha_used": 1.0, "matched_to": "none (alpha=1)",
                          "energy_target": np.nan, "feasible": True})
        tc = plan.get("target_at_control_energy", {}).get(c["name"])
        if tc:
            conds.append({"condition": f"targeted_at_{c['name']}_energy", "factory": mk(basis_t, float(tc["alpha"])),
                          "kind": "targeted", "draw": c["draw"], "alpha_used": float(tc["alpha"]),
                          "matched_to": f"{c['name']} energy at alpha=1",
                          "energy_target": tc.get("target", np.nan), "feasible": bool(tc.get("feasible", True))})
    return conds


# ---------------------------------------------------------------------------
# massive-activation account
# ---------------------------------------------------------------------------

def calibration_stats(model, tok, sentences: list[str], fmt: str = "raw") -> dict[int, dict]:
    """Per layer and coordinate, over the calibration prompts: mean absolute activation (positions
    >= 1; position 0 kept apart), its coefficient of variation across prompts (input stability),
    the ratio to the median coordinate, the position-0 ratio, and the pooled variance (the old
    criterion, reported for overlap only)."""
    per_prompt: dict[int, list] = {}
    s1: dict[int, np.ndarray] = {}; s2: dict[int, np.ndarray] = {}; n_pos: dict[int, int] = {}
    bos: dict[int, list] = {}
    for s in sentences:
        text = I.build_input(tok, s, fmt, None)
        _, hs = I.forward_hidden(model, tok, text, None, None)
        for l, h in hs.items():
            x = h.numpy().astype(np.float64)
            body = x[1:] if x.shape[0] > 1 else x
            per_prompt.setdefault(l, []).append(np.abs(body).mean(0))
            bos.setdefault(l, []).append(np.abs(x[0]))
            s1[l] = s1.get(l, 0) + body.sum(0); s2[l] = s2.get(l, 0) + (body ** 2).sum(0)
            n_pos[l] = n_pos.get(l, 0) + body.shape[0]
    stats = {}
    for l, rows in per_prompt.items():
        A = np.stack(rows, 0)
        mean_abs = A.mean(0); cv = A.std(0) / (mean_abs + 1e-9)
        med = float(np.median(mean_abs)) + 1e-9
        var = s2[l] / n_pos[l] - (s1[l] / n_pos[l]) ** 2
        stats[l] = {"mean_abs": mean_abs, "cv": cv, "ratio": mean_abs / med, "median": med,
                    "bos_ratio": np.stack(bos[l], 0).mean(0) / med, "var": var}
    return stats


def identify_massive(stats: dict, n: int, cv_max: float, ratio_min: float) -> pd.DataFrame:
    """Top-n coordinates by magnitude ratio among input-stable ones (cv <= cv_max) per layer,
    with a flag for whether each reaches ratio_min (the stated massive criterion)."""
    rows = []
    for l, st in stats.items():
        cand = np.where(st["cv"] <= cv_max)[0]
        order = cand[np.argsort(-st["ratio"][cand])][:n]
        topvar = set(np.argsort(-st["var"])[:n].tolist())
        for r, dim in enumerate(order):
            rows.append({"layer": int(l), "dim": int(dim), "rank_in_layer": r + 1, "mean_abs": float(st["mean_abs"][dim]),
                         "ratio_to_median": float(st["ratio"][dim]), "cv_across_prompts": float(st["cv"][dim]),
                         "bos_ratio": float(st["bos_ratio"][dim]), "qualifies_ratio": bool(st["ratio"][dim] >= ratio_min),
                         "in_old_top_variance": bool(dim in topvar), "selected": True,
                         "n_stable_candidates": int(len(cand)), "cv_max": cv_max, "ratio_min": ratio_min})
    return pd.DataFrame(rows)


def coords_from_table(tab: pd.DataFrame) -> dict[int, list[int]]:
    sel = tab[tab["selected"].astype(bool)]
    return {int(l): [int(x) for x in g.sort_values("rank_in_layer")["dim"]] for l, g in sel.groupby("layer")}


def random_coords(coords: dict[int, list[int]], d: int, draw: int) -> dict[int, list[int]]:
    out = {}
    for l, dims in coords.items():
        rng = np.random.default_rng(K.RANDOM_SEED_V2 + 1000 * draw + l)
        pool = np.setdiff1d(np.arange(d), np.asarray(dims, int))
        out[l] = [int(x) for x in rng.choice(pool, size=min(len(dims), len(pool)), replace=False)]
    return out


def basis_mass_on_coords(basis: dict, coords: dict[int, list[int]]) -> pd.DataFrame:
    rows = []
    for l, U in basis.items():
        dims = coords.get(l, [])
        peaks = [int(np.abs(U[k]).argmax()) for k in range(U.shape[0])]
        rows.append({"layer": l, "n_coords": len(dims),
                     "basis_mass_fraction_on_coords": float((U[:, dims] ** 2).sum() / max(1, U.shape[0])) if dims else 0.0,
                     "peak_coordinate_hits": int(sum(p in dims for p in peaks)), "rank": int(U.shape[0]),
                     "peak_dims": json.dumps(peaks)})
    return pd.DataFrame(rows)


class RestoredEdit:
    """Erasure (p1_intervention.HookedEdit) followed by a controlled restoration hook at the same
    positions: h'' = h' + r, r = alpha * M (U^T U h) with h the PRE-erasure state captured by a
    snapshot hook registered before the erasure hook, and M a coordinate mask.

    modes  massive        M = identified coordinates
           random         M = random coordinates of the same count
           equal_energy   M = random coordinates, r rescaled per token to ||alpha M_ref (U^T U h)||
           complement     M = every coordinate except the identified ones (erase them only)
           all            M = 1: numerical identity check (h'' must equal h)
    Exposes the HookedEdit interface the P1 helpers call (set_targets, context manager, summary,
    changed_log with the net delta plus removed and restored norms). A diagnostic, not a method."""

    MODES = ("massive", "random", "equal_energy", "complement", "all")

    def __init__(self, model, basis_by_layer: dict, alpha: float, site: str, coords_by_layer: dict[int, list[int]],
                 mode: str, ref_coords_by_layer: dict[int, list[int]] | None = None, record: bool = True):
        import torch
        if mode not in self.MODES:
            raise ValueError(mode)
        self.model = model; self.mode = mode; self.record = record
        self.edit = I.HookedEdit(model, basis_by_layer, alpha=alpha, site=site, record=record)
        dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
        self.mask, self.ref_mask = {}, {}
        for li, U in self.edit.U.items():
            d = U.shape[1]
            m = torch.zeros(d, device=dev, dtype=dt)
            idx = [i for i in coords_by_layer.get(li, []) if 0 <= i < d]
            if mode == "all":
                m[:] = 1
            elif mode == "complement":
                m[:] = 1
                if idx:
                    m[idx] = 0
            elif idx:
                m[idx] = 1
            self.mask[li] = m
            if mode == "equal_energy":
                rm = torch.zeros(d, device=dev, dtype=dt)
                ref = [i for i in (ref_coords_by_layer or {}).get(li, []) if 0 <= i < d]
                if ref:
                    rm[ref] = 1
                self.ref_mask[li] = rm
        self._snap: dict[int, dict] = {}
        self.handles = []
        self.changed_log: list[dict] = []

    # HookedEdit interface -------------------------------------------------
    def set_targets(self, positions_by_row, pad_offsets=None):
        self.edit.set_targets(positions_by_row, pad_offsets)

    @property
    def alpha(self):
        return self.edit.alpha

    @property
    def U(self):
        return self.edit.U

    @property
    def rank(self):
        return self.edit.rank

    @property
    def n_decode_edits(self):
        return self.edit.n_decode_edits

    def _snap_hook(self, li: int):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            B, T, _ = hs.shape
            prefill = T > 1
            snaps = {}
            for b in range(B):
                pos = self.edit._positions_for(b, T, prefill)
                if pos:
                    snaps[b] = (pos, hs[b, pos, :].clone(), prefill)
            self._snap[li] = snaps
            return output
        return hook

    def _restore_hook(self, li: int):
        U = self.edit.U[li]; alpha = self.edit.alpha; mask = self.mask[li]; ref = self.ref_mask.get(li)

        def hook(module, inputs, output):
            import torch
            hs = output[0] if isinstance(output, tuple) else output
            for b, (pos, v, prefill) in self._snap.pop(li, {}).items():
                comp = alpha * (v @ U.transpose(0, 1)) @ U            # the removed component, n x d
                r = comp * mask
                if ref is not None:
                    rn = r.float().norm(dim=-1, keepdim=True)
                    refn = (comp * ref).float().norm(dim=-1, keepdim=True)
                    r = (r.float() * (refn / rn.clamp_min(1e-12))).to(r.dtype)
                hs[b, pos, :] = hs[b, pos, :] + r
                if self.record:
                    with torch.no_grad():
                        net = (hs[b, pos, :] - v).float().norm(dim=-1); hn = v.float().norm(dim=-1)
                        rn_ = r.float().norm(dim=-1); cn = comp.float().norm(dim=-1)
                        for p, n_, h_, r_, c_ in zip(pos, net.tolist(), hn.tolist(), rn_.tolist(), cn.tolist()):
                            self.changed_log.append({"layer": li, "row": b, "position": int(p), "prefill": prefill,
                                                     "delta_norm": n_, "h_norm": h_, "removed_norm": c_,
                                                     "restored_norm": r_})
            return output
        return hook

    def __enter__(self):
        layers = I._decoder_layers(self.model)
        active = [li for li, U in self.edit.U.items() if U.shape[0] > 0]
        for li in active:
            self.handles.append(layers[li].register_forward_hook(self._snap_hook(li)))
        self.edit.__enter__()                                    # erasure hooks, registered second
        for li in active:
            self.handles.append(layers[li].register_forward_hook(self._restore_hook(li)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.edit.__exit__(*exc)
        self._snap = {}
        return False

    def energy_by_layer(self) -> dict[int, float]:
        num, den = {}, {}
        for r in self.changed_log:
            num[r["layer"]] = num.get(r["layer"], 0.0) + r["delta_norm"] ** 2
            den[r["layer"]] = den.get(r["layer"], 0.0) + r["h_norm"] ** 2
        return {l: (num[l] / den[l] if den[l] > 0 else float("nan")) for l in num}

    def summary(self) -> dict:
        e = self.energy_by_layer()
        return {"site": self.edit.site, "alpha": self.edit.alpha, "rank": self.edit.rank, "mode": self.mode,
                "n_layers_edited": len(self.edit.U), "n_edits": len(self.changed_log),
                "n_decode_step_edits": self.edit.n_decode_edits,
                "energy_mean_over_layers": float(np.mean(list(e.values()))) if e else float("nan"),
                "energy_by_layer": e,
                "removed_energy_pooled": pooled_energy_of_log(self.changed_log, "removed_norm"),
                "restored_energy_pooled": pooled_energy_of_log(self.changed_log, "restored_norm")}


def massive_conditions(model, basis: dict, alpha: float, coords: dict[int, list[int]], d: int, n_draws: int) -> list[dict]:
    def mk_restore(mode, cby, ref=None):
        return lambda: RestoredEdit(model, basis, alpha, "span", cby, mode, ref)
    n_coords = int(np.mean([len(v) for v in coords.values()])) if coords else 0
    conds = [{"condition": "unedited", "factory": None, "restore_mode": "none", "draw": 0, "n_coords": 0, "alpha_used": 0.0},
             {"condition": "erased", "factory": lambda: I.HookedEdit(model, basis, alpha=alpha, site="span"),
              "restore_mode": "none", "draw": 0, "n_coords": 0, "alpha_used": alpha},
             {"condition": "restore_massive", "factory": mk_restore("massive", coords), "restore_mode": "massive",
              "draw": 0, "n_coords": n_coords, "alpha_used": alpha},
             {"condition": "erase_massive_only", "factory": mk_restore("complement", coords), "restore_mode": "complement",
              "draw": 0, "n_coords": n_coords, "alpha_used": alpha},
             {"condition": "restore_all", "factory": mk_restore("all", coords), "restore_mode": "all", "draw": 0,
              "n_coords": d, "alpha_used": alpha}]
    for i in range(1, n_draws + 1):
        rc = random_coords(coords, d, i)
        conds.append({"condition": f"restore_random_{i}", "factory": mk_restore("random", rc), "restore_mode": "random",
                      "draw": i, "n_coords": n_coords, "alpha_used": alpha})
        conds.append({"condition": f"restore_equal_energy_{i}", "factory": mk_restore("equal_energy", rc, coords),
                      "restore_mode": "equal_energy", "draw": i, "n_coords": n_coords, "alpha_used": alpha})
    return conds


# ---------------------------------------------------------------------------
# statistics and summaries
# ---------------------------------------------------------------------------

def spearman_cluster_ci(x, y, clusters, n_boot: int = 1000, seed: int = K.RANDOM_SEED_V2) -> tuple[float, float, float, int]:
    """Spearman correlation with a seed-cluster percentile bootstrap (Efron and Tibshirani 1993)."""
    import warnings
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float); cl = np.asarray(clusters)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, cl = x[ok], y[ok], cl[ok]
    if len(x) < 4:
        return float("nan"), float("nan"), float("nan"), int(len(x))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # constant resamples give nan, dropped below
        point = float(spearmanr(x, y).correlation)
        codes, uniq = pd.factorize(cl, sort=False)
        k = len(uniq)
        if k < 2:
            return point, float("nan"), float("nan"), int(len(x))
        idx_by = [np.where(codes == i)[0] for i in range(k)]
        rng = np.random.default_rng(seed)
        draws = np.empty(n_boot)
        for b in range(n_boot):
            take = rng.integers(0, k, size=k)
            idx = np.concatenate([idx_by[i] for i in take])
            draws[b] = spearmanr(x[idx], y[idx]).correlation
        if not np.isfinite(draws).any():
            return point, float("nan"), float("nan"), int(len(x))
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi), int(len(x))


def mean_ci(x, clusters) -> tuple[float, float, float]:
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), float); cl = np.asarray(clusters)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    return K.ratio_bootstrap(x[ok], np.ones(int(ok.sum())), cl[ok])


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)


def _bool01(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].map({True: 1.0, False: 0.0}).astype(float) if col in df else pd.Series(np.nan, index=df.index)


def sides_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (item, side) with option correctness, p_gold and margin; used for accuracy."""
    parts = []
    for side in ("A", "B"):
        if f"opt_correct_{side}" not in df:
            continue
        d = df[PAIR_KEY + ["model_name", "phase", "condition", "benchmark"]].copy()
        d["side"] = side
        d["correct"] = _bool01(df, f"opt_correct_{side}")
        d["p_gold"] = _num(df, f"p_gold_{side}")
        d["margin"] = _num(df, f"opt_margin_{side}")
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SIDE_KEY + ["correct", "p_gold", "margin"])


def paired_diff(cond_df: pd.DataFrame, ref_df: pd.DataFrame, col: str, keys: list[str]) -> tuple[float, float, float, int]:
    """Mean of (cond - ref) over items matched on `keys` (seed_id included), seed-cluster CI."""
    if not len(cond_df) or not len(ref_df) or col not in cond_df or col not in ref_df:
        return float("nan"), float("nan"), float("nan"), 0
    m = cond_df[keys + [col]].merge(ref_df[keys + [col]], on=keys, suffixes=("", "_ref"))
    x = pd.to_numeric(m[col], errors="coerce") - pd.to_numeric(m[col + "_ref"], errors="coerce")
    ok = x.notna()
    if ok.sum() == 0:
        return float("nan"), float("nan"), float("nan"), 0
    p, lo, hi = K.ratio_bootstrap(x[ok].to_numpy(), np.ones(int(ok.sum())), m.loc[ok, "seed_id"].to_numpy())
    return p, lo, hi, int(ok.sum())


def _first(df: pd.DataFrame, col: str, default=""):
    return df[col].iloc[0] if col in df and len(df) else default


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Drop failed rows and rows whose demographic span did not resolve on either prompt (an
    unresolved span means the edit had nothing to act on, so the row would read as unedited)."""
    n0 = len(df)
    if "error" in df:
        df = df[df["error"].isna()]
    if "span_len_A" in df and "span_len_B" in df:
        df = df[(_num(df, "span_len_A") > 0) & (_num(df, "span_len_B") > 0)]
    if len(df) < n0:
        log.info("summaries use %d of %d rows (%d failed or span-unresolved rows excluded)", len(df), n0, n0 - len(df))
    return df


def summarise(df: pd.DataFrame, account: str) -> pd.DataFrame:
    """Per (model, phase, condition): |C|, option accuracy, p_gold, energy, each with a seed-cluster
    CI, and paired changes versus the unedited rows of the same items. Losses are positive when
    the edit hurts."""
    df = usable(df)
    rows = []
    for (m, ph), g in df.groupby(["model_name", "phase"]):
        ref = g[g["condition"] == "unedited"]
        ref_s = sides_long(ref)
        for cond, gc in g.groupby("condition", sort=False):
            gs = sides_long(gc)
            absC = mean_ci(gc["absC"], gc["seed_id"])
            acc = mean_ci(gs["correct"], gs["seed_id"])
            pg = mean_ci(gs["p_gold"], gs["seed_id"])
            dC = paired_diff(gc, ref, "absC", PAIR_KEY)
            dacc = paired_diff(gs, ref_s, "correct", SIDE_KEY)
            dpg = paired_diff(gs, ref_s, "p_gold", SIDE_KEY)
            r = {"model_name": m, "phase": ph, "account": account, "condition": cond, "n_items": len(gc),
                 "n_seeds": int(gc["seed_id"].nunique()), "n_prompts_scored": int(gs["correct"].notna().sum()),
                 "absC_mean": absC[0], "absC_mean_lo": absC[1], "absC_mean_hi": absC[2],
                 "absC_change_vs_unedited": dC[0], "absC_change_vs_unedited_lo": dC[1], "absC_change_vs_unedited_hi": dC[2],
                 "opt_acc": acc[0], "opt_acc_lo": acc[1], "opt_acc_hi": acc[2],
                 "acc_loss_vs_unedited": -dacc[0], "acc_loss_vs_unedited_lo": -dacc[2], "acc_loss_vs_unedited_hi": -dacc[1],
                 "n_paired_prompts": dacc[3],
                 "p_gold_mean": pg[0], "p_gold_loss_vs_unedited": -dpg[0], "p_gold_loss_vs_unedited_lo": -dpg[2],
                 "p_gold_loss_vs_unedited_hi": -dpg[1],
                 "energy_total_A_mean": float(_num(gc, "energy_total_A").mean()),
                 "energy_audit_mean": float(_num(gc, "energy_audit").mean()),
                 "alpha": float(_num(gc, "alpha").median()), "sec_mean": float(_num(gc, "sec_total").mean())}
            if account == "depth":
                for k in ("first_dev_layer_span", "first_dev_layer_last", "first_dev_layer_postspan", "first_margin_dev_cutoff"):
                    v = _num(gc, k)
                    r[k + "_median"] = float(v[v >= 0].median()) if (v >= 0).any() else np.nan
                    r[k + "_frac_never"] = float((v < 0).mean()) if v.notna().any() else np.nan
                r["n_layers_edited"] = int(_num(gc, "n_layers_edited").median()) if _num(gc, "n_layers_edited").notna().any() else 0
                r["prespan_untouched_frac"] = float(_bool01(gc, "prespan_untouched").mean())
                r["fitting"] = _first(gc, "fitting")
            elif account == "magnitude":
                r["kind"] = _first(gc, "kind"); r["matched_to"] = _first(gc, "matched_to")
                r["energy_target"] = float(_num(gc, "energy_target").median())
                r["feasible"] = bool(_first(gc, "feasible", True))
            else:
                r["restore_mode"] = _first(gc, "restore_mode")
                r["n_coords"] = int(_num(gc, "n_coords").median()) if _num(gc, "n_coords").notna().any() else 0
                r["removed_energy_A_mean"] = float(_num(gc, "removed_energy_A").mean())
                r["restored_energy_A_mean"] = float(_num(gc, "restored_energy_A").mean())
            rows.append(r)
    return pd.DataFrame(rows)


def energy_relationships(df: pd.DataFrame) -> list[dict]:
    """Per-item removed energy versus per-item loss (p_gold, correctness) and |C| change, pooled
    over edited conditions and within kinds, with seed-cluster Spearman CIs."""
    df = usable(df)
    out = []
    for (m, ph), g in df.groupby(["model_name", "phase"]):
        ref = g[g["condition"] == "unedited"][PAIR_KEY + ["absC", "p_gold_A", "opt_correct_A"]]
        ed = g[g["condition"] != "unedited"].merge(ref, on=PAIR_KEY, suffixes=("", "_ref"))
        if not len(ed):
            continue
        ed["p_gold_loss_A"] = _num(ed, "p_gold_A_ref") - _num(ed, "p_gold_A")
        ed["acc_loss_A"] = _bool01(ed, "opt_correct_A_ref") - _bool01(ed, "opt_correct_A")
        ed["absC_change"] = _num(ed, "absC") - _num(ed, "absC_ref")
        kind = ed["kind"] if "kind" in ed else pd.Series("", index=ed.index)
        for scope, sub in (("all_edited_conditions", ed), ("targeted_only", ed[kind == "targeted"]),
                           ("random_only", ed[kind == "random"]), ("neutral_only", ed[kind == "neutral"])):
            if not len(sub):
                continue
            for y in ("p_gold_loss_A", "acc_loss_A", "absC_change"):
                rho, lo, hi, n = spearman_cluster_ci(_num(sub, "energy_total_A"), sub[y], sub["seed_id"])
                out.append({"model_name": m, "phase": ph, "scope": scope, "x": "energy_total_A", "y": y,
                            "spearman": rho, "lo": lo, "hi": hi, "n": n, "n_seeds": int(sub["seed_id"].nunique())})
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def upsert_section(path: Path, header: str, body: list[str]) -> None:
    """Replace the section starting with `header` (a '## ' line) or append it."""
    text = path.read_text(encoding="utf-8") if path.exists() else \
        "# P3 explanation report\n\nOne account per section. Development sections choose; test sections confirm once.\n"
    lines = text.split("\n")
    start = next((i for i, l in enumerate(lines) if l.strip() == header.strip()), None)
    if start is not None:
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
        lines = lines[:start] + lines[end:]
    while lines and lines[-1] == "":
        lines.pop()
    lines += ["", header, ""] + body + [""]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("updated %s [%s]", _rel(path), header)


def _f(v, nd: int = 3) -> str:
    try:
        return "nan" if v is None or not np.isfinite(float(v)) else f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _row(summ: pd.DataFrame, cond: str):
    s = summ[summ["condition"] == cond]
    return s.iloc[0] if len(s) else None


def _ci(r, key: str) -> str:
    return "%s [%s, %s]" % (_f(r[key]), _f(r[key + "_lo"]), _f(r[key + "_hi"]))


def _phase_note(phase: str) -> str:
    return (" Development phase: this chooses whether the account is confirmed on the P2 test seeds."
            if phase == "dev" else " Test phase: single confirmation on P2's test seeds; nothing was selected on them.")


def decision_depth(summ: pd.DataFrame, per: pd.DataFrame, model: str, phase: str) -> list[str]:
    s = summ[(summ["model_name"] == model) & (summ["phase"] == phase)]
    lines = ["Layer choices are diagnostic controls, not localisation: no layer block is claimed to hold the "
             "demographic computation and no Patchscope-style statement is made.", "",
             "| condition | layers | fitting | acc loss vs unedited [95% CI] | abs C change [95% CI] | energy(A) | "
             "first dev layer: span / last / post-span (median) | first margin cutoff (median) |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        ed = r["condition"] != "unedited"
        lines.append("| %s | %d | %s | %s | %s | %s | %s / %s / %s | %s |" % (
            r["condition"], int(r.get("n_layers_edited", 0) or 0), r.get("fitting", ""),
            _ci(r, "acc_loss_vs_unedited") if ed else "-", _ci(r, "absC_change_vs_unedited") if ed else "-",
            _f(r["energy_total_A_mean"]), _f(r.get("first_dev_layer_span_median"), 0),
            _f(r.get("first_dev_layer_last_median"), 0), _f(r.get("first_dev_layer_postspan_median"), 0),
            _f(r.get("first_margin_dev_cutoff_median"), 0)))
    fr, sq = _row(s, "full_frozen"), _row(s, "full_sequential")
    lines.append("")
    if fr is not None:
        lines.append("Sanity: positions before the span were untouched in %s of profiled items (must be 1.000)."
                     % _f(fr.get("prespan_untouched_frac")))
    if fr is None or sq is None:
        lines.append("Decision: incomplete (full_frozen or full_sequential rows missing).")
        return lines
    pe = usable(per[(per["model_name"] == model) & (per["phase"] == phase)])
    d = paired_diff(sides_long(pe[pe["condition"] == "full_sequential"]), sides_long(pe[pe["condition"] == "full_frozen"]),
                    "correct", SIDE_KEY)
    lines.append("Accuracy under full_sequential minus full_frozen (paired, seed-cluster 95%% CI): %s [%s, %s], n = %d prompts."
                 % (_f(d[0]), _f(d[1]), _f(d[2]), d[3]))
    lf, ls = float(fr["acc_loss_vs_unedited"]), float(sq["acc_loss_vs_unedited"])
    if np.isfinite(lf) and np.isfinite(ls) and lf > 0 and ls <= 0.5 * lf and np.isfinite(d[1]) and d[1] > 0:
        verdict = ("SUPPORTED (stale representations): sequential fitting removes at least half of the accuracy loss "
                   "with an interval excluding zero. The conclusion concerns stale representations under repeated "
                   "frozen edits, not a property of demographic erasure as such.")
    elif np.isfinite(lf) and np.isfinite(ls) and ls >= lf:
        verdict = "NOT supported: the harm persists when every layer's basis is refitted on already-edited activations."
    else:
        verdict = "INCONCLUSIVE: the sequential and frozen losses are not separated by the interval."
    lines += ["", "Decision: " + verdict + _phase_note(phase)]
    return lines


def decision_magnitude(summ: pd.DataFrame, rel: list[dict], plan: dict | None, model: str, phase: str) -> list[str]:
    s = summ[(summ["model_name"] == model) & (summ["phase"] == phase)]
    lines = ["| condition | kind | alpha | energy target | energy(A) | feasible | acc loss vs unedited [95% CI] | "
             "p_gold loss [95% CI] | abs C change [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        ed = r["condition"] != "unedited"
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["condition"], r.get("kind", ""), _f(r["alpha"]), _f(r.get("energy_target")), _f(r["energy_total_A_mean"]),
            r.get("feasible", True), _ci(r, "acc_loss_vs_unedited") if ed else "-",
            _ci(r, "p_gold_loss_vs_unedited") if ed else "-", _ci(r, "absC_change_vs_unedited") if ed else "-"))
    lines += ["", "Per-item removed energy versus per-item damage (Spearman, seed-cluster 95% CI):", "",
              "| scope | y | rho | lo | hi | n |", "|---|---|---|---|---|---|"]
    for r in rel:
        if r["model_name"] == model and r["phase"] == phase:
            lines.append("| %s | %s | %s | %s | %s | %d |" % (r["scope"], r["y"], _f(r["spearman"]), _f(r["lo"]), _f(r["hi"]), r["n"]))
    t = _row(s, "targeted")
    ctrl = s[s["condition"].str.endswith("_matched")]
    lines.append("")
    if plan:
        lines.append("Targeted energy on dev = %s at alpha = %s (%s)." % (_f(plan.get("target_energy")), _f(plan.get("target_alpha")),
                                                                        plan.get("matched_on", "")))
    if t is None or not len(ctrl):
        lines.append("Decision: incomplete (targeted or matched-control rows missing).")
        return lines
    lt = float(t["acc_loss_vs_unedited"])
    feas = ctrl[ctrl["feasible"].astype(bool)]
    lines.append("Matched controls feasible at alpha <= 1: %d of %d (an infeasible control cannot reach the targeted "
                 "energy without alpha > 1, so the reverse match 'targeted_at_<control>_energy' carries that comparison)."
                 % (len(feas), len(ctrl)))
    lc = float(feas["acc_loss_vs_unedited"].mean()) if len(feas) else float("nan")
    rev = s[s["condition"].str.startswith("targeted_at_")]
    ctrl1 = s[s["condition"].str.endswith("_alpha1") | (s["condition"].str.endswith("_matched") & ~s["feasible"].astype(bool))]
    lrev = float(rev["acc_loss_vs_unedited"].mean()) if len(rev) else float("nan")
    lc1 = float(ctrl1["acc_loss_vs_unedited"].mean()) if len(ctrl1) else float("nan")
    pooled = next((r for r in rel if r["model_name"] == model and r["phase"] == phase
                   and r["scope"] == "all_edited_conditions" and r["y"] == "p_gold_loss_A"), None)
    rho_ok = pooled is not None and np.isfinite(pooled["lo"]) and pooled["lo"] > 0 and pooled["spearman"] >= 0.3
    controls_match = (np.isfinite(lc) and np.isfinite(lt) and lt > 0 and lc >= 0.5 * lt) or \
                     (np.isfinite(lrev) and np.isfinite(lc1) and abs(lrev - lc1) <= 0.5 * max(abs(lt), 1e-9))
    if controls_match and rho_ok:
        verdict = ("MAGNITUDE ACCOUNT SUPPORTED: damage tracks removed energy and matched-energy controls reproduce at "
                   "least half of the targeted loss. Report non-specific perturbation; a special demographic mechanism "
                   "claim is weakened.")
    elif np.isfinite(lt) and lt > 0 and not controls_match and not rho_ok:
        verdict = ("Damage exceeds matched-energy controls and does not track energy: the magnitude account does not "
                   "explain it. This does not by itself establish a demographic mechanism.")
    else:
        verdict = "INCONCLUSIVE: the matched-control and energy-correlation evidence do not agree."
    lines += ["", "Decision: " + verdict + (" Strengths were matched on dev and frozen." if phase == "dev"
                                             else " Strengths frozen from dev; single confirmation.")]
    return lines


def decision_massive(summ: pd.DataFrame, per: pd.DataFrame, coords_tab: pd.DataFrame | None,
                     mass: pd.DataFrame | None, model: str, phase: str) -> list[str]:
    s = summ[(summ["model_name"] == model) & (summ["phase"] == phase)]
    lines = ["Coordinates were identified on %d neutral calibration sentences (no demographic terms) by mean absolute "
             "activation and input stability (CV across prompts), not by variance; the top-variance overlap is "
             "reported for comparison with the old diagnostic only." % len(NEUTRAL_CALIBRATION), ""]
    if coords_tab is not None and len(coords_tab):
        q = coords_tab[coords_tab["selected"].astype(bool)]
        lines.append("Identified coordinates: %d per layer (median); %s of them reach the stated ratio criterion "
                     "(ratio_min = %s); %s overlap the old top-variance set; distinct coordinates across layers: %d; "
                     "median CV = %s." % (int(q.groupby("layer").size().median()), _f(q["qualifies_ratio"].mean()),
                                          _f(q["ratio_min"].iloc[0], 0), _f(q["in_old_top_variance"].mean()),
                                          int(q["dim"].nunique()), _f(q["cv_across_prompts"].median())))
    if mass is not None and len(mass):
        lines.append("Targeted basis mass on the identified coordinates: mean %s over layers; peak-coordinate hits "
                     "(old criterion) %d of %d rows." % (_f(mass["basis_mass_fraction_on_coords"].mean()),
                                                         int(mass["peak_coordinate_hits"].sum()), int(mass["rank"].sum())))
    lines += ["", "| condition | mode | n coords | removed E | restored E | net E | acc loss vs unedited [95% CI] | "
              "p_gold loss [95% CI] | abs C mean [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        ed = r["condition"] != "unedited"
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["condition"], r.get("restore_mode", ""), r.get("n_coords", 0), _f(r.get("removed_energy_A_mean")),
            _f(r.get("restored_energy_A_mean")), _f(r["energy_total_A_mean"]),
            _ci(r, "acc_loss_vs_unedited") if ed else "-", _ci(r, "p_gold_loss_vs_unedited") if ed else "-",
            _ci(r, "absC_mean")))
    un, er, rm = _row(s, "unedited"), _row(s, "erased"), _row(s, "restore_massive")
    lines.append("")
    if un is None or er is None or rm is None:
        lines.append("Decision: incomplete (unedited, erased or restore_massive rows missing).")
        return lines
    pe = usable(per[(per["model_name"] == model) & (per["phase"] == phase)])
    ers = sides_long(pe[pe["condition"] == "erased"])

    def gain(cond):
        return paired_diff(sides_long(pe[pe["condition"] == cond]), ers, "correct", SIDE_KEY)
    g_m = gain("restore_massive")
    g_r = [gain(c) for c in s["condition"] if c.startswith("restore_random_")]
    g_e = [gain(c) for c in s["condition"] if c.startswith("restore_equal_energy_")]
    ra = _row(s, "restore_all")
    lines.append("Accuracy gain of restore_massive over erased (paired, 95%% CI): %s [%s, %s], n = %d prompts; "
                 "random-coordinate restorations: %s; equal-energy restorations: %s." % (
                     _f(g_m[0]), _f(g_m[1]), _f(g_m[2]), g_m[3],
                     ", ".join(_f(g[0]) for g in g_r) or "none", ", ".join(_f(g[0]) for g in g_e) or "none"))
    if ra is not None:
        lines.append("Identity check restore_all: accuracy loss vs unedited %s, abs C change %s (numerical tolerance of "
                     "the model dtype; not an experimental condition)." % (_f(ra["acc_loss_vs_unedited"]),
                                                                            _f(ra["absC_change_vs_unedited"])))
    red_er = float(un["absC_mean"]) - float(er["absC_mean"])
    red_rm = float(un["absC_mean"]) - float(rm["absC_mean"])
    retained = red_rm / red_er if np.isfinite(red_er) and abs(red_er) > 1e-9 else float("nan")
    lines.append("Share of the erased abs C reduction retained under restore_massive: %s." % _f(retained))
    best_ctrl = max([g[0] for g in g_r + g_e if np.isfinite(g[0])], default=float("nan"))
    if np.isfinite(g_m[1]) and g_m[1] > 0 and (not np.isfinite(best_ctrl) or g_m[0] > best_ctrl) \
            and np.isfinite(retained) and retained >= 0.5:
        verdict = ("SUPPORTED as a diagnostic explanation: restoring the removed component on the identified high-"
                   "magnitude, input-stable coordinates recovers option accuracy while retaining at least half of the "
                   "abs C suppression, and random-coordinate and equal-energy restorations do not match it.")
    elif np.isfinite(g_m[1]) and g_m[1] > 0:
        verdict = ("PARTIAL: restoration recovers accuracy but also undoes most of the suppression, or a control "
                   "matches it; the removed component on those coordinates carries both, so this is a trade-off, "
                   "not a mechanism that separates them.")
    else:
        verdict = ("NOT supported by restoration: only coordinate overlap is available, so the massive-activation "
                   "explanation stays a hypothesis.")
    verdict += (" A diagnostic restoration is not a protected-erasure algorithm and is not proposed as one. The word "
                "'bias' in Sun et al. (2024) and related titles denotes a constant attention offset, not social bias.")
    lines += ["", "Decision: " + verdict + (" Development phase." if phase == "dev"
                                             else " Test phase: coordinates frozen before the run; single confirmation.")]
    return lines


def write_report(out: Path, account: str, models: list[str], phase: str, extras: dict) -> None:
    per_path = out / f"p3_{account}_per_item.parquet"
    if not per_path.exists():
        log.warning("no per-item results for %s yet", account); return
    per = pd.read_parquet(per_path)
    summ = summarise(per, account)
    wcsv(summ, out / f"p3_{account}_summary.csv")
    rel = energy_relationships(per) if account == "magnitude" else []
    if rel:
        wcsv(pd.DataFrame(rel), out / "p3_magnitude_energy_relationships.csv")
    for m in models:
        if not ((per["model_name"] == m) & (per["phase"] == phase)).any():
            continue
        header = "## account=%s model=%s phase=%s" % (account, K.DISPLAY.get(m, m), phase)
        n_seeds = int(per[(per["model_name"] == m) & (per["phase"] == phase)]["seed_id"].nunique())
        body = ["Generated %s. Items: largest-|C_pre| clean demographic pair per seed; %d seeds. Edit: P2 frozen basis "
                "'%s' rank %s, span site, sequential source state." % (K.utc_now(), n_seeds,
                                                                        extras.get("target_condition", ""), extras.get("rank", "")), ""]
        if account == "depth":
            body += decision_depth(summ, per, m, phase)
        elif account == "magnitude":
            pp = out / f"p3_matched_alphas_{m}.json"
            plan = extras.get("plans", {}).get(m) or (json.loads(pp.read_text(encoding="utf-8")) if pp.exists() else None)
            body += decision_magnitude(summ, rel, plan, m, phase)
        else:
            cp = out / f"p3_massive_coords_{m}.csv"
            mp = out / f"p3_massive_basis_mass_{m}.csv"
            body += decision_massive(summ, per, pd.read_csv(cp) if cp.exists() else None,
                                     pd.read_csv(mp) if mp.exists() else None, m, phase)
        upsert_section(out / "p3_report.md", header, body)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    d = None
    for attempt in range(5):                 # a git checkout can rewrite the file for an instant
        try:
            d = pd.read_parquet(path, columns=KEY_COLS); break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 + 3 * attempt)
    return set(map(tuple, d.to_numpy().tolist()))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Design")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--account", choices=ACCOUNTS, required=True)
    ap.add_argument("--models", nargs="*", default=["qwen2.5-7b-instruct", "llama-3.1-8b-instruct"])
    ap.add_argument("--phase", choices=PHASES, default="dev")
    ap.add_argument("--fmt", choices=["chat", "raw"], default="chat")
    ap.add_argument("--target-condition", default="cure_centred_svd", help="P2 basis condition name")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=None,
                    help="override the operating strength (else p2_matched_alphas.json, else 1.0)")
    ap.add_argument("--n-seeds", type=int, default=60, help="dev-phase cap, benchmark-stratified")
    ap.add_argument("--n-fit-pairs", type=int, default=120, help="FIT-split pairs for sequential fitting (depth)")
    ap.add_argument("--seq-block", type=int, default=4, help="layers per sequential-fitting block (1 = exact)")
    ap.add_argument("--dev-tol", type=float, default=0.05, help="relative-norm deviation tolerance (depth)")
    ap.add_argument("--margin-tol", type=float, default=0.5, help="gold-option margin tolerance in nats (depth)")
    ap.add_argument("--margin-scan-stride", type=int, default=4)
    ap.add_argument("--n-random-draws", type=int, default=3)
    ap.add_argument("--n-massive", type=int, default=5, help="coordinates per layer (massive)")
    ap.add_argument("--massive-cv-max", type=float, default=0.5, help="input-stability ceiling (CV across prompts)")
    ap.add_argument("--massive-ratio", type=float, default=50.0, help="ratio to the median coordinate that counts as massive")
    ap.add_argument("--calib-fmt", choices=["raw", "chat"], default="raw")
    ap.add_argument("--gpu-hours-cap", type=float, default=6.0)
    ap.add_argument("--out-dir", default=None, help="override results/v2 (CPU smoke tests only)")
    ap.add_argument("--report-only", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    global _FMT
    _FMT = args.fmt                      # P2 basis files are named by prompt format
    out = Path(args.out_dir) if args.out_dir else K.OUT_V2
    out.mkdir(parents=True, exist_ok=True)
    extras = {"target_condition": args.target_condition, "rank": args.rank, "plans": {}}
    if args.report_only:
        write_report(out, args.account, args.models, args.phase, extras); return

    t_start = time.time()
    cap_s = args.gpu_hours_cap * 3600.0

    def clock() -> bool:
        over = (time.time() - t_start) > cap_s
        if over:
            log.warning("GPU-hours cap reached; stopping cleanly (everything done is on disk)")
        return over

    # P2 must have run: a frozen basis for every requested model, and the test seed list for test
    matched_p2 = read_matched(out)
    for m in args.models:
        require(basis_path(out, m, args.target_condition, args.rank),
                "P3 runs only after P2: run p2 first so its frozen basis files exist "
                "(p2_basis_<model>_<condition>_r<rank>.npz).")
    seeds = select_seeds(out, args.phase, args.n_seeds, K.RANDOM_SEED_V2)
    per_item = out / f"p3_{args.account}_per_item.parquet"
    done = done_keys(per_item)
    c, _ = P._pentad_c()
    cdva = K.read_cdva()

    for mname in args.models:
        if clock():
            break
        log.info("=== P3 %s: %s (%s phase, %d seeds) ===", args.account, mname, args.phase, len(seeds))
        elig = pd.read_csv(K.OUT_P0 / "pair_sets" / f"{mname}_eligible.csv")
        elig_keys = set(zip(elig["seed_id"], elig["subvariant_A"], elig["subvariant_B"]))
        items = select_items(mname, seeds, cdva, c, elig_keys)
        basis_t = load_basis(basis_path(out, mname, args.target_condition, args.rank), args.rank)
        alpha_t, alpha_src = operating_alpha(matched_p2, mname, args.target_condition, args.alpha)
        log.info("operating alpha = %.3f (%s); basis layers = %d; items = %d", alpha_t, alpha_src, len(basis_t), len(items))
        t_load = time.time()
        cfg, model, tok = P.load(mname)
        log.info("loaded %s in %.0fs", mname, time.time() - t_load)
        try:
            d_model = next(iter(basis_t.values())).shape[1]
            if args.account == "depth":
                seq_path = out / f"p3_basis_{mname}_sequential_b{args.seq_block}_r{args.rank}.npz"
                if seq_path.exists():
                    basis_seq = load_basis(seq_path); log.info("sequential basis loaded from %s", seq_path.name)
                else:
                    fp = P.fit_pairs(mname, args.n_fit_pairs, K.RANDOM_SEED_V2)
                    basis_seq, sinfo = fit_sequential_basis(model, tok, args.fmt, fp, c, args.rank, alpha_t, args.seq_block)
                    save_basis(seq_path, basis_seq)
                    wjson(sinfo, seq_path.with_suffix(".json"))
                run_depth(model, tok, args, mname, items, basis_t, basis_seq, alpha_t, per_item, done, clock)
            elif args.account == "magnitude":
                controls = control_bases(out, mname, args.rank, sorted(basis_t), d_model, args.n_random_draws)
                plan = magnitude_plan(model, tok, args, mname, items, basis_t, alpha_t, controls, out, matched_p2)
                extras["plans"][mname] = plan
                conds = magnitude_conditions(model, basis_t, alpha_t, controls, plan)
                run_conditions(model, tok, args, mname, "magnitude", items, conds, per_item, done, clock,
                               ["kind", "draw", "matched_to", "energy_target", "feasible"])
            else:
                cp = out / f"p3_massive_coords_{mname}.csv"
                if cp.exists():
                    tab = pd.read_csv(cp); log.info("massive coordinates frozen from %s", cp.name)
                else:
                    stats = calibration_stats(model, tok, NEUTRAL_CALIBRATION, args.calib_fmt)
                    tab = identify_massive(stats, args.n_massive, args.massive_cv_max, args.massive_ratio)
                    wcsv(tab, cp)
                coords = coords_from_table(tab)
                wcsv(basis_mass_on_coords(basis_t, coords), out / f"p3_massive_basis_mass_{mname}.csv")
                conds = massive_conditions(model, basis_t, alpha_t, coords, d_model, args.n_random_draws)
                run_conditions(model, tok, args, mname, "massive", items, conds, per_item, done, clock,
                               ["restore_mode", "draw", "n_coords"])
        finally:
            from load_osm import unload_model
            unload_model(mname)
    write_report(out, args.account, args.models, args.phase, extras)


if __name__ == "__main__":
    main()
