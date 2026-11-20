"""
x0_extension_manifest.py -- the item set and the declared protocol of the extension run (after the ICLR
2027 discussion; aimed at the next submission). CPU, author's machine, deterministic.

Why this run exists. On the 160 held-out BBQ pairs, the reduction in summed-score sensitivity T is
supported on Llama-3.1-8B and Phi-4-mini only, the length-normalised sensitivity T_mean shows no
supported fall, and the generation flip rate shows no supported change. A noise split of those rows
(between-template and within-template variance) shows that most of the noise is within templates,
so more pairs per template raise precision even though only 82 unused templates exist. This run adds
those pairs and tests the three readouts again. It can return supported effects, tighter nulls or
supported reversals; every outcome is reported.

Items. Ambiguous BBQ sibling pairs built by Next_Run/f1_data.py's frozen rule (same construction,
same exclusion of templates touched by any earlier pool), restricted to pairs never used in this
study (no final, development or control seed), from templates not used by the development or control
sets. Up to 12 pairs per template: one per question polarity first, then distinct descriptor pairs,
shuffled under random_seed 20261120. No model output enters the selection.

Output: results/extension_20261120/x_manifest.csv and x_protocol.json (with hash).
Usage: python x0_extension_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Next_Run"))
import nr_common as N   # noqa: E402
import f1_data as F     # noqa: E402

SEED = 20261120
K_MAX = 12
OUT = N.FINAL.parent / "extension_20261120"


def select(cands: list[dict], used: set, banned_templates: set) -> list[dict]:
    rng = random.Random(SEED)
    by_t = {}
    for c in cands:
        if c["seed_id"] in used or c["template"] in banned_templates:
            continue
        by_t.setdefault(c["template"], []).append(c)
    chosen = []
    for t, v in sorted(by_t.items()):
        v = sorted(v, key=lambda c: c["seed_id"]); rng.shuffle(v)
        pick, pols, pairs = [], set(), set()
        for c in v:
            if c["question_polarity"] not in pols and len(pick) < K_MAX:
                pick.append(c); pols.add(c["question_polarity"]); pairs.add((c["descriptor_A"], c["descriptor_B"]))
        for c in v:
            if len(pick) >= K_MAX:
                break
            key = (c["descriptor_A"], c["descriptor_B"])
            if c not in pick and key not in pairs:
                pick.append(c); pairs.add(key)
        chosen += pick
    return chosen


def protocol(man: pd.DataFrame, old_templates: set) -> dict:
    p = {
        "cycle": "extension_20261120",
        "purpose": "test again the three answer readouts that the 160-pair held-out set left unsupported or underpowered",
        "items": {"n_pairs": int(len(man)), "n_templates": int(man.template.nunique()),
                  "templates_shared_with_the_inspected_held_out_set": int(len(set(man.template) & old_templates)),
                  "per_template_max": K_MAX, "selection_seed": SEED,
                  "status": "new pairs, never scored; most templates were inspected through other pairs, so the items are new but the templates are not"},
        "models": ["llama-3.1-8b-instruct", "gemma-2-2b-it", "qwen2.5-7b-instruct", "phi-4-mini-instruct"],
        "conditions": {"B": "no edit", "P_span@1": "matched-position erasure, rank-one frozen basis, alpha 1",
                       "CV_span@1": "contrast vector at the term, energy matched per side to P_span@1 within 5 percent, no alpha cap"},
        "kernels": "bfloat16, flash attention 2.8.3, torch 2.5.1, transformers 5.17.0 (as the rebuttal cycle)",
        "readouts": {"T": "total variation between the two versions' option distributions, summed option log-likelihood",
                     "T_mean": "the same with mean log-likelihood per token",
                     "flip": "1 when both generated answers are valid and select different options, 0 when valid and equal; pairs with an invalid side are dropped"},
        "primary_family": {"tests": ["pooled T", "pooled T_mean", "pooled flip"],
                           "statistic": "per template, the mean over the four models of the per-template mean of B minus P_span@1; one-sided (greater), studentised bootstrap over templates, 10,000 resamples",
                           "correction": "Holm over the three"},
        "secondary_families": {
            "per_model_erasure": "per readout, B minus P_span@1 for each model, one-sided (greater), Holm over the four models",
            "per_model_vector": "per readout, B minus CV_span@1 for each model, one-sided (greater), Holm over the four models",
            "erasure_vs_vector": "per readout, P_span@1 minus CV_span@1 for each model, two-sided, descriptive"},
        "combined_with_inspected_set": "reported separately as exploratory, never as the confirmatory result",
        "outcome_rules": "supported when the test rejects in the stated direction after Holm; a rejection in the opposite direction is reported as a supported reversal; otherwise limited. Every number is reported whichever way it falls.",
        "expected_power_note": ("From the 160-pair noise split at about 11 pairs per template over 82 templates: the T fall is well powered on "
                                "Llama-3.1-8B and Phi-4-mini, weakly on Gemma-2-2B, not on Qwen2.5-7B; T_mean is expected to show reversals on "
                                "Llama-3.1-8B and Gemma-2-2B; the flip test is adequately powered only on Qwen2.5-7B. The pooled tests are the primary family for that reason."),
        "smoke": "two new pairs plus two inspected pairs per model; the inspected pairs must reproduce their stored B and P_span@1 option scores",
        "written_utc": N.utc_now(),
    }
    core = {k: v for k, v in p.items() if k != "written_utc"}
    p["protocol_hash"] = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]
    return p


def main() -> None:
    bbq = pd.read_parquet(N.BBQ_RAW)
    bbq["question_index"] = bbq["question_index"].astype(str)
    templates, contexts, _ = F.touched_templates(bbq)
    inv, _ = F.build_candidates(bbq, templates, contexts, "invariance")
    old = pd.read_csv(N.FINAL / "source_manifest.csv")
    banned = set(old[old.group.isin(["dev", "control"])].template)
    rows = select(inv, set(old.seed_id), banned)
    man = pd.DataFrame(rows)
    man["group"] = "ext"
    man["in_tier_100"] = False
    man["in_diag_32"] = False
    man["random_seed"] = SEED
    assert not (set(man.seed_id) & set(old.seed_id)), "a used seed was selected"
    assert not (set(man.template) & banned), "a development or control template was selected"
    # two inspected pairs per run go into the smoke test only, to check reproduction
    repro = old[old.group == "final"].sort_values("seed_id").head(2).copy()
    repro["group"] = "repro"
    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(old.columns)
    full = pd.concat([man.reindex(columns=cols), repro[cols]], ignore_index=True)
    full.to_csv(OUT / "x_manifest.csv", index=False)
    p = protocol(man, set(old[old.group == "final"].template))
    (OUT / "x_protocol.json").write_text(json.dumps(p, indent=1), encoding="utf-8")
    print("pairs %d, templates %d (shared with the inspected set %d), per-template median %.0f, min %d; protocol %s"
          % (len(man), man.template.nunique(), p["items"]["templates_shared_with_the_inspected_held_out_set"],
             man.groupby("template").size().median(), man.groupby("template").size().min(), p["protocol_hash"]))


if __name__ == "__main__":
    main()
