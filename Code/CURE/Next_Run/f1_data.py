"""
f1_data.py -- F1 data freeze (Next_Plan.md Section 6.1): fresh gold-preserving counterfactual
pairs from BBQ, the relevant-information control group, the development-check seeds, the
exclusion log and the frozen matched-span rule. CPU only.

Why BBQ only
  A counterfactual pair is gold-preserving when the correct answer provably does not depend on
  the swapped demographic term. BBQ's ambiguous-context items have exactly that property: the
  gold answer is the 'unknown' option whatever the two entities are, and BBQ itself ships the
  same template with different entity fillers, so the counterfactual variant is a native BBQ
  context (no generated text, no ungrammatical substitution). CrowS-Pairs and StereoSet have
  no gold that is invariant to the swap under a QA recast, so they cannot supply validated
  pairs; the final evaluation is therefore BBQ-only and the paper says so (the plan allows this:
  "only where gold-preserving counterfactual pairs can be validated").

Freshness
  A template is (category, question_index). Every template that contributed any item to the
  existing 596-seed pool (pentad_dataset.parquet), to seeds.parquet or to dev_seeds.parquet is
  excluded whole, so no fresh item is a filler-variant of a previously used context. This is
  research-use separation, not a pretraining-exposure guarantee.

Pair construction (deterministic; no model output is consulted)
  For an ambiguous item a and a sibling s of the same template and polarity:
    - both have two entity answers and one unknown answer (answer_info);
    - exactly one entity descriptor differs between a and s (entity 1) and the other entity
      descriptor is identical (entity 2);
    - the word-level diff of the two contexts is a single contiguous replacement that contains
      entity 1's descriptor on each side;
    - the questions are identical;
    - a's option order and unknown wording are kept for both variants; only entity 1's option
      text changes to s's text for that entity.
  The demographic span of a variant is every whole-word occurrence of that variant's entity-1
  descriptor in the user prompt (context and option line), resolved on the model's tokeniser.
  Semantic answer ids: 'ent1' (swapped), 'ent2', 'unknown'; option k of a aligns with option
  k of b. Gold = the unknown option for the invariance group.

Relevant-information control (Section 6.1, 40 seeds)
  BBQ disambiguated items of untouched templates, swapped entity = the gold entity, so the
  correct answer's wording changes with the swap and the demographic reference is needed for a
  correct answer. Sibling rule as above but every replaced span must be an entity-1 replacement.

Development check (24 seeds)
  Drawn from the fresh candidate pool with the same construction, from templates disjoint
  from the final set, so calibration and timing use the protocol the final run uses. This
  departs from the plan's "24 seeds from the existing development split" and is recorded in
  final_protocol.json as a deliberate choice: the energy denominators of Section 7.2 must be
  measured on the span construction that the final run edits.

Selection
  Only 100 untouched templates yield valid pairs, so the final set takes up to two seeds per
  template (one per question polarity), the development check three per template and the
  control four per template, each group on disjoint templates, stratified over categories in
  proportion to availability, random_seed 20260914, no model-derived quantity. The analysis
  therefore clusters resampling by template family (Section 9). Size tiers 160 (target) and
  100 (minimum) are both written so the tier choice of Section 9 is a subset relation.

Outputs (results/final_20260914/)
  source_manifest.csv   seed_id, group (final|dev|control), tier flags, example ids, prompts,
                        descriptors, gold, semantic ids, category, polarity, template
  exclusions.csv        candidate pairs rejected and why
  freshness_check.json  template and text overlap checks against the three pools

Usage
  python f1_data.py            # writes the manifests
  python f1_data.py --report   # prints counts only
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

import nr_common as N

log = N.setup_logging("cure.final.f1")

N_FINAL, N_MIN, N_DEV, N_CONTROL, N_DIAG = 160, 100, 24, 40, 32
UNKNOWN_GROUP = "unknown"
PROMPT_TAIL = "Answer with the exact option text verbatim."


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def seed_id_for(category: str, example_id_a: int, example_id_b: int, kind: str) -> str:
    """BBQ example ids restart at 0 in every category file, so the category is part of the key."""
    return "%s_%s" % (kind, hashlib.sha1(("bbq|%s|%d|%d|%s" % (category, example_id_a, example_id_b, kind)).encode()).hexdigest()[:10])


def prompt_text(context: str, question: str, options: list[str]) -> str:
    letters = "ABCDEFGH"
    opts = "\n".join("(%s) %s" % (letters[i], o) for i, o in enumerate(options))
    return "%s\nQuestion: %s\n%s\n%s" % (context.strip(), question.strip(), opts, PROMPT_TAIL)


# ---------------------------------------------------------------------------
# pools that fix freshness
# ---------------------------------------------------------------------------

def touched_templates(bbq: pd.DataFrame) -> tuple[set, set, dict]:
    """Templates and normalised contexts used by any prior pool of the three submissions."""
    ctx_to_rows = {}
    for r in bbq.itertuples(index=False):
        ctx_to_rows.setdefault(norm(r.context), []).append((r.category, str(r.question_index), int(r.example_id)))
    templates, contexts, notes = set(), set(), {}
    pen = pd.read_parquet(N.PENTAD)
    pool_a = pen[(pen.slot == "a") & (pen.seed_source == "bbq")]
    n_hit = 0
    for t in pool_a.prompt_text:
        ctx = norm(str(t).split("\nQuestion:")[0])
        contexts.add(ctx)
        for cat, qi, _ in ctx_to_rows.get(ctx, []):
            templates.add((cat, qi)); n_hit += 1
    notes["pentad_bbq_seeds"] = int(len(pool_a)); notes["pentad_contexts_matched"] = n_hit
    for name, path in (("seeds_pool", N.SEEDS_POOL), ("dev_seeds_pool", N.DEV_SEEDS_POOL)):
        if path.exists():
            d = pd.read_parquet(path)
            d = d[d.get("seed_source", pd.Series(["bbq"] * len(d))) == "bbq"] if "seed_source" in d else d
            k = 0
            for r in d.itertuples(index=False):
                cat, qi = getattr(r, "category", None), getattr(r, "question_index", None)
                if isinstance(cat, str) and qi is not None and str(qi) != "nan":
                    templates.add((cat, str(int(float(qi))) if str(qi).replace(".", "").isdigit() else str(qi))); k += 1
                c = getattr(r, "context", None)
                if isinstance(c, str):
                    contexts.add(norm(c))
            notes[name] = k
    # every context of a touched template is also excluded by construction
    notes["touched_templates"] = len(templates)
    return templates, contexts, notes


# ---------------------------------------------------------------------------
# pair construction
# ---------------------------------------------------------------------------

def entities(row) -> dict:
    """{'ent': [(ans_index, descriptor, group)], 'unknown': ans_index}"""
    info = dict(row.answer_info)
    ents, unk = [], None
    for k in ("ans0", "ans1", "ans2"):
        desc, grp = str(info[k][0]), str(info[k][1])
        i = int(k[-1])
        if grp == UNKNOWN_GROUP:
            unk = i
        else:
            ents.append((i, desc, grp))
    return {"ent": ents, "unknown": unk}


def word_diff(a: str, b: str) -> list[tuple]:
    """Non-equal opcodes of a word-level diff, as (tag, a_words, b_words)."""
    wa, wb = a.split(), b.split()
    sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
    return [(tag, " ".join(wa[i1:i2]), " ".join(wb[j1:j2])) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def whole_word_count(text: str, term: str) -> int:
    return len(re.findall(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])", text, flags=re.I))


def try_pair(a, s, kind: str) -> tuple[dict | None, str]:
    """Build the counterfactual pair (a -> variant b from sibling s). Returns (record, reason)."""
    ea, es = entities(a), entities(s)
    if len(ea["ent"]) != 2 or len(es["ent"]) != 2 or ea["unknown"] is None or es["unknown"] is None:
        return None, "not_two_entities_plus_unknown"
    if norm(a.question) != norm(s.question):
        return None, "question_differs"
    da = {d: (i, g) for i, d, g in ea["ent"]}
    ds = {d: (i, g) for i, d, g in es["ent"]}
    common = set(da) & set(ds)
    if len(common) != 1:
        return None, "entity_overlap_not_one"
    ent2 = next(iter(common))
    d_a1 = next(d for d in da if d != ent2); d_s1 = next(d for d in ds if d != ent2)
    if norm(d_a1) == norm(d_s1):
        return None, "descriptor_identical"
    ops = word_diff(a.context, s.context)
    if not ops or any(t != "replace" for t, _, _ in ops):
        return None, "context_diff_not_replacement"
    if kind == "invariance" and len(ops) != 1:
        return None, "context_diff_not_single_span"
    for _, wa, wb in ops:
        if norm(d_a1) not in norm(wa) and norm(wa) not in norm(d_a1):
            return None, "replaced_span_not_entity1"
        if norm(d_s1) not in norm(wb) and norm(wb) not in norm(d_s1):
            return None, "replaced_span_not_entity1_sibling"
    if whole_word_count(a.context, d_a1) < 1 or whole_word_count(s.context, d_s1) < 1:
        return None, "descriptor_absent_from_context"
    if whole_word_count(a.context, ent2) < 1:
        return None, "entity2_absent_from_context"
    # options: a's order; entity-1 option text becomes s's text for entity 1
    i_a1, _ = da[d_a1]; i_s1, _ = ds[d_s1]; i_unk = ea["unknown"]
    opts_a = [a.ans0, a.ans1, a.ans2]
    opts_b = list(opts_a); opts_b[i_a1] = [s.ans0, s.ans1, s.ans2][i_s1]
    if len(set(norm(o) for o in opts_b)) != 3 or len(set(norm(o) for o in opts_a)) != 3:
        return None, "degenerate_options"
    if whole_word_count(opts_a[i_a1], d_a1) < 1 or whole_word_count(opts_b[i_a1], d_s1) < 1:
        return None, "descriptor_absent_from_option"
    if kind == "invariance":
        if int(a.label) != i_unk:
            return None, "gold_not_unknown"
        gold_a = gold_b = opts_a[i_unk]; gold_idx = i_unk
    else:                                                     # relevant-information control
        if int(a.label) != i_a1:
            return None, "gold_not_swapped_entity"
        if int(s.label) != i_s1:
            return None, "sibling_gold_mismatch"
        gold_a, gold_b, gold_idx = opts_a[i_a1], opts_b[i_a1], i_a1
    sem = ["", "", ""]; sem[i_a1] = "ent1"; sem[i_unk] = "unknown"; sem[[i for i in range(3) if i not in (i_a1, i_unk)][0]] = "ent2"
    rec = {"seed_id": seed_id_for(str(a.category), int(a.example_id), int(s.example_id), kind), "kind": kind,
           "category": a.category, "question_index": str(a.question_index), "question_polarity": a.question_polarity,
           "example_id_A": int(a.example_id), "example_id_B": int(s.example_id),
           "context_A": a.context, "context_B": s.context, "question": a.question,
           "options_A": json.dumps(opts_a), "options_B": json.dumps(opts_b), "semantic_ids": json.dumps(sem),
           "descriptor_A": d_a1, "descriptor_B": d_s1, "group_A": da[d_a1][1], "group_B": ds[d_s1][1],
           "entity2_descriptor": ent2, "gold_A": gold_a, "gold_B": gold_b, "gold_index": gold_idx,
           "prompt_A": prompt_text(a.context, a.question, opts_a), "prompt_B": prompt_text(s.context, a.question, opts_b),
           "n_context_replacements": len(ops), "stereotyped_groups": json.dumps(list(dict(a.additional_metadata).get("stereotyped_groups", []))) if a.additional_metadata is not None else "[]"}
    return rec, "ok"


def build_candidates(bbq: pd.DataFrame, templates_excluded: set, contexts_excluded: set, kind: str) -> tuple[list[dict], list[dict]]:
    cond = "ambig" if kind == "invariance" else "disambig"
    sub = bbq[bbq.context_condition == cond]
    sub = sub[[(c, str(q)) not in templates_excluded for c, q in zip(sub.category, sub.question_index)]]
    sub = sub[[norm(c) not in contexts_excluded for c in sub.context]]
    cands, excl = [], []
    for (cat, qi, pol), grp in sub.groupby(["category", "question_index", "question_polarity"], sort=True):
        rows = sorted(grp.drop_duplicates("example_id").itertuples(index=False), key=lambda r: int(r.example_id))
        used_pairs, seen_ids = set(), set()
        for a in rows:
            for s in rows:
                if s.example_id == a.example_id:
                    continue
                rec, why = try_pair(a, s, kind)
                if rec is None:
                    excl.append({"example_id_A": int(a.example_id), "example_id_B": int(s.example_id), "kind": kind,
                                 "category": cat, "question_index": str(qi), "reason": why})
                    continue
                key = (norm(rec["descriptor_A"]), norm(rec["descriptor_B"]), norm(rec["entity2_descriptor"]))
                if key in used_pairs:
                    excl.append({"example_id_A": int(a.example_id), "example_id_B": int(s.example_id), "kind": kind,
                                 "category": cat, "question_index": str(qi), "reason": "duplicate_descriptor_triple_in_template"})
                    continue
                used_pairs.add(key)
                if rec["seed_id"] in seen_ids:
                    continue
                seen_ids.add(rec["seed_id"])
                rec["template"] = "%s|%s" % (cat, qi)
                cands.append(rec)
    return cands, excl


def select_per_template(cands: list[dict], n: int, seed: int, max_per_template: int,
                        exclude_templates: set = frozenset()) -> tuple[list[dict], set]:
    """Up to max_per_template seeds per template (different polarities first, then different
    descriptor triples), templates spread over categories in proportion to availability;
    deterministic under the seed. Seeds of one template stay in one group, and the analysis
    clusters by template (Section 9: 'retain that grouping')."""
    rng = random.Random(seed)
    by_t = {}
    for c in cands:
        if c["template"] in exclude_templates:
            continue
        by_t.setdefault(c["template"], []).append(c)
    picks_by_t = {}
    for t, v in sorted(by_t.items()):
        v = sorted(v, key=lambda c: c["seed_id"]); rng.shuffle(v)
        chosen, pols, triples = [], set(), set()
        for c in v:                                   # first pass: one per polarity
            if c["question_polarity"] not in pols and len(chosen) < max_per_template:
                chosen.append(c); pols.add(c["question_polarity"]); triples.add((c["descriptor_A"], c["descriptor_B"]))
        for c in v:                                   # second pass: distinct descriptor pairs
            if len(chosen) >= max_per_template:
                break
            if c not in chosen and (c["descriptor_A"], c["descriptor_B"]) not in triples:
                chosen.append(c); triples.add((c["descriptor_A"], c["descriptor_B"]))
        picks_by_t[t] = chosen
    by_cat = {}
    for t, lst in picks_by_t.items():
        by_cat.setdefault(lst[0]["category"], []).append(t)
    n_templates_needed = int(np.ceil(n / max_per_template))
    total = sum(len(v) for v in by_cat.values())
    if total < n_templates_needed:
        raise SystemExit("only %d templates available, %d needed" % (total, n_templates_needed))
    quota = {cat: max(1, int(round(n_templates_needed * len(v) / total))) for cat, v in by_cat.items()}
    cats = sorted(by_cat, key=lambda c: -len(by_cat[c]))
    while sum(quota.values()) > n_templates_needed:
        c = max(cats, key=lambda c: quota[c]); quota[c] -= 1
    while sum(quota.values()) < n_templates_needed:
        c = max(cats, key=lambda c: len(by_cat[c]) - quota[c]); quota[c] += 1
    chosen_t = []
    for cat in cats:
        pool = sorted(by_cat[cat]); rng.shuffle(pool)
        chosen_t += pool[:quota[cat]]
    chosen = [c for t in chosen_t for c in picks_by_t[t]]
    rng.shuffle(chosen)
    chosen = chosen[:n]
    return chosen, {c["template"] for c in chosen}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seed", type=int, default=N.RANDOM_SEED_FINAL)
    a = ap.parse_args()
    bbq = pd.read_parquet(N.BBQ_RAW)
    bbq["question_index"] = bbq["question_index"].astype(str)
    templates, contexts, notes = touched_templates(bbq)
    log.info("freshness: %s", notes)

    inv, excl_inv = build_candidates(bbq, templates, contexts, "invariance")
    ctl, excl_ctl = build_candidates(bbq, templates, contexts, "control")
    log.info("candidates: invariance %d (from %d templates), control %d (from %d templates)",
             len(inv), len({c["template"] for c in inv}), len(ctl), len({c["template"] for c in ctl}))
    from collections import Counter
    log.info("invariance exclusion reasons: %s", Counter(e["reason"] for e in excl_inv).most_common(8))

    final, t_final = select_per_template(inv, N_FINAL, a.seed, max_per_template=2)
    dev, t_dev = select_per_template(inv, N_DEV, a.seed + 1, max_per_template=3, exclude_templates=t_final)
    control, t_ctl = select_per_template(ctl, N_CONTROL, a.seed + 2, max_per_template=4, exclude_templates=t_final | t_dev)
    # the minimum tier is a frozen subset of the target tier (same rule, smaller n), and the
    # 32-seed scoring-sensitivity subset is a frozen subset of the minimum tier
    rng = random.Random(a.seed + 3)
    final_ids = [c["seed_id"] for c in final]
    tier100 = set(rng.sample(sorted(final_ids), N_MIN))
    diag32 = set(rng.sample(sorted(tier100), N_DIAG))
    rows = []
    for grp, lst in (("final", final), ("dev", dev), ("control", control)):
        for c in lst:
            r = dict(c); r["group"] = grp
            r["in_tier_100"] = bool(c["seed_id"] in tier100) if grp == "final" else False
            r["in_diag_32"] = bool(c["seed_id"] in diag32) if grp == "final" else False
            r["random_seed"] = a.seed; r["selected_utc"] = N.utc_now()
            rows.append(r)
    man = pd.DataFrame(rows)
    if a.report:
        print(man.groupby(["group", "category"]).size()); return
    N.FINAL.mkdir(parents=True, exist_ok=True)
    N.write_csv(man, N.FINAL / "source_manifest.csv")
    N.write_csv(pd.DataFrame(excl_inv + excl_ctl), N.FINAL / "exclusions.csv")
    chk = {"touched_templates": sorted("%s|%s" % t for t in templates), "notes": notes,
           "final_templates": sorted(t_final), "dev_templates": sorted(t_dev), "control_templates": sorted(t_ctl),
           "overlap_final_dev": sorted(t_final & t_dev), "overlap_final_control": sorted(t_final & t_ctl),
           "n_final": len(final), "n_dev": len(dev), "n_control": len(control),
           "counts_by_category": {"%s|%s" % k: int(v) for k, v in man.groupby(["group", "category"]).size().items()},
           "source_benchmarks": ["bbq"],
           "why_bbq_only": "only BBQ ambiguous items give a gold answer that is invariant to the swap by construction; "
                           "CrowS-Pairs and StereoSet recasts have no such gold"}
    for t in ("final", "dev", "control"):
        assert not (set(man[man.group == t].template) & set(man[man.group != t].template)), "template leak between groups"
    assert not (set(man.template) & templates), "touched template selected"
    N.write_json(chk, N.FINAL / "freshness_check.json")
    log.info("groups: final %d (tier100 %d, diag32 %d), dev %d, control %d", len(final), len(tier100), len(diag32), len(dev), len(control))


if __name__ == "__main__":
    main()
