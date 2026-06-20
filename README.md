# CURE: Causal Audit and Repair of Social Bias in Language Models

CURE is a two-stage instrument for language-model fairness. It first **audits** a bias
benchmark to find the items whose answer routes causally through a protected attribute,
even when the surface answer looks fair. It then **repairs** that routing by erasing the
audited direction, and re-audits to confirm. The audit score doubles as the repair
target, so one causal signal does both jobs.

The audit half lives in `Code/audit`. The repair half, the comparison against recent
debiasing methods, the prognosis, and the held-out and behavioural tests live in
`Code/CURE`. A reader can reproduce the whole pipeline from this file alone.

---

## 0. Honest headline (read this first)

The contribution is the **causal audit and its prognosis**, not a debiasing method.

1. **Audit.** A benchmark score hides a large validity gap. The causal commutator finds
   items that look fair but compute unfairly, which no behavioural audit can detect.
2. **Prognosis.** The audit score predicts how hard a model is to repair. Per pair the
   audit score predicts the erasure rank needed (Spearman 0.58 to 0.63 over 508 pairs per
   model); per model a severer audit tracks a costlier repair.
3. **Repair is conditional, not general.** Erasing the audited subspace removes the
   causal signal, and the removal **generalises to held-out seeds** (CURE removes the most
   causal swap effect out of sample on all four models). But an **independent behavioural
   test** shows the same erasure **lowers task accuracy and raises answer inconsistency**
   on every model. The audited direction is **load-bearing**: it overlaps the
   massive-activation structure the model computes with. So linear erasure is a
   conditional repair, and the audit says in advance when it is safe.

The mechanism is massive-activation entanglement. On Qwen2.5-7B the leading bias
direction peaks on hidden dimension 458, a massive-activation dimension, where the
mid-layer singular spectrum shows a near 70-fold gap (32,895 against 466).

---

## 1. What the two folders do

| Folder | Role | Summary |
|--------|------|---------|
| `Code/audit` | Diagnosis | The causal discriminative-validity audit. A behavioural probe over the five-slot "pentad" plus a causal intervention (CDVA) that patches the protected-attribute residual at every layer and reads the change in the answer logit. Produces the validity leaderboard, the commutator results, and the residual of items that pass behaviourally yet fail causally. |
| `Code/CURE`  | Repair + study | Estimates the bias subspace from the audit's counterfactual activations, erases it at inference, re-audits, measures the utility cost, compares against eight recent debiasing methods on an independent signal, fits the audit-to-repair prognosis, and runs the held-out and behavioural rebuttal tests. |

The repair runs on the same four open models, the same three datasets, and the same
causal stack as the audit, so every comparison is fair.

**Models** (all instruction-tuned): Llama-3.1-8B and Gemma-2-2B via TransformerLens;
Qwen2.5-7B and Phi-4-mini via NNsight. **Datasets**: BBQ, CrowS-Pairs, StereoSet, folded
into a 596-seed "pentad" over ten demographic axes.

---

## 2. The audit: procedure (`Code/audit`)

### 2.1 The pentad dataset
Each seed is a template with a demographic slot, expanded into five slots (a to e) and
several sub-variants (`Dataset/seeds/pentad_dataset.parquet`). Slot `a` carries the clean
(disambiguated) prompt; slot `c` carries the demographic swaps used by the commutator;
other slots add surface and order controls. Gold answers are attached per prompt.

### 2.2 Behavioural evaluation
`GPU_CPU/osm_behavioral.py::evaluate_osm_model` runs each model on the prompts with
greedy decoding, then parses the option answer. Parsing is deterministic first, with a
judge model as a fallback only when the deterministic parse fails (Section 7). This gives
the native pass rate and the behavioural fairness signals.

### 2.3 The causal commutator (CDVA)
`GPU_CPU/cdva_patching.py` is the heart of the audit. For a counterfactual pair, the
demographic token is swapped (`a -> b`). Activation patching writes the residual stream at
the swapped-token position, taken from the run on `a`, into the run on `b` **at every
decoder layer**, then reads the change in the gold-option logit:

```
C(a, b) = logit_gold( swap(a -> b) ) - logit_gold( a )      # the commutator
```

A model that satisfies causal swap invariance returns `C ~ 0` for swaps that should not
change the answer. A large `|C|` means the prediction moves with the protected attribute.
The threshold `tau = 0.7644` is the 75th percentile of `|C|` over all audited pairs.

### 2.4 Aggregates
- **Severity** = mean `|C|` over a model's pairs.
- **Commutativity index** = fraction of seeds whose every pair stays below `tau`.
- **Validity gap** = native pass rate minus the causal-audit-robust pass rate; it measures
  how much a benchmark overstates a model's fairness.

### 2.5 Run the audit
```bash
cd Code/audit
python3 GPU_CPU/run_gpu_pipeline.py     # behavioural eval + CDVA patching (GPU)
python3 run_cpu_full.py                 # scoring, leaderboard, statistics (CPU)
```
Outputs land in `Code/audit/results/` (`cdva_results.parquet`, `behavioral_results.parquet`,
`leaderboard.parquet`, `validity_gap_leaderboard.parquet`, `scored_results.parquet`).

---

## 3. The repair: procedure (`Code/CURE`)

The repair reuses the audit's counterfactual activations. All steps are inference only;
there is no fine-tuning.

| ID | Step | File | Output |
|----|------|------|--------|
| E1 | Estimate the bias subspace: collect the per-layer activation difference at the swapped position, SVD, take the top-`r` directions. The SVD is computed once and sliced to every rank. | `erase.py` (`bases_at_ranks`) | per-(layer, rank) basis |
| E2 | Erase: project the residual at the swapped position onto the complement of the rank-`r` subspace, **at every decoder layer** (LEACE-style concept scrubbing). | `erase.py` (`ErasureContext`, `erased_commutator`) | inference hook |
| E3 | Re-audit: recompute the commutator under erasure. | `experiments.py` (`e3_reaudit`) | `cure_recovery_*.parquet` |
| E4 | Utility: four-option accuracy under erasure, the drop from the unedited model. | `experiments.py` (`native_accuracy`) | `cure_utility_*.parquet` |
| E5 | Compare against eight debiasing baselines. | `baselines.py` | `cure_final_*.parquet` |
| E6 | Prognosis: regress the per-pair audit score on the erasure rank needed to repair it. | `experiments.py` (`e6_prognosis`) | `cure_prognosis_*.{parquet,json}` |

### 3.1 Utility-aware operating rank
`run_cure.py::_utility_aware_rank` picks, per model, the rank that removes the most bias
among ranks whose accuracy drop stays at or below `MAX_UTILITY_COST` (0.15). If no rank
meets the budget, because the bias is entangled with directions the model needs, the
least-damaging rank is used. One rank per model fixes every reported number
(`cure_rankcurve_*.json`). The operating ranks are Llama 8, Qwen 4, Gemma 1, Phi 1.

### 3.2 Fair baselines on an independent signal
The eight baselines (`baselines.py`) derive their bias direction from an **independent**
set of 310 demographic-contrast templates, not from the audit pairs. They are faithful
re-implementations on one shared erasure-or-steering protocol: prompt self-debias,
generic erase, mean-difference steering, FairSteer, BiasGym, SAE-Debias, H-SAL, and
logit-space steering. **CURE alone reads the causal audit signal**, so any advantage
isolates the value of that signal, not the implementation.

### 3.3 Run the repair
```bash
cd Code/CURE
python3 run_cure.py --mode dry        # validate the whole environment on two pairs
python3 run_cure.py --mode main       # E1-E6 for all four models, 15-min GitHub checkpoints
python3 run_cure.py --mode baselines  # CURE vs eight baselines under one harness
python3 run_cure.py --mode diagnose   # massive-activation diagnostic (anomaly_diagnostic.json)
```

---

## 4. Held-out and behavioural rebuttal (`run_tacl_extra.py`)

This run answers two reviewer objections without re-fitting the headline run.

- **Held-out (out-of-sample).** The CURE subspace is fitted on a **train** split of seeds
  and the causal residual removed is scored on a **disjoint test** split. CURE removes the
  most causal swap effect out of sample on all four models (Llama 0.55, Qwen 0.55, Gemma
  0.48, Phi 0.43), so the removal is not an in-sample artefact.
- **Independent behavioural readout.** On the same held-out seeds the model is generated
  under erasure and scored by two output-level metrics that do **not** use the commutator:
  disambiguated accuracy, and the answer-flip rate (the fraction of seeds whose generated
  answer changes across demographic sub-variants). CURE lowers accuracy and raises the
  flip rate on every model; the gentle baselines preserve both. Removing the audited
  causal direction therefore does not produce behavioural fairness.

```bash
cd Code/CURE
python3 run_tacl_extra.py --mode dry    # two seeds per split, every model
python3 run_tacl_extra.py --mode main   # full held-out + behavioural run, 15-min checkpoints
```
Outputs: `results/tacl_extra_<model>.parquet` (per method: held-out bias removed,
behavioural accuracy, behavioural flip rate).

---

## 5. Environment (exact)

The OS must match the precompiled flash-attention wheel.

- OS: Ubuntu 24.04 LTS, x86_64 (CUDA image `nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04`).
- Python 3.12, Torch `2.5.1` (cu124), CUDA 12.x driver, a single 24 GB or larger GPU
  (48 GB removes all memory headroom worries for the 8B models).
- No virtual environment; install globally with `--break-system-packages`.

```bash
pip3 install --break-system-packages torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install --break-system-packages -r Code/CURE/requirements_cure.txt --extra-index-url https://download.pytorch.org/whl/cu124
pip3 install --break-system-packages --no-deps transformer_lens==2.18.0
# precompiled flash-attention (do not build from source)
pip3 install --break-system-packages --no-deps \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

Cloud GPU bootstrap scripts are provided: `Code/CURE/bootstrap.sh` (full audit+repair)
and `Code/CURE/vast_tacl_bootstrap.sh` (held-out+behavioural). Both pin the environment
above, download the four models, run the dry check, then the main run, with 15-minute
GitHub checkpoints. Flash-attention is verified by a real forward pass; the dry run fails
loud if it is missing.

---

## 6. Secrets and the `.env` contract

Every key is read from the environment. No secret is ever written into a tracked file.
Copy `Code/CURE/.env.example` to `Code/CURE/.env` and fill the values. The `.env` is
git-ignored and never pushed.

| Variable | Purpose |
|----------|---------|
| `HUGGINGFACE_TOKEN` | Model download. |
| `Github_Classic_Token` | Checkpoint pushes. |
| `RANDOM_SEED` | Reproducibility (default 20260101). |
| `GEMINI_API_KEY_1..4` | Primary judge, gemini-2.5-flash. |
| `DEEPSEEK_API_KEY_1..2`, `MISTRAL_API_KEY1..2`, `OPENROUTER_API_KEY_1..2` | Fallback judge tiers. |
| `AWS_ACCESS_KEY`, `AWS_SECRET_KEY` | Required by the audit config; unused by CURE. |

---

## 7. Judge and answer-extraction design

The judge extracts a structured answer only when the deterministic JSON parse fails. One
active tier is chosen by `CURE_JUDGE_PROVIDER` (default `gemini`); keys are round-robined
within the tier. There is no automatic fallback between tiers, so every judgement in a run
comes from one model, which protects reproducibility.

---

## 8. Results (where each number comes from)

- **Audit validity gaps** (up to 0.51 on BBQ for Qwen2.5-7B): `audit/results/validity_gap_leaderboard.parquet`.
- **Severity, commutativity, per-axis `|C|`**: `audit/results/cdva_results.parquet`.
- **Headline repair (in-distribution)**: `CURE/results/cure_final_<model>.parquet`.
- **Operating rank per model**: `CURE/results/cure_rankcurve_<model>.json`.
- **Prognosis** (Spearman 0.58-0.63): `CURE/results/cure_prognosis_<model>.{parquet,json}`.
- **Held-out + behavioural**: `CURE/results/tacl_extra_<model>.parquet`.
- **Massive-activation mechanism**: `CURE/results/anomaly_diagnostic.json`.

---

## 9. Integrity and resume

Every run starts with `integrity.py`, which quarantines corrupt parquets (recomputed
next run) and removes duplicate primary keys. The checkpoint pusher (`checkpoint.py`)
force-adds `results/` and `logs/` every 15 minutes and after each unit, but never the
dry-run test results or the quarantine. A released VM resumes from the last pushed,
de-duplicated results.

---

## 10. Repository map

```
Code/
  audit/                  the causal discriminative-validity audit (diagnosis)
    Dataset/seeds/        the pentad dataset
    GPU_CPU/              behavioural evaluation (osm_behavioral) and CDVA patching (cdva_patching)
    CPU_Only/             scoring, statistics, leaderboard
    results/              cdva_results, behavioral_results, leaderboards
  CURE/                   the repair, comparison, prognosis, and rebuttal (this work)
    erase.py              E1 subspace extraction, E2 erasure hooks (all layers)
    experiments.py        E3 re-audit, E4 utility, E6 prognosis, demographic signal
    baselines.py          E5 eight baselines on the independent signal
    run_cure.py           entry point (dry, main, baselines, diagnose)
    run_tacl_extra.py     held-out + behavioural rebuttal (dry, main)
    config_cure.py        loads .env, reuses the audit models and dataset
    judge_api.py          judge and answer extraction (round-robin, no cross-tier fallback)
    integrity.py          duplicate and corruption checks
    checkpoint.py         resume-safe 15-minute GitHub pushes
    bootstrap.sh          GPU VM entrypoint (full audit+repair)
    vast_tacl_bootstrap.sh GPU VM entrypoint (held-out+behavioural)
    results/              all repair, prognosis, and rebuttal artifacts
README.md                 this file
```

---

## 11. Citations

The repair builds on LEACE concept erasure (Belrose et al. 2023, arXiv:2306.03819) and
activation patching (Meng et al. 2022, arXiv:2202.05262; Zhang and Nanda 2025,
arXiv:2309.16042), grounded in the interventional account of explanation (Pearl 2009).
The load-bearing reading of the audited direction follows the massive-activation
literature (Sun et al. 2024, arXiv:2402.17762; Yu et al. 2024, arXiv:2411.07191; Oh et
al. 2024, arXiv:2410.01866). The eight baselines carry their own citations in
`Code/CURE/baselines.py`. The audit half is the causal discriminative-validity audit
described in the accompanying paper.
