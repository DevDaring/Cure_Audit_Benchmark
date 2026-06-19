# CURE: Causal Audit and Repair of Bias Benchmarks

CURE is a two-stage instrument for large language model fairness. It first audits a
bias benchmark to find items whose decision routes through a protected attribute, even
when the surface answer looks fair, and it then repairs that routing with a surgical,
inference-time intervention confirmed by the same causal test. The audit half lives in
`Code/audit`; the repair half, and the comparison against recent debiasing methods,
lives in `Code/CURE`.

This repository is a one-stop solution. A reader can reproduce the whole pipeline from
this file alone.

---

## 1. What the two folders do

| Folder | Role | Summary |
|--------|------|---------|
| `Code/audit` | Diagnosis | The causal discriminative-validity audit. A five-slot behavioural probe (the pentad) plus a causal intervention (CDVA) that patches the protected-attribute representation and reads the change in the answer logit. Produces the validity leaderboard and the residual of items that pass behaviourally yet fail causally. |
| `Code/CURE`  | Repair   | Uses the audit's causal direction to erase the protected subspace at inference, re-runs the audit to confirm removal, measures the utility cost at a utility-aware operating rank, and compares against eight debiasing methods that read an independent demographic signal. Also fits the relation between the audit score and the repair effort. |

The repair runs on the same four open models, the same three datasets, and the same
causal stack as the audit, so every comparison is fair.

---

## 2. The result story: diagnose, then cure

1. Diagnose. A benchmark score hides a large validity gap, and behavioural signals do
   not predict the causal outcome. There is an invisible residual: items that look fair
   but compute unfairly, which no behavioural audit can detect.
2. Cure. The same causal direction that detects the bias is projected out of the
   residual stream at the protected position. Re-running the audit shows the residual
   commutator drop. CURE removes more causal bias than the eight baselines on three of
   the four models, but the gain sits on a fairness-utility frontier: on the severest
   model the audited bias subspace coincides with massive-activation directions, so
   removing it costs task accuracy. The repair pays its cost at a utility-aware operating
   rank, and reports that cost honestly rather than hiding it.
3. Prognosis. The audit score predicts the repair effort, so an auditor can estimate the
   cost of fixing a model from the audit alone. Per pair the audit score forecasts the
   rank needed to repair it (Spearman 0.58 to 0.63); per model the audit severity tracks
   the utility cost of repair.

The headline claim: CURE removes causal failures that no behavioural method can detect,
removes more causal bias than eight recent debiasing methods on most models, and the
audit score forecasts both the effort and the cost of repair. The full write-up is the
TACL submission in `Submission2/` (see `Submission2/submission_notes.md`).

---

## 3. Environment (exact)

The operating system must match the precompiled flash-attention wheel.

- OS: Ubuntu 24.04 LTS, x86_64 (CUDA image `nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04`).
- Python: 3.12.
- Torch: `torch==2.5.1` from the cu124 index.
- CUDA: 12.x.
- GPU: a single 24 GB or larger card is enough for the 2 to 8 billion parameter models.
- No virtual environment. Install globally with `--break-system-packages`.

Install sequence (also automated by `Code/CURE/bootstrap.sh`):

```bash
pip3 install --break-system-packages torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip3 install --break-system-packages -r Code/CURE/requirements_cure.txt
pip3 install --break-system-packages --no-deps transformer_lens==2.18.0
# precompiled flash-attention (do not build from source)
wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl -O /tmp/fa.whl
pip3 install --break-system-packages --no-deps /tmp/fa.whl
```

Flash-attention is required for run speed. The dry run verifies it with a real forward
pass and fails loud if it is missing. A debug override `CURE_ALLOW_SDPA=1` exists but
logs a loud warning and runs slowly.

---

## 4. Secrets and the `.env` contract

Every key is read from the environment. No secret is ever written into a tracked file.
Copy `Code/CURE/.env.example` to `Code/CURE/.env` and fill the values. The `.env` is
git-ignored and never pushed.

| Variable | Purpose |
|----------|---------|
| `HUGGINGFACE_TOKEN` | Model download. |
| `Github_Classic_Token` | Checkpoint pushes. |
| `RANDOM_SEED` | Reproducibility (default 20260101). |
| `GEMINI_API_KEY_1..4` | Primary judge, gemini-2.5-flash (the four Gemini / GCP keys). |
| `DEEPSEEK_API_KEY_1..2` | Secondary judge, deepseek-chat. |
| `MISTRAL_API_KEY1..2` | Tertiary judge, mistral-small-latest. |
| `OPENROUTER_API_KEY_1..2` | Alternative gateway. |
| `AWS_ACCESS_KEY`, `AWS_SECRET_KEY` | Required by the audit config; unused by CURE (dummy values are fine). |

---

## 5. Judge and answer-extraction design

The judge is used to extract a structured answer when the deterministic JSON parse fails,
and for any model-as-judge step. One active tier is chosen by `CURE_JUDGE_PROVIDER`
(default `gemini`). Keys are round-robined within the active tier. There is no automatic
fallback between tiers: if the active tier fails for an item, the item is recorded as a
judge failure and logged. This keeps every judgement in a run from one model, which
protects reproducibility.

| Tier | Provider | Model | Keys |
|------|----------|-------|------|
| Primary | Google Gemini | gemini-2.5-flash | `GEMINI_API_KEY_1..4` |
| Secondary | DeepSeek | deepseek-chat | `DEEPSEEK_API_KEY_1..2` |
| Tertiary | Mistral | mistral-small-latest | `MISTRAL_API_KEY1..2` |
| Gateway | OpenRouter | configurable | `OPENROUTER_API_KEY_1..2` |

---

## 6. How to run

```bash
cd Code/CURE
python3 run_cure.py --mode dry     # validate the whole environment on two pairs
python3 run_cure.py --mode main    # run E1 to E6 for all four models
```

The dry run validates: API connectivity across every provider and key, the per-model
code path on two pairs for all four models, flash-attention by a real forward pass, a
secret scan of the tracked tree, and an integrity self-test. It exits non-zero on any
failure.

The main run checkpoints to GitHub every 15 minutes and after each model. Resume skips
any unit whose result parquet is present and non-empty, so a released VM restarts from
the last correct results. A `results/DONE` marker is written when all models complete.

The experiments:

| ID | Experiment | Output |
|----|-----------|--------|
| E1 | Extract the bias subspace from the audit counterfactual activations | per-(model, rank) basis |
| E2 | Surgical erasure at the protected position | inference hook |
| E3 | Re-audit the erased model | `cure_recovery_sweep_*.parquet` |
| E4 | Utility cost across erasure rank; pick the utility-aware operating rank | `cure_rankcurve_*.json` |
| E5 | Eight debiasing baselines on an independent demographic signal | `cure_final_*.parquet` |
| E6 | Audit score versus repair effort | `cure_prognosis_*.parquet`, `.json` |

### Cost controls (safe, statistically sound)

The run is expedited without weakening any reported number. The bias subspace is
estimated once from a bounded subset and sliced to every rank (no per-rank
re-extraction). The headline residual-removed and the per-model recovery run on the
full pair set at one operating rank, so they keep the full audit sample and stay in
harmony with the audit. The multi-rank sweep, the fairness-utility curve, and the
six-baseline head-to-head run on a fixed-seed, benchmark-stratified subset of about a
thousand pairs, which gives tight confidence intervals for the prognosis and the
comparison. Knobs (with safe defaults) in `.env`: `CURE_HEADLINE_RANK`,
`CURE_SUBSPACE_PAIRS`, `CURE_SWEEP_SUBSET`, `CURE_E4_MAX_TOKENS`, `CURE_E4_LIMIT`.

The eight baselines all derive their bias direction from an independent set of 310
demographic-contrast templates, not from the audit pairs, so CURE alone reads the causal
audit signal and any advantage isolates the value of that signal. They are: prompt
self-debiasing, generic non-audit-guided erasure, mean-difference steering, FairSteer
(arXiv:2504.14492), BiasGym (arXiv:2508.08855), SAE-Debias (arXiv:2511.00177), H-SAL
(arXiv:2606.12088), and logit-space steering from the No Free Lunch study
(arXiv:2511.18635). The published-method adapters in `Code/CURE/baselines.py` carry
citations and are wired with the official code or a faithful re-implementation.
Faithful-Patchscopes (arXiv:2602.00300) is implemented but kept out of the head-to-head,
since its layer-localisation mechanism is not comparable on the shared erasure protocol.

---

## 7. Integrity guarantees

Every run starts with `integrity.py`, which checks each result parquet for corruption
(corrupt files are quarantined and recomputed) and removes duplicate primary keys, then
writes `results/integrity_report.json`. Dry-run test artifacts under `results/dryrun` are
never pushed.

---

## 8. Repository map

```
Code/
  audit/                 the causal discriminative-validity audit (diagnosis)
    GPU_CPU/             behavioural evaluation and CDVA patching
    CPU_Only/            scoring, statistics, leaderboard
    Dataset/             pentad generator and seed manifests
    results/             behavioural and CDVA result artifacts
  CURE/                  the repair extension (this work)
    run_cure.py          single entry point (dry, main)
    config_cure.py       loads .env, reuses the audit models and dataset
    judge_api.py         judge and answer extraction (round-robin, no cross-tier fallback)
    erase.py             E1 subspace extraction, E2 erasure hooks
    experiments.py       E3 re-audit, E4 utility, E6 prognosis
    baselines.py         E5 comparative methods
    integrity.py         duplicate and corruption checks
    checkpoint.py        resume-safe 15-minute GitHub pushes
    dry_checks.py        all dry-run validations
    bootstrap.sh         GPU VM entrypoint
    requirements_cure.txt
    .env.example
Submission2/             the TACL paper (LaTeX), figures, references, submission_notes.md
README.md                this file
```

---

## 9. Troubleshooting

- Flash-attention import fails: the wheel must match the OS, Python, torch, and CUDA. The
  pinned wheel targets Ubuntu 24.04, Python 3.12, torch 2.5.x, CUDA 12.x. Change the
  wheel tag if you change any of these.
- A judge key is dead: the dry-run `results/dryrun/api_check.json` reports each key. The
  run does not cross tiers; switch `CURE_JUDGE_PROVIDER` or replace the key.
- A quarantined parquet: a corrupt result is moved to `results/quarantine` and recomputed
  on the next run; no action is needed.

---

## 10. Citations

The repair builds on LEACE concept erasure (Belrose et al. 2023, arXiv:2306.03819),
activation patching (Meng et al. 2022, arXiv:2202.05262), and the interventional account
of explanation (Pearl 2009). The audit half is the causal discriminative-validity audit
described in the accompanying paper.
