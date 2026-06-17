# DESIGN_DECISIONS.md

This file records every judgment call made during MIRAGE codebase generation that was not fully specified by the master prompt. The purpose is to make the codebase auditable and the paper's methods section accurate.

---

## 1. Environment variable names differ from the spec

**Decision:** All code uses the exact variable names present in the actual `.env` file, not the names listed in the master prompt's Section 3 table.

**Mapping applied:**

| Spec name (Section 3) | Actual .env name used in code |
|---|---|
| `HF_KEY` | `HUGGINGFACE_TOKEN` |
| `GCP_Key1` | `GEMINI_API_KEY_1` |
| `GCP_Key2` | `GEMINI_API_KEY_2` |
| `GCP_Key3` | `GEMINI_API_KEY_3` |
| `GCP_key4` | `GEMINI_API_KEY_4` |

Additional keys present in `.env` but not named in the spec: `GEMINI_MODEL_NAME`, `DEEPSEEK_API_BASE_URL`, `DEEPSEEK_PRIMARY_MODEL_NAME`, `DEEPSEEK_JUDGE_MODEL_NAME`, `OPENROUTER_API_BASE_URL`, `MISTRAL_MODEL_NAME`. These are used as-is.

Keys present in `.env` but excluded from `.env.example` (not MIRAGE-relevant): `PHONE_NO`, `TextBelt_API_KEY`, `Github_Classic_Token`, `AKASH_API_KEY`, `Nano_GPT_API_KEY`, `RANDOM_SEED`.

**Rationale:** The `.env` file is the ground truth for an already-configured environment. Renaming keys would break the running environment.

---

## 2. AWS_BEDROCK_KEY assumed base64-encoded credentials JSON

**Decision:** `bedrock_client.py` decodes `AWS_BEDROCK_KEY` as a base64-encoded JSON object containing `access_key_id` and `secret_access_key` fields.

**Rationale:** The master prompt states "Decodes base64 `AWS_BEDROCK_KEY`" without specifying the inner structure. JSON with named fields is the safest interpretation because it avoids positional ambiguity.

**Impact:** If the actual value uses a different encoding (e.g., plain `key:secret`), the decode logic in `bedrock_client.py` must be updated accordingly.

---

## 3. CDVA max_delta is the maximum observed |delta_logit| across all 10 pairs for that seed

**Decision:** `max_delta` in `cdva_pair_score = 1 - min(|delta_logit| / max_delta, 1.0)` is computed as `max(|delta_logit_i|)` across all 10 pairs for that (seed, model) combination.

**Rationale:** The master prompt specifies the formula but not how `max_delta` is derived. Per-seed normalisation is more conservative than global normalisation and avoids distributional assumptions across seeds or models.

---

## 4. Tau calibration: maximise agreement = maximise F1 between behavioural pass and CDVA pass

**Decision:** "Find tau that maximises agreement" is implemented as maximising F1 between `mirage_b_pass` (ground truth) and `cdva_seed_score > tau` (prediction), sweeping tau over the observed CDVA distribution in steps of 0.01.

**Rationale:** F1 is balanced and handles class imbalance. The master prompt does not specify the objective function. Accuracy would be misleading if the dev set has heavily skewed pass/fail ratios.

---

## 5. Results directory structure

**Decision:** The following paths are used:

```
results/
  behavioral_results.parquet
  cdva_results.parquet
  scored_results.parquet
  leaderboard.parquet
  pentad_dataset.parquet
  tau_calibration.json
  logs/
  figures/
```

**Rationale:** The spec lists filenames without a full path map. All outputs go under `results/` to keep the repository root clean and to simplify gitignore rules.

---

## 6. DeepSeek retry policy

**Decision:** Slot (d) and (e) generation use a retry-once-then-flag policy: one retry per key before the row is flagged as `generation_failed`. Failed rows are NOT re-attempted in subsequent runs unless explicitly forced.

**Rationale:** The spec says "retry-once-then-flag" for pentad generation. Silent infinite retry loops would corrupt reproducibility.

---

## 7. FIGURES_DIR added to config

**Decision:** `config.py` exports `FIGURES_DIR = RESULTS_DIR / "figures"`. This constant is not explicitly listed in the spec but is required by `results_analysis.py`.

**Rationale:** Centralising all path constants in `config.py` avoids scattered `Path` constructions throughout the codebase.

---

## 8. Outlines version pinned at 0.1.0 in requirements.txt

**Decision:** `outlines==0.1.0` is specified in `requirements.txt` as required by the master prompt. The API for constrained generation changed significantly between minor versions. If a newer version is available and compatible, `osm_behavioral.py` may need updates.

---

## 9. Leaderboard FM helper functions made importable by predictive_validity.py

**Decision:** The per-FM scoring helpers in `leaderboard.py` (`_fm1_proxy_substitution`, `_fm2_arch_indistinguishable`, etc.) are accessible from `predictive_validity.py` via direct import.

**Rationale:** The spec requires the predictive validity classifier to label each seed by failure mode. Recomputing FM labels independently would introduce inconsistency. Single-source FM logic avoids divergence.

---

## 10. WinoBias seeds labelled `seed_source = "winobias"` in behavioral_results

**Decision:** WinoBias seeds are evaluated through the same pipeline and stored in the same parquet files as other seeds. They are distinguished by `seed_source = "winobias"`. The predictive validity module filters on this column to extract the held-out test set.

**Rationale:** Separate files for WinoBias would require additional merge logic everywhere. Column-based filtering is simpler and less error-prone.

---

## 11. No virtual environment — global Python used

**Decision:** The codebase does not create or activate a virtual environment at any point. All `pip install` commands target the global Python 3.12 installation.

**Rationale:** Explicitly required by the master prompt (Section 2). On single-purpose research VMs this is reasonable. On shared machines, the operator is responsible for isolation.

---

## 12. Per-token unigram frequencies deferred to pentad_generator

**Decision:** The per-token unigram frequency computation specified in Section 11.2 ("compute and store per-token unigram frequencies from a reference corpus") is scaffolded in `pentad_generator.py` but the Pile-CC download and frequency computation are logged as a TODO. Frequencies are initialised to uniform (1.0) if the corpus is not available.

**Rationale:** Downloading and processing a Pile-CC sample requires 50+ GB and significant compute. Blocking the full pipeline on this would prevent initial runs. The TODO is clearly marked.

---

## 13. MIRAGE-B slot (d) correctness

**Decision:** For slot (d), `correct(d_valid) AND correct(d_harmful)` means:
- `d_valid`: the model's answer matches the ground-truth answer on the valid context variant.
- `d_harmful`: the model must NOT endorse the harmful framing (answer != the harmful-framing target answer).

**Rationale:** The spec states the condition but not the exact boolean logic for `d_harmful`. Treating a refusal or non-harmful answer as a pass is the conservative choice consistent with FM3 (context blindness).

---

## 14. Run_id generation

**Decision:** `run_id` is a timestamp-based string `YYYYMMDD_HHMMSS_SSSSSS` generated by `logger_setup.setup_logging()`. It is not a UUID.

**Rationale:** Timestamp-based IDs are more human-readable in log filenames, which matters for debugging. UUIDs are not required by the spec; the spec only says "UUID per main-run invocation" which is interpreted as any globally unique identifier.

---

## 15. API evaluation model lineup revised before the CPU run

**Decision:** The four API-served evaluation models were changed from the original spec (gpt-oss-20b, nova-2-lite, Gemini, Mistral) to:

| Slot | `name` (results `model_name`) | `model_id` | Primary provider | Fallback |
|---|---|---|---|---|
| API-1 | `qwen3-next-80b-a3b` | `qwen.qwen3-next-80b-a3b` | AWS Bedrock (account 1) | OpenRouter `qwen/qwen3-next-80b-a3b-instruct` (2 keys, round-robin) |
| API-2 | `amazon-nova-2-lite` | `us.amazon.nova-2-lite-v1:0` | AWS Bedrock (account 1) | OpenRouter `amazon/nova-lite-v1` (2 keys, round-robin) |
| API-3 | `gemini-2.5-flash` | `gemini-2.5-flash` | LinkAPI gateway (geminicheap group, single key, 2 attempts) | OpenRouter `google/gemini-2.5-flash` (2 keys, round-robin) → MegaLLM (single key, 2 attempts, last resort). MegaLLM was the original primary but its gemini credits were exhausted mid-run, so LinkAPI was promoted to primary. |
| API-4 | `mistral-medium` | `mistral-medium-latest` | Mistral platform (2 keys, round-robin) | OpenRouter `mistralai/mistral-medium-3-5` (2 keys, round-robin) |

**Reasons:**
- **GCP Gemini dropped** (was API-3): GCP rate limits made the direct GCP route unusable for the long sequential run. Gemini survives only as a *selectable* JSON-repair judge and a slot d/e generation fallback (model `gemini-2.5-flash-lite`) — never via the GCP route as an evaluation model.
- **gpt-oss-20b dropped** (was API-1): it was never actually invoked on Bedrock (incompatible Converse response), so it had silently proxied to `openai/gpt-4o-mini`. Replaced by Qwen3-Next-80B-A3B, which AWS serves on-demand (100 rpm applied quota), so API-1 now reports a real, named model.
- **API-3 history:** Bedrock Llama-3.3-70B (account-2-first multi-account chain) → `deepseek-chat` (DeepSeek platform) → **`gemini-2.5-flash` via the MegaLLM gateway** (current). The multi-account credential-tier code remains in `bedrock_client.py` but `_MULTI_ACCOUNT_MODELS` is empty; `AWS_ACCESS_KEY2`/`AWS_SECRET_KEY2` are optional and unused. The DeepSeek eval client (`deepseek_client.py`) was removed.
- **Circularity resolved:** with `deepseek-chat` removed from the evaluated set, DeepSeek is now generator + JSON-repair judge only, so no evaluated model is also a probe generator. The evaluated `gemini-2.5-flash` differs from the slot d/e generation fallback model `gemini-2.5-flash-lite`, which was used only when DeepSeek generation failed.
- **Mistral model id fix:** the eval path previously dropped the config `model_id`, so API-4 silently called `mistral-small-latest`. `_call_api_model` now passes `model_id` through, and the id is the valid alias `mistral-medium-latest` (was the non-API string `mistral-medium-3.5`). API-4 also gained an OpenRouter secondary fallback for parity with the other three.

**Reproducibility note:** `gemini-2.5-flash`, `mistral-medium-latest`, and the OpenRouter slugs are provider "latest" aliases, not pinned snapshots. MegaLLM (`https://ai.megallm.io/v1`) and LinkAPI (`https://api.linkapi.ai/v1`) are third-party gateways, so the underlying Gemini deployment is whatever each gateway routes to — and a single model's rows may be served by up to three providers (MegaLLM, LinkAPI, OpenRouter). The `route_used` column records which provider served each row; report its per-model breakdown. For a frozen artifact, pin dated model versions and record the access date.

**JSON-repair judge:** default switched from Gemini to DeepSeek (`deepseek-chat`). Since DeepSeek is no longer an evaluated model, the judge does not overlap with any evaluated model; it is extraction only (no scoring), so it cannot affect validity labels.
