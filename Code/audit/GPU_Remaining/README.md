# GPU_Remaining — finishing the GPU-only review items

Runs the three review items that need a GPU, on a single rented H200/H100, then
checkpoints results back to GitHub every 15 minutes so a lost VM resumes from the
last push.

| Item | What it does | Output |
|---|---|---|
| **T0.2** patch-site recovery | For each CDVA pair, `recovery = patch_delta / (logit_A − logit_B)`. A value near 1 proves the patched protected position carries the demographic effect (the site is not inert). | `results/t02_recovery_<model>.parquet`, `results/recovery_summary.json` |
| **T0.1** temperature sweep | Re-runs slot-a at **T=1.0** (6 samples) to recompute FM4 and show it is not a T=0.7 artefact. | `results/t01_temp_<model>.parquet` |
| **T1.2** option-order | Re-runs MCQ slot-a with a different random option order per sample to separate FM4 content-instability from positional bias. | `results/t12_order_<model>.parquet` |

All three **reuse the production code** (`config.OSM_MODELS`, `GPU_CPU.load_osm.load_model`,
`GPU_CPU.utils_attention.patch_activation`, `GPU_CPU.osm_behavioral.evaluate_osm_model`) so
behaviour matches the original run.

## Run order (handled automatically by `bootstrap.sh`)

1. Install deps (torch 2.5.1 cu124, transformers 4.50.3, nnsight 0.3.7, then
   transformer_lens 2.18.0 via `--no-deps` so it keeps torch 2.5.1 / transformers 4.50.3 --
   this is the version the production CDVA run used, README sec 9.4) and the precompiled
   flash-attn wheel (cu12/torch2.5/cp312; optional, sdpa fallback is numerically identical).
2. Download the 4 OSM models (the pentad dataset + `cdva_results.parquet` ship in the repo).
3. **Dry run** (`--mode dry`): exercises all three tasks on 2 models (one TransformerLens,
   one nnsight) with 2 instances each. On pass, deletes `results/dryrun` + test logs.
4. **Main run** (`--mode main`): all 4 models, sequential, checkpoint + resume; pushes
   `results/` to `main` every 15 minutes and after each model.

## Provisioning

```bash
python deploy_akash.py                 # tries H200, then H100
python deploy_akash.py --status        # lease/deployment status
python deploy_akash.py --close         # tear down + refund escrow
```

Secrets live in `GPU_Remaining/.env` (gitignored) and are injected into the Akash SDL
`env` at submit time only — never committed.

## Monitoring

Watch `Code/audit/GPU_Remaining/results/` on the `main` branch. `STATUS.json` lists
completed `(model, task)` units; `DONE` appears when all four models finish.
