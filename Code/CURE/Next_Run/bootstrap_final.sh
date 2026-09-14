#!/usr/bin/env bash
# bootstrap_final.sh -- GPU VM entry point for Code/CURE/Next_Run (Next_Plan.md, final cycle).
#
# ONE MODEL PER VM (NR_MODEL = gemma-2-2b-it | llama-3.1-8b-instruct). Every output goes to
# results/final_20260914/ with model-suffixed file names, so two VMs never write the same file;
# f4_analysis.py merges on the author's machine. Non-markdown outputs are pushed to GitHub at
# every stage end and every 20 minutes.
#
# Sequence: python env (torch 2.5.1 cu124 from the image, precompiled flash-attention wheel,
# requirements) -> predownload -> SMOKE (2 seeds through every stage, --smoke; a failure stops
# with a FATAL status) -> checks -> calibrate -> timing (tier decision) -> final -> control ->
# baseline -> DONE marker. Resumable: rerunning skips finished rows.
#
# Env knobs: NR_MODEL (required), NR_MODEL_HOURS (default 12), NR_GPU_RATE (USD/h, recorded),
#            NR_TIER (160|100 to force), NR_SKIP_SMOKE=1
set -uo pipefail
WORK=${WORK:-/workspace}; REPO=${REPO:-"$WORK/Cure_Audit_Benchmark"}
CURE="$REPO/Code/CURE"; NR="$CURE/Next_Run"; FINAL="$CURE/results/final_20260914"
export HF_HOME=${HF_HOME:-"$WORK/hf"} DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PIP="pip3 install --break-system-packages --no-cache-dir"
MODEL=${NR_MODEL:?NR_MODEL is required}
STATUS_REL="Code/CURE/results/NR_STATUS_${MODEL}.txt"
mkdir -p "$CURE/logs" "$FINAL"; cd "$NR"
echo "[nr] $(date -u) model=$MODEL"
(apt-get update -y && apt-get install -y --no-install-recommends git wget ca-certificates) >/dev/null 2>&1 || true

python3 - <<'PY'
import os
real = ["HUGGINGFACE_TOKEN", "Github_Classic_Token", "RANDOM_SEED"]
dummy = ["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2", "GEMINI_API_KEY_1", "GEMINI_API_KEY_2",
         "GEMINI_API_KEY_3", "GEMINI_API_KEY_4", "MISTRAL_API_KEY1", "MISTRAL_API_KEY2", "OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2"]
with open("../.env", "w") as f:
    for k in real:
        v = os.environ.get(k, "")
        if v: f.write(f"{k}={v}\n")
    for k in dummy:
        if not os.environ.get(k): f.write(f"{k}=unused\n")
PY
git config --global --add safe.directory "$REPO"
git -C "$REPO" config user.name "CURE Runner"; git -C "$REPO" config user.email "koushikdeb2009@gmail.com"
git -C "$REPO" config pull.rebase false
[ -n "${Github_Classic_Token:-}" ] && git -C "$REPO" remote set-url origin "https://${Github_Classic_Token}@github.com/DevDaring/Cure_Audit_Benchmark.git"

LOCK=/tmp/nr_git.lock
git_sync_push() {
  local n=0
  git -C "$REPO" commit -q -m "$1" >/dev/null 2>&1 || true
  while [ $n -lt 6 ]; do
    git -C "$REPO" add -u -- Code/CURE/results >/dev/null 2>&1 || true
    git -C "$REPO" commit -q -m "$1 (tracked)" >/dev/null 2>&1 || true
    git -C "$REPO" pull --no-rebase --no-edit -q origin main >/dev/null 2>&1 || git -C "$REPO" merge --abort >/dev/null 2>&1
    git -C "$REPO" push -q origin main >/dev/null 2>&1 && { echo "[nr] pushed: $1"; return 0; }
    n=$((n+1)); sleep $((10 * n))
  done
  echo "[nr] WARNING: push failed: $1"; return 1
}
_push_status_locked() { printf '%s @ %s\n' "$1" "$(date -u)" > "$REPO/$STATUS_REL"; git -C "$REPO" add -f "$STATUS_REL" >/dev/null 2>&1; git_sync_push "nr[$MODEL]: $1"; }
_push_results_locked() {
  find "$FINAL" -type f ! -name '*.md' ! -name '*.log' -print0 | xargs -0 -r git -C "$REPO" add -f >/dev/null 2>&1
  git -C "$REPO" add -f "$STATUS_REL" >/dev/null 2>&1; git_sync_push "nr-results[$MODEL]: $1"; }
push_status()  { ( flock 9; _push_status_locked  "$1" ) 9>"$LOCK"; }
push_results() { ( flock 9; _push_results_locked "$1" ) 9>"$LOCK"; }
checkpoint_loop() { while true; do sleep 1200; push_results "checkpoint $(date -u +%H:%M)"; done; }
push_status "container started ($(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1))"

# ---------------------------------------------------------------- python env
if ! python3 -c "import torch; assert torch.__version__.startswith('2.5.1')" >/dev/null 2>&1; then
  $PIP torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124; fi
printf 'torch==2.5.1\n' > /tmp/torch_pin.txt
$PIP --upgrade pip >/dev/null 2>&1
$PIP -r "$CURE/Extended_Research_Codes/requirements_extended.txt" scikit-learn -c /tmp/torch_pin.txt --extra-index-url https://download.pytorch.org/whl/cu124 > "$CURE/logs/nr_pip.log" 2>&1 || {
  tail -20 "$CURE/logs/nr_pip.log"; push_status "FATAL: pip install failed"; sleep infinity; }
PYTAG=$(python3 -c 'import sys;print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
FA_OK=0
for ABI in FALSE TRUE; do
  $PIP --no-deps --force-reinstall "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl" >> "$CURE/logs/nr_pip.log" 2>&1
  python3 -c "import flash_attn, torch; assert torch.cuda.is_available()" >/dev/null 2>&1 && { echo "[nr] flash-attn OK"; FA_OK=1; break; }
done
[ "$FA_OK" = "1" ] || echo "[nr] flash-attn wheel not usable; SDPA fallback (recorded in final_protocol.json)"
python3 - > "$FINAL/environment_${MODEL}.txt" 2>&1 <<'PY'
import sys, torch, transformers
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available(), "| transformers", transformers.__version__)
if torch.cuda.is_available(): print("device", torch.cuda.get_device_name(0), "| mem GB", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
for pkg in ("flash_attn", "concept_erasure", "sklearn"):
    try: m = __import__(pkg); print(pkg, getattr(m, "__version__", "ok"))
    except Exception as e: print(pkg, "MISSING:", e)
PY
cat "$FINAL/environment_${MODEL}.txt"

echo "[nr] predownload $MODEL"
python3 - "$MODEL" <<'PY'
import sys; sys.path.insert(0, "..")
import config_cure as C
from huggingface_hub import snapshot_download
hf = C.model_cfg(sys.argv[1])["hf_id"]; print("downloading", hf, flush=True); snapshot_download(hf, token=C.HUGGINGFACE_TOKEN); print("present")
PY
push_status "setup complete"

run_stage() {  # $1 label, $2 log, rest = command; 3 attempts
  local label=$1 logn=$2; shift 2; local attempt=0
  while true; do
    attempt=$((attempt+1))
    "$@" > "$CURE/logs/$logn.log" 2>&1 && { tail -3 "$CURE/logs/$logn.log"; return 0; }
    tail -25 "$CURE/logs/$logn.log"
    push_status "$label attempt $attempt failed: $(tail -3 "$CURE/logs/$logn.log" | tr '\n' ' ' | tail -c 240)"
    [ "$attempt" -ge 3 ] && { push_status "$label gave up after 3 attempts"; return 1; }
    sleep 60
  done
}
HOURS=${NR_MODEL_HOURS:-12}; RATE=${NR_GPU_RATE:-}; TIERF=""; [ -n "${NR_TIER:-}" ] && TIERF="--tier $NR_TIER"
RATEF=""; [ -n "$RATE" ] && RATEF="--gpu-rate $RATE"

# ---------------------------------------------------------------- SMOKE (real model, 2 seeds, every stage, separate output dir)
if [ "${NR_SKIP_SMOKE:-0}" != "1" ]; then
  push_status "SMOKE starting"
  export NR_FINAL_DIR="final_smoke_${MODEL}"; mkdir -p "$CURE/results/$NR_FINAL_DIR"
  cp "$FINAL/source_manifest.csv" "$CURE/results/$NR_FINAL_DIR/"
  if run_stage "SMOKE" "nr_smoke" python3 f2_runner.py --model "$MODEL" --stage all --smoke --model-hours 1 $RATEF; then
    push_status "SMOKE PASSED"
  else
    push_status "FATAL: SMOKE FAILED"; sleep infinity
  fi
  unset NR_FINAL_DIR
fi

# ---------------------------------------------------------------- FULL
checkpoint_loop & CKPT=$!
ok=1
for st in checks calibrate timing final control baseline; do
  push_status "stage $st starting"
  run_stage "$st" "nr_$st" python3 f2_runner.py --model "$MODEL" --stage "$st" --model-hours "$HOURS" $RATEF $TIERF || { ok=0; break; }
  push_results "stage $st"
done
kill "$CKPT" >/dev/null 2>&1 || true
if [ "$ok" = "1" ]; then
  date -u > "$FINAL/DONE_${MODEL}.txt"; push_results "ALL STAGES DONE"; push_status "ALL COMPLETE for $MODEL"
else
  push_results "partial"; push_status "FATAL: a stage gave up; outputs so far are pushed"
fi
echo "[nr] done; container kept alive"; sleep infinity
