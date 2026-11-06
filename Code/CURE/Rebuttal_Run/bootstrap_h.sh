#!/usr/bin/env bash
# bootstrap_h.sh -- GPU VM entry point for Code/CURE/Rebuttal_Run (ICLR 2027 rebuttal cycle).
#
# One VM runs the models listed in HH_MODELS (space separated) in sequence. For every model:
# SMOKE (two seeds through the model's whole stage sequence into a separate directory; a failure
# stops the VM with a FATAL status) -> the model's stage sequence (run_all.py). Outputs go to
# results/rebuttal_20261107/ with model-suffixed names; non-markdown outputs are pushed to
# GitHub after every stage and every 30 minutes. Resumable.
#
# Env: HH_MODELS (required), HH_SKIP_SMOKE=1, HH_GPU_RATE (recorded), HH_ONLY_STAGE (optional)
set -uo pipefail
WORK=${WORK:-/workspace}; REPO=${REPO:-"$WORK/Cure_Audit_Benchmark"}
CURE="$REPO/Code/CURE"; HH="$CURE/Rebuttal_Run"; OUTD="$CURE/results/rebuttal_20261107"
export HF_HOME=${HF_HOME:-"$WORK/hf"} DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PIP="pip3 install --break-system-packages --no-cache-dir"
MODELS=${HH_MODELS:?HH_MODELS is required}
TAG=$(echo "$MODELS" | tr ' ' '+')
STATUS_REL="Code/CURE/results/HH_STATUS_${TAG}.txt"
mkdir -p "$CURE/logs" "$OUTD"; cd "$HH"
echo "[hh] $(date -u) models=$MODELS"
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
git -C "$REPO" config user.name "CURE Runner"; git -C "$REPO" config user.email "runner@localhost"
git -C "$REPO" config pull.rebase false

LOCK=/tmp/hh_git.lock
git_sync_push() {
  local n=0
  git -C "$REPO" commit -q -m "$1" >/dev/null 2>&1 || true
  while [ $n -lt 6 ]; do
    git -C "$REPO" add -u -- Code/CURE/results >/dev/null 2>&1 || true
    git -C "$REPO" commit -q -m "$1 (tracked)" >/dev/null 2>&1 || true
    git -C "$REPO" pull --no-rebase --no-edit -q origin main >/dev/null 2>&1 || git -C "$REPO" merge --abort >/dev/null 2>&1
    git -C "$REPO" push -q origin main >/dev/null 2>&1 && { echo "[hh] pushed: $1"; return 0; }
    n=$((n+1)); sleep $((10 * n))
  done
  echo "[hh] WARNING: push failed: $1"; return 1
}
_push_status_locked() { printf '%s @ %s\n' "$1" "$(date -u)" > "$REPO/$STATUS_REL"; git -C "$REPO" add -f "$STATUS_REL" >/dev/null 2>&1; git_sync_push "hh[$TAG]: $1"; }
_push_results_locked() {
  find "$OUTD" -type f ! -name '*.md' ! -name '*.log' -print0 | xargs -0 -r git -C "$REPO" add -f >/dev/null 2>&1
  for d in "$CURE"/results/rebuttal_smoke_*; do [ -d "$d" ] && find "$d" -type f ! -name '*.md' ! -name '*.log' -print0 | xargs -0 -r git -C "$REPO" add -f >/dev/null 2>&1; done
  git -C "$REPO" add -f "$STATUS_REL" >/dev/null 2>&1; git_sync_push "hh-results[$TAG]: $1"; }
push_status()  { ( flock 9; _push_status_locked  "$1" ) 9>"$LOCK"; }
push_results() { ( flock 9; _push_results_locked "$1" ) 9>"$LOCK"; }
checkpoint_loop() { while true; do sleep 1800; push_results "checkpoint $(date -u +%H:%M)"; done; }
push_status "container started ($(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1))"

# ---------------------------------------------------------------- python env (torch 2.5.1 cu124 image + precompiled flash-attention)
if ! python3 -c "import torch; assert torch.__version__.startswith('2.5.1')" >/dev/null 2>&1; then
  $PIP torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124; fi
printf 'torch==2.5.1\ntransformers==5.17.0\n' > /tmp/torch_pin.txt
$PIP --upgrade pip >/dev/null 2>&1
$PIP -r "$CURE/Extended_Research_Codes/requirements_extended.txt" scikit-learn scipy concept-erasure -c /tmp/torch_pin.txt --extra-index-url https://download.pytorch.org/whl/cu124 > "$CURE/logs/hh_pip.log" 2>&1 || {
  tail -20 "$CURE/logs/hh_pip.log"; push_status "FATAL: pip install failed"; sleep infinity; }
PYTAG=$(python3 -c 'import sys;print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
FA_OK=0
for ABI in FALSE TRUE; do
  $PIP --no-deps --force-reinstall "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl" >> "$CURE/logs/hh_pip.log" 2>&1
  python3 -c "import flash_attn, torch; assert torch.cuda.is_available()" >/dev/null 2>&1 && { echo "[hh] flash-attn OK"; FA_OK=1; break; }
done
[ "$FA_OK" = "1" ] || echo "[hh] flash-attn wheel not usable; SDPA fallback"
python3 - > "$OUTD/environment_${TAG}.txt" 2>&1 <<'PY'
import sys, torch, transformers
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available(), "| transformers", transformers.__version__)
if torch.cuda.is_available(): print("device", torch.cuda.get_device_name(0), "| mem GB", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
for pkg in ("flash_attn", "concept_erasure", "sklearn", "scipy"):
    try: m = __import__(pkg); print(pkg, getattr(m, "__version__", "ok"))
    except Exception as e: print(pkg, "MISSING:", e)
PY
cat "$OUTD/environment_${TAG}.txt"

for M in $MODELS; do
  echo "[hh] predownload $M"
  python3 - "$M" <<'PY'
import sys; sys.path.insert(0, ".")
import hh_common as H
import config_cure as C
from huggingface_hub import snapshot_download
hf = C.model_cfg(sys.argv[1])["hf_id"]; print("downloading", hf, flush=True); snapshot_download(hf, token=C.HUGGINGFACE_TOKEN); print("present")
PY
done
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

stages_of() {
  case "$1" in
    gemma-2-2b-it|llama-3.1-8b-instruct) echo "h1a h7 h4 h2 h3 h1b";;
    qwen2.5-7b-instruct|phi-4-mini-instruct) echo "h5 h1a h7 h4 h3 h1b";;
    *) echo "h3";;
  esac
}

checkpoint_loop & CKPT=$!
ok=1
for M in $MODELS; do
  # ---------------------------------------------------------------- SMOKE (two seeds through every stage, separate directory)
  if [ "${HH_SKIP_SMOKE:-0}" != "1" ]; then
    push_status "SMOKE $M starting"
    export HH_OUT_DIR="rebuttal_smoke_${M}"; mkdir -p "$CURE/results/$HH_OUT_DIR"
    if run_stage "SMOKE $M" "hh_smoke_$M" python3 run_all.py --stage all --model "$M" --smoke; then
      push_status "SMOKE PASSED for $M"; push_results "smoke $M"
    else
      push_status "FATAL: SMOKE FAILED for $M"; push_results "smoke $M failed"; ok=0; unset HH_OUT_DIR; break
    fi
    unset HH_OUT_DIR
  fi
  # the first launch stops here so the smoke rows can be read before any full stage starts;
  # deploy_h.py restart --skip-smoke clears HH_SMOKE_ONLY and starts the full sequence
  if [ "${HH_SMOKE_ONLY:-0}" = "1" ]; then continue; fi
  # ---------------------------------------------------------------- FULL
  for st in $(stages_of "$M"); do
    if [ -n "${HH_ONLY_STAGE:-}" ] && [ "$st" != "$HH_ONLY_STAGE" ]; then continue; fi
    push_status "stage $st $M starting"
    run_stage "$st $M" "hh_${st}_$M" python3 run_all.py --stage "$st" --model "$M" || { ok=0; break; }
    push_results "stage $st $M done"
  done
  [ "$ok" = "1" ] || break
  date -u > "$OUTD/DONE_${M}.txt"; push_results "model $M done"
done
kill "$CKPT" >/dev/null 2>&1 || true
if [ "$ok" = "1" ] && [ "${HH_SMOKE_ONLY:-0}" = "1" ]; then
  push_results "smoke rows"; push_status "SMOKE ONLY DONE for $TAG; holding for review"
elif [ "$ok" = "1" ]; then
  push_results "ALL STAGES DONE"; push_status "ALL COMPLETE for $TAG"
else
  push_results "partial"; push_status "FATAL: a stage gave up; outputs so far are pushed"
fi
echo "[hh] done; container kept alive"; sleep infinity
