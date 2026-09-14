#!/usr/bin/env bash
# bootstrap_g.sh -- GPU VM entry point for Code/CURE/Next2Next_Run (ICLR-feedback cycle).
#
# One VM runs the models listed in NN_MODELS (space separated) in sequence. For every model:
# SMOKE (two seeds through every stage into a separate directory; a failure stops the VM with
# a FATAL status) -> g3 -> g1 -> g2 -> g5 (g2 / g5 / g3-b only for the two fresh models).
# Outputs go to results/feedback_20260915/ with model-suffixed names; non-markdown outputs are
# pushed to GitHub after every stage and every 20 minutes. Resumable.
#
# Env: NN_MODELS (required), NN_SKIP_SMOKE=1, NN_GPU_RATE (recorded)
set -uo pipefail
WORK=${WORK:-/workspace}; REPO=${REPO:-"$WORK/Cure_Audit_Benchmark"}
CURE="$REPO/Code/CURE"; NN="$CURE/Next2Next_Run"; OUTD="$CURE/results/feedback_20260915"
export HF_HOME=${HF_HOME:-"$WORK/hf"} DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PIP="pip3 install --break-system-packages --no-cache-dir"
MODELS=${NN_MODELS:?NN_MODELS is required}
TAG=$(echo "$MODELS" | tr ' ' '+')
STATUS_REL="Code/CURE/results/NN_STATUS_${TAG}.txt"
mkdir -p "$CURE/logs" "$OUTD"; cd "$NN"
echo "[nn] $(date -u) models=$MODELS"
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

LOCK=/tmp/nn_git.lock
git_sync_push() {
  local n=0
  git -C "$REPO" commit -q -m "$1" >/dev/null 2>&1 || true
  while [ $n -lt 6 ]; do
    git -C "$REPO" add -u -- Code/CURE/results >/dev/null 2>&1 || true
    git -C "$REPO" commit -q -m "$1 (tracked)" >/dev/null 2>&1 || true
    git -C "$REPO" pull --no-rebase --no-edit -q origin main >/dev/null 2>&1 || git -C "$REPO" merge --abort >/dev/null 2>&1
    git -C "$REPO" push -q origin main >/dev/null 2>&1 && { echo "[nn] pushed: $1"; return 0; }
    n=$((n+1)); sleep $((10 * n))
  done
  echo "[nn] WARNING: push failed: $1"; return 1
}
_push_status_locked() { printf '%s @ %s\n' "$1" "$(date -u)" > "$REPO/$STATUS_REL"; git -C "$REPO" add -f "$STATUS_REL" >/dev/null 2>&1; git_sync_push "nn[$TAG]: $1"; }
_push_results_locked() {
  find "$OUTD" -type f ! -name '*.md' ! -name '*.log' -print0 | xargs -0 -r git -C "$REPO" add -f >/dev/null 2>&1
  git -C "$REPO" add -f "$STATUS_REL" >/dev/null 2>&1; git_sync_push "nn-results[$TAG]: $1"; }
push_status()  { ( flock 9; _push_status_locked  "$1" ) 9>"$LOCK"; }
push_results() { ( flock 9; _push_results_locked "$1" ) 9>"$LOCK"; }
checkpoint_loop() { while true; do sleep 1200; push_results "checkpoint $(date -u +%H:%M)"; done; }
push_status "container started ($(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1))"

# ---------------------------------------------------------------- python env (torch 2.5.1 cu124 image + precompiled flash-attention)
if ! python3 -c "import torch; assert torch.__version__.startswith('2.5.1')" >/dev/null 2>&1; then
  $PIP torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124; fi
printf 'torch==2.5.1\n' > /tmp/torch_pin.txt
$PIP --upgrade pip >/dev/null 2>&1
$PIP -r "$CURE/Extended_Research_Codes/requirements_extended.txt" scikit-learn scipy -c /tmp/torch_pin.txt --extra-index-url https://download.pytorch.org/whl/cu124 > "$CURE/logs/nn_pip.log" 2>&1 || {
  tail -20 "$CURE/logs/nn_pip.log"; push_status "FATAL: pip install failed"; sleep infinity; }
PYTAG=$(python3 -c 'import sys;print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
FA_OK=0
for ABI in FALSE TRUE; do
  $PIP --no-deps --force-reinstall "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl" >> "$CURE/logs/nn_pip.log" 2>&1
  python3 -c "import flash_attn, torch; assert torch.cuda.is_available()" >/dev/null 2>&1 && { echo "[nn] flash-attn OK"; FA_OK=1; break; }
done
[ "$FA_OK" = "1" ] || echo "[nn] flash-attn wheel not usable; SDPA fallback"
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
  echo "[nn] predownload $M"
  python3 - "$M" <<'PY'
import sys; sys.path.insert(0, "..")
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

checkpoint_loop & CKPT=$!
ok=1
for M in $MODELS; do
  # ---------------------------------------------------------------- SMOKE (two seeds through every stage, separate directory)
  if [ "${NN_SKIP_SMOKE:-0}" != "1" ]; then
    push_status "SMOKE $M starting"
    export NN_OUT_DIR="feedback_smoke_${M}"; mkdir -p "$CURE/results/$NN_OUT_DIR"
    if run_stage "SMOKE $M" "nn_smoke_$M" python3 run_all.py --stage all --model "$M" --smoke; then
      push_status "SMOKE PASSED for $M"
    else
      push_status "FATAL: SMOKE FAILED for $M"; ok=0; unset NN_OUT_DIR; break
    fi
    unset NN_OUT_DIR
  fi
  # ---------------------------------------------------------------- FULL
  for st in g3 g1 g2 g5; do
    push_status "stage $st $M starting"
    run_stage "$st $M" "nn_${st}_$M" python3 run_all.py --stage "$st" --model "$M" || { ok=0; break; }
    push_results "stage $st $M"
  done
  [ "$ok" = "1" ] || break
  date -u > "$OUTD/DONE_${M}.txt"; push_results "model $M done"
done
kill "$CKPT" >/dev/null 2>&1 || true
if [ "$ok" = "1" ]; then
  push_results "ALL STAGES DONE"; push_status "ALL COMPLETE for $TAG"
else
  push_results "partial"; push_status "FATAL: a stage gave up; outputs so far are pushed"
fi
echo "[nn] done; container kept alive"; sleep infinity
