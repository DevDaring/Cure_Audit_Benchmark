#!/usr/bin/env bash
# bootstrap_extended.sh -- GPU VM entry point for Code/CURE/Extended_Research_Codes.
#
# ONE MODEL PER VM. Each VM runs every stage for its model and pushes its outputs to a
# model-specific directory, results/${EXT_V2_DIR} (default results/v2_${EXT_MODEL}), so four
# VMs never touch the same file. merge_v2.py combines the directories into results/v2 on the
# author's machine, where P4 (CPU) and the claims ledger then run.
#
# Sequence (all resumable; a stage that already has its rows on disk adds nothing):
#   P0 (CPU)    evidence repair, regenerated locally for the manifests (deterministic; the
#               committed copy is restored afterwards so git stays clean)
#   tests (CPU) the edit's functional tests, must print ALL PASS
#   SMOKE       every GPU stage below on the REAL model with 2-3 seeds and tiny probes, into
#               results/v2_smoke_${EXT_MODEL}; any failure stops here with a FATAL status so
#               the code can be fixed before the paid run starts (merge_v2.py ignores smoke dirs)
#   P1  (GPU)   24-seed protocol pilot for this model
#   P2  (GPU)   dev phase (energy matching, rank ladder, frozen + sequential LEACE, capability
#               probes), then the test phase
#   P3  (GPU)   magnitude, depth and massive-activation accounts, dev then test
#   done        DONE marker pushed; container kept alive for inspection until destroyed
#
# Speed: torch 2.5.1 cu124 comes with the pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel image;
# flash-attention is installed from the PRECOMPILED wheel matching the interpreter (no
# compilation). load_osm.load_model prefers flash_attention_2 and falls back to SDPA, and the
# attention implementation actually used is recorded in every protocol file.
#
# Pushes never include *.md, *.log, .env or the HF cache: results are CSV / JSON / parquet /
# npz / txt only.
#
# Env knobs (secrets arrive as container env vars; nothing is written outside the VM):
#   EXT_MODEL              required, one of llama-3.1-8b-instruct qwen2.5-7b-instruct
#                          gemma-2-2b-it phi-4-mini-instruct
#   EXT_V2_DIR             output subdirectory under results/ (default v2_${EXT_MODEL})
#   EXT_P1_HOURS=6  EXT_P2_DEV_HOURS=24  EXT_P2_TEST_HOURS=24  EXT_P3_HOURS=10
#   EXT_EXTEND_RANKS=1     also run ranks 2 and 4 in P2 (rank 8 is not run)
#   EXT_CAP_POLICY=all     capability probes under the global application policy
#   EXT_SKIP_P1=0 EXT_SKIP_P2=0 EXT_SKIP_P3=0
set -uo pipefail

WORK=${WORK:-/workspace}
REPO=${REPO:-"$WORK/Cure_Audit_Benchmark"}
CURE="$REPO/Code/CURE"
EXT="$CURE/Extended_Research_Codes"
export HF_HOME=${HF_HOME:-"$WORK/hf"}
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
PIP="pip3 install --break-system-packages --no-cache-dir"

MODEL=${EXT_MODEL:?EXT_MODEL is required}
export EXT_V2_DIR=${EXT_V2_DIR:-"v2_${MODEL}"}
V2="$CURE/results/$EXT_V2_DIR"
STATUS_REL="Code/CURE/results/EXT_STATUS_${MODEL}.txt"
mkdir -p "$CURE/logs" "$V2" "$CURE/results/reanalysis_v2"
cd "$EXT"

echo "[ext] $(date -u) model=$MODEL out=$EXT_V2_DIR"
echo "[ext] system deps"
(apt-get update -y && apt-get install -y --no-install-recommends git wget ca-certificates) >/dev/null 2>&1 || true

echo "[ext] write Code/CURE/.env from injected secrets (gitignored, VM-local)"
python3 - <<'PY'
import os
real = ["HUGGINGFACE_TOKEN", "Github_Classic_Token", "RANDOM_SEED"]
dummy = ["AWS_ACCESS_KEY", "AWS_SECRET_KEY", "GEMINI_API_KEY_1", "DEEPSEEK_API_KEY_1",
         "MISTRAL_API_KEY1", "OPENROUTER_API_KEY_1"]      # names the audit config expects; unused
with open("../.env", "w") as f:
    for k in real:
        v = os.environ.get(k, "")
        if v:
            f.write(f"{k}={v}\n")
    for k in dummy:
        if not os.environ.get(k):
            f.write(f"{k}=unused-by-extended\n")
print("wrote Code/CURE/.env")
PY

git config --global --add safe.directory "$REPO"
git -C "$REPO" config user.name "CURE Runner"
git -C "$REPO" config user.email "koushikdeb2009@gmail.com"
git -C "$REPO" config pull.rebase true
if [ -n "${Github_Classic_Token:-}" ]; then
  git -C "$REPO" remote set-url origin "https://${Github_Classic_Token}@github.com/DevDaring/Cure_Audit_Benchmark.git"
fi

# ---------------------------------------------------------------- git helpers (no .md ever)
LOCK=/tmp/ext_git.lock
git_sync_push() {   # $1 = commit message; adds what is already staged; serialised by flock
  local n=0
  git -C "$REPO" commit -q -m "$1" >/dev/null 2>&1 || true
  while [ $n -lt 6 ]; do
    git -C "$REPO" pull --rebase -q origin main >/dev/null 2>&1 || git -C "$REPO" rebase --abort >/dev/null 2>&1
    git -C "$REPO" push -q origin main >/dev/null 2>&1 && { echo "[ext] pushed: $1"; return 0; }
    n=$((n+1)); sleep $((10 * n))
  done
  echo "[ext] WARNING: push failed after retries: $1"; return 1
}
# Both pushers run under one lock so the 20-minute checkpoint loop and a stage-end push
# never interleave their git operations.
_push_status_locked() {
  printf '%s @ %s\n' "$1" "$(date -u)" > "$REPO/$STATUS_REL"
  git -C "$REPO" add -f "$STATUS_REL" >/dev/null 2>&1
  git_sync_push "ext[$MODEL]: $1"
}
_push_results_locked() {
  # every non-markdown file under this VM's own output directory (incremental parquet rows
  # included, so a stage interrupted by a credit stop keeps its finished items)
  find "$CURE/results/$EXT_V2_DIR" -type f ! -name '*.md' ! -name '*.log' -print0 | xargs -0 -r git -C "$REPO" add -f >/dev/null 2>&1
  git -C "$REPO" add -f "$STATUS_REL" >/dev/null 2>&1
  git_sync_push "ext-results[$MODEL]: $1"
}
push_status()  { ( flock 9; _push_status_locked  "$1" ) 9>"$LOCK"; }
push_results() { ( flock 9; _push_results_locked "$1" ) 9>"$LOCK"; }
checkpoint_loop() {   # background: push whatever exists every 20 minutes
  while true; do sleep 1200; push_results "checkpoint $(date -u +%H:%M)"; done
}
push_status "container started ($(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1))"

# ---------------------------------------------------------------- python environment
echo "[ext] python $(python3 --version) | torch present: $(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)"
if ! python3 -c "import torch; assert torch.__version__.startswith('2.5.1')" >/dev/null 2>&1; then
  echo "[ext] installing torch 2.5.1 cu124"
  $PIP torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
fi
printf 'torch==2.5.1\n' > /tmp/torch_pin.txt
$PIP --upgrade pip >/dev/null 2>&1
$PIP -r requirements_extended.txt -c /tmp/torch_pin.txt --extra-index-url https://download.pytorch.org/whl/cu124 > "$CURE/logs/ext_pip.log" 2>&1 || {
  tail -20 "$CURE/logs/ext_pip.log"; push_status "FATAL: pip install failed"; sleep infinity; }

echo "[ext] precompiled flash-attention wheel for this interpreter"
PYTAG=$(python3 -c 'import sys;print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
FA_BASE="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3"
FA_OK=0
for ABI in FALSE TRUE; do
  FA_URL="$FA_BASE/flash_attn-2.8.3+cu12torch2.5cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
  $PIP --no-deps --force-reinstall "$FA_URL" >> "$CURE/logs/ext_pip.log" 2>&1
  if python3 -c "import flash_attn, torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    echo "[ext] flash-attn OK (cxx11abi${ABI}, ${PYTAG})"; FA_OK=1; break
  fi
done
[ "$FA_OK" = "1" ] || echo "[ext] flash-attn wheel not usable; load_osm will fall back to SDPA (recorded in protocol)"

DIAG="$CURE/logs/ext_diag.txt"; : > "$DIAG"
python3 - >> "$DIAG" 2>&1 <<'PY'
import sys, torch, transformers
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| avail", torch.cuda.is_available(), "| transformers", transformers.__version__)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "| cap", torch.cuda.get_device_capability(0),
          "| mem GB", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
for pkg in ("flash_attn", "concept_erasure", "datasets", "sklearn", "accelerate"):
    try:
        m = __import__(pkg); print(pkg, getattr(m, "__version__", "ok"))
    except Exception as e:
        print(pkg, "MISSING:", e)
PY
cat "$DIAG"
cp "$DIAG" "$V2/environment_${MODEL}.txt"

# ---------------------------------------------------------------- P0 + tests (CPU)
echo "[ext] P0 (CPU, deterministic; committed copy restored afterwards)"
python3 run_all.py --stage p0 > "$CURE/logs/ext_p0.log" 2>&1; RC=$?
tail -5 "$CURE/logs/ext_p0.log"
if [ "$RC" -ne 0 ]; then push_status "FATAL: P0 failed rc=$RC"; sleep infinity; fi
git -C "$REPO" checkout -- Code/CURE/results/reanalysis_v2 >/dev/null 2>&1 || true
git -C "$REPO" clean -fdq Code/CURE/results/reanalysis_v2 >/dev/null 2>&1 || true

echo "[ext] CPU tests"
python3 run_all.py --stage test > "$CURE/logs/ext_test.log" 2>&1; RC=$?
tail -3 "$CURE/logs/ext_test.log"
if [ "$RC" -ne 0 ]; then push_status "FATAL: edit tests failed"; sleep infinity; fi

echo "[ext] predownload $MODEL"
python3 - "$MODEL" <<'PY'
import sys; sys.path.insert(0, "..")
import config_cure as C
from huggingface_hub import snapshot_download
hf = C.model_cfg(sys.argv[1])["hf_id"]; print("downloading", hf, flush=True)
snapshot_download(hf, token=C.HUGGINGFACE_TOKEN)
print("present")
PY
push_status "setup complete"

# ---------------------------------------------------------------- stage runner (retry x3, resume-aware)
run_stage() {  # $1 label, $2 log name, rest = command; non-zero after 3 failed attempts
  local label=$1 logn=$2; shift 2
  local attempt=0
  while true; do
    attempt=$((attempt+1))
    "$@" > "$CURE/logs/$logn.log" 2>&1 && { tail -3 "$CURE/logs/$logn.log"; return 0; }
    tail -25 "$CURE/logs/$logn.log"
    push_status "$label attempt $attempt exited non-zero: $(tail -3 "$CURE/logs/$logn.log" | tr '\n' ' ' | tail -c 240)"
    [ "$attempt" -ge 3 ] && { push_status "$label gave up after 3 attempts"; return 1; }
    sleep 60
  done
}

# Remote control: Code/CURE/results/EXT_CONTROL_${MODEL}.env in the repo (pushed from the
# author's machine) is sourced before every stage. It may set EXT_EXTEND_RANKS, EXT_SKIP_P1,
# EXT_SKIP_P2, EXT_SKIP_P3, EXT_P1_HOURS, EXT_P2_DEV_HOURS, EXT_P2_TEST_HOURS, EXT_P3_HOURS,
# EXT_P3_ACCOUNTS ("magnitude depth massive") and EXT_STOP=1 (finish the current stage, push,
# then idle).
CONTROL_REL="Code/CURE/results/EXT_CONTROL_${MODEL}.env"
refresh_control() {
  ( flock 9; git -C "$REPO" pull --rebase -q origin main >/dev/null 2>&1 || git -C "$REPO" rebase --abort >/dev/null 2>&1 ) 9>"$LOCK"
  if [ -f "$REPO/$CONTROL_REL" ]; then
    # shellcheck disable=SC1090
    set -a; . "$REPO/$CONTROL_REL"; set +a
    echo "[ext] control file applied: $(tr '\n' ' ' < "$REPO/$CONTROL_REL")"
  fi
  if [ "${EXT_STOP:-0}" = "1" ]; then
    push_results "stopped by control file"; push_status "STOPPED by EXT_CONTROL (EXT_STOP=1); idle"
    sleep infinity
  fi
}

# run_sequence smoke|full : P1, P2 dev, P2 test, P3 x {magnitude, depth, massive} x {dev, test}
run_sequence() {
  local MODE=$1 P1F P2F P3F EXTEND DEVCAP TESTCAP ACC PH
  refresh_control
  if [ "$MODE" = "smoke" ]; then
    export EXT_V2_DIR="v2_smoke_${MODEL}"
    P1F="--n-per-bench 1 --n-fit-pairs 8 --gpu-hours-cap 1"
    P2F="--n-dev 2 --n-test 2 --n-fit-pairs 8 --match-energy --match-energy-prompts 2 --max-new-tokens 8 --n-mmlu 4 --n-wiki-tokens 1024 --capability-policy all --extend-ranks --leace-sequential"
    P3F="--n-seeds 2 --n-fit-pairs 8 --n-random-draws 1 --n-massive 2 --gpu-hours-cap 1"
    DEVCAP=1; TESTCAP=1
  else
    export EXT_V2_DIR="v2_${MODEL}"
    P1F="--gpu-hours-cap ${EXT_P1_HOURS:-6}"
    EXTEND=""; [ "${EXT_EXTEND_RANKS:-1}" = "1" ] && EXTEND="--extend-ranks"
    P2F="--match-energy $EXTEND --leace-sequential --capability-policy ${EXT_CAP_POLICY:-all}"
    P3F="--gpu-hours-cap ${EXT_P3_HOURS:-10}"
    DEVCAP="${EXT_P2_DEV_HOURS:-24}"; TESTCAP="${EXT_P2_TEST_HOURS:-24}"
  fi
  mkdir -p "$CURE/results/$EXT_V2_DIR"
  local TAG="[$MODE]"

  if [ "${EXT_SKIP_P1:-0}" != "1" ]; then
    push_status "$TAG P1 pilot starting"
    run_stage "$TAG P1" "ext_${MODE}_p1" python3 run_all.py --stage p1 --models "$MODEL" -- $P1F || return 1
    push_results "$TAG P1 pilot"
  fi
  refresh_control
  if [ "$MODE" = "full" ]; then
    EXTEND=""; [ "${EXT_EXTEND_RANKS:-1}" = "1" ] && EXTEND="--extend-ranks"
    P2F="--match-energy $EXTEND --leace-sequential --capability-policy ${EXT_CAP_POLICY:-all}"
    DEVCAP="${EXT_P2_DEV_HOURS:-24}"; TESTCAP="${EXT_P2_TEST_HOURS:-24}"
  fi
  if [ "${EXT_SKIP_P2:-0}" != "1" ]; then
    push_status "$TAG P2 dev starting"
    run_stage "$TAG P2-dev" "ext_${MODE}_p2_dev" python3 run_all.py --stage p2 --phase dev --models "$MODEL" \
        --gpu-hours-cap "$DEVCAP" -- $P2F || return 1
    push_results "$TAG P2 dev"
    refresh_control
    push_status "$TAG P2 test starting"
    run_stage "$TAG P2-test" "ext_${MODE}_p2_test" python3 run_all.py --stage p2 --phase test --models "$MODEL" \
        --gpu-hours-cap "$TESTCAP" -- $P2F || return 1
    push_results "$TAG P2 test"
  fi
  refresh_control
  if [ "$MODE" = "full" ]; then P3F="--gpu-hours-cap ${EXT_P3_HOURS:-10}"; fi
  if [ "${EXT_SKIP_P3:-0}" != "1" ]; then
    for ACC in ${EXT_P3_ACCOUNTS:-magnitude depth massive}; do
      for PH in dev test; do
        refresh_control
        [ "${EXT_SKIP_P3:-0}" = "1" ] && break 2
        push_status "$TAG P3 $ACC $PH starting"
        run_stage "$TAG P3-$ACC-$PH" "ext_${MODE}_p3_${ACC}_${PH}" python3 run_all.py --stage p3 --account "$ACC" \
            --phase "$PH" --models "$MODEL" -- $P3F || return 1
        push_results "$TAG P3 $ACC $PH"
      done
    done
  fi
  return 0
}

# ---------------------------------------------------------------- SMOKE first (real model, tiny sample)
push_status "SMOKE sequence starting (2-3 seeds through every stage)"
if run_sequence smoke; then
  push_status "SMOKE PASSED; full run starting"
else
  push_status "FATAL: SMOKE FAILED (see the EXT_STATUS history and logs/ext_smoke_*.log on the VM)"
  echo "[ext] smoke failed; container kept alive for inspection"; sleep infinity
fi

# ---------------------------------------------------------------- FULL run
checkpoint_loop &
CKPT_PID=$!
if run_sequence full; then
  kill "$CKPT_PID" >/dev/null 2>&1 || true
  date -u > "$CURE/results/$EXT_V2_DIR/DONE_${MODEL}.txt"
  push_results "ALL STAGES DONE"
  push_status "ALL COMPLETE for $MODEL"
else
  kill "$CKPT_PID" >/dev/null 2>&1 || true
  push_results "partial (a full-run stage gave up)"
  push_status "FATAL: a full-run stage gave up after 3 attempts; outputs so far are pushed"
fi
echo "[ext] done; container kept alive"; sleep infinity
