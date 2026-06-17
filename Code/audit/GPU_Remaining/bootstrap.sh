#!/usr/bin/env bash
# bootstrap.sh -- VM entrypoint for the GPU_Remaining run on Akash (H200/H100).
# Secrets (HUGGINGFACE_TOKEN, Github_Classic_Token) arrive as container env vars
# injected by the SDL; they are never committed to the repo.
#
# Steps: system deps -> python deps -> precompiled flash-attn -> download models
#        (dataset is already in the cloned repo) -> DRY RUN (2 instances) ->
#        on pass, clean test artifacts -> MAIN run under a restart supervisor.
set -uo pipefail

WORK=/workspace
REPO="$WORK/Audit_Benchmark"
GR="$REPO/Code/audit/GPU_Remaining"
export HF_HOME="$WORK/hf"
export DEBIAN_FRONTEND=noninteractive
export PIP="pip3 install --break-system-packages"

echo "[bootstrap] system deps"
apt-get update -y
apt-get install -y --no-install-recommends git wget ca-certificates python3 python3-pip python3-venv build-essential

echo "[bootstrap] python: $(python3 --version)"   # expect 3.12 on Ubuntu 24.04
cd "$GR"

echo "[bootstrap] write .env from injected secrets (gitignored, local only)"
python3 - <<'PY'
import os
real_keys = ["HUGGINGFACE_TOKEN", "Github_Classic_Token", "RANDOM_SEED"]
# config.py _require()s these 12 at import; the OSM GPU job never calls them,
# so dummy non-empty values satisfy validation without exposing real API keys.
dummy_keys = ["DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2",
              "OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2",
              "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4",
              "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "MISTRAL_API_KEY1", "MISTRAL_API_KEY2"]
n = 0
with open(".env", "w") as f:
    for k in real_keys:
        v = os.environ.get(k, "")
        if v:
            f.write(f"{k}={v}\n"); n += 1
    for k in dummy_keys:
        f.write(f"{k}=unused-by-osm-gpu-job\n")
print(f"wrote .env with {n} real secrets + {len(dummy_keys)} dummy api keys")
PY

echo "[bootstrap] configure git for result/log pushes"
git config --global --add safe.directory "$REPO"
git -C "$REPO" config user.name "MIRAGE GPU Runner"
git -C "$REPO" config user.email "koushikdeb2009@gmail.com"
git -C "$REPO" config pull.rebase true
if [ -n "${Github_Classic_Token:-}" ]; then
  git -C "$REPO" remote set-url origin "https://${Github_Classic_Token}@github.com/DevDaring/Cure_Audit_Benchmark.git"
fi
push_logs() {
  mkdir -p "$GR/results" "$GR/logs"
  echo "$1 @ $(date -u)" > "$GR/results/BOOT_STATUS.txt"
  git -C "$REPO" add -f Code/audit/GPU_Remaining/results Code/audit/GPU_Remaining/logs >/dev/null 2>&1
  # never push the dry-run TEST results -- only real main-run results belong on GitHub
  git -C "$REPO" reset -q -- Code/audit/GPU_Remaining/results/dryrun >/dev/null 2>&1
  git -C "$REPO" commit -q -m "gpu-boot: $1" >/dev/null 2>&1
  git -C "$REPO" pull --rebase -q origin main >/dev/null 2>&1
  git -C "$REPO" push -q origin main >/dev/null 2>&1 && echo "[bootstrap] pushed: $1"
}

push_logs "container started; installing deps (nvidia-smi: $(nvidia-smi -L 2>/dev/null | head -1))"

echo "[bootstrap] torch 2.5.1 (cu124)"
$PIP --upgrade pip
$PIP torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
echo "[bootstrap] python deps"
$PIP -r requirements_gpu.txt

echo "[bootstrap] transformer_lens 2.18.0 (--no-deps so it keeps torch 2.5.1 / transformers 4.50.3)"
$PIP --no-deps transformer_lens==2.18.0

echo "[bootstrap] verify patching libraries import (fail loud BEFORE the run)"
python3 -c "import transformer_lens, nnsight; import importlib.metadata as m; print('TL', m.version('transformer_lens'), '| nnsight', m.version('nnsight'))" > "$GR/logs/verify.log" 2>&1
VRC=$?
cat "$GR/logs/verify.log"
git -C "$REPO" add -f Code/audit/GPU_Remaining/logs/verify.log >/dev/null 2>&1
if [ "$VRC" -ne 0 ]; then
    echo "[bootstrap] FATAL: transformer_lens / nnsight import failed -- container kept alive for inspection"
    push_logs "FATAL: patching libs import failed (full traceback in logs/verify.log)"
    sleep infinity
fi

echo "[bootstrap] precompiled flash-attention"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
wget -q "$FA_URL" -O /tmp/fa.whl && $PIP --no-deps /tmp/fa.whl \
  && echo "[bootstrap] flash-attn installed" \
  || echo "[bootstrap] WARN flash-attn install failed; will fall back to sdpa/eager"

echo "[bootstrap] verify GPU"
python3 -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo "[bootstrap] download OSM models (dataset already in repo)"
python3 - <<'PY'
import os, sys
sys.path.insert(0, os.path.abspath(".."))
from config import OSM_MODELS, HUGGINGFACE_TOKEN
from huggingface_hub import snapshot_download
for m in OSM_MODELS:
    print("downloading", m["hf_id"], flush=True)
    snapshot_download(m["hf_id"], token=HUGGINGFACE_TOKEN)
print("models present")
PY

echo "[bootstrap] setup complete -- marker push"
push_logs "setup complete; starting dry-run"

echo "[bootstrap] DRY RUN (2 instances)"
python3 run_gpu_remaining.py --mode dry > "$GR/logs/dryrun_console.log" 2>&1; DRY_RC=$?
tail -50 "$GR/logs/dryrun_console.log"
echo "[bootstrap] dry-run rc=$DRY_RC"
push_logs "dry-run rc=$DRY_RC"
if [ "$DRY_RC" -ne 0 ]; then
    echo "[bootstrap] DRY FAILED -- container kept alive (logs pushed for inspection)"
    sleep infinity
fi
echo "[bootstrap] DRY PASSED -- cleaning test artifacts (results + logs)"
rm -rf results/dryrun
rm -f logs/dryrun_console.log
: > logs/run_gpu_remaining.log || true
# remove the now-stale dry-run test artifacts from git too, so only the real
# main-run results/logs are tracked (push happens at the first main checkpoint).
git -C "$REPO" rm -r --cached --ignore-unmatch Code/audit/GPU_Remaining/results/dryrun Code/audit/GPU_Remaining/logs/dryrun_console.log >/dev/null 2>&1 || true

echo "[bootstrap] MAIN run (restart supervisor)"
ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT+1))
    echo "[bootstrap] main attempt $ATTEMPT"
    python3 run_gpu_remaining.py --mode main > "$GR/logs/main_console.log" 2>&1 && break
    tail -30 "$GR/logs/main_console.log"; push_logs "main attempt $ATTEMPT exited non-zero"
    echo "[bootstrap] main exited non-zero; retry in 60s"
    sleep 60
done

echo "[bootstrap] COMPLETE"
sleep infinity
