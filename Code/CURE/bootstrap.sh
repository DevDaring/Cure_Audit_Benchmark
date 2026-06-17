#!/usr/bin/env bash
# bootstrap.sh -- VM entrypoint for the CURE run on a rented GPU (Ubuntu 24.04 / py3.12).
# Secrets arrive as container env vars injected by the SDL; they are written to a local
# .env (gitignored) and never committed. Flash-attention is installed from a precompiled
# wheel and verified before the run. No virtual environment: installs are global.
# Monitoring uses a tiny results/BOOT_STATUS.txt marker; logs and test results are never
# pushed. Models are predownloaded before the dry run; the dataset ships in Code/audit.
set -uo pipefail

WORK=/workspace
REPO="$WORK/Cure_Audit_Benchmark"
CURE="$REPO/Code/CURE"
export HF_HOME="$WORK/hf"
export DEBIAN_FRONTEND=noninteractive
export PIP="pip3 install --break-system-packages"

echo "[bootstrap] system deps"
apt-get update -y
apt-get install -y --no-install-recommends git wget ca-certificates python3 python3-pip build-essential
echo "[bootstrap] python: $(python3 --version)"   # expect 3.12 on Ubuntu 24.04
cd "$CURE"
# the cloned repo has no empty logs/ dir; create it so every '> logs/...' redirect works
mkdir -p "$CURE/logs" "$CURE/results"

echo "[bootstrap] write .env from injected secrets (gitignored, local only)"
python3 - <<'PY'
import os
real = ["HUGGINGFACE_TOKEN", "Github_Classic_Token", "RANDOM_SEED", "CURE_JUDGE_PROVIDER",
        "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4",
        "DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2", "DEEPSEEK_API_BASE_URL", "DEEPSEEK_JUDGE_MODEL_NAME",
        "MISTRAL_API_KEY1", "MISTRAL_API_KEY2",
        "OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2", "OPENROUTER_API_BASE_URL"]
dummy = ["AWS_ACCESS_KEY", "AWS_SECRET_KEY"]   # required by the audit config; unused by CURE
with open(".env", "w") as f:
    for k in real:
        v = os.environ.get(k, "")
        if v:
            f.write(f"{k}={v}\n")
    for k in dummy:
        f.write(f"{k}=unused-by-cure\n")
print("wrote .env")
PY

echo "[bootstrap] configure git"
git config --global --add safe.directory "$REPO"
git -C "$REPO" config user.name "CURE Runner"
git -C "$REPO" config user.email "koushikdeb2009@gmail.com"
git -C "$REPO" config pull.rebase true
if [ -n "${Github_Classic_Token:-}" ]; then
  git -C "$REPO" remote set-url origin "https://${Github_Classic_Token}@github.com/DevDaring/Cure_Audit_Benchmark.git"
fi

# Tiny status marker for monitoring (status only -- never pushes logs or test results).
push_status() {
  mkdir -p "$CURE/results"
  printf '%s @ %s\n' "$1" "$(date -u)" > "$CURE/results/BOOT_STATUS.txt"
  git -C "$REPO" add -f Code/CURE/results/BOOT_STATUS.txt >/dev/null 2>&1
  git -C "$REPO" commit -q -m "cure-boot: $1" >/dev/null 2>&1
  git -C "$REPO" pull --rebase -q origin main >/dev/null 2>&1
  git -C "$REPO" push -q origin main >/dev/null 2>&1 && echo "[bootstrap] status: $1"
}

push_status "container started (nvidia-smi: $(nvidia-smi -L 2>/dev/null | head -1))"

echo "[bootstrap] torch 2.5.1 (cu124)"
$PIP --upgrade pip
$PIP torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
echo "[bootstrap] python deps"
$PIP -r requirements_cure.txt
echo "[bootstrap] transformer_lens 2.18.0 (--no-deps so it keeps torch 2.5.1 / transformers 4.50.3)"
$PIP --no-deps transformer_lens==2.18.0

DIAG="$CURE/logs/diag.txt"; : > "$DIAG"
echo "[bootstrap] environment diagnostics" | tee -a "$DIAG"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | tee -a "$DIAG" || echo "no nvidia-smi" | tee -a "$DIAG"
python3 - >> "$DIAG" 2>&1 <<'PY'
import torch, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| torch.version.cuda", torch.version.cuda,
      "| cuda_available", torch.cuda.is_available(),
      "| cxx11abi", torch._C._GLIBCXX_USE_CXX11_ABI)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
PY
cat "$DIAG"

echo "[bootstrap] precompiled flash-attention (try both cxx11abi, capture every failure)"
FA_BASE="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3"
FA_OK=0
for ABI in TRUE FALSE; do
  echo "=== flash_attn cxx11abi${ABI} ===" >> "$DIAG"
  wget -q "$FA_BASE/flash_attn-2.8.3+cu12torch2.5cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl" -O /tmp/fa.whl
  echo "wheel_size=$(stat -c%s /tmp/fa.whl 2>/dev/null) bytes" >> "$DIAG"
  $PIP --no-deps --force-reinstall /tmp/fa.whl >> "$DIAG" 2>&1
  python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__)" >> "$DIAG" 2>&1
  echo "import_exit=$? (139=segfault, 132=illegal-instruction, 1=ImportError)" >> "$DIAG"
  if python3 -c "import flash_attn" >/dev/null 2>&1; then
    echo "[bootstrap] flash-attn OK (cxx11abi${ABI})"; FA_OK=1; break
  fi
done

echo "[bootstrap] verify patching libs + flash-attn import (fail loud)"
python3 -c "import transformer_lens, nnsight, flash_attn, importlib.metadata as m; print('TL', m.version('transformer_lens'), '| nnsight', m.version('nnsight'), '| flash_attn', m.version('flash_attn'))" >> "$DIAG" 2>&1
VRC=$?
tail -40 "$DIAG"
if [ "$VRC" -ne 0 ]; then
  TS=$(date +%s)
  cp "$DIAG" "$CURE/results/VERIFY_FAIL_${TS}.txt"
  git -C "$REPO" add -f "Code/CURE/results/VERIFY_FAIL_${TS}.txt" >/dev/null 2>&1
  push_status "FATAL: lib import failed FA_OK=$FA_OK (see results/VERIFY_FAIL_${TS}.txt)"
  echo "[bootstrap] FATAL: a required library failed to import -- container kept alive"; sleep infinity
fi

echo "[bootstrap] download OSM models (predownload; dataset ships in Code/audit)"
python3 - <<'PY'
import config_cure as C
from huggingface_hub import snapshot_download
for m in C.OSM_MODELS:
    print("downloading", m["hf_id"], flush=True)
    snapshot_download(m["hf_id"], token=C.HUGGINGFACE_TOKEN)
print("models present")
PY

push_status "setup complete; starting dry-run"
echo "[bootstrap] DRY RUN (2 instances per dataset)"
python3 run_cure.py --mode dry > "$CURE/logs/dryrun_console.log" 2>&1; DRY_RC=$?
tail -40 "$CURE/logs/dryrun_console.log"
echo "[bootstrap] dry-run rc=$DRY_RC"
if [ "$DRY_RC" -ne 0 ]; then
  push_status "dry-run rc=$DRY_RC FAILED: $(tail -8 "$CURE/logs/dryrun_console.log" | tr '\n' ' ' | tail -c 400)"
  echo "[bootstrap] DRY FAILED -- container kept alive for inspection"; sleep infinity
fi
push_status "dry-run rc=0 PASSED; cleaning test artifacts"
# remove dry-run test results and logs before the real run
rm -rf results/dryrun
rm -f logs/dryrun_console.log
: > logs/run_cure.log || true

echo "[bootstrap] MAIN run (restart supervisor)"
ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT+1))
  push_status "main attempt $ATTEMPT running"
  python3 run_cure.py --mode main > "$CURE/logs/main_console.log" 2>&1 && break
  tail -30 "$CURE/logs/main_console.log"
  push_status "main attempt $ATTEMPT exited non-zero: $(tail -5 "$CURE/logs/main_console.log" | tr '\n' ' ' | tail -c 300)"
  echo "[bootstrap] main exited non-zero; retry in 60s"; sleep 60
done
push_status "ALL COMPLETE"
echo "[bootstrap] COMPLETE"; sleep infinity
