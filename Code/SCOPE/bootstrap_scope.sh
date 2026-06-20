#!/usr/bin/env bash
# bootstrap_scope.sh -- single-launch VM entrypoint for the SCOPE study. Reuses the proven
# CURE environment (torch 2.5.1 cu124 + precompiled flash-attn + TransformerLens/nnsight),
# writes a gitignored Code/CURE/.env from injected secrets, then runs the SCOPE dry check
# (two pairs per model) followed by the full SCOPE run, with 15-minute GitHub checkpoints
# to Cure_Audit_Benchmark. Setup survives a dry/main failure (pull+retry), so a fix needs
# no redeploy. One launch produces every scope_* artifact in Code/CURE/results/.
set -uo pipefail

WORK=/workspace
REPO="$WORK/Cure_Audit_Benchmark"
CURE="$REPO/Code/CURE"
SCOPE="$REPO/Code/SCOPE"
export HF_HOME="$WORK/hf"
export DEBIAN_FRONTEND=noninteractive
export PIP="pip3 install --break-system-packages"

echo "[boot] system deps"
apt-get update -y
apt-get install -y --no-install-recommends git wget ca-certificates python3 python3-pip build-essential
echo "[boot] python: $(python3 --version)"
mkdir -p "$CURE/logs" "$CURE/results"
cd "$CURE"

echo "[boot] write .env from injected secrets (gitignored, local only)"
python3 - <<'PY'
import os
real = ["HUGGINGFACE_TOKEN", "Github_Classic_Token", "RANDOM_SEED", "CURE_JUDGE_PROVIDER",
        "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4",
        "DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2", "DEEPSEEK_API_BASE_URL", "DEEPSEEK_JUDGE_MODEL_NAME",
        "MISTRAL_API_KEY1", "MISTRAL_API_KEY2",
        "OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2", "OPENROUTER_API_BASE_URL"]
dummy = ["AWS_ACCESS_KEY", "AWS_SECRET_KEY"]
with open(".env", "w") as f:
    for k in real:
        v = os.environ.get(k, "")
        if v:
            f.write(f"{k}={v}\n")
    for k in dummy:
        f.write(f"{k}=unused-by-cure\n")
print("wrote .env")
PY

echo "[boot] configure git -> Cure_Audit_Benchmark"
git config --global --add safe.directory "$REPO"
git -C "$REPO" config user.name "SCOPE Runner"
git -C "$REPO" config user.email "koushikdeb2009@gmail.com"
git -C "$REPO" config pull.rebase true
if [ -n "${Github_Classic_Token:-}" ]; then
  git -C "$REPO" remote set-url origin "https://${Github_Classic_Token}@github.com/DevDaring/Cure_Audit_Benchmark.git"
fi

push_status() {
  mkdir -p "$CURE/results"
  printf '%s @ %s\n' "$1" "$(date -u)" > "$CURE/results/SCOPE_BOOT_STATUS.txt"
  git -C "$REPO" add -f Code/CURE/results/SCOPE_BOOT_STATUS.txt >/dev/null 2>&1
  git -C "$REPO" commit -q -m "scope-boot: $1" >/dev/null 2>&1
  git -C "$REPO" pull --rebase -q origin main >/dev/null 2>&1
  git -C "$REPO" push -q origin main >/dev/null 2>&1 && echo "[boot] status: $1"
}
push_status "container started ($(nvidia-smi -L 2>/dev/null | head -1))"

echo "[boot] torch 2.5.1 (cu124) + deps + TL/nnsight"
$PIP --upgrade pip
$PIP torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
printf 'torch==2.5.1\n' > /tmp/torch_pin.txt
$PIP -r requirements_cure.txt -c /tmp/torch_pin.txt --extra-index-url https://download.pytorch.org/whl/cu124
$PIP --no-deps transformer_lens==2.18.0
TORCH_V=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ "${TORCH_V#2.5.1}" = "$TORCH_V" ]; then
  $PIP --force-reinstall --no-deps torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
fi

DIAG="$CURE/logs/scope_diag.txt"; : > "$DIAG"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | tee -a "$DIAG" || true
python3 - >> "$DIAG" 2>&1 <<'PY'
import torch, sys
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda,
      "| avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "| cap", torch.cuda.get_device_capability(0))
PY
cat "$DIAG"

echo "[boot] precompiled flash-attention (try both cxx11abi)"
FA_BASE="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3"
for ABI in TRUE FALSE; do
  $PIP --no-deps --force-reinstall "$FA_BASE/flash_attn-2.8.3+cu12torch2.5cxx11abi${ABI}-cp312-cp312-linux_x86_64.whl" >> "$DIAG" 2>&1
  if python3 -c "import flash_attn" >/dev/null 2>&1; then echo "[boot] flash-attn OK (cxx11abi${ABI})"; break; fi
done
python3 -c "import transformer_lens, nnsight, flash_attn" >> "$DIAG" 2>&1; VRC=$?
tail -25 "$DIAG"
if [ "$VRC" -ne 0 ]; then push_status "FATAL: lib import failed"; echo "[boot] FATAL libs"; sleep infinity; fi

echo "[boot] download OSM models"
python3 - <<'PY'
import config_cure as C
from huggingface_hub import snapshot_download
for m in C.OSM_MODELS:
    print("downloading", m["hf_id"], flush=True)
    snapshot_download(m["hf_id"], token=C.HUGGINGFACE_TOKEN)
print("models present")
PY

cd "$SCOPE"
push_status "setup complete; SCOPE dry-run (2 instances)"
echo "[boot] DRY RUN (run_scope --mode dry; pull+retry on failure)"
while true; do
  python3 run_scope.py --mode dry > "$CURE/logs/scope_dry_console.log" 2>&1; DRY_RC=$?
  tail -45 "$CURE/logs/scope_dry_console.log"
  [ "$DRY_RC" -eq 0 ] && break
  TS=$(date +%s); cp "$CURE/logs/scope_dry_console.log" "$CURE/results/SCOPE_DRYFAIL_${TS}.txt"
  git -C "$REPO" add -f "Code/CURE/results/SCOPE_DRYFAIL_${TS}.txt" >/dev/null 2>&1
  push_status "dry rc=$DRY_RC FAILED (results/SCOPE_DRYFAIL_${TS}.txt); pull+retry in 60s"
  sleep 60; git -C "$REPO" pull --rebase -q origin main >/dev/null 2>&1
done
push_status "dry rc=0 PASSED; cleaning test artifacts"
rm -rf "$CURE/results/dryrun"
rm -f "$CURE/logs/scope_dry_console.log" "$CURE/results/SCOPE_DRYFAIL_"*.txt
: > "$CURE/logs/run_scope.log" || true

echo "[boot] MAIN run (restart supervisor; 15-min checkpoints inside run_scope)"
ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT+1))
  push_status "scope-main attempt $ATTEMPT running"
  python3 run_scope.py --mode main > "$CURE/logs/scope_main_console.log" 2>&1 && break
  tail -30 "$CURE/logs/scope_main_console.log"
  push_status "scope-main attempt $ATTEMPT non-zero: $(tail -4 "$CURE/logs/scope_main_console.log" | tr '\n' ' ' | tail -c 260)"
  sleep 60; git -C "$REPO" pull --rebase -q origin main >/dev/null 2>&1
done
push_status "SCOPE COMPLETE"
echo "[boot] COMPLETE"; sleep infinity
