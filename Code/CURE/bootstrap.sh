#!/usr/bin/env bash
# bootstrap.sh -- VM entrypoint for the CURE run on a rented GPU (Ubuntu 24.04 / py3.12).
# Secrets arrive as container env vars injected by the SDL; they are written to a local
# .env (gitignored) and never committed. Flash-attention is installed from a precompiled
# wheel and verified before the run. No virtual environment: installs are global.
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

echo "[bootstrap] write .env from injected secrets (gitignored, local only)"
python3 - <<'PY'
import os
real = ["HUGGINGFACE_TOKEN", "Github_Classic_Token", "RANDOM_SEED",
        "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4",
        "DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2", "DEEPSEEK_API_BASE_URL", "DEEPSEEK_JUDGE_MODEL_NAME",
        "MISTRAL_API_KEY1", "MISTRAL_API_KEY2",
        "OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2", "OPENROUTER_API_BASE_URL"]
# the audit config _require()s a few keys CURE never calls; dummy values satisfy it.
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

echo "[bootstrap] configure git"
git config --global --add safe.directory "$REPO"
git -C "$REPO" config user.name "CURE Runner"
git -C "$REPO" config user.email "koushikdeb2009@gmail.com"
git -C "$REPO" config pull.rebase true
if [ -n "${Github_Classic_Token:-}" ]; then
  git -C "$REPO" remote set-url origin "https://${Github_Classic_Token}@github.com/DevDaring/Cure_Audit_Benchmark.git"
fi

echo "[bootstrap] torch 2.5.1 (cu124)"
$PIP --upgrade pip
$PIP torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
echo "[bootstrap] python deps"
$PIP -r requirements_cure.txt
echo "[bootstrap] transformer_lens 2.18.0 (--no-deps so it keeps torch 2.5.1 / transformers 4.50.3)"
$PIP --no-deps transformer_lens==2.18.0

echo "[bootstrap] precompiled flash-attention (Ubuntu 24.04 / py3.12 / torch2.5 / cu12)"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
wget -q "$FA_URL" -O /tmp/fa.whl && $PIP --no-deps /tmp/fa.whl \
  && echo "[bootstrap] flash-attn installed" \
  || echo "[bootstrap] WARN flash-attn wheel install failed"

echo "[bootstrap] verify patching libs + flash-attn import"
python3 -c "import transformer_lens, nnsight, flash_attn, importlib.metadata as m; print('TL', m.version('transformer_lens'), '| nnsight', m.version('nnsight'), '| flash_attn', m.version('flash_attn'))" || {
  echo "[bootstrap] FATAL: a required library failed to import -- container kept alive"; sleep infinity; }

echo "[bootstrap] download OSM models (dataset ships in Code/audit)"
python3 - <<'PY'
import config_cure as C
from huggingface_hub import snapshot_download
for m in C.OSM_MODELS:
    print("downloading", m["hf_id"], flush=True)
    snapshot_download(m["hf_id"], token=C.HUGGINGFACE_TOKEN)
print("models present")
PY

echo "[bootstrap] DRY RUN"
python3 run_cure.py --mode dry > "$CURE/logs/dryrun_console.log" 2>&1; DRY_RC=$?
tail -40 "$CURE/logs/dryrun_console.log"
echo "[bootstrap] dry-run rc=$DRY_RC"
if [ "$DRY_RC" -ne 0 ]; then
  echo "[bootstrap] DRY FAILED -- container kept alive for inspection"; sleep infinity
fi
rm -rf results/dryrun; : > logs/run_cure.log || true

echo "[bootstrap] MAIN run (restart supervisor)"
ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT+1))
  echo "[bootstrap] main attempt $ATTEMPT"
  python3 run_cure.py --mode main > "$CURE/logs/main_console.log" 2>&1 && break
  tail -30 "$CURE/logs/main_console.log"
  echo "[bootstrap] main exited non-zero; retry in 60s"; sleep 60
done
echo "[bootstrap] COMPLETE"; sleep infinity
