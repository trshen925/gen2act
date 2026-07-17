#!/usr/bin/env bash
# Formal C30 camera-corrected evaluation: 800 val windows, 32 ODE steps, 16 samples.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gen2act/bin/python}"
CONFIG="${CONFIG:-configs/droidexFULL_C30_camera_corrected_cont12_lr3e5_eval32.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/droidexFULL_C30_camera_corrected_cont12_lr3e5/c30_camera_corrected_ep11.pt}"
GPU_ID="${GPU_ID:-0}"
MAX_WINDOWS="${MAX_WINDOWS:-800}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LOG_FILE="${LOG_FILE:-outputs/droidexFULL_C30_camera_corrected_cont12_lr3e5/eval32_ep11_800w.log}"

for path in "$PYTHON_BIN" "$CONFIG" "$CHECKPOINT"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path not found: $path" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="${HF_HOME:-/mnt/pfs/share/pretrained_model/.cache/huggingface}"
mkdir -p "$(dirname "$LOG_FILE")"

echo "C30 corrected eval: checkpoint=$CHECKPOINT, max_windows=$MAX_WINDOWS, GPU_ID=$GPU_ID"
set -o pipefail
"$PYTHON_BIN" scripts/diagnose_actions.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split val \
  --max-windows "$MAX_WINDOWS" \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  "$@" 2>&1 | tee "$LOG_FILE"
