#!/usr/bin/env bash
# Formal offline evaluation for the deployable C28 epoch-17 checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gen2act/bin/python}"
CONFIG="${CONFIG:-configs/droidexFULL_C28_deploy_eval32.yaml}"
CHECKPOINT="${CHECKPOINT:-outputs/droidexFULL_C28_bigdit_scratch/latest.pt}"
GPU_ID="${GPU_ID:-0}"
MAX_WINDOWS="${MAX_WINDOWS:-800}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LOG_FILE="${LOG_FILE:-outputs/droidexFULL_C28_bigdit_scratch/eval32_latest_ep17_800w.log}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="${HF_HOME:-/mnt/pfs/share/pretrained_model/.cache/huggingface}"
mkdir -p "$(dirname "$LOG_FILE")"

echo "C28 eval: checkpoint=$CHECKPOINT, max_windows=$MAX_WINDOWS, GPU_ID=$GPU_ID"
set -o pipefail
"$PYTHON_BIN" scripts/diagnose_actions.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split val \
  --max-windows "$MAX_WINDOWS" \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  "$@" 2>&1 | tee "$LOG_FILE"
