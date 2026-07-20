#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gen2act/bin/python}"
CONFIG="${CONFIG:-configs/droidexFULL_C33_front_wrist_history_eval32.yaml}"
CHECKPOINT="${1:-${CHECKPOINT:-outputs/droidexFULL_C33_front_wrist_history_cont10_lr3e5/latest.pt}}"
if [[ $# -gt 0 ]]; then shift; fi
GPU_ID="${GPU_ID:-0}"
MAX_WINDOWS="${MAX_WINDOWS:-800}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LOG_FILE="${LOG_FILE:-outputs/droidexFULL_C33_front_wrist_history_cont10_lr3e5/eval32_800w_seed0.log}"

for path in "$PYTHON_BIN" "$CONFIG" "$CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "Required path not found: $path" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME="${HF_HOME:-/mnt/pfs/share/pretrained_model/.cache/huggingface}"
mkdir -p "$(dirname "$LOG_FILE")"

"$PYTHON_BIN" scripts/diagnose_actions.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --split val \
  --max-windows "$MAX_WINDOWS" --batch-size "$BATCH_SIZE" \
  --device cuda --seed 0 "$@" 2>&1 | tee "$LOG_FILE"

