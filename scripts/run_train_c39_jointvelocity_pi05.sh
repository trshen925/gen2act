#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/droidexFULL_C39_jointvelocity_pi05_fulltrain.yaml}"
INDEX_JSON="${INDEX_JSON:-artifacts/c39_jointvelocity_window_index.json}"
WARM_START="${WARM_START:-outputs/droidexFULL_C38_singlecurrent_wristletterbox_fronttranslate_fulltrain/best_model.pt}"
LOG_DIR="outputs/droidexFULL_C39_jointvelocity_pi05_fulltrain/logs"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$(date -u +%Y%m%d_%H%M%S).log}"

for required in "$CONFIG" "$INDEX_JSON" "$WARM_START"; do
    if [[ ! -f "$required" ]]; then
        echo "required file not found: $ROOT/$required" >&2
        exit 1
    fi
done

if ! command -v nvidia-smi >/dev/null 2>&1 || [[ "$(nvidia-smi -L 2>/dev/null | wc -l)" -lt 1 ]]; then
    echo "no visible NVIDIA GPU; set CUDA_VISIBLE_DEVICES inside a GPU allocation" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "C39 training"
echo "  config:  $ROOT/$CONFIG"
echo "  index:   $ROOT/$INDEX_JSON"
echo "  init:    $ROOT/$WARM_START"
echo "  GPUs:    ${CUDA_VISIBLE_DEVICES:-all visible GPUs}"
echo "  log:     $ROOT/$LOG_FILE"
echo
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || true
echo

export PYTHONUNBUFFERED=1

set +e
bash scripts/run_distributed_train.sh "$CONFIG" "$@" 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]]; then
    echo "C39 training exited with status $status; see $ROOT/$LOG_FILE" >&2
    exit "$status"
fi

echo "C39 training completed; log: $ROOT/$LOG_FILE"
