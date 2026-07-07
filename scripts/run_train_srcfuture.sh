#!/usr/bin/env bash
# Launcher for Exp 3 — the leak-exploitability test (source_sampling: future_window).
# Same machinery as run_train.sh (auto-detect GPUs; 1 GPU -> python, >1 -> torchrun/DDP),
# but defaults to the srcfuture config.
#
# Usage:
#   bash scripts/run_train_srcfuture.sh [CONFIG] [EXTRA_ARGS...]
#   CONFIG defaults to the srcfuture config below.
set -euo pipefail

cd "$(dirname "$0")/.."   # project root

PYTHON=/root/miniconda3/envs/gen2act/bin/python
TORCHRUN=/root/miniconda3/envs/gen2act/bin/torchrun
CONFIG="${1:-configs/droid2000new_future5_chunk4_pose6d_regression_qpos_ft4dinov2_latent128_srcfuture.yaml}"
shift || true

# Proxy + allow HF downloads (DINOv2 weights). Comment out if running fully offline with cached weights.
export http_proxy="${http_proxy:-http://192.168.48.17:18000}"
export https_proxy="${https_proxy:-http://192.168.48.17:18000}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

# Count GPUs (respect CUDA_VISIBLE_DEVICES if the user set it).
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NGPU=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NGPU=${NGPU:-0}

echo "=============================================="
echo " config : $CONFIG"
echo " GPUs   : $NGPU"
echo " extra  : $*"
echo "=============================================="

if [[ "$NGPU" -le 1 ]]; then
  exec "$PYTHON" scripts/train.py --config "$CONFIG" "$@"
else
  exec "$TORCHRUN" --standalone --nnodes=1 --nproc_per_node="$NGPU" \
    scripts/train.py --config "$CONFIG" "$@"
fi
