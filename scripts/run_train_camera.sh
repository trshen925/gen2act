#!/usr/bin/env bash
# Launcher for the camera-frame action experiment on droid-ex with extrinsics.
#   - Auto-detects the number of GPUs on this machine and uses all of them.
#   - 1 GPU  -> plain python (single process).
#   - >1 GPU -> torchrun with one process per GPU (DDP). batch_size in the config is PER-GPU.
#   - Periodic inference on the held-out set is driven by train.eval_every_epochs in the config;
#     train vs inference loss curves are written to <output_dir>/loss_curve.png each epoch.
#
# Usage:
#   bash scripts/run_train.sh [CONFIG] [EXTRA_ARGS...]
#   CONFIG defaults to the droid-ex camera-frame chunk4 pose6d config below.
set -euo pipefail

cd "$(dirname "$0")/.."   # project root

PYTHON=/root/miniconda3/envs/gen2act/bin/python
TORCHRUN=/root/miniconda3/envs/gen2act/bin/torchrun
CONFIG="${1:-configs/droidex2000_future5_chunk4_pose6d_camera_regression_qpos_ft4dinov2_latent128_aug50.yaml}"
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
