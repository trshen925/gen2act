#!/usr/bin/env bash
# Single-node launcher. train.batch_size is PER GPU.
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_distributed_train.sh CONFIG.yaml
#   NPROC_PER_NODE=4 bash scripts/run_distributed_train.sh CONFIG.yaml
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/root/miniconda3/envs/gen2act/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/envs/gen2act/bin/torchrun}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONFIG.yaml [train.py arguments...]" >&2
  exit 2
fi

CONFIG="$1"
shift

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  NGPU="$NPROC_PER_NODE"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  NGPU=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
else
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
fi

if [[ "${NGPU:-0}" -lt 1 ]]; then
  echo "no visible GPU found" >&2
  exit 1
fi

export http_proxy="${http_proxy:-http://192.168.48.17:18000}"
export https_proxy="${https_proxy:-http://192.168.48.17:18000}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

PREFLIGHT_ARGS=(--config "$CONFIG" --expected-gpus "$NGPU")
if [[ "${AUTO_INSTALL_DEPS:-1}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--install)
fi
"$PYTHON" scripts/check_train_env.py "${PREFLIGHT_ARGS[@]}"

echo "config=$CONFIG nproc_per_node=$NGPU per_gpu_batch=from-config"

if [[ "$NGPU" -eq 1 ]]; then
  exec "$PYTHON" scripts/train.py --config "$CONFIG" "$@"
fi

exec "$TORCHRUN" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$NGPU" \
  scripts/train.py --config "$CONFIG" "$@"
