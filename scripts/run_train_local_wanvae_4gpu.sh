#!/usr/bin/env bash
# Train the localized Wan-VAE baseline on the node's rear four GPUs only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STORAGE="${STORAGE:-/mnt/project/simvla/world_policy_storage/gen2act}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/home/sujiayi/miniconda3/envs/gen2act/bin/python}"
export CONFIG="${CONFIG:-configs/local_wanvae_full_droid_4gpu.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export WAN21_ROOT="${WAN21_ROOT:-$STORAGE/third_party/Wan2.1}"
export WAN_VAE_CHECKPOINT="${WAN_VAE_CHECKPOINT:-$STORAGE/pretrained/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth}"
export PYTHONUNBUFFERED=1

# Keep cluster credentials out of version control. The node may already export
# http_proxy/https_proxy; otherwise callers can provide one URL through
# GEN2ACT_HTTP_PROXY without changing this launcher.
if [[ -n "${GEN2ACT_HTTP_PROXY:-}" ]]; then
    export http_proxy="$GEN2ACT_HTTP_PROXY"
    export https_proxy="$GEN2ACT_HTTP_PROXY"
fi
unset all_proxy ALL_PROXY HF_ENDPOINT

for required in \
    "$CONFIG" \
    "$WAN21_ROOT/wan/modules/vae.py" \
    "$WAN_VAE_CHECKPOINT" \
    "$STORAGE/droid-decompressed-1.0.1" \
    "$STORAGE/artifacts/raw_droid_1_0_1_pi05_manifest.json" \
    "$STORAGE/artifacts/c40_raw_droid_jointvelocity_window_index.json"; do
    if [[ ! -e "$required" ]]; then
        echo "required local asset is missing: $required" >&2
        exit 1
    fi
done

PYTHON="$PYTHON_BIN" \
TORCHRUN="$(dirname "$PYTHON_BIN")/torchrun" \
AUTO_INSTALL_DEPS=0 \
    exec bash scripts/run_distributed_train.sh "$CONFIG" "$@"
