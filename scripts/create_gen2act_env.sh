#!/usr/bin/env bash
set -euo pipefail

# PyTorch is installed separately because its CUDA build must match the target
# machine's driver. PyPI chooses the platform-compatible default; set
# TORCH_INDEX_URL explicitly when a site provides a CUDA- or CPU-specific index.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/environment.gen2act.yaml}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$ROOT/requirements.gen2act.txt}"
ENV_NAME="${1:-gen2act}"
CONDA_BIN="${CONDA_BIN:-conda}"

if ! command -v "$CONDA_BIN" >/dev/null 2>&1; then
    echo "conda executable not found: $CONDA_BIN" >&2
    exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
    echo "environment file not found: $ENV_FILE" >&2
    exit 1
fi
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "requirements file not found: $REQUIREMENTS_FILE" >&2
    exit 1
fi

"$CONDA_BIN" env create --name "$ENV_NAME" --file "$ENV_FILE"
run_in_env() {
    env -u PIP_CONSTRAINT "$CONDA_BIN" run --name "$ENV_NAME" "$@"
}

run_in_env python -m pip install --upgrade pip

TORCH_ARGS=(torch torchvision)
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    TORCH_ARGS+=(--index-url "$TORCH_INDEX_URL")
fi
run_in_env python -m pip install "${TORCH_ARGS[@]}"
run_in_env python -m pip install -r "$REQUIREMENTS_FILE"
run_in_env python -m pip install -e "$ROOT"

run_in_env python -c '
import torch
print(f"torch={torch.__version__} bundled_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()} gpu_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"gpu0={torch.cuda.get_device_name(0)} bf16={torch.cuda.is_bf16_supported()}")
'

echo "Created environment: $ENV_NAME"
echo "Run: $CONDA_BIN run --name $ENV_NAME python scripts/check_train_env.py --config configs/droidFULL_C40_jointvelocity_pi05_letterbox_fulltrain.yaml --expected-gpus <count>"
