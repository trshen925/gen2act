#!/usr/bin/env bash
# Launch the local from-scratch C40 run on this cluster's 8x H100 node.
#
# Adapted from scripts/run_train_c40_raw_droid_pi05.sh. The differences are the
# ones the migration forces: local data/manifest/index paths, no C39 warm-start
# checkpoint to require, this machine's conda env and proxy, and a W&B entity
# taken from the API key's own default instead of the original author's team.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STORAGE="${STORAGE:-/mnt/project/simvla/world_policy_storage/gen2act}"
CONFIG="${CONFIG:-configs/local_C40_jointvelocity_pi05_letterbox_fromscratch.yaml}"
MANIFEST="${MANIFEST:-$STORAGE/artifacts/raw_droid_1_0_1_pi05_manifest.json}"
INDEX_JSON="${INDEX_JSON:-$STORAGE/artifacts/c40_raw_droid_jointvelocity_window_index.json}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/home/sujiayi/miniconda3/envs/gen2act/bin/python}"
WANDB_CREDENTIAL="${WANDB_CREDENTIAL:-/mnt/home/sujiayi/.credentials/wandb_api.txt}"
LOG_DIR="${LOG_DIR:-$STORAGE/outputs/local_C40_jointvelocity_pi05_letterbox_fromscratch/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$(date -u +%Y%m%d_%H%M%S).log}"

# This cluster's proxy. run_distributed_train.sh only fills in its own default
# when these are unset, and that default is not reachable from here.
export http_proxy="${http_proxy:-http://galbot:sK0aZ5bZ9v@10.119.176.202:3128}"
export https_proxy="${https_proxy:-http://galbot:sK0aZ5bZ9v@10.119.176.202:3128}"
# The login shells here export a SOCKS all_proxy that httpx cannot use without
# socksio, and an HF_ENDPOINT mirror that does not serve the timm DINOv2 repo.
# Both break backbone download; huggingface.co works through the HTTP proxy.
unset all_proxy ALL_PROXY HF_ENDPOINT

# GPUs 4-7 of this node are reserved for other work; run_distributed_train.sh
# derives nproc_per_node from this list. Override CUDA_VISIBLE_DEVICES to change.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "filter manifest not found: $MANIFEST" >&2
    echo "build it with: bash scripts/build_local_c40_manifest.sh" >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi
# Unlike the original launcher, the window index is not required up front: rank 0
# builds it from the manifest on the first run and the other ranks wait for the
# atomic rename. Only report which path will be used.
if [[ ! -f "$INDEX_JSON" ]]; then
    echo "note: window index absent; rank 0 will build $INDEX_JSON on this run"
fi
mkdir -p "$(dirname "$INDEX_JSON")"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "gen2act Python not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" && ! -s "$WANDB_CREDENTIAL" ]]; then
    echo "W&B API key is missing: set WANDB_API_KEY or create $WANDB_CREDENTIAL" >&2
    echo "To train without W&B instead, set WANDB_MODE=offline." >&2
    [[ "${WANDB_MODE:-}" == "offline" ]] || exit 1
elif [[ -z "${WANDB_API_KEY:-}" ]]; then
    WANDB_API_KEY="$(<"$WANDB_CREDENTIAL")"
    export WANDB_API_KEY
fi

# Confirm the key resolves to an account before burning a GPU allocation. The
# config leaves entity null, so the key's own default entity is used.
if [[ "${WANDB_MODE:-online}" != "offline" ]]; then
    if ! "$PYTHON_BIN" - <<'PY'
import wandb
api = wandb.Api()
account = getattr(api.viewer, "username", None) or api.default_entity
print(f"W&B authentication OK: account={account} entity={api.default_entity}")
PY
    then
        echo "W&B authentication failed; training was not started." >&2
        echo "Fix the key, or rerun with WANDB_MODE=offline." >&2
        exit 1
    fi
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || [[ "$(nvidia-smi -L 2>/dev/null | wc -l)" -lt 1 ]]; then
    echo "no visible NVIDIA GPU; run inside a GPU allocation" >&2
    exit 1
fi
if [[ "${C40_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "C40 local preflight OK; training was not started."
    exit 0
fi

mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
echo "C40 local from-scratch training"
echo "  config:   $CONFIG"
echo "  manifest: $MANIFEST"
echo "  index:    $INDEX_JSON"
echo "  gpus:     $(nvidia-smi -L | wc -l)"
echo "  log:      $LOG_FILE"

set +e
PYTHON="$PYTHON_BIN" \
TORCHRUN="$(dirname "$PYTHON_BIN")/torchrun" \
    bash scripts/run_distributed_train.sh "$CONFIG" "$@" 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
if [[ "$status" -ne 0 ]]; then
    echo "training exited with status $status; see $LOG_FILE" >&2
    exit "$status"
fi
echo "training completed; log: $LOG_FILE"
