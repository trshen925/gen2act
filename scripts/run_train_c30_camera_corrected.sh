#!/usr/bin/env bash
# Continue C30 total-epoch-24 on camera-corrected C28 data semantics.
set -euo pipefail

cd "$(dirname "$0")/.."
exec bash scripts/run_distributed_train.sh \
  configs/droidexFULL_C30_camera_corrected_cont12_lr3e5.yaml "$@"
