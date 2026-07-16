#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec bash scripts/run_distributed_train.sh \
  configs/droidexFULL_C30_cont12_lr3e5.yaml "$@"
