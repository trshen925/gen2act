#!/usr/bin/env bash
# Continue the completed 12-epoch C30 run for 12 additional epochs at 3e-5.
# Loads the original epoch-12 latest.pt weights into a fresh optimizer/scheduler
# and writes to outputs/droidexFULL_C30_cont12_lr3e5/.
set -euo pipefail

cd "$(dirname "$0")/.."
exec bash scripts/run_distributed_train.sh \
  configs/droidexFULL_C30_cont12_lr3e5.yaml "$@"
