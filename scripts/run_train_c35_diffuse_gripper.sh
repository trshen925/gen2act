#!/usr/bin/env bash
set -euo pipefail

# C35 shares C34's data/cache prerequisites and changes only the action head.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CONFIG="${CONFIG:-configs/droidexFULL_C35_diffuse_gripper_fulltrain.yaml}"
exec bash scripts/run_train_c34_current_gripper.sh "$@"
