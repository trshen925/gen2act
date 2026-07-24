#!/usr/bin/env bash
set -euo pipefail

# C36 reuses C35's RGB/wrist inputs but deliberately skips FoundationStereo depth.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG="${CONFIG:-configs/droidexFULL_C36_no_frontdepth_diffuse_gripper_fulltrain.yaml}"
exec bash scripts/run_distributed_train.sh "$CONFIG" "$@"
