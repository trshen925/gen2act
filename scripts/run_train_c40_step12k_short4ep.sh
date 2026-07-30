#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CONFIG="${CONFIG:-configs/droidFULL_C40_step12k_short4ep.yaml}"
export WARM_START="${WARM_START:-outputs/droidFULL_C40_jointvelocity_pi05_letterbox_fulltrain/step_000012000.pt}"
export LOG_DIR="${LOG_DIR:-outputs/droidFULL_C40_step12k_short4ep/logs}"

exec bash scripts/run_train_c40_raw_droid_pi05.sh "$@"
