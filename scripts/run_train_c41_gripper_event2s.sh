#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CONFIG="${CONFIG:-configs/droidFULL_C41_gripper_event2s_short4ep.yaml}"
export RUN_LABEL="${RUN_LABEL:-C41}"
export INDEX_JSON="${INDEX_JSON:-artifacts/c41_raw_droid_gripper_target_pre3s_post1s_window_index.json}"
export WARM_START="${WARM_START:-outputs/droidFULL_C40_jointvelocity_pi05_letterbox_fulltrain/step_000012000.pt}"
export LOG_DIR="${LOG_DIR:-outputs/droidFULL_C41_gripper_event2s_short4ep/logs}"

exec bash scripts/run_train_c40_raw_droid_pi05.sh "$@"
