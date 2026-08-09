#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export RUN_LABEL="${RUN_LABEL:-C42}"
export CONFIG="${CONFIG:-configs/droidFULL_C42_c39_subset_step12k_short4ep.yaml}"
# The C42 mapping is self-contained; /dev/null satisfies the shared C40
# wrapper's legacy preflight while RawDroidDataset skips the optional manifest.
export MANIFEST="${MANIFEST:-/dev/null}"
export INDEX_JSON="${INDEX_JSON:-metadata/c42_c39_to_raw_droid_mapping.json}"
export WARM_START="${WARM_START:-outputs/droidFULL_C40_jointvelocity_pi05_letterbox_fulltrain/step_000012000.pt}"
export LOG_DIR="${LOG_DIR:-outputs/droidFULL_C42_c39_subset_step12k_short4ep/logs}"

exec bash scripts/run_train_c40_raw_droid_pi05.sh "$@"
