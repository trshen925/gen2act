#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python scripts/preprocess_wrist_frames.py \
  --root /mnt/pfs/data/shentingrui/droid-ex-3000-out \
  --raw-root /mnt/pfs/data/fenghaoran/droid/decompressed/1.0.1 \
  --start-id "${START_ID:-0}" \
  --end-id "${END_ID:-35696}" \
  --workers "${WORKERS:-16}" \
  "$@"
