#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/mnt/pfs/data/fenghaoran/droid/decompressed/1.0.1}"
OUTPUT="${OUTPUT:-artifacts/raw_droid_1_0_1_pi05_manifest.json}"
WORKERS="${WORKERS:-16}"

exec /root/miniconda3/envs/gen2act/bin/python scripts/build_raw_droid_pi05_manifest.py \
    --root "$DATA_ROOT" \
    --output "$OUTPUT" \
    --workers "$WORKERS" \
    "$@"

