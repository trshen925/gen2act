#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DEPTH_ROOT="${DEPTH_ROOT:-/mnt/pfs/data/shentingrui/droid-ex-3000-foundation-depth}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gen2act/bin/python}"
MIN_GEOMETRY="${MIN_GEOMETRY:-32000}"
COUNT=$("$PYTHON_BIN" -c \
  'from pathlib import Path; import sys; root=Path(sys.argv[1]); print(sum((root / f"{i:05d}" / "patch_geometry_v1.npy").exists() for i in range(35696)))' \
  "$DEPTH_ROOT")
if [[ "$COUNT" -lt "$MIN_GEOMETRY" ]]; then
  echo "Only $COUNT patch-geometry files found in frozen IDs 00000..35695; expected at least $MIN_GEOMETRY." >&2
  echo "Run: bash scripts/run_preprocess_c32_patch_geometry.sh" >&2
  exit 1
fi
echo "C32 frozen-set patch geometry files: $COUNT / 35696"

exec bash scripts/run_distributed_train.sh \
  configs/droidexFULL_C32_wrist_frontdepth_cont12_lr3e5.yaml "$@"
