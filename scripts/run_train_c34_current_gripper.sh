#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gen2act/bin/python}"
DEPTH_ROOT="${DEPTH_ROOT:-/mnt/pfs/data/shentingrui/droid-ex-3000-foundation-depth}"
MIN_GEOMETRY="${MIN_GEOMETRY:-39000}"

# C34 requires the newly added clips' wrist cache. The raw wrist videos are
# available for all 43,415 clips; run preprocess_wrist_frames.py beforehand.
COUNT=$("$PYTHON_BIN" -c 'from pathlib import Path; import sys; r=Path(sys.argv[1]); print(sum((r / f"{i:05d}" / "patch_geometry_v1.npy").exists() for i in range(43415)))' "$DEPTH_ROOT")
if [[ "$COUNT" -lt "$MIN_GEOMETRY" ]]; then
  echo "Only $COUNT/43415 patch-geometry files found; preprocess the new clips first." >&2
  exit 1
fi
echo "C34 patch geometry files: $COUNT / 43415"

WRIST_CHECK=$("$PYTHON_BIN" -c '
from pathlib import Path
import json, sys
r = Path(sys.argv[1]); ok = 0; bad = []
for i in range(43415):
    clip = r / f"{i:05d}"
    meta = clip / "meta.json"; frames = clip / "wrist_frames"
    try:
        n = int(json.loads(meta.read_text())["num_frames"])
    except Exception:
        bad.append(f"{i:05d}:meta"); continue
    if n > 0 and (frames / "000000.jpg").exists() and (frames / f"{n - 1:06d}.jpg").exists():
        ok += 1
    else:
        bad.append(f"{i:05d}")
print(ok)
if bad:
    print(" ".join(bad[:10]), file=sys.stderr)
' /mnt/pfs/data/shentingrui/droid-ex-3000-out)
WRIST_COUNT=$(tail -n 1 <<< "$WRIST_CHECK")
if [[ "$WRIST_COUNT" -ne 43415 ]]; then
  echo "Only $WRIST_COUNT/43415 complete wrist frame caches found; run scripts/preprocess_wrist_frames.py for START_ID=35696 END_ID=43415 first." >&2
  exit 1
fi
echo "C34 wrist frame caches: $WRIST_COUNT / 43415"

exec bash scripts/run_distributed_train.sh \
  configs/droidexFULL_C34_current_gripper_fulltrain.yaml "$@"
