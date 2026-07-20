#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# C33 reuses C32's precomputed current-frame geometry. The trainer auto-detects
# all visible GPUs; train.batch_size is per GPU. The config resumes the saved
# C33 epoch-1 checkpoint with its optimizer state and continues at epoch 2.
DEPTH_ROOT="${DEPTH_ROOT:-/mnt/pfs/data/shentingrui/droid-ex-3000-foundation-depth}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gen2act/bin/python}"
MIN_GEOMETRY="${MIN_GEOMETRY:-32000}"
RESUME_CHECKPOINT="$ROOT/outputs/droidexFULL_C33_front_wrist_history_cont10_lr3e5/latest.pt"
if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "C33 resume checkpoint not found: $RESUME_CHECKPOINT" >&2
  exit 1
fi
"$PYTHON_BIN" -c \
  'import sys, torch; p=sys.argv[1]; c=torch.load(p, map_location="cpu", weights_only=False); e=int(c.get("epoch", -1)); assert 1 <= e < 10, f"expected an incomplete C33 checkpoint at epoch 1..9, found epoch {e}: {p}"; assert c.get("optimizer_state_dict") is not None, f"optimizer state missing: {p}"; print(f"C33 full resume checkpoint: {p} (completed epoch {e}; continuing at epoch {e + 1})")' \
  "$RESUME_CHECKPOINT"
COUNT=$("$PYTHON_BIN" -c \
  'from pathlib import Path; import sys; root=Path(sys.argv[1]); print(sum((root / f"{i:05d}" / "patch_geometry_v1.npy").exists() for i in range(35696)))' \
  "$DEPTH_ROOT")
if [[ "$COUNT" -lt "$MIN_GEOMETRY" ]]; then
  echo "Only $COUNT patch-geometry files found in frozen IDs 00000..35695; expected at least $MIN_GEOMETRY." >&2
  exit 1
fi
echo "C33 frozen-set patch geometry files: $COUNT / 35696"

exec bash scripts/run_distributed_train.sh \
  configs/droidexFULL_C33_front_wrist_history_cont10_lr3e5.yaml "$@"
