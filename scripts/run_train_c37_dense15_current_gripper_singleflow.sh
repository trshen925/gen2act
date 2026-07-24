#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG="${CONFIG:-configs/droidexFULL_C37_dense15_current_gripper_singleflow_fulltrain.yaml}"
PYTHON_BIN="${PYTHON:-/root/miniconda3/envs/gen2act/bin/python}"
# The verified training environment already has all dependencies. Avoid a
# network-dependent pip upgrade during launch (which can fail on a transient proxy error).
export AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-0}"

# Some immutable cluster image revisions omit pyarrow even though it is needed
# by the DROID parquet adapter. Bootstrap only that known dependency from the
# PFS-mounted wheel: this is offline, deterministic, and runs before torchrun.
if ! "$PYTHON_BIN" -c 'import pyarrow' >/dev/null 2>&1; then
  PYARROW_WHEEL="$ROOT/third_party/wheels/pyarrow-19.0.1-cp311-cp311-manylinux_2_28_x86_64.whl"
  if [[ ! -f "$PYARROW_WHEEL" ]]; then
    echo "Missing offline pyarrow wheel: $PYARROW_WHEEL" >&2
    exit 1
  fi
  echo "49a3aecb62c1be1d822f8bf629226d4a96418228a42f5b40835c1f10d42e4db6  $PYARROW_WHEEL" | sha256sum --check --status || {
    echo "Offline pyarrow wheel checksum mismatch: $PYARROW_WHEEL" >&2
    exit 1
  }
  echo "[c37 bootstrap] installing pyarrow from local wheel (no network)"
  "$PYTHON_BIN" -m pip install --no-index --no-deps "$PYARROW_WHEEL"
fi
"$PYTHON_BIN" -c 'import pyarrow; print("[c37 bootstrap] pyarrow=" + pyarrow.__version__)'

exec bash scripts/run_distributed_train.sh "$CONFIG" "$@"
