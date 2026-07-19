#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gen2act/bin/python}"
exec "$PYTHON_BIN" scripts/preprocess_c32_patch_geometry.py "$@"

