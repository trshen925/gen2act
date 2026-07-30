#!/usr/bin/env bash
# Build the pi0.5 non-idle manifest for the locally rebuilt decompressed DROID.
#
# Same builder as scripts/build_raw_droid_pi05_manifest.sh, pointed at this
# machine's data root and at the KarlP annotation files staged under $STORAGE
# (episode_id_to_path.json is git-LFS in the public mirror, so it was fetched
# from huggingface.co/KarlP/droid separately).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STORAGE="${STORAGE:-/mnt/project/simvla/world_policy_storage/gen2act}"
DATA_ROOT="${DATA_ROOT:-$STORAGE/droid-decompressed-1.0.1}"
OUTPUT="${OUTPUT:-$STORAGE/artifacts/raw_droid_1_0_1_pi05_manifest.json}"
KARLP="${KARLP:-$STORAGE/karlp}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/home/sujiayi/miniconda3/envs/gen2act/bin/python}"
WORKERS="${WORKERS:-64}"

mkdir -p "$(dirname "$OUTPUT")"
exec "$PYTHON_BIN" scripts/build_raw_droid_pi05_manifest.py \
    --root "$DATA_ROOT" \
    --output "$OUTPUT" \
    --episode-id-to-path "$KARLP/episode_id_to_path.json" \
    --language-annotations "$KARLP/droid_language_annotations.json" \
    --workers "$WORKERS" \
    "$@"
