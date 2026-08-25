#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_LABEL="${RUN_LABEL:-C40}"
CONFIG="${CONFIG:-configs/droidFULL_C40_jointvelocity_pi05_letterbox_fulltrain.yaml}"
MANIFEST="${MANIFEST:-artifacts/raw_droid_1_0_1_pi05_manifest.json}"
INDEX_JSON="${INDEX_JSON:-artifacts/c40_raw_droid_jointvelocity_window_index.json}"
WARM_START="${WARM_START:-outputs/droidexFULL_C39_jointvelocity_pi05_fulltrain/latest_model.pt}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gen2act/bin/python}"
WANDB_CREDENTIAL="${WANDB_CREDENTIAL:-/mnt/pfs/users/shentingrui/.credentials/wandb_api_trshen.txt}"
WANDB_ENTITY="${WANDB_ENTITY:-trshen925-peking-university}"
WANDB_VENDOR_DIR="${WANDB_VENDOR_DIR:-/mnt/pfs/users/shentingrui/.cache/gen2act/wandb-0.20.1-py311}"
LOG_DIR="${LOG_DIR:-outputs/droidFULL_C40_jointvelocity_pi05_letterbox_fulltrain/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/train_$(date -u +%Y%m%d_%H%M%S).log}"

if [[ "$MANIFEST" != "/dev/null" && ! -f "$MANIFEST" ]]; then
    echo "filter manifest not found: $ROOT/$MANIFEST" >&2
    echo "build it first with: bash scripts/build_raw_droid_pi05_manifest.sh" >&2
    exit 1
fi
for required in "$CONFIG" "$WARM_START"; do
    if [[ ! -f "$required" ]]; then
        echo "required file not found: $ROOT/$required" >&2
        exit 1
    fi
done
if [[ ! -f "$INDEX_JSON" ]]; then
    echo "sampling index not found; rank 0 will build it: $ROOT/$INDEX_JSON"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "gen2act Python not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c 'import pytest' >/dev/null 2>&1; then
    echo "pytest is missing; installing it into $PYTHON_BIN"
    "$PYTHON_BIN" -m pip install --no-deps \
        pytest==8.1.1 iniconfig==2.0.0 pluggy==1.5.0
fi

# New environments install W&B from requirements.gen2act.txt. Retain the
# isolated shared install only as a fallback for older environments.
if ! "$PYTHON_BIN" -c 'import wandb, google.protobuf; assert wandb.__version__ == "0.20.1"' >/dev/null 2>&1; then
    export PYTHONPATH="$WANDB_VENDOR_DIR${PYTHONPATH:+:$PYTHONPATH}"
    echo "installing isolated W&B runtime into $WANDB_VENDOR_DIR"
    mkdir -p "$WANDB_VENDOR_DIR"
    "$PYTHON_BIN" -m pip install --target "$WANDB_VENDOR_DIR" --upgrade --no-deps \
        wandb==0.20.1 \
        click==8.1.8 \
        gitpython==3.1.57 gitdb==4.0.12 smmap==5.0.3 \
        packaging==23.2 platformdirs==4.3.6 protobuf==4.24.4 psutil==7.0.0 \
        pydantic==2.10.6 pydantic-core==2.27.2 annotated-types==0.7.0 \
        pyyaml==6.0.2 requests==2.32.3 certifi==2025.1.31 \
        charset-normalizer==3.4.1 idna==3.10 urllib3==2.0.7 \
        sentry-sdk==2.66.1 setproctitle==1.3.7 typing-extensions==4.12.2
fi
if ! "$PYTHON_BIN" -c 'import wandb, google.protobuf; assert wandb.__version__ == "0.20.1"' >/dev/null 2>&1; then
    echo "isolated wandb 0.20.1 runtime could not be imported; full traceback:" >&2
    "$PYTHON_BIN" -c 'import wandb, google.protobuf; print(wandb.__version__)' >&2 || true
    exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" && ! -s "$WANDB_CREDENTIAL" ]]; then
    echo "W&B API key is missing: $WANDB_CREDENTIAL" >&2
    echo "Inject WANDB_API_KEY into the cluster job or create the shared credential file before submission." >&2
    exit 1
elif [[ -z "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY="$(<"$WANDB_CREDENTIAL")"
fi
export WANDB_ENTITY
if ! "$PYTHON_BIN" - <<'PY'
import os
import wandb
from wandb_gql import gql

expected = os.environ["WANDB_ENTITY"]
api = wandb.Api()
viewer = getattr(api.viewer, "username", None) or api.default_entity
data = api.client.execute(gql(
    "query ViewerTeams { viewer { username teams { edges { node { name } } } } }"
))
team_edges = ((data.get("viewer") or {}).get("teams") or {}).get("edges", [])
teams = {str(edge["node"]["name"]) for edge in team_edges}
if expected not in teams and expected not in {viewer, api.default_entity}:
    raise SystemExit(
        f"W&B account {viewer!r} is not a member of target entity {expected!r}; "
        "read access is insufficient to create runs"
    )
print(f"W&B authentication OK: account={viewer} target={expected}/gen2act")
PY
then
    echo "W&B authentication check failed; $RUN_LABEL training was not started." >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || [[ "$(nvidia-smi -L 2>/dev/null | wc -l)" -lt 1 ]]; then
    echo "no visible NVIDIA GPU; run inside a GPU allocation" >&2
    exit 1
fi
if [[ "${C40_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "$RUN_LABEL preflight OK; training was not started."
    exit 0
fi

mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
echo "$RUN_LABEL raw DROID training"
echo "  config:   $ROOT/$CONFIG"
echo "  manifest: $ROOT/$MANIFEST"
echo "  index:    $ROOT/$INDEX_JSON"
echo "  init:     $ROOT/$WARM_START"
echo "  pytest:   $("$PYTHON_BIN" -m pytest --version)"
echo "  wandb:    $("$PYTHON_BIN" -c 'import wandb; print(wandb.__version__)')"
echo "  log:      $ROOT/$LOG_FILE"

TRAIN_ARGS=("$@")
if [[ -n "${RESUME_FULL_CHECKPOINT:-}" ]]; then
    if [[ ! -f "$RESUME_FULL_CHECKPOINT" ]]; then
        echo "resume checkpoint not found: $RESUME_FULL_CHECKPOINT" >&2
        exit 1
    fi
    echo "  resume:   $RESUME_FULL_CHECKPOINT"
    TRAIN_ARGS=(--resume-full-checkpoint "$RESUME_FULL_CHECKPOINT" "${TRAIN_ARGS[@]}")
fi

set +e
PYTHON="$PYTHON_BIN" \
TORCHRUN="${TORCHRUN:-$(dirname "$PYTHON_BIN")/torchrun}" \
    bash scripts/run_distributed_train.sh "$CONFIG" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
if [[ "$status" -ne 0 ]]; then
    echo "$RUN_LABEL training exited with status $status; see $ROOT/$LOG_FILE" >&2
    exit "$status"
fi
