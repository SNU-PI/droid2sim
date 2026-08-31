#!/usr/bin/env bash
set -euo pipefail

# usage: ./run_sweep.sh <egl_device> <libero_stats.json> [output_dir]
ROOT=$(cd "$(dirname "$0")" && pwd)
DEVICE=${1:?usage: ./run_sweep.sh <egl_device> <libero_stats.json> [output_dir]}
STATS=${2:?usage: ./run_sweep.sh <egl_device> <libero_stats.json> [output_dir]}
OUTPUT=${3:-artifacts/physics_sweep}

cd "$ROOT"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$DEVICE"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python src/gen/make_sweep.py --stats "$STATS" --output-dir "$OUTPUT"
python src/gen/make_gifs.py --output-dir artifacts/threshold_gifs
