#!/usr/bin/env bash
set -euo pipefail

# usage: ./run_vwm.sh <model_dir> <model-480p-16fps.pt> [sweep_root] [output_dir]
ROOT=$(cd "$(dirname "$0")" && pwd)
MODEL_DIR=${1:?usage: ./run_vwm.sh <model_dir> <checkpoint.pt> [sweep_root] [output_dir]}
CHECKPOINT=${2:?usage: ./run_vwm.sh <model_dir> <checkpoint.pt> [sweep_root] [output_dir]}
SWEEP_ROOT=${3:-artifacts/physics_sweep}
OUTPUT=${4:-$SWEEP_ROOT/cosmos_v2w}

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python src/exp/cosmos_v2w_sweep.py \
  --model-dir "$MODEL_DIR" \
  --original-checkpoint "$CHECKPOINT" \
  --manifest "$SWEEP_ROOT/manifest.jsonl" \
  --sweep-root "$SWEEP_ROOT" \
  --output-dir "$OUTPUT"

