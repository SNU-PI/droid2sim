#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="/home/sangjunpark/miniconda3/bin/conda"

DROID_SIM_DATASET_ROOT="${DROID_SIM_DATASET_ROOT:-${SCRIPT_DIR}/lerobot_data/sponge_pick_place}"
DROID_SIM_REPO_ID="${DROID_SIM_REPO_ID:-ssangjunpark/droid_sim_sponge_pick_place}"
DROID_SIM_TASK_DESCRIPTION="${DROID_SIM_TASK_DESCRIPTION:-Pick up the sponge and place it in the frying pan.}"
ISAAC_VISIBLE_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! "${ISAAC_VISIBLE_DEVICE}" =~ ^[0-9]+$ ]]; then
  echo "[pipeline] ERROR: CUDA_VISIBLE_DEVICES must select exactly one numeric physical GPU." >&2
  exit 2
fi

echo "[pipeline] Isaac rollout: physical GPU ${ISAAC_VISIBLE_DEVICE} only (logical cuda:0)"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" CUDA_VISIBLE_DEVICES="${ISAAC_VISIBLE_DEVICE}" \
"${CONDA_BIN}" run --no-capture-output -n env_isaaclab \
python "${SCRIPT_DIR}/run_data_gen.py" \
  --headless \
  --enable_cameras \
  --save-waypoints \
  --camera-fps 10 \
  --width 256 \
  --height 256 \
  --task-description "${DROID_SIM_TASK_DESCRIPTION}" \
  "$@"

echo "[pipeline] Isaac exited; exporting latest successful rollout on CPU only"
CUDA_VISIBLE_DEVICES="" \
"${CONDA_BIN}" run --no-capture-output -n lerobot \
python "${SCRIPT_DIR}/export_lerobot.py" \
  --run-dir latest \
  --dataset-root "${DROID_SIM_DATASET_ROOT}" \
  --repo-id "${DROID_SIM_REPO_ID}" \
  --task "${DROID_SIM_TASK_DESCRIPTION}" \
  --fps 10 \
  --width 256 \
  --height 256

echo "[pipeline] Complete: ${DROID_SIM_DATASET_ROOT}"
