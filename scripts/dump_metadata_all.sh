#!/usr/bin/env bash
# Sequentially dump metadata + trajectory for all six PolaRiS environments.
# Each env runs in its own Isaac Sim process to avoid USD-reload hangs.
#
# Expects a policy server already listening on the port given to
# scripts/dump_metadata.py (default 8001 in PolicyArgs).
#
# Required env vars are set per invocation here (NVIDIA Vulkan/EGL ICDs,
# CUDA_VISIBLE_DEVICES, EULA acceptance, no DISPLAY).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POLARIS_DIR="${POLARIS_DIR:-${REPO_ROOT}/repos/polaris}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
LOG_DIR="${REPO_ROOT}/notes"
mkdir -p "${LOG_DIR}"

if [[ ! -d "${POLARIS_DIR}" ]]; then
    echo "polaris working tree not found at ${POLARIS_DIR}" >&2
    echo "set POLARIS_DIR=... or symlink polaris/ with a .venv" >&2
    exit 1
fi

ENVS=(
    food_bussing
    block_stack_kitchen
    pan_clean
    move_latte_cup
    organize_tools
    tape_into_container
)

for env_short in "${ENVS[@]}"; do
    echo "=== ${env_short} (GPU ${GPU}) ==="
    log="${LOG_DIR}/dump_${env_short}.log"
    (
        cd "${POLARIS_DIR}"
        unset DISPLAY
        VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
        __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
        __GLX_VENDOR_LIBRARY_NAME=nvidia \
        CUDA_VISIBLE_DEVICES="${GPU}" OMNI_KIT_ACCEPT_EULA=YES \
        uv run python "${REPO_ROOT}/scripts/dump_metadata.py" "${env_short}"
    ) 2>&1 | tee "${log}"
    status=${PIPESTATUS[0]}
    if [[ ${status} -ne 0 ]]; then
        echo "!!! ${env_short} exited ${status} (see ${log})"
    fi
done
echo "all done"
