#!/bin/bash

set -e

cd "$(dirname ${BASH_SOURCE[0]})/../.."
source docker/.env

usage() {
  echo "Usage: $0 [--gpu GPU_DEVICE]"
  echo "       $0 [-g GPU_DEVICE]"
}

GPU_DEVICE="${GPU_DEVICE:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --gpu requires a value" >&2
        usage >&2
        exit 1
      fi
      GPU_DEVICE="$2"
      shift 2
      ;;
    --gpu=*)
      GPU_DEVICE="${1#*=}"
      shift
      ;;
    -g)
      if [[ -z "${2:-}" ]]; then
        echo "Error: -g requires a value" >&2
        usage >&2
        exit 1
      fi
      GPU_DEVICE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

profile_name=isim$GPU_DEVICE
export REPO_DIR="$(pwd)"
export ISAAC_SIM_DATA="${ISAAC_SIM_DATA}-${GPU_DEVICE}"
export GPU_DEVICE=$GPU_DEVICE
export ISAACSIM_SIGNAL_PORT=$((49100 + GPU_DEVICE))
export ISAACSIM_STREAM_PORT=$((47998 - GPU_DEVICE))
export WEB_VIEWER_PORT=$((8210 + GPU_DEVICE))

# Pre-create the persistent data dir so the container user (uid 1234) can write
# its caches; without this Isaac Sim never reaches AppReady. Only chmod the dirs
# we just created (non-recursive) — the container fills them with 1234-owned
# files afterwards, which we neither own nor need to touch.
for sub in cache/main cache/computecache config data logs pkg; do
  mkdir -p "${ISAAC_SIM_DATA}/${sub}"
  chmod 777 "${ISAAC_SIM_DATA}/${sub}" 2>/dev/null || true
done
chmod 777 "${ISAAC_SIM_DATA}" 2>/dev/null || true

# Make repo data world-readable so the container user (uid 1234) can read USD
# assets and textures. Files copied/downloaded with a restrictive umask land as
# mode 600, which the container cannot read -> UJITSO texture loads fail and RTX
# renders everything teal. a+rX (read + dir-traverse) fixes it without touching
# write bits.
[ -d "${REPO_DIR}/data" ] && chmod -R a+rX "${REPO_DIR}/data" 2>/dev/null || true

docker compose -f docker/docker-compose.yml -p $profile_name up -d

echo "ISAAC_SIM_IMAGE: $ISAAC_SIM_IMAGE"
echo "ISAAC_SIM_DATA: $ISAAC_SIM_DATA"
echo "ISAACSIM_HUB_CACHE_PATH: $ISAACSIM_HUB_CACHE_PATH"
echo "ISAACSIM_HOST: $ISAACSIM_HOST"
echo "ISAACSIM_SIGNAL_PORT: $ISAACSIM_SIGNAL_PORT"
echo "ISAACSIM_STREAM_PORT: $ISAACSIM_STREAM_PORT"
echo "WEB_VIEWER_PORT: $WEB_VIEWER_PORT"
echo "GPU_DEVICE: $GPU_DEVICE"
