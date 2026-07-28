#!/bin/bash
#
# Wait for the Isaac Sim container to be healthy, then load the PanClean scene
# into the running GUI so the browser shows the kitchen right away.
#
# Run standalone (re-open the scene) or let launch.sh call it in the background.
#   ./scripts/docker/open_scene.sh -g 0

set -e

cd "$(dirname ${BASH_SOURCE[0]})/../.."

GPU_DEVICE="${GPU_DEVICE:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpu) GPU_DEVICE="$2"; shift 2 ;;
    --gpu=*)  GPU_DEVICE="${1#*=}"; shift ;;
    -h|--help) echo "Usage: $0 [-g GPU]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

container="isim${GPU_DEVICE}-isaac-sim-1"

# Wait until the container reports healthy (AppReady), up to ~5 min.
echo "[open_scene] waiting for ${container} to be healthy..."
for i in $(seq 1 60); do
  state="$(docker inspect -f '{{.State.Health.Status}}' "${container}" 2>/dev/null || echo missing)"
  [ "${state}" = "healthy" ] && break
  [ "${state}" = "missing" ] && { echo "[open_scene] ${container} not found"; exit 1; }
  sleep 5
done
if [ "${state}" != "healthy" ]; then
  echo "[open_scene] ${container} never became healthy (last: ${state})" >&2
  exit 1
fi

echo "[open_scene] loading scene into live GUI..."
docker exec -i "${container}" \
  /isaac-sim/kit/python/bin/python3 /work/scripts/docker_render/gui_send.py \
  /work/scripts/docker_render/open_scene.py
echo "[open_scene] done — open http://\${ISAACSIM_HOST}:$((8210 + GPU_DEVICE))/"
