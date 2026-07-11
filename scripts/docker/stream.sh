set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU_DEVICE=0
USD=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpu) GPU_DEVICE="$2"; shift 2 ;;
    --usd)    USD="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [-g GPU] [--usd /work/path/to/scene.usd]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

profile_name=isim$GPU_DEVICE
container="${profile_name}-isaac-sim-1"

EXTRA=""
[ -n "$USD" ] && EXTRA="--/app/livestream/allowResize=true --exec \"open_stage $USD\""

echo "streaming ${container}  (web-viewer: http://<host>:$((8210 + GPU_DEVICE))/ )"
docker exec -i \
  -e ISAACSIM_HOST="${ISAACSIM_HOST:-127.0.0.1}" \
  -e ISAACSIM_SIGNAL_PORT=$((49100 + GPU_DEVICE)) \
  -e ISAACSIM_STREAM_PORT=$((47998 - GPU_DEVICE)) \
  "$container" /isaac-sim/runheadless.sh $EXTRA