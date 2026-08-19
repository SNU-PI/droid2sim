#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="${DROID_SIM_PIPELINE_SCRIPT:-${SCRIPT_DIR}/run_and_export_lerobot.sh}"
PYTHON_BIN="/home/sangjunpark/miniconda3/envs/lerobot/bin/python"

DROID_SIM_DATASET_ROOT="${DROID_SIM_DATASET_ROOT:-${SCRIPT_DIR}/lerobot_data/sponge_pick_place_50}"
DROID_SIM_REPO_ID="${DROID_SIM_REPO_ID:-ssangjunpark/droid_sim_sponge_pick_place_50}"
DROID_SIM_TASK_DESCRIPTION="${DROID_SIM_TASK_DESCRIPTION:-Pick up the sponge and place it in the frying pan.}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DROID_SIM_COLLECTION_LOG_DIR="${DROID_SIM_COLLECTION_LOG_DIR:-${SCRIPT_DIR}/collection_logs/$(basename -- "${DROID_SIM_DATASET_ROOT}")}"

START_INDEX=0
END_INDEX=49
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: collect_50_episodes.sh [options] [-- additional-run_data_gen-options]

Collects sponge source positions sequentially and appends only successful
rollouts to one local LeRobot dataset. Existing successful source positions in
the dataset manifest are skipped, so rerunning this command retries only the
missing positions.

Options:
  --start-index N   First source-position index (default: 0)
  --end-index N     Last source-position index, inclusive (default: 49)
  --dry-run         Print the resume plan without launching Isaac or exporting
  -h, --help        Show this help

Environment:
  CUDA_VISIBLE_DEVICES             Exactly one physical GPU number
  DROID_SIM_DATASET_ROOT           Output LeRobot dataset directory
  DROID_SIM_REPO_ID                Repo id stored in local metadata
  DROID_SIM_TASK_DESCRIPTION       Language instruction
  DROID_SIM_COLLECTION_LOG_DIR     Logs/state outside the dataset directory
  DROID_SIM_PIPELINE_SCRIPT        Pipeline wrapper override (primarily for testing)
EOF
}

while (($#)); do
  case "$1" in
    --start-index)
      [[ $# -ge 2 ]] || { echo "[collector] ERROR: --start-index needs a value." >&2; exit 2; }
      START_INDEX="$2"
      shift 2
      ;;
    --end-index)
      [[ $# -ge 2 ]] || { echo "[collector] ERROR: --end-index needs a value." >&2; exit 2; }
      END_INDEX="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      echo "[collector] ERROR: unknown collector option: $1" >&2
      echo "[collector] Put run_data_gen.py options after --." >&2
      exit 2
      ;;
  esac
done

if [[ ! "${START_INDEX}" =~ ^[0-9]+$ ]] || [[ ! "${END_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "[collector] ERROR: start/end indices must be integers." >&2
  exit 2
fi
if ((START_INDEX < 0 || END_INDEX > 49 || START_INDEX > END_INDEX)); then
  echo "[collector] ERROR: require 0 <= start-index <= end-index <= 49." >&2
  exit 2
fi
if [[ ! "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]]; then
  echo "[collector] ERROR: CUDA_VISIBLE_DEVICES must be exactly one numeric physical GPU." >&2
  exit 2
fi
for argument in "${EXTRA_ARGS[@]}"; do
  if [[ "${argument}" == "--episode-index" || "${argument}" == --episode-index=* ]]; then
    echo "[collector] ERROR: the collector owns --episode-index; do not pass it after --." >&2
    exit 2
  fi
done
if [[ ! -x "${PIPELINE_SCRIPT}" ]]; then
  echo "[collector] ERROR: pipeline script is missing or not executable: ${PIPELINE_SCRIPT}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[collector] ERROR: LeRobot Python is missing: ${PYTHON_BIN}" >&2
  exit 2
fi

export DROID_SIM_DATASET_ROOT DROID_SIM_REPO_ID DROID_SIM_TASK_DESCRIPTION
export CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES

mkdir -p -- "${DROID_SIM_COLLECTION_LOG_DIR}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${DROID_SIM_COLLECTION_LOG_DIR}/collection_${RUN_STAMP}.log"
ATTEMPTS_LOG="${DROID_SIM_COLLECTION_LOG_DIR}/attempts.tsv"
if [[ ! -f "${ATTEMPTS_LOG}" ]]; then
  printf 'timestamp\tsource_episode_index\tresult\texit_code\tepisode_log\n' > "${ATTEMPTS_LOG}"
fi

log() {
  printf '[collector] %s\n' "$*" | tee -a "${RUN_LOG}"
}

# Print one completed source-position index per line. This also verifies that
# the LeRobot episode count and provenance manifest agree before collection.
completed_source_indices() {
  "${PYTHON_BIN}" - "${DROID_SIM_DATASET_ROOT}" "${DROID_SIM_REPO_ID}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_repo_id = sys.argv[2]
info_path = root / "meta" / "info.json"
manifest_path = root / "meta" / "droid_sim_manifest.json"

if not info_path.exists() and not manifest_path.exists():
    raise SystemExit(0)
if info_path.exists() != manifest_path.exists():
    raise RuntimeError(
        "Dataset metadata and droid_sim_manifest.json do not both exist; "
        "refusing an ambiguous resume."
    )

info = json.loads(info_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("repo_id") != expected_repo_id:
    raise RuntimeError(
        f"Existing provenance manifest repo_id does not match {expected_repo_id!r}."
    )

episodes = manifest.get("episodes", [])
total_episodes = int(info.get("total_episodes", -1))
if total_episodes != len(episodes):
    raise RuntimeError(
        f"LeRobot has {total_episodes} episodes but provenance manifest has "
        f"{len(episodes)}; refusing to risk a duplicate or mislabelled append."
    )

completed = []
for episode in episodes:
    if "source_episode_index" not in episode:
        raise RuntimeError(
            "Existing episode lacks source_episode_index provenance. Use a fresh "
            "production dataset root for the 0-49 collection."
        )
    index = int(episode["source_episode_index"])
    if not 0 <= index <= 49:
        raise RuntimeError(f"Invalid source_episode_index in manifest: {index}")
    completed.append(index)
if len(completed) != len(set(completed)):
    raise RuntimeError("Duplicate source_episode_index values exist in the manifest.")
for index in sorted(completed):
    print(index)
PY
}

load_completed() {
  local output
  if ! output="$(completed_source_indices)"; then
    log "ERROR: dataset resume validation failed."
    return 1
  fi
  COMPLETED=()
  while IFS= read -r index; do
    [[ -n "${index}" ]] && COMPLETED["${index}"]=1
  done <<< "${output}"
  return 0
}

declare -A COMPLETED=()
load_completed

log "Dataset: ${DROID_SIM_DATASET_ROOT}"
log "Repo id: ${DROID_SIM_REPO_ID}"
log "Physical GPU: ${CUDA_VISIBLE_DEVICES} (Isaac sees logical cuda:0)"
log "Requested source positions: ${START_INDEX}-${END_INDEX}"
log "Successful positions already present: ${#COMPLETED[@]}"
log "Attempt ledger: ${ATTEMPTS_LOG}"

attempted=0
succeeded=0
skipped=0
failed=()
interrupted=0
trap 'interrupted=1' INT TERM

for ((source_index=START_INDEX; source_index<=END_INDEX; source_index++)); do
  if [[ -n "${COMPLETED[${source_index}]+yes}" ]]; then
    skipped=$((skipped + 1))
    log "SKIP source ${source_index}/49: already exported successfully."
    continue
  fi

  if ((DRY_RUN)); then
    log "PLAN source ${source_index}/49: would run and export."
    continue
  fi

  attempted=$((attempted + 1))
  episode_log="${DROID_SIM_COLLECTION_LOG_DIR}/source_${source_index}_${RUN_STAMP}.log"
  log "START source ${source_index}/49 (attempt ${attempted}); output: ${episode_log}"

  set +e
  "${PIPELINE_SCRIPT}" --episode-index "${source_index}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${episode_log}"
  pipeline_status=${PIPESTATUS[0]}
  set -e

  timestamp="$(date --iso-8601=seconds)"
  if ((interrupted)); then
    printf '%s\t%d\tinterrupted\t%d\t%s\n' \
      "${timestamp}" "${source_index}" "${pipeline_status}" "${episode_log}" >> "${ATTEMPTS_LOG}"
    log "INTERRUPTED during source ${source_index}/49; stopping without starting another episode."
    exit 130
  fi

  if ((pipeline_status != 0)); then
    # A normal Isaac failure occurs before export and leaves dataset counts
    # unchanged. If an exporter died after mutating the dataset, this consistency
    # check fails and we stop instead of risking bad source provenance.
    if ! load_completed; then
      printf '%s\t%d\tdataset_inconsistent\t%d\t%s\n' \
        "${timestamp}" "${source_index}" "${pipeline_status}" "${episode_log}" >> "${ATTEMPTS_LOG}"
      log "FATAL source ${source_index}/49: pipeline failed and dataset consistency is uncertain."
      exit 3
    fi
    if [[ -n "${COMPLETED[${source_index}]+yes}" ]]; then
      printf '%s\t%d\texport_committed_but_pipeline_failed\t%d\t%s\n' \
        "${timestamp}" "${source_index}" "${pipeline_status}" "${episode_log}" >> "${ATTEMPTS_LOG}"
      log "FATAL source ${source_index}/49: export exists although the pipeline returned failure."
      exit 3
    fi
    failed+=("${source_index}")
    printf '%s\t%d\tfailed_not_exported\t%d\t%s\n' \
      "${timestamp}" "${source_index}" "${pipeline_status}" "${episode_log}" >> "${ATTEMPTS_LOG}"
    log "FAIL source ${source_index}/49: not exported; continuing to the next position."
    continue
  fi

  if ! load_completed; then
    printf '%s\t%d\tdataset_inconsistent\t0\t%s\n' \
      "${timestamp}" "${source_index}" "${episode_log}" >> "${ATTEMPTS_LOG}"
    log "FATAL source ${source_index}/49: pipeline returned success but resume validation failed."
    exit 3
  fi
  if [[ -z "${COMPLETED[${source_index}]+yes}" ]]; then
    printf '%s\t%d\tmissing_after_success\t0\t%s\n' \
      "${timestamp}" "${source_index}" "${episode_log}" >> "${ATTEMPTS_LOG}"
    log "FATAL source ${source_index}/49: pipeline returned success without dataset provenance."
    exit 3
  fi

  succeeded=$((succeeded + 1))
  printf '%s\t%d\tsuccess\t0\t%s\n' \
    "${timestamp}" "${source_index}" "${episode_log}" >> "${ATTEMPTS_LOG}"
  log "PASS source ${source_index}/49: rollout and LeRobot export completed."
done

if ((DRY_RUN)); then
  log "Dry run complete; Isaac was not launched and no dataset was changed."
  exit 0
fi

load_completed
missing=()
for ((source_index=START_INDEX; source_index<=END_INDEX; source_index++)); do
  if [[ -z "${COMPLETED[${source_index}]+yes}" ]]; then
    missing+=("${source_index}")
  fi
done

log "Run complete: attempted=${attempted}, succeeded=${succeeded}, already_present=${skipped}, failed=${#failed[@]}."
if ((${#failed[@]})); then
  log "Failed this run (not exported): ${failed[*]}"
fi
if ((${#missing[@]})); then
  log "Still missing: ${missing[*]}"
  log "Rerun the same command to retry only these missing source positions."
  exit 1
fi

log "SUCCESS: every requested source position ${START_INDEX}-${END_INDEX} is present."
