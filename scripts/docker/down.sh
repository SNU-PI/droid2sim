#!/bin/bash

set -e

cd "$(dirname ${BASH_SOURCE[0]})/../.."

usage() {
  echo "Usage: $0 [--gpu GPU_DEVICE]"
  echo "       $0 [-g GPU_DEVICE]"
}

GPU_DEVICE=0

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

docker compose -f docker/docker-compose.yml -p $profile_name down
