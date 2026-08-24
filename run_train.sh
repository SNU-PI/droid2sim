#!/usr/bin/env bash
# Train the six world-model variants. GPUs 2 and 3, three jobs each.
source ~/miniforge3/etc/profile.d/conda.sh && conda activate trellis2
cd "$(dirname "$0")"
STEPS=${STEPS:-6000}
mkdir -p out/logs

launch () {  # gpu, args...
  local gpu=$1; shift
  local tag=$1; shift
  CUDA_VISIBLE_DEVICES=$gpu python -u src/train.py --steps "$STEPS" --tag "$tag" "$@" \
      > "out/logs/${tag}.log" 2>&1
}

( launch 2 k1  --k 1
  launch 2 k4  --k 4
  launch 2 k16 --k 16 ) &
( launch 3 k2  --k 2
  launch 3 k8  --k 8
  launch 3 oracle --k 1 --oracle ) &
wait
echo "ALL TRAINING DONE"
tail -n 2 out/logs/*.log
