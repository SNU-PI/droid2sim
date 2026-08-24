#!/usr/bin/env bash
# camera-augmented TRAINING data, to test whether the probe's one weakness is fixable
source ~/miniforge3/etc/profile.d/conda.sh && conda activate trellis2
cd "$(dirname "$0")"
DEV=$1; shift
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$DEV
export EP_DIR=/data/pgc/simdroid/episodes
LOG=out/multi/gen_aug_dev$DEV.log
for fam in "$@"; do
  for pass in $(seq 1 250); do
    python -u src/gen_episodes.py --family $fam --split train --variant camera \
           --total 250 --batch 5 >> $LOG 2>&1
    n=$(ls $EP_DIR/$fam/train_camera/*.npz 2>/dev/null | wc -l)
    [ "$n" -ge 250 ] && break
  done
  echo "[dev$DEV] $fam train/camera -> $(ls $EP_DIR/$fam/train_camera/*.npz 2>/dev/null | wc -l)/250"
done
echo "[dev$DEV] AUG COMPLETE"
