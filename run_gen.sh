#!/usr/bin/env bash
# usage: run_gen.sh <egl_device> <family> [family ...]
source ~/miniforge3/etc/profile.d/conda.sh && conda activate trellis2
cd "$(dirname "$0")"
DEV=$1; shift
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$DEV
export EP_DIR=/data/pgc/simdroid/episodes
LOG=out/multi/gen_dev$DEV.log
for fam in "$@"; do
  for spec in "train clean 250" "test clean 80" "test camera 80" "test light 80"; do
    set -- $spec; split=$1; var=$2; tot=$3
    for pass in $(seq 1 250); do
      python -u src/gen/gen_episodes.py --family $fam --split $split --variant $var \
             --total $tot --batch 5 >> $LOG 2>&1
      n=$(ls $EP_DIR/$fam/${split}_${var}/*.npz 2>/dev/null | wc -l)
      [ "$n" -ge "$tot" ] && break
    done
    echo "[dev$DEV] $fam $split/$var -> $(ls $EP_DIR/$fam/${split}_${var}/*.npz 2>/dev/null | wc -l)/$tot"
    set -- "$@"
  done
done
echo "[dev$DEV] GENERATION COMPLETE"
