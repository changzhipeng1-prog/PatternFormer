#!/bin/bash
# Per-param solution counts over ALL 1083 params, 8-GPU sharded, for the 2x3
# distribution figure (stop/det/noise) and the random-vs-ours comparison.
# Usage: bash run_grid.sh <method> [extra args...]
set -u
PY=/home/zfc5231/.conda/envs/torch124/bin/python
N=1083; SH=8; STEP=$(( (N + SH - 1) / SH ))
M=$1; shift
EXTRA="$@"
mkdir -p results_grid
echo "[$(date)] method=$M extra='$EXTRA' step=$STEP"
for g in $(seq 0 7); do
  s=$((g*STEP)); e=$((s+STEP))
  CUDA_VISIBLE_DEVICES=$g $PY eval_grid.py --method $M --split all --idx_start $s --idx_end $e \
      $EXTRA --out results_grid/${M}_shard${g}.json 2>results_grid/${M}_shard${g}.log &
  sleep 8   # stagger 7B model loads to avoid a CPU/IO peak
done
wait
echo "[$(date)] method=$M DONE — $(ls results_grid/${M}_shard*.json | wc -l) shards"
