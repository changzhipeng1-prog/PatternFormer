#!/bin/bash
#SBATCH --job-name=gs_grid_detnoise
#SBATCH --output=results_grid/detnoise_%j.out
#SBATCH --error=results_grid/detnoise_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=10:00:00

# Per-param solution counts (total/matched/beyond) over ALL 1083 params, 8-GPU sharded.
# Phase 1: ordered-det (fixed-K=24).  Phase 2: ordered+noise (det + 5 noise passes).
set -u
source activate torch124
cd /home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/try
PY=/home/zfc5231/.conda/envs/torch124/bin/python
N=1083; SH=8; STEP=$(( (N + SH - 1) / SH ))

run () {  # $1=method  $2..=extra
  local M=$1; shift; local EXTRA="$@"
  echo "=== $M over $N params, step=$STEP $(date) ==="
  for g in $(seq 0 7); do
    s=$((g*STEP)); e=$((s+STEP))
    CUDA_VISIBLE_DEVICES=$g $PY eval_grid.py --method $M --split all --idx_start $s --idx_end $e \
        $EXTRA --out results_grid/${M}_shard${g}.json 2>results_grid/${M}_shard${g}.log &
    sleep 8
  done
  wait
  echo "=== $M DONE $(date) — $(ls results_grid/${M}_shard*.json | wc -l) shards ==="
}

run det   --code ../code --ckpt ../ckpt/epoch_60
run noise --code ../code --ckpt ../ckpt/epoch_60 --passes 5 --noise 0.1
echo "=== ALL DONE $(date) ==="
