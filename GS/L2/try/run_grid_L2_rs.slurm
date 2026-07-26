#!/bin/bash
#SBATCH --job-name=gs_L2_rs
#SBATCH --output=results_grid/L2_rs_%j.out
#SBATCH --error=results_grid/L2_rs_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=6:00:00

# L=2: random + STOP on the 292 TEST params (completes the 4-method 1x4).
# det = full_chunks_L2 ; noise already done. Same patient refiner 30000/0.
set -u
source activate torch124
TRY=/path/to/PatternFormer/GS/L2/try
cd "$TRY"
PY=/path/to/envs/torch124/bin/python
N=292; SH=8; STEP=$(( (N + SH - 1) / SH ))
MI="--maxiter 30000 --early_iter 0"

run () {  # $1=method  $2..=extra
  local M=$1; shift; local EXTRA="$@"
  echo "=== L2 $M over $N TEST params, step=$STEP $(date) ==="
  for g in $(seq 0 7); do
    s=$((g*STEP)); e=$((s+STEP))
    CUDA_VISIBLE_DEVICES=$g $PY eval_grid.py --L2 --method $M --split test --idx_start $s --idx_end $e \
        $MI $EXTRA --out results_grid/${M}_shard${g}.json 2>results_grid/${M}_shard${g}.log &
    sleep 8
  done
  wait
  echo "=== L2 $M DONE $(date) — $(ls results_grid/${M}_shard*.json 2>/dev/null | wc -l) shards ==="
}

run random
run stop --code ../baseline_stop --ckpt ../ckpt/stop_base

echo "=== L2 random+stop DONE $(date); plotting L2 1x4 (+ all L2 grid figs) ==="
$PY make_grid_figs.py || echo "L2 plot failed"
echo "=== DONE $(date) ==="
