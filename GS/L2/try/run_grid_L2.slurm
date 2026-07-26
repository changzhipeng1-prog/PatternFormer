#!/bin/bash
#SBATCH --job-name=gs_grid_L2
#SBATCH --output=results_grid/grid_L2_%j.out
#SBATCH --error=results_grid/grid_L2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# L=2 per-param solution counts (total/matched/beyond) over ALL 1930 params, 8-GPU sharded.
# Methods: random / stop / det / noise (det+5).  L=2 patient-ish refiner (18000/5000).
# At the end: plot BOTH L1 and L2 grid figures (L1 grid data is ready — this job depends on it).
set -u
source activate torch124
TRY=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L2/try
cd "$TRY"
PY=/home/zfc5231/.conda/envs/torch124/bin/python
N=292; SH=8; STEP=$(( (N + SH - 1) / SH ))   # L=2 noise on the 292 TEST params (option 2)
MI="--maxiter 30000 --early_iter 0"   # same patient refiner as fig1 L=2 noise (full consistency)

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

# L=2: only NOISE (det+5, sigma=0.1) — same method as fig1 / fig_noise_multipass.
# random + stop comparison is taken from L=1; deterministic from full_chunks_L2.
run noise --code ../code --ckpt ../ckpt/epoch_60 --passes 5 --noise 0.1

echo "=== ALL L2 GRID DONE $(date); plotting L1 + L2 figures ==="
cd /home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/try && $PY make_grid_figs.py || echo "L1 plot failed"
cd "$TRY" && $PY make_grid_figs.py || echo "L2 plot failed"
echo "=== PLOTTING DONE $(date) ==="
