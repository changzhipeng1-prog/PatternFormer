#!/bin/bash
#SBATCH --job-name=gs_noise_grid
#SBATCH --output=results_grid/noise_grid_%j.out
#SBATCH --error=results_grid/noise_grid_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=8:00:00

# ordered + noise (det + 5 noise passes), per-param solution counts over ALL 1083 params,
# 8-GPU sharded. Writes results_grid/noise_shard{0..7}.json (total/matched/beyond per param).
set -u
source activate torch124
cd /path/to/PatternFormer/GS/L1/try
PY=/path/to/envs/torch124/bin/python
N=1083; SH=8; STEP=$(( (N + SH - 1) / SH ))
echo "=== noise grid (det+5 noise) over $N params, step=$STEP $(date) ==="
for g in $(seq 0 7); do
  s=$((g*STEP)); e=$((s+STEP))
  CUDA_VISIBLE_DEVICES=$g $PY eval_grid.py --method noise --code ../code --ckpt ../ckpt/epoch_60 \
      --passes 5 --noise 0.1 --split all --idx_start $s --idx_end $e \
      --out results_grid/noise_shard${g}.json 2>results_grid/noise_shard${g}.log &
  sleep 8
done
wait
echo "=== noise grid DONE $(date) — $(ls results_grid/noise_shard*.json | wc -l) shards ==="
