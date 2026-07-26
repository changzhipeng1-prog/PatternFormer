#!/bin/bash
#SBATCH --job-name=gs_curve_L1
#SBATCH --output=results_grid/curve_L1_%j.out
#SBATCH --error=results_grid/curve_L1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=6:00:00

# L=1 noise-sweep curve: TEST 163 params, sigmas 0.05/0.1/0.2, 6 noise passes.
set -u
source activate torch124
cd /path/to/PatternFormer/GS/L1/try
PY=/path/to/envs/torch124/bin/python
N=163; SH=8; STEP=$(( (N + SH - 1) / SH ))
echo "=== L1 noise curve, TEST $N, step=$STEP $(date) ==="
for g in $(seq 0 7); do
  s=$((g*STEP)); e=$((s+STEP))
  CUDA_VISIBLE_DEVICES=$g $PY eval_noise_curve.py --code ../code --ckpt ../ckpt/epoch_60 \
      --sigmas 0.05,0.1,0.2 --passes 6 --maxiter 8000 --early_iter 3000 \
      --idx_start $s --idx_end $e --out results_grid/curve_shard${g}.json 2>results_grid/curve_shard${g}.log &
  sleep 8
done
wait
echo "=== L1 curve DONE $(date); plotting ==="
$PY make_noise_curve_fig.py || echo "plot failed"
echo "=== DONE $(date) ==="
