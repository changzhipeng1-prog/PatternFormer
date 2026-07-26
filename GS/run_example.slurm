#!/bin/bash
#SBATCH --job-name=gs_example
#SBATCH --output=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/example_%j.out
#SBATCH --error=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/example_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=5:00:00

# Unified single-param example: fixed seed, det + sigma{0.05,0.1,0.2} x5 passes.
# L1 ti=576 / L2 ti=595 (each setup's max-distinct TEST param). Produces BOTH
# Fig1 (sigma=0.1 union) and noise-curve (per-sigma curves) from ONE run.
set -u
source activate torch124
cd /home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS
PY=/home/zfc5231/.conda/envs/torch124/bin/python
ROOT=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS
echo "=== example gen start $(date) ==="
CUDA_VISIBLE_DEVICES=0 $PY _example_gen.py --setup L1 --ti 576 \
    --ckpt $ROOT/L1/ckpt/epoch_60 > example_L1.log 2>&1 &
sleep 8
CUDA_VISIBLE_DEVICES=1 $PY _example_gen.py --setup L2 --ti 595 \
    --ckpt $ROOT/L2/ckpt/epoch_60 --L2 --maxiter 30000 --early_iter 0 > example_L2.log 2>&1 &
wait
echo "=== example gen DONE $(date); replotting fig1 + noise curve ==="
$PY make_fig1.py --l1_ti 576 --l2_ti 595 --noise 2>&1
$PY make_noise_curve_example.py 2>&1
echo "=== ALL DONE $(date) ==="