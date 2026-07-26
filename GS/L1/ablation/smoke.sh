#!/bin/bash
#SBATCH --job-name=gsabl_smoke
#SBATCH --output=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/ablation/logs/smoke_%j.out
#SBATCH --error=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/ablation/logs/smoke_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=0:30:00
set -euo pipefail
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
CODE=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/code
AE=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/ckpt/epoch_60/autoencoder2d.pt
cd "$CODE"
echo "=== SMOKE: random-init fixed-K Phase2 (with-context), 5 steps $(date) ==="
python train.py \
    --batch_size 4 --grad_accum 1 --num_epochs 1 \
    --no_context_prob 0.0 --context_mode local \
    --ss_prob 0.25 --unet_checkpoint "$AE" \
    --random_init_backbone 1 \
    --output_dir /home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/ablation/_smoke_out \
    --smoke 5
echo "=== SMOKE done $(date) ==="
