#!/bin/bash
#SBATCH --job-name=ex3_scratch
#SBATCH --output=logs/ex3_scratch_%j.out
#SBATCH --error=logs/ex3_scratch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=72:00:00

# CROSS-PDE WARM-START ABLATION — Ex2->Ex3 transition, SCRATCH arm.
#   Trains 2D (Ex3) with the SAME two-stage recipe as the delivered warm-start
#   model (run_twostage_chain.sh), but initialized FRESH from the general
#   pretrained Qwen (no --resume_from, no --init_from_v2) instead of warm-starting
#   from the Ex2 (a2a4) checkpoint. Everything else identical (4 GPU, batch 8,
#   accum 4 -> eff 128, same as the delivered warm model).
#   Compare against the delivered warm model 2D/best_ckpt/best_model (eval via
#   2D/test/run_generate_test.sh + run_compute_stats.sh).
set -euo pipefail
export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 2D lr_plateau_patience=25 (config.py); EARLY_STOP must exceed it (same as warm).
export EARLY_STOP_PATIENCE=50

WORK_DIR="/home/zfc5231/work/BBBB_qwen_pde_branch/paper/2D/code"
cd "${WORK_DIR}"
mkdir -p logs checkpoints_scratch_s1 checkpoints_scratch outputs/logs outputs/test_results

AE_CKPT="${WORK_DIR}/../best_ckpt/best_model/autoencoder2d.pt"
if [ ! -f "${AE_CKPT}" ]; then echo "ERROR: 2D AE missing at ${AE_CKPT}"; exit 1; fi

echo "=== [Ex3 SCRATCH] STAGE 1 withcontext (FRESH Qwen, no warm-start, lr 1e-4/2e-5) start $(date) ==="
torchrun --nproc_per_node=4 --master_port=29539 train.py \
    --batch_size 8 --grad_accum 4 --num_epochs 200 \
    --lr_projector 1e-4 --lr_lora 2e-5 \
    --lambda_mse 0.5 --lambda_pde 0.05 --pde_use_gt_ref 1 \
    --target_repeat_per_epoch 10 \
    --no_context_prob 0.0 --val_no_context_prob 0.0 \
    --unet_checkpoint "${AE_CKPT}" \
    --output_dir "./checkpoints_scratch_s1"

if [ ! -d "./checkpoints_scratch_s1/best_model" ]; then echo "ERROR: stage1 best_model missing"; exit 1; fi

echo "=== [Ex3 SCRATCH] STAGE 2 nocontext (from stage1, lr 2e-5/5e-6) start $(date) ==="
torchrun --nproc_per_node=4 --master_port=29540 train.py \
    --batch_size 8 --grad_accum 4 --num_epochs 200 \
    --lr_projector 2e-5 --lr_lora 5e-6 \
    --lambda_mse 0.5 --lambda_pde 0.05 --pde_use_gt_ref 1 \
    --target_repeat_per_epoch 10 \
    --no_context_prob 1.0 --val_no_context_prob 1.0 \
    --unet_checkpoint "${AE_CKPT}" --resume_from "./checkpoints_scratch_s1/best_model" \
    --output_dir "./checkpoints_scratch"

echo "=== [Ex3 SCRATCH] two-stage complete $(date) ==="
