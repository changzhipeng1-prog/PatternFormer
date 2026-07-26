#!/bin/bash
#SBATCH --job-name=a2a4_pde_2stage
#SBATCH --output=logs/a2a4_2stage_%j.out
#SBATCH --error=logs/a2a4_2stage_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=72:00:00

# CHAIN STEP 2 — 1D_(a4,a2), GT-referenced physics with h²-scaling UNIFIED with 1D_p
# (BC stays ghost = data-matched).  TWO STAGES, same lr scheme as 1D_p:
#   Stage1 = WITHCONTEXT, warm-start from 1D_p two-stage best (fresh input_proj 1->2),
#            lr 1e-4/2e-5.
#   Stage2 = WITHOUTCONTEXT finetune of Stage1, lr 2e-5/5e-6, no_context=1.
# AE = a2a4's own frozen Phase-1 UNet (read-only baseline).  Early stop per stage.
set -euo pipefail
export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EARLY_STOP_PATIENCE=15

# Retrain lives in the PAPER project (self-contained): train.py + config.py here,
# data read from paper/a2a4/data (the FDM-corrected dataset).
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$HERE"
cd "${WORK_DIR}"
mkdir -p logs checkpoints_pde_s1 checkpoints_pde outputs/logs outputs/test_results

PRETRAINED_1DP="$HERE/../../1D_p/best_ckpt/best_model"
# AE = the freshly retrained UNet on the FDM-corrected, zero-free data (Phase 1 above)
AE="$HERE/checkpoints/unet_best.pt"
if [ ! -d "${PRETRAINED_1DP}" ]; then echo "ERROR: 1D_p two-stage best missing at ${PRETRAINED_1DP}"; exit 1; fi
if [ ! -f "${AE}" ]; then echo "ERROR: a2a4 AE missing at ${AE}"; exit 1; fi

echo "=== [a2a4] STAGE 1 withcontext (warm from 1D_p, lr 1e-4/2e-5) start $(date) ==="
torchrun --nproc_per_node=8 --master_port=29533 train.py \
    --phase phase2 --num_epochs 60 --batch_size 4 --grad_accum 2 \
    --lr_projector 1e-4 --lr_lora 2e-5 \
    --lambda_pde 0.05 --pde_use_gt_ref 1 --pde_warmup_start 5 --pde_warmup_end 30 \
    --target_repeat_per_epoch 1 \
    --no_context_prob 0.0 --val_no_context_prob 0.0 \
    --pretrained_1dp "${PRETRAINED_1DP}" --unet_checkpoint "${AE}" \
    --output_dir "./checkpoints_pde_s1"

if [ ! -d "./checkpoints_pde_s1/best_model" ]; then echo "ERROR: stage1 best_model missing"; exit 1; fi

echo "=== [a2a4] STAGE 2 nocontext (from stage1, lr 2e-5/5e-6) start $(date) ==="
torchrun --nproc_per_node=8 --master_port=29534 train.py \
    --phase phase2 --num_epochs 60 --batch_size 4 --grad_accum 2 \
    --lr_projector 2e-5 --lr_lora 5e-6 \
    --lambda_pde 0.05 --pde_use_gt_ref 1 --pde_warmup_start 5 --pde_warmup_end 30 \
    --target_repeat_per_epoch 1 \
    --no_context_prob 1.0 --val_no_context_prob 1.0 \
    --unet_checkpoint "${AE}" \
    --resume_from "./checkpoints_pde_s1/best_model" \
    --output_dir "./checkpoints_pde"

echo "=== [a2a4] two-stage complete $(date) ==="
