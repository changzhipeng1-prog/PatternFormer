#!/bin/bash
#SBATCH --job-name=1dp_pde_2stage
#SBATCH --output=logs/1dp_2stage_%j.out
#SBATCH --error=logs/1dp_2stage_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=72:00:00

# CHAIN ROOT — 1D_p, GT-referenced (h²) physics, TWO STAGES matching the deconstructed
# 1D_p recipe:  Stage1 = WITHCONTEXT (from general Qwen, fresh LoRA, lr 1e-4/2e-5),
#               Stage2 = WITHOUTCONTEXT finetune of Stage1 (lr 2e-5/5e-6, no_context=1).
# Early stopping (EARLY_STOP_PATIENCE, default 15) auto-stops each stage.  Stage2's
# best_model is the chain's 1D_p deliverable that a2a4 warm-starts from.
set -euo pipefail
export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EARLY_STOP_PATIENCE=15

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-$HERE}"
cd "${WORK_DIR}"
mkdir -p logs checkpoints_pde_s1 checkpoints_pde outputs/logs outputs/test_results

AE="$HERE/../best_ckpt/best_model/unet.pt"
if [ ! -f "${AE}" ]; then echo "ERROR: AE missing at ${AE}"; exit 1; fi

echo "=== [1D_p] STAGE 1 withcontext (from-HF, lr 1e-4/2e-5) start $(date) ==="
torchrun --nproc_per_node=8 --master_port=29531 train.py \
    --phase phase2 --batch_size 8 --grad_accum 2 --num_epochs 100 \
    --lr_projector 1e-4 --lr_lora 2e-5 \
    --lambda_pde 0.05 --pde_use_gt_ref 1 --pde_warmup_start 5 --pde_warmup_end 30 \
    --target_repeat_per_epoch 1 \
    --no_context_prob 0.0 --val_no_context_prob 0.0 \
    --unet_checkpoint "${AE}" \
    --output_dir "./checkpoints_pde_s1"

if [ ! -d "./checkpoints_pde_s1/best_model" ]; then echo "ERROR: stage1 best_model missing"; exit 1; fi

echo "=== [1D_p] STAGE 2 nocontext (from stage1, lr 2e-5/5e-6) start $(date) ==="
torchrun --nproc_per_node=8 --master_port=29532 train.py \
    --phase phase2 --batch_size 8 --grad_accum 2 --num_epochs 100 \
    --lr_projector 2e-5 --lr_lora 5e-6 \
    --lambda_pde 0.05 --pde_use_gt_ref 1 --pde_warmup_start 5 --pde_warmup_end 30 \
    --target_repeat_per_epoch 1 \
    --no_context_prob 1.0 --val_no_context_prob 1.0 \
    --unet_checkpoint "${AE}" \
    --resume_from "./checkpoints_pde_s1/best_model" \
    --output_dir "./checkpoints_pde"

echo "=== [1D_p] two-stage complete $(date) ==="
