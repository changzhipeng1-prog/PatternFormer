#!/bin/bash
#SBATCH --job-name=ex2_scratch
#SBATCH --output=logs/ex2_scratch_%j.out
#SBATCH --error=logs/ex2_scratch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=72:00:00

# CROSS-PDE WARM-START ABLATION — Ex1->Ex2 transition, SCRATCH arm.
#   Trains a2a4 (Ex2, 1D two-parameter) with the SAME two-stage recipe as the
#   delivered warm-start model (run_twostage_pde.sh), but initialized FRESH from
#   the general pretrained Qwen (--no_warmstart 1) instead of warm-starting from
#   the Ex1 (1D_p) checkpoint. Everything else identical.
#   Effective batch matched to the 8-GPU deliverable: 4 GPU x batch 4 x accum 4 = 64
#   (deliverable was 8 GPU x batch 4 x accum 2 = 64).
#   Compare against the delivered warm model a2a4/best_ckpt/best_model (eval via
#   a2a4/test/run_generate_test.sh + run_compute_stats.sh).
set -euo pipefail
export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EARLY_STOP_PATIENCE=15

WORK_DIR="/home/zfc5231/work/BBBB_qwen_pde_branch/paper/a2a4/code"
cd "${WORK_DIR}"
mkdir -p logs checkpoints_scratch_s1 checkpoints_scratch outputs/logs outputs/test_results

AE="${WORK_DIR}/checkpoints/unet_best.pt"
if [ ! -f "${AE}" ]; then echo "ERROR: a2a4 AE missing at ${AE}"; exit 1; fi

echo "=== [Ex2 SCRATCH] STAGE 1 withcontext (FRESH Qwen, no warm-start, lr 1e-4/2e-5) start $(date) ==="
torchrun --nproc_per_node=4 --master_port=29537 train.py \
    --phase phase2 --num_epochs 60 --batch_size 4 --grad_accum 4 \
    --lr_projector 1e-4 --lr_lora 2e-5 \
    --lambda_pde 0.05 --pde_use_gt_ref 1 --pde_warmup_start 5 --pde_warmup_end 30 \
    --target_repeat_per_epoch 1 \
    --no_context_prob 0.0 --val_no_context_prob 0.0 \
    --no_warmstart 1 --unet_checkpoint "${AE}" \
    --output_dir "./checkpoints_scratch_s1"

if [ ! -d "./checkpoints_scratch_s1/best_model" ]; then echo "ERROR: stage1 best_model missing"; exit 1; fi

echo "=== [Ex2 SCRATCH] STAGE 2 nocontext (from stage1, lr 2e-5/5e-6) start $(date) ==="
torchrun --nproc_per_node=4 --master_port=29538 train.py \
    --phase phase2 --num_epochs 60 --batch_size 4 --grad_accum 4 \
    --lr_projector 2e-5 --lr_lora 5e-6 \
    --lambda_pde 0.05 --pde_use_gt_ref 1 --pde_warmup_start 5 --pde_warmup_end 30 \
    --target_repeat_per_epoch 1 \
    --no_context_prob 1.0 --val_no_context_prob 1.0 \
    --unet_checkpoint "${AE}" \
    --resume_from "./checkpoints_scratch_s1/best_model" \
    --output_dir "./checkpoints_scratch"

echo "=== [Ex2 SCRATCH] two-stage complete $(date) ==="
