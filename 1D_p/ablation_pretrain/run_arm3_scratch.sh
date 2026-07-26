#!/bin/bash
#SBATCH --job-name=1dp_abl_scratch
#SBATCH --output=logs/arm3_scratch_%j.out
#SBATCH --error=logs/arm3_scratch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=72:00:00

# ABLATION ARM 3 — do you even need a 7B LLM?
#   A SMALL transformer (hidden 1024, 12 layers, ~0.2B), random-init, FULLY trained from
#   scratch (no LoRA: every backbone param is trainable). Tests whether a purpose-built
#   small model trained properly can match the frozen-pretrained-7B + LoRA.
#   NOTE: from-scratch full-FT is lr-sensitive; lr_lora here is the BACKBONE lr (all
#   backbone params land in that optimizer group). Tune if it does not converge.
set -euo pipefail
export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EARLY_STOP_PATIENCE=20

ABL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
CODE_DIR="$(cd "${ABL_DIR}/../code" && pwd)"
AE="${ABL_DIR}/../best_ckpt/best_model/unet.pt"
mkdir -p "${ABL_DIR}/logs"
[ -f "${AE}" ] || { echo "ERROR: AE missing at ${AE}"; exit 1; }
cd "${CODE_DIR}"

OUT1="${ABL_DIR}/ckpt_arm3_scratch_s1"
OUT2="${ABL_DIR}/ckpt_arm3_scratch"
SCR="--full_finetune_backbone 1 --scratch_hidden 1024 --scratch_layers 12"

echo "=== [ARM3 scratch] STAGE 1 withcontext (backbone lr 3e-4) $(date) ==="
torchrun --nproc_per_node=4 --master_port=29555 train.py \
    --phase phase2 --batch_size 8 --grad_accum 4 --num_epochs 150 \
    --lr_projector 1e-4 --lr_lora 3e-4 \
    --lambda_pde 0.05 --pde_use_gt_ref 1 --pde_warmup_start 5 --pde_warmup_end 30 \
    --target_repeat_per_epoch 1 \
    --no_context_prob 0.0 --val_no_context_prob 0.0 \
    ${SCR} \
    --unet_checkpoint "${AE}" \
    --output_dir "${OUT1}"

[ -d "${OUT1}/best_model" ] || { echo "ERROR: stage1 best_model missing"; exit 1; }

echo "=== [ARM3 scratch] STAGE 2 nocontext (backbone lr 1e-4) $(date) ==="
torchrun --nproc_per_node=4 --master_port=29556 train.py \
    --phase phase2 --batch_size 8 --grad_accum 4 --num_epochs 150 \
    --lr_projector 5e-5 --lr_lora 1e-4 \
    --lambda_pde 0.05 --pde_use_gt_ref 1 --pde_warmup_start 5 --pde_warmup_end 30 \
    --target_repeat_per_epoch 1 \
    --no_context_prob 1.0 --val_no_context_prob 1.0 \
    ${SCR} \
    --unet_checkpoint "${AE}" \
    --resume_from "${OUT1}/best_model" \
    --output_dir "${OUT2}"

echo "=== [ARM3 scratch] two-stage complete $(date) ==="
