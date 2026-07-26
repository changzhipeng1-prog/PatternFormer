#!/bin/bash
#SBATCH --job-name=1dp_abl_random
#SBATCH --output=logs/arm2_random_%j.out
#SBATCH --error=logs/arm2_random_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=72:00:00

# ABLATION ARM 2 — does the PRETRAINED backbone matter?
#   SAME architecture (Qwen2.5-7B), SAME LoRA/projectors/heads, SAME recipe as the
#   delivered 1D_p model, but the frozen backbone is RANDOM-initialised, not pretrained.
#   The only difference vs Arm 1 is --random_init_backbone 1.
#   eff-batch 8*4*4 = 128 matches the delivered 8-GPU run (8*2*8 = 128).
set -euo pipefail
export PATH=/path/to/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/path/to/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EARLY_STOP_PATIENCE=15

ABL_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
CODE_DIR="$(cd "${ABL_DIR}/../code" && pwd)"
AE="${ABL_DIR}/../best_ckpt/best_model/unet.pt"
mkdir -p "${ABL_DIR}/logs"
[ -f "${AE}" ] || { echo "ERROR: AE missing at ${AE}"; exit 1; }
cd "${CODE_DIR}"

OUT1="${ABL_DIR}/ckpt_arm2_random_s1"
OUT2="${ABL_DIR}/ckpt_arm2_random"

echo "=== [ARM2 random-init] STAGE 1 withcontext (lr 1e-4/2e-5) $(date) ==="
torchrun --nproc_per_node=4 --master_port=29551 train.py \
    --phase phase2 --batch_size 8 --grad_accum 4 --num_epochs 100 \
    --lr_projector 1e-4 --lr_lora 2e-5 \
    --lambda_pde 0.05 --pde_use_gt_ref 1 --pde_warmup_start 5 --pde_warmup_end 30 \
    --target_repeat_per_epoch 1 \
    --no_context_prob 0.0 --val_no_context_prob 0.0 \
    --random_init_backbone 1 \
    --unet_checkpoint "${AE}" \
    --output_dir "${OUT1}"

[ -d "${OUT1}/best_model" ] || { echo "ERROR: stage1 best_model missing"; exit 1; }

echo "=== [ARM2 random-init] STAGE 2 nocontext (lr 2e-5/5e-6) $(date) ==="
torchrun --nproc_per_node=4 --master_port=29552 train.py \
    --phase phase2 --batch_size 8 --grad_accum 4 --num_epochs 100 \
    --lr_projector 2e-5 --lr_lora 5e-6 \
    --lambda_pde 0.05 --pde_use_gt_ref 1 --pde_warmup_start 5 --pde_warmup_end 30 \
    --target_repeat_per_epoch 1 \
    --no_context_prob 1.0 --val_no_context_prob 1.0 \
    --random_init_backbone 1 \
    --unet_checkpoint "${AE}" \
    --resume_from "${OUT1}/best_model" \
    --output_dir "${OUT2}"

echo "=== [ARM2 random-init] two-stage complete $(date) ==="
