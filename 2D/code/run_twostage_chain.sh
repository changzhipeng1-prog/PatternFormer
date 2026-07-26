#!/bin/bash
#SBATCH --job-name=2d_pde_2stage
#SBATCH --output=logs/2d_2stage_%j.out
#SBATCH --error=logs/2d_2stage_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=72:00:00

# CHAIN STEP 3 — 2D (-Δu-u²=-s·sinπx·sinπy), GT-referenced FEM physics KEPT as-is
# (problem-specific stiffness-matrix residual, NOT unified to 1D's FDM form).  TWO
# STAGES, same lr scheme as 1D_p:
#   Stage1 = WITHCONTEXT, warm-start from a2a4 two-stage best (LoRA+outproj+heads+
#            special transferred, input_proj fresh via param_dim 2->1), lr 1e-4/2e-5.
#   Stage2 = WITHOUTCONTEXT finetune of Stage1, lr 2e-5/5e-6, no_context=1.
# AE = 2D's own frozen autoencoder.  Early stop per stage.
set -euo pipefail
export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NOTE: 2D's lr_plateau_patience=25 (config.py), so EARLY_STOP must exceed it or the
# early-stop pre-empts every LR decay (the bug that crippled job 45761: LR never stepped,
# stopped at ep58/val 0.077 while still descending).  Set 50 to allow LR to decay 1-2x.
export EARLY_STOP_PATIENCE=50

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-$HERE}"
cd "${WORK_DIR}"
mkdir -p logs checkpoints_chain_s1 checkpoints_chain outputs/logs outputs/test_results

A2A4_CKPT="$HERE/../../a2a4/best_ckpt/best_model_OLD_baseline_jun22"
AE_CKPT="$HERE/../best_ckpt/best_model/autoencoder2d.pt"
INIT="./init_chain_a2a4"
if [ ! -d "${A2A4_CKPT}" ]; then echo "ERROR: a2a4 two-stage best missing at ${A2A4_CKPT}"; exit 1; fi
if [ ! -f "${AE_CKPT}" ]; then echo "ERROR: 2D AE missing"; exit 1; fi

# ---- assemble warm-start init: 2D's own AE + a2a4 Qwen-side (NO input_proj -> fresh) ----
rm -rf "${INIT}"; mkdir -p "${INIT}"
cp "${AE_CKPT}" "${INIT}/autoencoder2d.pt"
for f in adapter_config.json adapter_model.safetensors output_projector.pt dual_head.pt special_tokens.pt; do
    cp "${A2A4_CKPT}/$f" "${INIT}/$f"
done
echo "init assembled:"; ls -1 "${INIT}"

echo "=== [2D] STAGE 1 withcontext (warm from a2a4, lr 1e-4/2e-5) start $(date) ==="
torchrun --nproc_per_node=4 --master_port=29535 train.py \
    --batch_size 8 --grad_accum 4 --num_epochs 200 \
    --lr_projector 1e-4 --lr_lora 2e-5 \
    --lambda_mse 0.5 --lambda_pde 0.05 --pde_use_gt_ref 1 \
    --target_repeat_per_epoch 10 \
    --no_context_prob 0.0 --val_no_context_prob 0.0 \
    --unet_checkpoint "${AE_CKPT}" --resume_from "${INIT}" \
    --output_dir "./checkpoints_chain_s1"

if [ ! -d "./checkpoints_chain_s1/best_model" ]; then echo "ERROR: stage1 best_model missing"; exit 1; fi

echo "=== [2D] STAGE 2 nocontext (from stage1, lr 2e-5/5e-6) start $(date) ==="
torchrun --nproc_per_node=4 --master_port=29536 train.py \
    --batch_size 8 --grad_accum 4 --num_epochs 200 \
    --lr_projector 2e-5 --lr_lora 5e-6 \
    --lambda_mse 0.5 --lambda_pde 0.05 --pde_use_gt_ref 1 \
    --target_repeat_per_epoch 10 \
    --no_context_prob 1.0 --val_no_context_prob 1.0 \
    --unet_checkpoint "${AE_CKPT}" --resume_from "./checkpoints_chain_s1/best_model" \
    --output_dir "./checkpoints_chain"

echo "=== [2D] two-stage complete $(date) ==="
