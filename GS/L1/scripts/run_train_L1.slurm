#!/bin/bash
#SBATCH --job-name=ordA_L1
#SBATCH --output=logs/ordA_L1_%j.out
#SBATCH --error=logs/ordA_L1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# qwen_ord A (L=1): ORDERED, NO-STOP, fixed-K=24 finetune from the STOP best.
#   head k<K_t : teacher forcing + ordered MSE + scheduled sampling + small hinge physics
#   tail k>=K_t: self-pred free-run + physics-only (best-effort discovery beyond GT)
# Eval = generate_fixed_k(24) (already the production path). No-context (matches base + eval).
set -euo pipefail
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1

cd "$(dirname "$(readlink -f "$0")")/../code"
mkdir -p logs
RESUME="$(dirname "$(readlink -f "$0")")/../ckpt/stop_base"
[ -d "$RESUME" ] || { echo "ERROR: resume dir $RESUME missing"; exit 1; }

echo "=== qwen_ord A (L=1) ordered no-STOP fixed-K finetune, 4-GPU DDP $(date) ==="
torchrun --nproc_per_node=4 --master_port=29531 train.py \
    --batch_size 8 --grad_accum 4 --num_epochs 60 \
    --lr_projector 3e-5 --lr_lora 1e-5 \
    --no_context_prob 1.0 --val_no_context_prob 1.0 --context_mode local \
    --resume_from "$RESUME" \
    --output_dir ./checkpoints_final
echo "=== qwen_ord A (L=1) done $(date) ==="
