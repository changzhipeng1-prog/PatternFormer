#!/bin/bash
#SBATCH --job-name=a2a4_phase1
#SBATCH --output=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/a2a4/code/logs/phase1_%j.out
#SBATCH --error=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/a2a4/code/logs/phase1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
# Phase 1: UNet autoencoder pretraining on the FDM-CORRECTED, zero-free a2a4
# dataset (paper/a2a4/data).  Must precede the two-stage Qwen training, whose
# frozen AE is this UNet.  Output: paper/a2a4/code/checkpoints/unet_best.pt
set -euo pipefail
export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$HERE"
cd "${WORK_DIR}"
mkdir -p logs checkpoints outputs/logs

echo "=== [a2a4] Phase 1 UNet AE pretrain (corrected data) start $(date) ==="
# IMPORTANT: write to the staging path (NOT the default best_ckpt/best_model/unet.pt,
# which would clobber the untouched baseline). Phase 2 reads this same path as its AE.
torchrun --nproc_per_node=8 --master_port=29601 train.py \
    --phase phase1 --phase1_epochs 200 \
    --unet_checkpoint ./checkpoints/unet_best.pt
echo "=== [a2a4] Phase 1 done $(date) ==="
[ -f "./checkpoints/unet_best.pt" ] && echo "unet_best.pt OK" || { echo "ERROR: unet_best.pt missing"; exit 1; }
