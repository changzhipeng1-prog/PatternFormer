#!/bin/bash
#SBATCH --job-name=L2_3a_mix
#SBATCH --output=logs/L2_3a_mix_%j.out
#SBATCH --error=logs/L2_3a_mix_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=24:00:00

# Step-3a: EXPANDED L2 data + MIXED init (L1-ord LoRA/special_tokens prior +
# L2 AE/dual_head/input_projector latent stack). Ordered no-STOP fixed-K=24.
set -euo pipefail
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1

cd "$(dirname "$(readlink -f "$0")")/../code"
mkdir -p logs
RESUME=./init_mixed_L1prior
D=../data
for f in "$D/gs_lookup_L2.pt" "$D/train_p_idx_L2.pt" "$D/val_p_idx_L2.pt" "$D/norm_stats_L2.pt"; do
  [ -f "$f" ] || { echo "ERROR: expanded data missing: $f"; exit 1; }
done

echo "=== Step-3a mixed-init on expanded L2 data, 2-GPU DDP $(date) ==="
torchrun --nproc_per_node=2 --master_port=29541 train.py \
    --batch_size 8 --grad_accum 4 --num_epochs 60 \
    --lr_projector 3e-5 --lr_lora 1e-5 \
    --no_context_prob 1.0 --val_no_context_prob 1.0 --context_mode local \
    --data_lookup "$D/gs_lookup_L2.pt" --train_idx "$D/train_p_idx_L2.pt" \
    --val_idx "$D/val_p_idx_L2.pt" --norm_stats "$D/norm_stats_L2.pt" \
    --resume_from "$RESUME" \
    --output_dir ./checkpoints_final
echo "=== Step-3a done $(date) ==="
