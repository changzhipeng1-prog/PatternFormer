#!/bin/bash
#SBATCH --job-name=gsabl
#SBATCH --output=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/ablation/logs/gsabl_%j.out
#SBATCH --error=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/ablation/logs/gsabl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
# GS L1 ablation: TWO-STAGE fixed-K from the BARE backbone (no stop warm-start).
#   RAND=0 -> pretrained backbone ; RAND=1 -> random-init frozen backbone + LoRA.
#   Phase2 = with-context (fresh) ; Phase3 = no-context (resume from Phase2 best).
#   Both arms share the SAME frozen conv-AE and identical recipe/seed.
set -euo pipefail
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ABL=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/ablation
CODE=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/code
AE=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/ckpt/epoch_60/autoencoder2d.pt
cd "$CODE"
RAND=${RAND:-0}; TAG=${TAG:-arm}; PORT=${PORT:-29560}
OUT2="$ABL/ckpt_${TAG}_p2"; OUT3="$ABL/ckpt_${TAG}_p3"

echo "=== [$TAG] PHASE2 with-context fixed-K (RAND=$RAND) $(date) ==="
torchrun --nproc_per_node=4 --master_port=$PORT train.py \
    --batch_size 8 --grad_accum 4 --num_epochs 80 \
    --lr_projector 1e-4 --lr_lora 2e-5 \
    --no_context_prob 0.0 --context_mode local \
    --ss_prob 0.25 --ss_warmup_start 0 --ss_warmup_end 15 \
    --unet_checkpoint "$AE" --random_init_backbone $RAND \
    --output_dir "$OUT2"

echo "=== [$TAG] PHASE3 no-context fixed-K (RAND=$RAND) $(date) ==="
torchrun --nproc_per_node=4 --master_port=$((PORT+1)) train.py \
    --batch_size 8 --grad_accum 4 --num_epochs 60 \
    --lr_projector 3e-5 --lr_lora 1e-5 \
    --no_context_prob 1.0 --val_no_context_prob 1.0 --context_mode local \
    --resume_from "$OUT2/best_model" --random_init_backbone $RAND \
    --output_dir "$OUT3"
echo "=== [$TAG] two-stage fixed-K complete $(date) ==="
