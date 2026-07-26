#!/bin/bash
#SBATCH --job-name=gsTB
#SBATCH --output=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L2/try/logs/gsTB_%j.out
#SBATCH --error=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L2/try/logs/gsTB_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# Appendix experiment: at TRAINING-set params, LLM-MSO det+noise -> refine -> dedup,
# label each distinct solution GT-matched vs beyond-data. Saves Fig3B-style data.
set -euo pipefail
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TRY=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L2/try
mkdir -p "$TRY/logs" "$TRY/results"
cd "$TRY"
python eval_train_beyond.py \
    --code ../code --ckpt ../ckpt/epoch_60 \
    --sigma 0.1 --passes 5 --n_params 8 \
    --maxiter 30000 --early_iter 0 \
    --out "$TRY/results/train_beyond.pt"
echo "=== DONE $(date) ==="
