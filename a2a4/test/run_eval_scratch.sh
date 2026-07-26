#!/bin/bash
#SBATCH --job-name=eval_ex2scr
#SBATCH --output=/path/to/PatternFormer/a2a4/test/eval_scratch_%j.out
#SBATCH --error=/path/to/PatternFormer/a2a4/test/eval_scratch_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
# Evaluate the Ex2 SCRATCH (cold-start, --no_warmstart) checkpoint on the a2a4 test
# set and write stats into test/scratch/ so the shipped warm-start baseline artifacts
# (test/generated_solutions.pt, test/stats.*) are NOT overwritten.
set -euo pipefail
export PATH=/path/to/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/path/to/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True STAT_NCPU=32

HERE="/path/to/PatternFormer/a2a4/test"
cd "$HERE"
mkdir -p scratch

export CKPT_DIR="/path/to/PatternFormer/a2a4/code/checkpoints_scratch/best_model"
export GEN_OUT="$HERE/scratch/generated_solutions.pt"
export STATS_OUT="$HERE/scratch/stats.pt"
export STATS_CSV="$HERE/scratch/stats.csv"

echo "=== [ex2 scratch] generate $(date) ===  CKPT=$CKPT_DIR"
python generate_test.py
echo "=== [ex2 scratch] stats $(date) ==="
python compute_stats.py
echo "=== [ex2 scratch eval] done $(date) ==="
