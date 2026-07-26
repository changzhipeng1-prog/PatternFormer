#!/bin/bash
#SBATCH --job-name=a2a4_timing
#SBATCH --output=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/a2a4/test/timing/a2a4_timing_%j.out
#SBATCH --error=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/a2a4/test/timing/a2a4_timing_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
set -euo pipefail
export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "=== [a2a4 timing] TRADITIONAL multi-start (CPU)  $(date) ==="
python measure_trad_a2a4.py

echo "=== [a2a4 timing] OURS (Qwen forward + Newton refine, GPU)  $(date) ==="
python measure_ours_a2a4.py

echo "=== [a2a4 timing] done $(date) ==="
