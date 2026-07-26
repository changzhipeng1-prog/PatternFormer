#!/bin/bash
#SBATCH --job-name=time_ours_1Dp
#SBATCH --output=/path/to/PatternFormer/1D_p/test/timing/time_ours_%j.out
#SBATCH --error=/path/to/PatternFormer/1D_p/test/timing/time_ours_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
set -euo pipefail
export PATH=/path/to/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/path/to/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
python measure_ours.py
echo "=== [1D_p ours timing] done $(date) ==="
