#!/bin/bash
#SBATCH --job-name=ours_2d_timing
#SBATCH --output=/path/to/PatternFormer/2D/test/timing/ours_2d_%j.out
#SBATCH --error=/path/to/PatternFormer/2D/test/timing/ours_2d_%j.err
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
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /path/to/PatternFormer/2D/test/timing
python measure_ours_2d.py
echo "=== [2D ours timing] done $(date) ==="
