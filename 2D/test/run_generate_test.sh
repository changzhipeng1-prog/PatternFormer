#!/bin/bash
#SBATCH --job-name=gen_2D
#SBATCH --output=/path/to/PatternFormer/2D/test/gen_2D_%j.out
#SBATCH --error=/path/to/PatternFormer/2D/test/gen_2D_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
set -euo pipefail
export PATH=/path/to/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/path/to/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /path/to/PatternFormer/2D/test
python generate_test.py
echo "=== [2D test generate] done $(date) ==="
