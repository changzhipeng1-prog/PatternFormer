#!/bin/bash
#SBATCH --job-name=gen_1D_p
#SBATCH --output=/path/to/PatternFormer/1D_p/test/gen_1D_p_%j.out
#SBATCH --error=/path/to/PatternFormer/1D_p/test/gen_1D_p_%j.err
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
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$HERE"
python generate_test.py
echo "=== [1D_p test generate] done $(date) ==="
