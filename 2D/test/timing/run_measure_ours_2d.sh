#!/bin/bash
#SBATCH --job-name=ours_2d_timing
#SBATCH --output=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/2D/test/timing/ours_2d_%j.out
#SBATCH --error=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/2D/test/timing/ours_2d_%j.err
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
cd /home/zfc5231/work/BBBB_qwen_pde_branch/paper/2D/test/timing
python measure_ours_2d.py
echo "=== [2D ours timing] done $(date) ==="
