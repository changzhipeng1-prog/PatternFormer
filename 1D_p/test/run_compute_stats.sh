#!/bin/bash
#SBATCH --job-name=stat_1Dp
#SBATCH --output=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/1D_p/test/stat_1Dp_%j.out
#SBATCH --error=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/1D_p/test/stat_1Dp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
set -euo pipefail
export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 STAT_NCPU=32
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
python compute_stats.py
echo "=== [1D_p stats] done $(date) ==="
