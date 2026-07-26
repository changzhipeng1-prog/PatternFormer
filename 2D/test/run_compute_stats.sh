#!/bin/bash
#SBATCH --job-name=stat_2D
#SBATCH --output=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/2D/test/stat_2D_%j.out
#SBATCH --error=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/2D/test/stat_2D_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
set -euo pipefail
export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 STAT_NCPU=32
cd /home/zfc5231/work/BBBB_qwen_pde_branch/paper/2D/test
python compute_stats.py
echo "=== [2D stats] done $(date) ==="
