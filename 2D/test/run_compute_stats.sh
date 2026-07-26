#!/bin/bash
#SBATCH --job-name=stat_2D
#SBATCH --output=/path/to/PatternFormer/2D/test/stat_2D_%j.out
#SBATCH --error=/path/to/PatternFormer/2D/test/stat_2D_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=4:00:00
set -euo pipefail
export PATH=/path/to/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/path/to/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 STAT_NCPU=32
cd /path/to/PatternFormer/2D/test
python compute_stats.py
echo "=== [2D stats] done $(date) ==="
