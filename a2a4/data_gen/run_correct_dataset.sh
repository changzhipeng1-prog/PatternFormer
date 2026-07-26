#!/bin/bash
#SBATCH --job-name=fix_a2a4_data
#SBATCH --output=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/a2a4/data_gen/fix_data_%j.out
#SBATCH --error=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/a2a4/data_gen/fix_data_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=4:00:00
set -euo pipefail
export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 NCPU=32
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
python correct_dataset_fdm.py
echo "=== [a2a4 data correction] done $(date) ==="
