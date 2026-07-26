#!/bin/bash
#SBATCH --job-name=steps_trad_1Dp
#SBATCH --output=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/1D_p/test/timing/steps_trad_%j.out
#SBATCH --error=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/1D_p/test/timing/steps_trad_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
set -euo pipefail
export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
python measure_traditional_steps.py
echo "=== [1D_p traditional steps] done $(date) ==="
