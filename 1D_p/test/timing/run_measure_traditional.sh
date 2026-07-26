#!/bin/bash
#SBATCH --job-name=time_trad_1Dp
#SBATCH --output=/path/to/PatternFormer/1D_p/test/timing/time_trad_%j.out
#SBATCH --error=/path/to/PatternFormer/1D_p/test/timing/time_trad_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
set -euo pipefail
export PATH=/path/to/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/path/to/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1
# match the OUR-method timing job (OMP_NUM_THREADS=8) so the Newton/BLAS cost
# is measured under identical settings -- apples-to-apples wall-clock.
export OMP_NUM_THREADS=8
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
python measure_traditional.py
echo "=== [1D_p traditional timing] done $(date) ==="
