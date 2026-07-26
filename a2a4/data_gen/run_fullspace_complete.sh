#!/bin/bash
#SBATCH --job-name=bvp_complete
#SBATCH --output=logs/fullspace_complete_%j.out
#SBATCH --error=logs/fullspace_complete_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:6
#SBATCH --mem=160G
#SBATCH --time=10:00:00

set -euo pipefail

export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1

# Auto-detect free GPUs (mem < 500MB, util == 0)
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 500 && $3 == 0 {printf "%s,", $1}' \
    | sed 's/,$//')
N_FREE=$(echo "$FREE_GPUS" | tr ',' '\n' | grep -c .)
echo "Free GPUs detected: $FREE_GPUS  (count=$N_FREE)"

if [ "$N_FREE" -lt 2 ]; then
    FREE_GPUS="1,2,4,5,6"
    echo "Falling back to CUDA_VISIBLE_DEVICES=$FREE_GPUS"
fi
export CUDA_VISIBLE_DEVICES="$FREE_GPUS"

HERE="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="${SLURM_SUBMIT_DIR:-$HERE}"
cd "${WORK_DIR}"
mkdir -p logs data

# 0-indexed device sequence within CUDA_VISIBLE_DEVICES
N_DEV=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
DEV_SEQ=$(seq 0 $((N_DEV-1)) | paste -sd,)

echo "=========================================="
echo "Full-domain BFS  (complete coverage)"
echo "  a4 in [0.05, 3.00] x 1200   Δa4 ≈ 0.00246"
echo "  a2 in [-40,  -2.0] x  380   Δa2 ≈ 0.100"
echo "  Total grid cells: 456,000"
echo "  Seeds: bvp_refined.pt + bvp_fullspace.pt (merged, best-k-wins)"
echo "  GPUs:  $CUDA_VISIBLE_DEVICES  (--devices $DEV_SEQ)"
echo "  min_k: 1  (save any cell with at least 1 solution)"
echo "Start: $(date)"
echo "=========================================="

nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true

python generate_fullspace_gpu.py \
    --output        ./data/bvp_complete.pt \
    --na4           1200 \
    --na2           380 \
    --a4_min        0.05 \
    --a4_max        3.00 \
    --a2_min       -40.0 \
    --a2_max        -2.0 \
    --seed_datasets ./data/bvp_refined.pt,./data/bvp_fullspace.pt \
    --min_k         1 \
    --devices       "$DEV_SEQ" \
    --batch_limit   8192

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
