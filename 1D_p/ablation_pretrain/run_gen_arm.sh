#!/bin/bash
#SBATCH --job-name=abl_gen
#SBATCH --output=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/1D_p/ablation_pretrain/logs/abl_gen_%j.out
#SBATCH --error=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/1D_p/ablation_pretrain/logs/abl_gen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
set -euo pipefail
export PATH=/home/zfc5231/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/home/zfc5231/anaconda3/lib:$LD_LIBRARY_PATH
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$HERE"
echo "=== [abl_gen] ARM_CKPT=${ARM_CKPT} ARM_RANDOM=${ARM_RANDOM} -> ${ARM_OUT} $(date) ==="
python generate_arm.py
echo "=== [abl_gen] done $(date) ==="
