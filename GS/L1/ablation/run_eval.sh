#!/bin/bash
#SBATCH --job-name=gsablEv
#SBATCH --output=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/ablation/logs/gsablEv_%j.out
#SBATCH --error=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/ablation/logs/gsablEv_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Ablation eval: #distinct solutions + coverage on BOTH p3 checkpoints
# (pretrained backbone vs random-init backbone). Deterministic (noise=0) is the
# headline comparison; noise=0.1 added for the diversity-vs-passes view.
set -euo pipefail
source activate torch124
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ABL=/home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/ablation
EVOUT="$ABL/eval"; mkdir -p "$EVOUT"
cd /home/zfc5231/work/BBBB_qwen_pde_branch/paper/GS/L1/code

for arm in arm1_pretrained arm2_random; do
  echo "===== $arm  DET (noise=0) $(date) ====="
  python eval_ord.py --ckpt "$ABL/ckpt_${arm}_p3/best_model" \
      --noise 0.0 --out "$EVOUT/${arm}_det.json"
  echo "===== $arm  NOISE=0.1 $(date) ====="
  python eval_ord.py --ckpt "$ABL/ckpt_${arm}_p3/best_model" \
      --noise 0.1 --out "$EVOUT/${arm}_noise01.json"
done
echo "=== DONE $(date) ==="
