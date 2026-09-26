#!/bin/bash
#SBATCH --job-name=pf_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
: "${PF_REPO:?Set the repository directory}"
: "${PF_RESULTS:?Set a new results root}"
: "${PF_PYTHON:?Set the Python interpreter}"
PROBLEMS=(1D_p a2a4 2D)
PROBLEM="${PROBLEMS[${SLURM_ARRAY_TASK_ID:?Submit as an array with indices 0-2}]}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
printf 'UTC start: '; date -u +%FT%TZ
printf 'job=%s task=%s problem=%s devices=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$PROBLEM" "${CUDA_VISIBLE_DEVICES:-unset}"
"$PF_PYTHON" "$PF_REPO/experiments/evaluate.py" \
  --problem "$PROBLEM" --output "$PF_RESULTS/$PROBLEM" \
  --limit "${PF_LIMIT:-2}" --cpus "$SLURM_CPUS_PER_TASK"
printf 'UTC end: '; date -u +%FT%TZ
