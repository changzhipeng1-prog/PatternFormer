#!/bin/bash
#SBATCH --job-name=pf_gs_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
set -euo pipefail
: "${PF_REPO:?Set repository directory}"
: "${PF_RESULTS:?Set new output directory}"
: "${PF_PYTHON:?Set Python interpreter}"
: "${PF_SETUP:?Set L1 or L2}"
if [ -e "$PF_RESULTS" ]; then echo "Output already exists: $PF_RESULTS"; exit 2; fi
mkdir -p "$PF_RESULTS"
cd "$PF_RESULTS"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
ARGS=(--ckpt "$PF_REPO/GS/$PF_SETUP/ckpt/epoch_60" --K 24 --idx_start "${PF_START:-0}" --idx_end "${PF_END:--1}" --n "${PF_N:-1}" --out "$PF_RESULTS/results.json")
case "$PF_SETUP" in
  L1) ARGS+=(--maxiter 8000 --early_iter 3000 --early_tol 0.05);;
  L2) ARGS+=(--L2 --maxiter 30000 --early_iter 0 --early_tol 0.05);;
  *) echo 'PF_SETUP must be L1 or L2'; exit 2;;
esac
printf 'UTC start: '; date -u +%FT%TZ
printf 'job=%s setup=%s start=%s end=%s n=%s devices=%s\n' "$SLURM_JOB_ID" "$PF_SETUP" "${PF_START:-0}" "${PF_END:--1}" "${PF_N:-1}" "${CUDA_VISIBLE_DEVICES:-unset}"
printf '%q ' "$PF_PYTHON" "$PF_REPO/GS/$PF_SETUP/code/eval_ord.py" "${ARGS[@]}"; printf '\n'
"$PF_PYTHON" -m pip freeze > environment.txt
"$PF_PYTHON" "$PF_REPO/GS/$PF_SETUP/code/eval_ord.py" "${ARGS[@]}"
printf 'UTC end: '; date -u +%FT%TZ
