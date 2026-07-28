#!/bin/bash
#SBATCH --job-name=tissue-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${DATA:?Set DATA to the dataset directory containing images/ and masks/}"

VENV="${VENV:-.venv}"
OUTPUT="${OUTPUT:-best_model.pt}"
LAST_OUTPUT="${LAST_OUTPUT:-last_model.pt}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CROP_SIZE="${CROP_SIZE:-512}"
WORKERS="${WORKERS:-4}"

if [[ -n "${MODULES:-}" ]]; then
    module load ${MODULES}
fi

source "${VENV}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ARGS=(
    --data "${DATA}"
    --device cuda
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --crop-size "${CROP_SIZE}"
    --workers "${WORKERS}"
    --output "${OUTPUT}"
    --last-output "${LAST_OUTPUT}"
)

if [[ -n "${VAL_DATA:-}" ]]; then
    ARGS+=(--val-data "${VAL_DATA}")
fi
if [[ -n "${RESUME:-}" ]]; then
    ARGS+=(--resume "${RESUME}")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a USER_ARGS <<< "${EXTRA_ARGS}"
    ARGS+=("${USER_ARGS[@]}")
fi

srun python -u train.py "${ARGS[@]}"
