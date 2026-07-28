#!/bin/bash
#SBATCH --job-name=tissue-predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${IMAGE:?Set IMAGE to the source image}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained checkpoint}"

VENV="${VENV:-.venv}"
OUTPUT_DIR="${OUTPUT_DIR:-prediction}"
TILE_SIZE="${TILE_SIZE:-512}"
OVERLAP="${OVERLAP:-128}"
BATCH_SIZE="${BATCH_SIZE:-2}"

if [[ -n "${MODULES:-}" ]]; then
    module load ${MODULES}
fi

source "${VENV}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ARGS=(
    --image "${IMAGE}"
    --checkpoint "${CHECKPOINT}"
    --output-dir "${OUTPUT_DIR}"
    --device cuda
    --tile-size "${TILE_SIZE}"
    --overlap "${OVERLAP}"
    --batch-size "${BATCH_SIZE}"
)

if [[ "${NO_SAVE_PROBABILITIES:-0}" == "1" ]]; then
    ARGS+=(--no-save-probabilities)
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a USER_ARGS <<< "${EXTRA_ARGS}"
    ARGS+=("${USER_ARGS[@]}")
fi

srun python -u predict.py "${ARGS[@]}"
