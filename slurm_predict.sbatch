#!/bin/bash
#SBATCH --job-name=tissue-predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT to the trained checkpoint}"
if [[ -z "${IMAGE:-}" && -z "${INPUT_DIR:-}" ]]; then
    echo "Set either IMAGE or INPUT_DIR" >&2
    exit 2
fi
if [[ -n "${IMAGE:-}" && -n "${INPUT_DIR:-}" ]]; then
    echo "Set IMAGE or INPUT_DIR, not both" >&2
    exit 2
fi

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
    --checkpoint "${CHECKPOINT}"
    --output-dir "${OUTPUT_DIR}"
    --device cuda
    --tile-size "${TILE_SIZE}"
    --overlap "${OVERLAP}"
    --batch-size "${BATCH_SIZE}"
)

if [[ -n "${IMAGE:-}" ]]; then
    ARGS+=(--image "${IMAGE}")
else
    ARGS+=(--input-dir "${INPUT_DIR}")
fi
if [[ "${RECURSIVE:-0}" == "1" ]]; then
    ARGS+=(--recursive)
fi
if [[ -n "${CZI_SCENE:-}" ]]; then
    ARGS+=(--czi-scene "${CZI_SCENE}")
fi

if [[ "${NO_SAVE_PROBABILITIES:-0}" == "1" ]]; then
    ARGS+=(--no-save-probabilities)
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a USER_ARGS <<< "${EXTRA_ARGS}"
    ARGS+=("${USER_ARGS[@]}")
fi

srun python -u predict.py "${ARGS[@]}"
