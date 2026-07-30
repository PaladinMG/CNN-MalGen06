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
CACHE_TRAINING_IMAGES="${CACHE_TRAINING_IMAGES:-0}"
CACHE_HANDCRAFTED_FEATURES="${CACHE_HANDCRAFTED_FEATURES:-0}"
CACHE_VRAM_RESERVE_GB="${CACHE_VRAM_RESERVE_GB:-10}"
CACHE_FEATURE_TILE_SIZE="${CACHE_FEATURE_TILE_SIZE:-1024}"
COMPILE_MODEL="${COMPILE_MODEL:-0}"
COMPILE_MODE="${COMPILE_MODE:-reduce-overhead}"
SPLIT_UNIT="${SPLIT_UNIT:-patch}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.15}"
PATCH_POOL_PER_IMAGE="${PATCH_POOL_PER_IMAGE:-}"

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
    --split-unit "${SPLIT_UNIT}"
    --validation-fraction "${VALIDATION_FRACTION}"
    --output "${OUTPUT}"
    --last-output "${LAST_OUTPUT}"
)

if [[ -n "${VAL_DATA:-}" ]]; then
    ARGS+=(--val-data "${VAL_DATA}")
fi
if [[ -n "${RESUME:-}" ]]; then
    ARGS+=(--resume "${RESUME}")
fi
if [[ -n "${PATCH_POOL_PER_IMAGE}" ]]; then
    ARGS+=(--patch-pool-per-image "${PATCH_POOL_PER_IMAGE}")
fi
if [[ "${CACHE_TRAINING_IMAGES}" == "1" ]]; then
    ARGS+=(
        --cache-training-images
        --cache-vram-reserve-gb "${CACHE_VRAM_RESERVE_GB}"
        --cache-feature-tile-size "${CACHE_FEATURE_TILE_SIZE}"
    )
fi
if [[ "${CACHE_HANDCRAFTED_FEATURES}" == "1" ]]; then
    if [[ "${CACHE_TRAINING_IMAGES}" != "1" ]]; then
        echo "CACHE_HANDCRAFTED_FEATURES=1 requires CACHE_TRAINING_IMAGES=1" >&2
        exit 2
    fi
    ARGS+=(--cache-handcrafted-features)
fi
if [[ "${COMPILE_MODEL}" == "1" ]]; then
    ARGS+=(--compile-model --compile-mode "${COMPILE_MODE}")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a USER_ARGS <<< "${EXTRA_ARGS}"
    ARGS+=("${USER_ARGS[@]}")
fi

srun python -u train.py "${ARGS[@]}"
