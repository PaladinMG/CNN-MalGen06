#!/bin/bash
#SBATCH --job-name=tissue-quantify
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${PREDICTION_DIR:?Set PREDICTION_DIR to the prediction output directory}"

VENV="${VENV:-.venv}"
OUTPUT="${OUTPUT:-${PREDICTION_DIR}/Quantification 3.xlsx}"
PIXEL_WIDTH_UM="${PIXEL_WIDTH_UM:-0.22}"
PIXEL_HEIGHT_UM="${PIXEL_HEIGHT_UM:-0.22}"
ANALYSIS_DOWNSAMPLE="${ANALYSIS_DOWNSAMPLE:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
GPU_MEMORY_FRACTION="${GPU_MEMORY_FRACTION:-0.80}"
GPU_CHUNK_ROWS="${GPU_CHUNK_ROWS:-0}"
GPU_MIN_PIXELS="${GPU_MIN_PIXELS:-262144}"

if [[ -n "${MODULES:-}" ]]; then
    module load ${MODULES}
fi

source "${VENV}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ARGS=(
    --prediction-dir "${PREDICTION_DIR}"
    --output "${OUTPUT}"
    --pixel-width-um "${PIXEL_WIDTH_UM}"
    --pixel-height-um "${PIXEL_HEIGHT_UM}"
    --analysis-downsample "${ANALYSIS_DOWNSAMPLE}"
    --device cuda
    --cuda-device "${CUDA_DEVICE}"
    --gpu-memory-fraction "${GPU_MEMORY_FRACTION}"
    --gpu-chunk-rows "${GPU_CHUNK_ROWS}"
    --gpu-min-pixels "${GPU_MIN_PIXELS}"
    --overwrite
)

if [[ -n "${ORIGINAL_DIR:-}" ]]; then
    ARGS+=(--original-dir "${ORIGINAL_DIR}")
fi
if [[ -n "${LOG_FILE:-}" ]]; then
    ARGS+=(--log-file "${LOG_FILE}")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a USER_ARGS <<< "${EXTRA_ARGS}"
    ARGS+=("${USER_ARGS[@]}")
fi

srun python -u "Quantification 3.py" "${ARGS[@]}"
