#!/bin/bash
# Baseline: Full (no PixelPrune) Evaluation
#
# Usage:
#   bash scripts/eval_full_baseline.sh [MODEL_PATH] [DATASETS] [GPU_IDS]

set -eux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="${1:-Qwen/Qwen3-VL-2B-Instruct}"
DATASETS="${2:-DocVQA_VAL AI2D_TEST ChartQA_TEST InfoVQA_VAL OCRBench MMLongBench_DOC olmOCRBench}"
# Default: use all available GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_IDS="${3:-$(seq -s, 0 $((NUM_GPUS - 1)))}"

# Explicitly disable PixelPrune
export PIXELPRUNE_ENABLED=false

# GUI grounding prompt/parser for Qwen3-VL
export GROUNDING_MODEL=qwen3vl

# Pass model path to VLMEvalKit config via env var
export model_path="$MODEL_PATH"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# --- VLMEvalKit is bundled under eval/ ---
VLMEVALKIT_DIR="${ROOT_DIR}/eval"

# Data & output paths
export LMUData="${VLMEVALKIT_DIR}/LMUData"
mkdir -p "$LMUData"
export pred_root="${VLMEVALKIT_DIR}/outputs/full_baseline"

echo "========================================"
echo "Baseline: Full (no PixelPrune)"
echo "Model:      $MODEL_PATH"
echo "Datasets:   $DATASETS"
echo "VLMEvalKit: $VLMEVALKIT_DIR"
echo "========================================"

cd "${VLMEVALKIT_DIR}"
read -ra DATASET_ARRAY <<< "$DATASETS"

# Detect GPU count from GPU_IDS
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
GPU_COUNT=${#GPU_ARRAY[@]}

if [ "$GPU_COUNT" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=$GPU_IDS \
    python run.py \
        --data "${DATASET_ARRAY[@]}" \
        --model Qwen3-VL-HF \
        --work-dir "$pred_root"
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS \
    torchrun --nproc_per_node="$GPU_COUNT" run.py \
        --data "${DATASET_ARRAY[@]}" \
        --model Qwen3-VL-HF \
        --work-dir "$pred_root"
fi

echo "========================================"
echo "Evaluation complete: $(date)"
echo "========================================"
