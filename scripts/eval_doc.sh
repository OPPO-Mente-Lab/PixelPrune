#!/bin/bash
# PixelPrune: Training-Free Evaluation on Document Understanding Benchmarks
#
# Usage:
#   bash scripts/eval_doc.sh [MODEL_PATH] [DATASETS] [GPU_IDS]
#
# Examples:
#   bash scripts/eval_doc.sh Qwen/Qwen3-VL-2B-Instruct
#   bash scripts/eval_doc.sh Qwen/Qwen3-VL-2B-Instruct "DocVQA_VAL ChartQA_TEST"
#   bash scripts/eval_doc.sh Qwen/Qwen3-VL-2B-Instruct "" "0,1,2,3"

set -eux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="${1:-Qwen/Qwen3-VL-2B-Instruct}"
DATASETS="${2:-DocVQA_VAL AI2D_TEST ChartQA_TEST InfoVQA_VAL OCRBench MMLongBench_DOC olmOCRBench}"
# Default: use all available GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_IDS="${3:-$(seq -s, 0 $((NUM_GPUS - 1)))}"

# PixelPrune settings (training-free)
export PIXELPRUNE_ENABLED=true
export PIXELPRUNE_METHOD=pred_2d
export PIXELPRUNE_METRIC=max
export PIXELPRUNE_THRESHOLD=0.0

# Pass model path to VLMEvalKit config via env var
export model_path="$MODEL_PATH"

# Ensure pixelprune is importable
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# --- VLMEvalKit is bundled under eval/ ---
VLMEVALKIT_DIR="${ROOT_DIR}/eval"

# Data & output paths
export LMUData="${VLMEVALKIT_DIR}/LMUData"
mkdir -p "$LMUData"
export pred_root="${VLMEVALKIT_DIR}/outputs/pixelprune_doc"

echo "========================================"
echo "PixelPrune: Document Understanding Eval"
echo "Model:      $MODEL_PATH"
echo "Datasets:   $DATASETS"
echo "GPUs:       $GPU_IDS"
echo "VLMEvalKit: $VLMEVALKIT_DIR"
echo "Method:     Pred-2D, τ=0"
echo "========================================"

cd "${VLMEVALKIT_DIR}"

# Convert space-separated datasets to array for proper argument passing
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
