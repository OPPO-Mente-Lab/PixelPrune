"""
vLLM Qwen3.5 monkey-patch：使用 PixelPrune 选择器计算 keep_indices。

Qwen3.5 在 vLLM 中通过 qwen3_5.py 定义，继承自 Qwen3VLForConditionalGeneration，
共享同一个 Qwen3VLMultiModalProcessor 和 Qwen3_VisionTransformer。

与 Qwen3-VL 的关键差异：
- 无 deepstack（embed_input_ids 不调用 _compute_deepstack_embeds）
- supports_multimodal_pruning = False（无 EVS 视频裁剪）
- _postprocess_image_embeds_evs 为 no-op（不追加 5 通道位置）
- 混合注意力架构（GatedDeltaNet + Full Attention）

继承关系：
  Qwen3_5ForConditionalGeneration → Qwen3VLForConditionalGeneration
  Qwen3_5MoeForConditionalGeneration → Qwen3_5ForConditionalGeneration

继承的方法（从 Qwen3VLForConditionalGeneration）：
  _parse_and_validate_image_input, _process_image_input,
  get_mrope_input_positions, embed_multimodal, _iter_mm_grid_hw

Qwen3.5 覆盖的方法：
  __init__, embed_input_ids, forward, load_weights, recompute_mrope_positions

环境变量：PIXELPRUNE_ENABLED, PIXELPRUNE_METHOD, PIXELPRUNE_METRIC, PIXELPRUNE_THRESHOLD, PIXELPRUNE_VERBOSE
在创建 vLLM LLM 前调用 apply_patches()。
"""

from __future__ import annotations

from typing import Any

try:
    from vllm.model_executor.models import qwen3_5
except ImportError:
    import importlib
    qwen3_5 = importlib.import_module("vllm.model_executor.models.qwen3_5")

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except Exception:
    logger = None

# Reuse all patch functions from qwen3_vl_vllm
from .qwen3_vl_vllm import (
    apply_patches as _apply_qwen3vl_patches,
    _mmp_init,
    _mmp_call_hf,
    _mmp_fields,
    _mmp_prompt_updates,
    _parse_image_input,
    _process_image_input,
    _get_mrope_input_positions,
    _vt_forward,
    _patch,
)

# Import qwen3_vl module for reference
try:
    from vllm.model_executor.models import qwen3_vl
except ImportError:
    import importlib
    qwen3_vl = importlib.import_module("vllm.model_executor.models.qwen3_vl")


def apply_patches() -> None:
    """对 vLLM Qwen3.5 应用 PixelPrune monkey-patch。

    1. 先 apply qwen3_vl 的 patch（patch 父类 + Processor + VisionTransformer）
    2. 再显式 patch Qwen3.5 自身的类（确保不依赖 Python MRO 隐式继承）
    """
    if getattr(qwen3_5.Qwen3_5ForConditionalGeneration, "__pixelprune_patched__", False):
        return

    # Step 1: Patch 父类（Qwen3VLForConditionalGeneration、Qwen3VLMultiModalProcessor、
    #          Qwen3_VisionTransformer）—— 这些是共享的基础组件
    _apply_qwen3vl_patches()

    # Step 2: 显式 patch Qwen3_5ForConditionalGeneration 上的方法。
    # 虽然通过 MRO 这些方法已经指向了 patched 版本，但显式 patch 确保：
    # (a) 即使 Qwen3.5 未来覆盖这些方法，patch 仍然生效
    # (b) 代码意图清晰，不依赖隐式继承
    _patch(qwen3_5.Qwen3_5ForConditionalGeneration,
           "_parse_and_validate_image_input", _parse_image_input)
    _patch(qwen3_5.Qwen3_5ForConditionalGeneration,
           "_process_image_input", _process_image_input)
    _patch(qwen3_5.Qwen3_5ForConditionalGeneration,
           "get_mrope_input_positions", _get_mrope_input_positions)

    # Step 3: 同样 patch MoE 变体
    if hasattr(qwen3_5, "Qwen3_5MoeForConditionalGeneration"):
        _patch(qwen3_5.Qwen3_5MoeForConditionalGeneration,
               "_parse_and_validate_image_input", _parse_image_input)
        _patch(qwen3_5.Qwen3_5MoeForConditionalGeneration,
               "_process_image_input", _process_image_input)
        _patch(qwen3_5.Qwen3_5MoeForConditionalGeneration,
               "get_mrope_input_positions", _get_mrope_input_positions)

    qwen3_5.Qwen3_5ForConditionalGeneration.__pixelprune_patched__ = True
    if logger:
        logger.info("Qwen3.5 PixelPrune (vLLM) patches applied successfully")


__all__ = ["apply_patches"]
