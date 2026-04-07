"""
Monkey-patch 入口：对 HuggingFace / vLLM 的 Qwen3-VL / Qwen3.5 应用 patch，
注入 PixelPrune 选择器逻辑。

使用前请设置环境变量（如 PIXELPRUNE_ENABLED=true），然后在加载模型前调用：

    from pixelprune.patches import apply_patches
    apply_patches(model='qwen3_vl')                 # HF Qwen3-VL
    apply_patches(model='qwen3_5')                  # HF Qwen3.5
    apply_patches(model='qwen3_vl', backend='vllm') # vLLM Qwen3-VL
    apply_patches(model='qwen3_5',  backend='vllm') # vLLM Qwen3.5
"""

from __future__ import annotations

from typing import Literal


def apply_patches(
    model: Literal["qwen3_vl", "qwen3_5"] = "qwen3_vl",
    backend: Literal["hf", "vllm"] = "hf",
) -> None:
    """对指定模型架构应用 patch。需在加载模型前调用。

    Args:
        model: 模型架构名称。
        backend: 推理后端，'hf' 为 HuggingFace，'vllm' 为 vLLM。
    """
    if model == "qwen3_vl":
        if backend == "vllm":
            from .qwen3_vl_vllm import apply_patches as _apply
        else:
            from .qwen3_vl_hf import apply_patches as _apply
        _apply()
    elif model == "qwen3_5":
        if backend == "vllm":
            from .qwen3_5_vllm import apply_patches as _apply
        else:
            from .qwen3_5_hf import apply_patches as _apply
        _apply()
    else:
        raise ValueError(f"model must be 'qwen3_vl' or 'qwen3_5', got {model!r}")


__all__ = ["apply_patches"]
