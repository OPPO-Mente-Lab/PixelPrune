"""
Monkey-patch 入口：对 HuggingFace Qwen3-VL / Qwen3.5 应用 patch，注入 PixelPrune 选择器逻辑。

使用前请设置环境变量（如 PIXELPRUNE_ENABLED=true），然后在加载模型前调用：

    from pixelprune.patches import apply_patches
    apply_patches(model='qwen3_vl')   # Qwen3-VL
    apply_patches(model='qwen3_5')    # Qwen3.5
"""

from __future__ import annotations

from typing import Literal


def apply_patches(model: Literal["qwen3_vl", "qwen3_5"] = "qwen3_vl") -> None:
    """对指定模型架构应用 patch。需在加载模型前调用。"""
    if model == "qwen3_vl":
        from .qwen3_vl_hf import apply_patches as _apply
        _apply()
    elif model == "qwen3_5":
        from .qwen3_5_hf import apply_patches as _apply
        _apply()
    else:
        raise ValueError(f"model must be 'qwen3_vl' or 'qwen3_5', got {model!r}")


__all__ = ["apply_patches"]
