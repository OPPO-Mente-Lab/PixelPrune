"""
Patch 选择器抽象基类。

所有 patch selection 方法都应继承 BasePatchSelector，并实现 select() 方法。
这提供了统一的接口，使得 HuggingFace / vLLM 的 monkey-patch 层可以
透明地调用任意选择策略。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch


class BasePatchSelector(ABC):
    """
    Patch 选择器的抽象基类。

    子类需要实现 `select()` 方法，根据 pixel_values 和 image_grid_thw
    计算每张图要保留的 **merged token** 索引列表。

    Attributes:
        method: 距离度量方法 ('mae'|'rmse'|'max'|'exact')
        threshold: 相似度/距离阈值
    """

    # 子类可在类定义中覆盖，表示该选择器在注册表中的名称
    name: str = ""
    # 别名列表，用于向后兼容（如 "showui" -> "rle_2d"）
    aliases: List[str] = []

    def __init__(
        self,
        method: str = "exact",
        threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.method = method
        self.threshold = threshold

    @abstractmethod
    def select(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> List[torch.Tensor]:
        """
        计算每张图要保留的 merged token 索引。

        Args:
            pixel_values: 已按 patch 展平的像素值，shape (total_patches, C*patch_size^2)
                其中 total_patches = sum(t*h*w for (t,h,w) in image_grid_thw)
            image_grid_thw: (B, 3)，每行 (t, h, w) 表示时间/高/宽 patch 数
            spatial_merge_size: 空间合并倍数，默认 2（Qwen3-VL）

        Returns:
            indices_list: List[Tensor]，长度等于 batch 中图像数；
                每个 Tensor 为该图保留的 merged token 索引（1D, sorted）
        """
        ...

    def _prepare_merged(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> tuple[torch.Tensor, List[int]]:
        """
        将 patch-level pixel_values 合并为 merged-level 表示。

        这是大多数选择器的共用预处理步骤：将每 spatial_merge_size^2 个
        相邻 patch 拼接为一个 merged token 的像素表示。

        Args:
            pixel_values: (total_patches, D)
            image_grid_thw: (B, 3)
            spatial_merge_size: 空间合并倍数

        Returns:
            merged_pv: (total_merged_tokens, D * merge_size^2)
            merged_lengths: 每张图的 merged token 数量列表
        """
        block_size = spatial_merge_size * spatial_merge_size
        merged_pv = pixel_values.reshape(-1, pixel_values.shape[-1] * block_size)
        merged_lengths = [
            int(t * h * w) // block_size
            for t, h, w in image_grid_thw.tolist()
        ]
        return merged_pv, merged_lengths

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"method={self.method!r}, threshold={self.threshold})"
        )
