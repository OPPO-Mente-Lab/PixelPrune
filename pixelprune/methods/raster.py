"""
Raster Scan Patch 选择器。

在行主序扫描下，对相邻的 merged token 进行连续去重：
如果相邻 token 的像素值"相同"（由 method/threshold 控制），
则只保留每段连续相同 run 的第一个 token。

对文档/GUI 等大面积重复区域的图像可以实现高达 99% 的 token 压缩。
"""

from __future__ import annotations

from typing import List

import torch

from ..dedup import deduplicate_packed_sequences
from .base import BasePatchSelector


class RasterSelector(BasePatchSelector):
    """
    Raster 扫描选择器：基于行主序扫描的连续去重。

    对 merged token 序列按行主序排列，相邻且相似的 token 只保留第一个。

    环境变量映射:
        PIXELPRUNE_METHOD=raster → 此选择器
    """

    name = "raster"
    aliases = ["rle", "rle_1d", "pixel_rle", "pixelrle"]

    def select(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> List[torch.Tensor]:
        merged_pv, merged_lengths = self._prepare_merged(
            pixel_values, image_grid_thw, spatial_merge_size
        )
        device = pixel_values.device

        _, merged_indices_list, _ = deduplicate_packed_sequences(
            merged_pv,
            lengths=merged_lengths,
            method=self.method,
            threshold=self.threshold,
        )
        return [t.to(device) for t in merged_indices_list]
