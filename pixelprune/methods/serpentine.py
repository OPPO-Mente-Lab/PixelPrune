"""
Serpentine (蛇形) Scan Patch 选择器。

在蛇形扫描顺序下，对相邻的 merged token 进行连续去重：
- 偶数行从左到右扫描
- 奇数行从右到左扫描

相比行主序 Raster 扫描，蛇形扫描可以更好地保留垂直方向上的连续性，
在某些图像类型（如文档、表格）中可能有更好的压缩效果。
"""

from __future__ import annotations

from typing import List

import torch

from ..dedup import deduplicate_consecutive
from .base import BasePatchSelector


class SerpentineSelector(BasePatchSelector):
    """
    Serpentine 扫描选择器：基于蛇形扫描的连续去重。

    对 merged token 序列按蛇形扫描顺序排列，相邻且相似的 token 只保留第一个。
    蛇形扫描顺序：偶数行从左到右，奇数行从右到左。

    环境变量映射:
        PIXELPRUNE_METHOD=serpentine → 此选择器
    """

    name = "serpentine"
    aliases = ["rle_1d_zig", "rle_zigzag", "zigzag", "rle_zig"]

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

        merged_indices_list = []
        offset = 0

        for length, (t, h, w) in zip(merged_lengths, image_grid_thw.tolist()):
            img_merged = merged_pv[offset:offset + length]
            merged_h = h // spatial_merge_size
            merged_w = w // spatial_merge_size
            indices = self._dedup_zigzag(img_merged, merged_h, merged_w, device)
            merged_indices_list.append(indices)
            offset += length

        return [t.to(device) for t in merged_indices_list]

    def _create_zigzag_indices(
        self, h: int, w: int, device: torch.device
    ) -> torch.Tensor:
        """创建蛇形扫描的索引映射。"""
        indices = []
        for row in range(h):
            row_start = row * w
            if row % 2 == 0:
                indices.extend(range(row_start, row_start + w))
            else:
                indices.extend(range(row_start + w - 1, row_start - 1, -1))
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _dedup_zigzag(
        self,
        merged_tokens: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """对单个图像的 merged tokens 进行蛇形扫描去重。"""
        zigzag_indices = self._create_zigzag_indices(h, w, device)
        tokens_zigzag = merged_tokens[zigzag_indices]
        dedup_indices_zigzag, _ = deduplicate_consecutive(
            tokens_zigzag,
            method=self.method,
            threshold=self.threshold,
        )
        original_indices = zigzag_indices[dedup_indices_zigzag]
        original_indices = torch.sort(original_indices)[0]
        return original_indices
