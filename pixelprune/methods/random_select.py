"""
Random Patch 选择器（Baseline）。

先用 Pred-2D (LOCO) 确定每张图应保留的 token 数量（即与 Pred-2D 相同的压缩率），
然后随机选择相同数量的 token。为 content-aware 方法提供公平的 random baseline。
"""

from __future__ import annotations

import os
from typing import List

import torch

from .base import BasePatchSelector
from .pred_2d import Pred2DSelector


class RandomSelector(BasePatchSelector):
    """
    随机选择器：与 Pred-2D 相同压缩率的随机 token 采样。

    环境变量映射:
        PIXELPRUNE_METHOD=random → 此选择器
    """

    name = "random"
    aliases = ["rand", "random_baseline"]

    def __init__(self, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        if seed is None:
            raw = os.environ.get("PIXELPRUNE_SEED", "42")
            self.seed = None if str(raw).lower() in ("none", "false", "0", "") else int(raw)
        else:
            self.seed = seed

    def select(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> List[torch.Tensor]:
        if self.seed is not None:
            torch.manual_seed(self.seed)
        merged_pv, merged_lengths = self._prepare_merged(
            pixel_values, image_grid_thw, spatial_merge_size
        )
        device = pixel_values.device

        pred2d_selector = Pred2DSelector(
            method=self.method,
            threshold=self.threshold,
        )
        pred2d_indices_list = pred2d_selector.select(
            pixel_values, image_grid_thw, spatial_merge_size
        )

        random_list: List[torch.Tensor] = []
        for seq_idx, pred2d_indices in enumerate(pred2d_indices_list):
            num_keep = len(pred2d_indices)
            total_tokens = merged_lengths[seq_idx]
            if num_keep >= total_tokens:
                random_indices = torch.arange(
                    total_tokens, dtype=torch.long, device=device
                )
            else:
                random_indices = (
                    torch.randperm(total_tokens, device=device)[:num_keep]
                    .sort()[0]
                )
            random_list.append(random_indices)
        return random_list
