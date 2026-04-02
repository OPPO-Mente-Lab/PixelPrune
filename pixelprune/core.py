"""
PixelPrune 统一调度入口 + 索引转换工具。

本模块提供:
    - compute_merged_keep_indices(): 统一入口，根据方法名调度到对应的选择器
    - merged_indices_to_patch_indices(): merged 索引 → patch 索引（ViT 输入层裁剪）
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch

from .methods import get_selector


# ---------------------------------------------------------------------------
# 统一入口：计算每张图保留的 merged token 索引
# ---------------------------------------------------------------------------


def compute_merged_keep_indices(
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
    method: Optional[str] = None,
    metric: Optional[str] = None,
    threshold: Optional[float] = None,
    **kwargs,
) -> List[torch.Tensor]:
    """
    根据 pixel_values 和 image_grid_thw 计算每张图要保留的 **merged token** 索引列表。

    Args:
        pixel_values: 已按 patch 展平的像素，shape (total_patches, C*patch_size^2)
        image_grid_thw: (B, 3)，每行 (t, h, w) 表示时间/高/宽 patch 数
        spatial_merge_size: 空间合并倍数，默认 2（Qwen3-VL）
        method: 扫描策略名称（如 'pred_2d', 'raster', 'serpentine'）。
            默认从环境变量 PIXELPRUNE_METHOD 读取，若未设置则为 'pred_2d'。
        metric: 距离度量方法 ('mae'|'rmse'|'max'|'exact')。
            默认从环境变量 PIXELPRUNE_METRIC 读取，若未设置则为 'max'。
        threshold: 阈值 τ。默认从环境变量 PIXELPRUNE_THRESHOLD 读取，若未设置则为 0.0。
        **kwargs: 传递给选择器构造函数的额外参数

    Returns:
        indices_list: List[Tensor]，长度为 batch 中图像数；
            每个 Tensor 为该图保留的 merged token 索引
    """
    if method is None:
        method = os.environ.get("PIXELPRUNE_METHOD", "pred_2d").lower()
    if metric is None:
        metric = os.environ.get("PIXELPRUNE_METRIC", "max")
    if threshold is None:
        threshold = float(os.environ.get("PIXELPRUNE_THRESHOLD", "0.0") or "0.0")

    selector = get_selector(
        method,
        method=metric,
        threshold=threshold,
        **kwargs,
    )
    return selector.select(pixel_values, image_grid_thw, spatial_merge_size)


# ---------------------------------------------------------------------------
# 索引转换工具
# ---------------------------------------------------------------------------


def merged_indices_to_patch_indices(
    merged_indices_list: List[torch.Tensor],
    block_size: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    将 merged 级保留索引展开为 patch 级索引（用于 ViT 输入层压缩）。

    Args:
        merged_indices_list: compute_merged_keep_indices 的返回值
        block_size: spatial_merge_size ** 2
        device: 输出张量设备

    Returns:
        patch_indices_list: 每个元素为 patch 级索引，长度 = len(merged_indices) * block_size
    """
    patch_indices_list: List[torch.Tensor] = []
    for merged_indices in merged_indices_list:
        patch_indices = []
        for midx in merged_indices.tolist():
            base = int(midx) * block_size
            patch_indices.extend(range(base, base + block_size))
        patch_indices_list.append(
            torch.tensor(patch_indices, device=device, dtype=torch.long)
        )
    return patch_indices_list
