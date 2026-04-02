"""
2D Predictive Coding Patch 选择器 — LOCO-I 局部边缘算子（PixelPrune 默认方法）。

改编自 JPEG-LS (ITU-T T.87) 中的 LOCO-I 预测器，将其应用于
merged token 的 2D 网格。对每个位置利用三个因果邻居
A(左)、B(上)、C(左上) 探测局部边缘方向，据此选择最优预测源。

与 1D Raster 扫描的本质区别：
    - Raster 仅做连续去重（当前 == 前驱?）
    - 本方法是 2D 预测编码：利用多邻居推断边缘方向，
      选择最可能正确的邻居作为预测，仅在预测失败时保留 token。

对于轴对齐矩形色块，锚点数量为 O(1)（仅角点），远优于 1D 方法的 O(h)。

精度说明：
    anchored=True（默认）：Python 循环，每个位置与锚点重建值比较，
        避免级联误差。适用于 near-exact（threshold > 0）场景。
    anchored=False 或 method='exact'：完全向量化快速路径。
        exact 下相似性具有传递性，滑动等价于锚定，无级联误差。
"""

from __future__ import annotations

import os
from typing import List

import torch

from .base import BasePatchSelector


def _sim2d(
    X: torch.Tensor,
    Y: torch.Tensor,
    method: str,
    threshold: float,
) -> torch.Tensor:
    """批量相似度比较。

    Args:
        X, Y: shape [..., D]，最后一维为特征维
        method: 'mae' / 'rmse' / 'max' / 'exact'
        threshold: 阈值（exact 内部固定为 0.0）

    Returns:
        bool tensor，shape [...] (去掉最后一维)，True 表示相似（可去重）
    """
    diff = X - Y
    if method in ('max', 'exact'):
        dist = diff.abs().amax(dim=-1)
        thr = 0.0 if method == 'exact' else threshold
    elif method == 'mae':
        dist = diff.abs().mean(dim=-1)
        thr = threshold
    elif method == 'rmse':
        dist = diff.pow(2).mean(dim=-1).sqrt()
        thr = threshold
    else:
        raise ValueError(
            f"Unknown similarity method: {method!r}，合法值为 mae / rmse / max / exact"
        )
    return dist <= thr


class Pred2DSelector(BasePatchSelector):
    """
    2D 预测编码选择器 (LOCO-I 变体)：基于局部边缘推断的 token 选择。

    在 merged token 的 2D 网格上对每个位置 (r, c)：
        1. 取 A=左, B=上, C=左上 三个邻居
        2. 通过 C 与 A/B 的相似关系推断边缘方向
        3. 选择最可能匹配的邻居作为预测
        4. 预测失败 → 保留该 token（放置锚点）

    这是 PixelPrune 论文推荐的默认方法。

    环境变量映射:
        PIXELPRUNE_METHOD=pred_2d → 此选择器
    """

    name = "pred_2d"
    aliases = ["pred_2d_loco", "loco", "loco_i", "pred_loco", "pixelprune"]

    def __init__(
        self,
        method: str = "exact",
        threshold: float = 0.0,
        anchored: bool | None = None,
        **kwargs,
    ) -> None:
        super().__init__(method, threshold, **kwargs)
        if anchored is None:
            anchored = os.environ.get("PIXELPRUNE_ANCHORED", "true").lower() in ("true", "1", "yes")
        self.anchored = anchored

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

            indices = self._select_2d_loco(img_merged, merged_h, merged_w, device)
            merged_indices_list.append(indices.to(device))
            offset += length

        return merged_indices_list

    def _select_2d_loco(
        self,
        tokens: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """分流：exact / (max,τ=0) / anchored=False → 向量化快速路径；否则 → 锚定模式。"""
        if self.method == 'exact' or (self.method == 'max' and self.threshold == 0) or not self.anchored:
            return self._select_2d_loco_fast(tokens, h, w, device)
        return self._select_2d_loco_anchored(tokens, h, w, device)

    # ------------------------------------------------------------------
    # 快速路径：完全向量化，O(1) kernel 调用
    # ------------------------------------------------------------------

    def _select_2d_loco_fast(
        self,
        tokens: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        向量化实现（滑动模式）。

        预测规则 (r > 0, c > 0 的一般情况)::

            C B
            A X

            similar(C, B) and not similar(C, A) → 水平分割线 → pred = A
            similar(C, A) and not similar(C, B) → 垂直分割线 → pred = B
            otherwise                           → 默认水平继承 → pred = A
        """
        D = tokens.shape[-1]
        g = tokens.view(h, w, D)
        method, threshold = self.method, self.threshold

        keep = torch.zeros(h, w, dtype=torch.bool, device=device)
        keep[0, 0] = True

        if w > 1:
            keep[0, 1:] = ~_sim2d(g[0:1, 1:], g[0:1, :-1], method, threshold)[0]

        if h > 1:
            keep[1:, 0] = ~_sim2d(g[1:, 0:1], g[:-1, 0:1], method, threshold)[:, 0]

        if h > 1 and w > 1:
            X = g[1:, 1:]
            A = g[1:, :-1]
            B = g[:-1, 1:]
            C = g[:-1, :-1]

            cb = _sim2d(C, B, method, threshold)
            ca = _sim2d(C, A, method, threshold)
            use_b = ca & ~cb

            pred = torch.where(use_b.unsqueeze(-1), B, A)
            keep[1:, 1:] = ~_sim2d(X, pred, method, threshold)

        return keep.flatten().nonzero(as_tuple=False)[:, 0]

    # ------------------------------------------------------------------
    # 锚定路径：Python 循环，追踪重建值，避免级联误差
    # ------------------------------------------------------------------

    def _select_2d_loco_anchored(
        self,
        tokens: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """锚定模式（Python 循环），保证任意位置的重建误差不超过单步 threshold。"""
        D = tokens.shape[-1]
        g = tokens.view(h, w, D)
        method, threshold = self.method, self.threshold

        keep = torch.zeros(h, w, dtype=torch.bool, device=device)
        keep[0, 0] = True

        anchor_grid = g.clone()

        def sim(a: torch.Tensor, b: torch.Tensor) -> bool:
            return _sim2d(a.unsqueeze(0), b.unsqueeze(0), method, threshold).item()

        for r in range(h):
            for c in range(w):
                if r == 0 and c == 0:
                    continue

                if c > 0:
                    A = anchor_grid[r, c - 1]
                elif r > 0:
                    A = anchor_grid[r - 1, 0]
                else:
                    A = None

                B = anchor_grid[r - 1, c] if r > 0 else A
                C = anchor_grid[r - 1, c - 1] if (r > 0 and c > 0) else B

                if r > 0 and c > 0:
                    cb = sim(C, B)
                    ca = sim(C, A)
                    pred = B if (ca and not cb) else A
                else:
                    pred = A

                X = g[r, c]
                if not sim(X, pred):
                    keep[r, c] = True
                    anchor_grid[r, c] = X
                else:
                    anchor_grid[r, c] = pred

        return keep.flatten().nonzero(as_tuple=False)[:, 0]
