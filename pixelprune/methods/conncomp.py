"""
ShowUI 风格 Patch 选择器（连通分量 baseline）。

受 ShowUI (Lin et al., CVPR 2025) 启发，将 merged token 排列在 2D 网格上，
通过 Union-Find 将相邻且相似的 token 合并为连通分量，然后对每个连通分量
按 skip_ratio 比例裁减 token，保留 max(1, N * (1 - skip_ratio)) 个。

与原始 ShowUI 的差异:
    - 原版在 attention 层通过 mask 跳过冗余 token，本实现直接删除 token。
    - 原版在 vision encoder 之前对原始像素 patch 构图，
      本实现在 spatial merge 之后对 merged token 构图。

可选功能:
    align_compression: 将压缩率与 Pred-2D 对齐。
        开启时，先用 Pred-2D 计算该图的压缩率，将其作为 skip_ratio
        喂给连通分量选择，从而近似对齐两者的 token 保留数量。
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import List

import torch

from ..dedup import is_similar
from .base import BasePatchSelector
from .pred_2d import Pred2DSelector


def _build_connected_components(
    patches: torch.Tensor,
    num_h: int,
    num_w: int,
    method: str = "max",
    threshold: float = 0.0,
) -> dict[int, list[int]]:
    """
    构建 2D 连通分量：相邻且相似的 patches 通过 Union-Find 合并。

    Args:
        patches: [N, D]，N = num_h * num_w
        num_h, num_w: 网格高宽（token 数）
        method, threshold: 相似度比较参数

    Returns:
        groups: {root_idx: [member_indices...]} 每个连通分量的成员列表
    """
    N = patches.shape[0]
    assert N == num_h * num_w, f"patches 数量 {N} 与网格 {num_h}x{num_w} 不匹配"
    parent = list(range(N))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(num_h):
        for j in range(num_w):
            idx = i * num_w + j
            if j + 1 < num_w:
                right_idx = i * num_w + (j + 1)
                if is_similar(patches[idx], patches[right_idx], method, threshold):
                    union(idx, right_idx)
            if i + 1 < num_h:
                down_idx = (i + 1) * num_w + j
                if is_similar(patches[idx], patches[down_idx], method, threshold):
                    union(idx, down_idx)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(N):
        groups[find(i)].append(i)
    return groups


def _select_from_components(
    groups: dict[int, list[int]],
    skip_ratio: float,
    rand: bool,
    device: torch.device,
) -> torch.Tensor:
    """
    从连通分量中按比例选择 token，对齐原始 ShowUI 的 get_select_mask 逻辑。

    对每个连通分量:
        num_to_skip = round(len(component) * skip_ratio)
        num_to_retain = max(1, len(component) - num_to_skip)

    Args:
        groups: 连通分量 {root: [indices...]}
        skip_ratio: 每个分量内跳过的 token 比例 (0.0 ~ 1.0)
        rand: True=随机采样, False=均匀间隔采样 (torch.linspace)
        device: 输出 tensor 的设备

    Returns:
        indices: 保留的 token 索引（已排序），形状 [L']
    """
    selected: list[int] = []
    for members in groups.values():
        n = len(members)
        if n == 1:
            selected.append(members[0])
            continue
        num_to_skip = int(round(n * skip_ratio))
        num_to_retain = max(1, n - num_to_skip)
        if rand:
            sampled = random.sample(members, num_to_retain)
        else:
            step_indices = torch.linspace(0, n - 1, steps=num_to_retain).long().tolist()
            sampled = [members[i] for i in step_indices]
        selected.extend(sampled)
    selected.sort()
    return torch.tensor(selected, dtype=torch.long, device=device)


class ConnCompSelector(BasePatchSelector):
    """
    ShowUI 风格选择器：基于 2D 连通分量的空间去重。

    受 ShowUI (Lin et al., CVPR 2025) 启发，将 token 排列为 2D 网格，
    通过 Union-Find 合并相邻且相似的 token 为连通分量，
    然后对每个分量按 skip_ratio 比例裁减，保留 max(1, N*(1-skip_ratio)) 个。

    与原始 ShowUI 的差异:
        - 原版通过 attention mask 跳过冗余 token，本实现直接删除。
        - 原版在 vision encoder 前对原始像素构图，本实现在 spatial merge 后构图。

    环境变量映射:
        PIXELPRUNE_METHOD=conncomp → 此选择器
    """

    name = "conncomp"
    aliases = ["showui", "connected_components", "cc"]

    def __init__(
        self,
        skip_ratio: float | None = None,
        rand: bool | None = None,
        align_compression: bool | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if skip_ratio is None:
            skip_ratio = float(os.environ.get("PIXELPRUNE_CONNCOMP_SKIP_RATIO", "0.0"))
        if rand is None:
            rand = os.environ.get(
                "PIXELPRUNE_CONNCOMP_RAND", "true"
            ).lower() in ("true", "1", "yes")
        if align_compression is None:
            align_compression = os.environ.get(
                "PIXELPRUNE_CONNCOMP_ALIGN", "true"
            ).lower() in ("true", "1", "yes")
        if seed is None:
            raw = os.environ.get("PIXELPRUNE_SEED", "42")
            self.seed = None if str(raw).lower() in ("none", "false", "0", "") else int(raw)
        else:
            self.seed = seed
        self.skip_ratio = skip_ratio
        self.rand = rand
        self.align_compression = align_compression

    def select(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> List[torch.Tensor]:
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
        merged_pv, merged_lengths = self._prepare_merged(
            pixel_values, image_grid_thw, spatial_merge_size
        )
        device = pixel_values.device

        loco_align_info: list[tuple[float, int]] | None = None
        if self.align_compression:
            loco_align_info = self._compute_loco_align_info(
                pixel_values, image_grid_thw, spatial_merge_size
            )

        merged_indices_list: List[torch.Tensor] = []
        offset = 0
        for img_idx, (length, (t, h, w)) in enumerate(zip(
            merged_lengths, image_grid_thw.tolist()
        )):
            patches = merged_pv[offset : offset + length]
            offset += length
            num_h = int(h) // spatial_merge_size
            num_w = int(w) // spatial_merge_size

            groups = _build_connected_components(
                patches, num_h, num_w,
                method=self.method,
                threshold=self.threshold,
            )

            if loco_align_info is not None:
                skip_ratio, target_num = loco_align_info[img_idx]
            else:
                skip_ratio, target_num = self.skip_ratio, None

            indices = _select_from_components(
                groups, skip_ratio, self.rand, device,
            )

            if target_num is not None:
                indices = self._fine_tune_count(
                    indices, target_num, length, device,
                )
            merged_indices_list.append(indices.to(device))
        return merged_indices_list

    def _compute_loco_align_info(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int,
    ) -> list[tuple[float, int]]:
        """
        用 Pred-2D 计算每张图的 skip_ratio 和精确目标 token 数。

        Returns:
            [(skip_ratio, target_num), ...] 每张图一个元组
        """
        pred2d_selector = Pred2DSelector(
            method=self.method,
            threshold=self.threshold,
        )
        pred2d_indices_list = pred2d_selector.select(
            pixel_values, image_grid_thw, spatial_merge_size
        )
        block_size = spatial_merge_size * spatial_merge_size
        merged_lengths = [
            int(t * h * w) // block_size
            for t, h, w in image_grid_thw.tolist()
        ]
        result: list[tuple[float, int]] = []
        for length, pred2d_indices in zip(merged_lengths, pred2d_indices_list):
            kept = len(pred2d_indices)
            ratio = 1.0 - kept / length if length > 0 else 0.0
            result.append((ratio, kept))
        return result

    @staticmethod
    def _fine_tune_count(
        indices: torch.Tensor,
        target_num: int,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """微调 token 数量，修正舍入偏差使其与 Pred-2D 精确对齐。"""
        current_num = len(indices)
        if current_num == target_num:
            return indices
        if current_num < target_num:
            all_idx = set(indices.cpu().tolist())
            remaining = list(set(range(length)) - all_idx)
            num_add = min(target_num - current_num, len(remaining))
            sampled = random.sample(remaining, num_add)
            indices = torch.cat([
                indices,
                torch.tensor(sampled, device=device, dtype=torch.long),
            ]).sort()[0]
        else:
            num_remove = current_num - target_num
            if num_remove < current_num:
                to_remove = random.sample(range(1, current_num), num_remove)
                mask = torch.ones(current_num, dtype=torch.bool, device=device)
                mask[torch.tensor(to_remove, device=device)] = False
                indices = indices[mask]
        return indices
