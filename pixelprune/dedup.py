"""
高效的连续去重工具，支持变长 pack 输入。

支持的去重方法（均为维度归一化，阈值在 [0,1] 像素空间直接可解释）：
    - mae      : Mean Absolute Error  = mean(|Δ|)，每像素平均绝对误差
    - rmse     : Root Mean Sq. Error  = sqrt(mean(Δ²))，每像素 RMS 误差
    - max      : L∞ 范数             = max(|Δ|)，最坏单像素误差上界
    - exact    : 精确匹配，等价于 max(τ=0)，向后兼容别名

三种度量均在 [0,1] 像素空间，阈值含义一致可比：
    mae  τ=0.05 → 平均每像素误差 < 5%
    rmse τ=0.05 → RMS 每像素误差 < 5%（比 mae 对大误差惩罚更重）
    max  τ=0.05 → 最大单像素误差 < 5%（最严格）

支持两种输入模式：
    1. 单序列：tensor shape [L, D]
    2. 变长 pack：tensor shape [L_total, D] + lengths list
"""

import torch
from typing import Tuple, List, Optional, Literal, Union

# 所有合法方法名（含别名）
_ALL_METHODS = frozenset({'mae', 'rmse', 'max', 'exact'})


# ============================================================================
# 核心差异计算
# ============================================================================

def _reduce_distance(
    delta: torch.Tensor,
    method: Literal['mae', 'rmse', 'max', 'exact']
) -> torch.Tensor:
    """
    将相邻元素差值归约为标量距离。

    所有度量均为维度归一化，在 [0,1] 像素空间中阈值语义直接可解释：
        mae   : mean(|Δ|)       — 每像素平均绝对误差，线性，对 outlier 不敏感
        rmse  : sqrt(mean(Δ²))  — 每像素 RMS 误差，对大误差惩罚更重，与 mae/max 同单位
        max   : max(|Δ|)        — 最坏单像素误差，保证无单像素超标
        exact : 等价于 max，threshold 固定为 0

    Args:
        delta: 相邻元素差值 [L-1, D]
        method: 距离类型

    Returns:
        distance: 标量距离 [L-1]
    """
    if method == 'mae':
        return torch.mean(torch.abs(delta), dim=1)
    elif method == 'rmse':
        return torch.sqrt(torch.mean(delta ** 2, dim=1))
    elif method in ('max', 'exact'):
        return torch.max(torch.abs(delta), dim=1)[0]
    else:
        raise ValueError(
            f"未知的距离度量: {method!r}，合法值为 {sorted(_ALL_METHODS)}"
        )


def _compute_adjacent_diff(
    tensor: torch.Tensor,
    method: Literal['mae', 'rmse', 'max', 'exact'],
    threshold: float
) -> torch.Tensor:
    """
    计算相邻元素的差异 mask（内部统一实现）。

    返回 True 表示相邻元素不同（应保留后者）。
    exact 语义固定 threshold=0，其余使用传入值。

    Args:
        tensor: [L, D]，要求 L >= 2
        method: 去重方法
        threshold: 距离阈值（exact 忽略，内部固定 0）

    Returns:
        diff: [L-1] bool 张量
    """
    delta = tensor[:-1] - tensor[1:]
    distance = _reduce_distance(delta, method)
    effective_threshold = 0.0 if method == 'exact' else threshold
    return distance > effective_threshold


# ============================================================================
# 单点对比
# ============================================================================

def is_similar(
    t1: torch.Tensor,
    t2: torch.Tensor,
    method: Literal['mae', 'rmse', 'max', 'exact'] = 'exact',
    threshold: float = 0.0,
) -> bool:
    """
    判断两个 token 向量是否相似（距离 <= threshold）。

    Args:
        t1, t2: 一维 token 向量
        method: 比较方法（mae / rmse / max / exact）
        threshold: 距离阈值；exact 固定为 0

    Returns:
        True 表示相似（应去重），False 表示不同（应保留）
    """
    stacked = torch.stack([t1, t2])
    is_different = _compute_adjacent_diff(stacked, method, threshold)
    return not is_different.item()


# ============================================================================
# 单序列去重
# ============================================================================

def deduplicate_consecutive(
    tensor: torch.Tensor,
    method: Literal['mae', 'rmse', 'max', 'exact'] = 'exact',
    threshold: float = 0,
    anchored: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    单序列连续去重。

    度量说明（像素值已归一化到 [0,1]）：
        mae   : 每像素平均绝对误差 < threshold → 去重
        rmse  : 每像素 RMS 误差   < threshold → 去重
        max   : 最大单像素误差    < threshold → 去重
        exact : 完全相同（max, threshold=0）

    Args:
        tensor: [L, D]
        method: 去重方法
        threshold: 距离阈值，越小越严格
        anchored: True（锚定模式）与当前 run 的锚点比较，避免误差累积；
                  False（滑动模式）仅与相邻元素比较，压缩率更高但误差可传递

    Returns:
        indices: [L'] 保留元素位置
        deduplicated: [L', D] 去重后张量
    """
    L = tensor.shape[0]
    device = tensor.device

    if L <= 1:
        return torch.arange(L, dtype=torch.long, device=device), tensor

    if not anchored:
        diff = _compute_adjacent_diff(tensor, method, threshold)
        keep_mask = torch.cat([torch.tensor([True], device=device), diff])
        indices = torch.where(keep_mask)[0]
    else:
        indices = []
        anchor_idx = 0
        indices.append(anchor_idx)
        effective_threshold = 0.0 if method == 'exact' else threshold

        for i in range(1, L):
            anchor_vec = tensor[anchor_idx:anchor_idx + 1]
            curr_vec = tensor[i:i + 1]
            stacked = torch.cat([anchor_vec, curr_vec])
            is_different = _compute_adjacent_diff(stacked, method, effective_threshold)[0]
            if is_different:
                indices.append(i)
                anchor_idx = i

        indices = torch.tensor(indices, dtype=torch.long, device=device)

    return indices, tensor[indices]


# ============================================================================
# 变长 pack 输入去重
# ============================================================================

def deduplicate_packed_sequences(
    tensor: torch.Tensor,
    lengths: Union[List[int], torch.Tensor],
    method: Literal['mae', 'rmse', 'max', 'exact'] = 'exact',
    threshold: float = 0,
    anchored: bool = True,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    """
    对打包的变长序列批量去重。

    Args:
        tensor: [L_total, D]，多序列拼接
        lengths: 每个序列的长度
        method: 去重方法（mae / rmse / max / exact）
        threshold: 距离阈值
        anchored: 是否使用锚定模式

    Returns:
        deduplicated_tensor: [L_total', D]
        indices_list: 每个序列保留的索引（相对原序列）
        new_lengths: 每个序列去重后的新长度
    """
    if isinstance(lengths, list):
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    else:
        lengths_tensor = lengths.to(tensor.device)

    assert tensor.shape[0] == lengths_tensor.sum().item(), (
        f"tensor 长度 ({tensor.shape[0]}) 与 lengths 总和 ({lengths_tensor.sum().item()}) 不匹配"
    )

    deduplicated_tensors, indices_list, new_lengths = [], [], []
    offset = 0
    for length in lengths_tensor:
        seq = tensor[offset:offset + length]
        idxs, dedup_seq = deduplicate_consecutive(seq, method, threshold, anchored)
        deduplicated_tensors.append(dedup_seq)
        indices_list.append(idxs)
        new_lengths.append(len(dedup_seq))
        offset += length

    if deduplicated_tensors:
        deduplicated_tensor = torch.cat(deduplicated_tensors, dim=0)
    else:
        deduplicated_tensor = torch.empty(0, tensor.shape[1], device=tensor.device)

    return deduplicated_tensor, indices_list, new_lengths


# ============================================================================
# 序列还原
# ============================================================================

def _restore_single_sequence(
    dedup_seq: torch.Tensor,
    indices: torch.Tensor,
    orig_len: int
) -> torch.Tensor:
    """前向填充还原单个序列（向量化实现）。"""
    device = dedup_seq.device
    marker = torch.zeros(orig_len, dtype=torch.long, device=device)
    marker[indices] = 1
    mapping = torch.cumsum(marker, dim=0) - 1
    return dedup_seq[mapping]


def restore_packed_sequences(
    deduplicated_tensor: torch.Tensor,
    indices_list: List[torch.Tensor],
    original_lengths: Union[List[int], torch.Tensor]
) -> torch.Tensor:
    """
    从去重的打包序列还原到原始长度（前向填充）。

    Args:
        deduplicated_tensor: [L_total', D]
        indices_list: 每个序列保留的索引列表
        original_lengths: 原始序列的长度列表

    Returns:
        restored_tensor: [L_total, D]
    """
    if isinstance(original_lengths, list):
        original_lengths = torch.tensor(original_lengths, dtype=torch.long)

    restored_sequences = []
    dedup_offset = 0
    for seq_idx, orig_len in enumerate(original_lengths):
        indices = indices_list[seq_idx]
        dedup_len = len(indices)
        dedup_seq = deduplicated_tensor[dedup_offset:dedup_offset + dedup_len]
        restored_sequences.append(
            _restore_single_sequence(dedup_seq, indices, int(orig_len))
        )
        dedup_offset += dedup_len

    if restored_sequences:
        return torch.cat(restored_sequences, dim=0)
    return torch.empty(0, deduplicated_tensor.shape[1], device=deduplicated_tensor.device)


# ============================================================================
# 统一接口
# ============================================================================

def deduplicate(
    tensor: torch.Tensor,
    method: Literal['mae', 'rmse', 'max', 'exact'] = 'exact',
    threshold: float = 0.01,
    lengths: Optional[Union[List[int], torch.Tensor]] = None,
    anchored: bool = True,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, List[torch.Tensor], List[int]]
]:
    """
    统一去重接口，自动检测单序列或打包序列。

    度量选择（像素值归一化到 [0,1]，threshold 在同一空间解释）：
        mae   : 每像素平均绝对误差，线性，对 outlier 不敏感
        rmse  : 每像素 RMS 误差，对大误差惩罚更重，与 mae/max 同单位可直接比较
        max   : 最大单像素误差，最严格，保证无单像素超标
        exact : 精确匹配（max, threshold=0）

    Args:
        tensor: 输入张量 [L, D] 或 [L_total, D]
        method: 去重方法
        threshold: 距离阈值（越小越严格）
        lengths: 提供时视为打包序列
        anchored: 是否使用锚定模式

    Returns:
        单序列: (indices, deduplicated)
        打包序列: (deduplicated_tensor, indices_list, new_lengths)
    """
    if lengths is None:
        return deduplicate_consecutive(tensor, method, threshold, anchored)
    return deduplicate_packed_sequences(tensor, lengths, method, threshold, anchored)


