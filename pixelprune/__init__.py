"""
PixelPrune: Pixel-Level Adaptive Visual Token Reduction via Predictive Coding.

为 Vision-Language Models 提供基于像素预测编码的 visual token 压缩，
在 ViT 编码器之前通过空间冗余检测移除重复 patch，实现高效推理与训练。

内置扫描策略:
    - raster:     行主序扫描，对相邻 token 连续去重
    - serpentine: 蛇形扫描，同时捕获水平与垂直方向重复
    - pred_2d:    LOCO-I 2D 预测编码（默认推荐），利用三个因果邻居推断边缘方向

快速使用::

    # 方式一：通过 monkey-patch 注入（推荐）
    import os
    os.environ["PIXELPRUNE_ENABLED"] = "true"
    os.environ["PIXELPRUNE_METHOD"] = "pred_2d"  # 默认

    from pixelprune import apply_pixelprune
    apply_pixelprune(model="qwen3_vl")   # 或 "qwen3_5"

    # 方式二：直接调用选择器
    from pixelprune import compute_merged_keep_indices
    indices_list = compute_merged_keep_indices(pixel_values, image_grid_thw)

扩展自定义方法::

    from pixelprune.methods import register_method, BasePatchSelector

    @register_method
    class MySelector(BasePatchSelector):
        name = "my_method"
        def select(self, pixel_values, image_grid_thw, spatial_merge_size=2):
            ...
"""

# --- 核心调度 & 索引工具 ---
from .core import (
    compute_merged_keep_indices,
    merged_indices_to_patch_indices,
)

# --- 底层去重算法 ---
from .dedup import (
    deduplicate,
    deduplicate_consecutive,
    deduplicate_consecutive_exact,
    deduplicate_consecutive_mse,
    deduplicate_consecutive_max_error,
    deduplicate_packed_sequences,
    restore_packed_sequences,
)

# --- 方法注册表 ---
from .methods import (
    BasePatchSelector,
    register_method,
    get_selector,
    list_methods,
    RasterSelector,
    SerpentineSelector,
    Pred2DSelector,
    RandomSelector,
    ConnCompSelector,
)


__version__ = "1.0.0"

__all__ = [
    # 核心 API
    "compute_merged_keep_indices",
    "merged_indices_to_patch_indices",
    "apply_pixelprune",
    # 方法注册表
    "BasePatchSelector",
    "register_method",
    "get_selector",
    "list_methods",
    "RasterSelector",
    "SerpentineSelector",
    "Pred2DSelector",
    "RandomSelector",
    "ConnCompSelector",
    # 底层去重
    "deduplicate",
    "deduplicate_consecutive",
    "deduplicate_consecutive_exact",
    "deduplicate_consecutive_mse",
    "deduplicate_consecutive_max_error",
    "deduplicate_packed_sequences",
    "restore_packed_sequences",
]


def apply_pixelprune(model: str = "qwen3_vl") -> None:
    """对 Qwen3-VL / Qwen3.5 应用 monkey-patch。需在加载模型前调用。"""
    from .patches import apply_patches as _apply
    _apply(model=model)
