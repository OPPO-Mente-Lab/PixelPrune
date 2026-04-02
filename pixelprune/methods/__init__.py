"""
PixelPrune 扫描策略注册表。

所有选择方法通过注册表管理，支持按名称创建选择器实例。
内置方法在模块加载时自动注册，用户也可以通过 register_method() 扩展。

使用方式::

    from pixelprune.methods import get_selector
    selector = get_selector("pred_2d", method="exact", threshold=0.0)
    indices_list = selector.select(pixel_values, image_grid_thw)

扩展方式::

    from pixelprune.methods import register_method
    from pixelprune.methods.base import BasePatchSelector

    class MySelector(BasePatchSelector):
        name = "my_method"
        def select(self, pixel_values, image_grid_thw, spatial_merge_size=2):
            ...

    register_method(MySelector)
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base import BasePatchSelector

# ---------------------------------------------------------------------------
# 全局注册表
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Type[BasePatchSelector]] = {}


def register_method(cls: Type[BasePatchSelector]) -> Type[BasePatchSelector]:
    """
    注册一个 patch 选择方法。

    可以作为装饰器使用::

        @register_method
        class MySelector(BasePatchSelector):
            name = "my_method"
            ...

    也可以直接调用::

        register_method(MySelector)
    """
    if not cls.name:
        raise ValueError(
            f"{cls.__name__} 必须定义 `name` 类属性"
        )
    _REGISTRY[cls.name.lower()] = cls
    for alias in getattr(cls, "aliases", []):
        _REGISTRY[alias.lower()] = cls
    return cls


def get_selector(
    name: str,
    **kwargs: Any,
) -> BasePatchSelector:
    """
    根据名称创建 patch 选择器实例。

    Args:
        name: 方法名称（不区分大小写），如 'pred_2d', 'raster', 'serpentine' 等
        **kwargs: 传递给选择器构造函数的参数（method, threshold 等）

    Returns:
        BasePatchSelector 的子类实例

    Raises:
        ValueError: 未知的方法名
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = sorted(set(_REGISTRY.keys()))
        raise ValueError(
            f"未知的 patch 选择方法: {name!r}。"
            f"可用方法: {available}"
        )
    return _REGISTRY[key](**kwargs)


def list_methods() -> list[str]:
    """返回所有已注册的方法名称（去重后的主名称列表）。"""
    seen = set()
    names = []
    for cls in _REGISTRY.values():
        if cls.name not in seen:
            seen.add(cls.name)
            names.append(cls.name)
    return sorted(names)


# ---------------------------------------------------------------------------
# 自动注册内置方法
# ---------------------------------------------------------------------------

from .raster import RasterSelector
from .serpentine import SerpentineSelector
from .pred_2d import Pred2DSelector
from .random_select import RandomSelector
from .conncomp import ConnCompSelector

register_method(RasterSelector)
register_method(SerpentineSelector)
register_method(Pred2DSelector)
register_method(RandomSelector)
register_method(ConnCompSelector)

__all__ = [
    "BasePatchSelector",
    "register_method",
    "get_selector",
    "list_methods",
    "RasterSelector",
    "SerpentineSelector",
    "Pred2DSelector",
    "RandomSelector",
    "ConnCompSelector",
]
