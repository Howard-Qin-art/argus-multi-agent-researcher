"""图编排入口（延迟导出）。

为什么 __init__ 不预先导入 orchestrator：
``orchestrator`` 会导入 ``src.agents``，而 ``src.agents`` 又要用到 ``src.graph.state``。
若在这里 eager import orchestrator，就会形成
``src.agents → src.graph.state → src.graph.__init__ → orchestrator → src.agents``
的导入环（partially initialized module 错误）。

改成 __getattr__ 惰性解析：断环的同时保持
``from src.graph import run_research`` 这种用法完全不变。
"""

from __future__ import annotations

import importlib
from typing import Any

from .state import ResearchState, notes_from_state, references_from_state

__all__ = [
    "build_graph",
    "run_research",
    "route_after_critic",
    "ResearchState",
    "notes_from_state",
    "references_from_state",
]

_LAZY_ATTRS: dict[str, str] = {
    "build_graph": ".orchestrator",
    "run_research": ".orchestrator",
    "route_after_critic": ".orchestrator",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path, __name__), name)
    globals()[name] = value  # 二次导入直接命中缓存
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
