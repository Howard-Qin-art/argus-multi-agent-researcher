"""统一工具注册表。

所有工具对外暴露一致的协议：``(query: str, **kwargs) -> ToolResult``。
注册表负责：
- 名称 → 实现的映射（模型永远只看到名称，拿不到真实对象）；
- 未知工具、调度失败、调用次数超限的统一兜底；
- 把 Router 输出的工具计划翻译成实际调用序列。
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Mapping

from ..config import Settings, load_settings
from ..schemas import ToolResult
from .calculator import SafeCalculator
from .retriever import LocalRetriever
from .web_search import WebSearchTool

__all__ = ["ToolRegistry", "build_default_registry"]

ToolFn = Callable[..., ToolResult]

# 给模型看的工具说明。写得越具体，路由越准。
TOOL_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "retriever": "本地知识库检索，适合术语定义、已有文档、概念解释",
        "web_search": "联网搜索，适合时效信息、外部公开数据与新闻",
        "calculator": "数值计算，适合占比/同比/环比等需要精确算的场景",
    }
)


class ToolRegistry:
    """工具容器 + 受限调度器。"""

    def __init__(self, tools: Mapping[str, ToolFn] | None = None) -> None:
        self._tools: dict[str, ToolFn] = dict(tools or {})

    # ---------- 管理 ----------
    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    # ---------- 调度 ----------
    def execute(self, name: str, query: str = "", **kwargs) -> ToolResult:
        """执行单个工具。任何异常都被收敛成失败的 ToolResult，绝不向上抛。"""
        if name not in self._tools:
            return ToolResult(tool=name, ok=False, error=f"未知工具: {name}")
        try:
            return self._tools[name](query, **kwargs)
        except Exception as exc:  # 兜底：工具实现自身的缺陷不应中断整图
            return ToolResult(
                tool=name, ok=False, error=f"{type(exc).__name__}: {exc}"
            )

    def run_plan(self, route: Mapping, max_calls: int = 3) -> list[ToolResult]:
        """按 Router 的计划执行工具。

        route 形如 ``{"tools": ["retriever"], "queries": {"retriever": "关键词"}}``。
        超出 max_calls 的部分直接丢弃，作为成本控制的一道闸门。
        工具返回但无任何输出时，不会被计入有效证据。
        """
        results: list[ToolResult] = []
        if not isinstance(route, Mapping):
            return results

        tools = [str(item) for item in route.get("tools", []) if str(item) in self]
        queries = route.get("queries") or {}

        for name in tools[:max_calls]:
            query = str(queries.get(name, "") if isinstance(queries, Mapping) else "")
            results.append(self.execute(name, query))
        return results


def build_default_registry(cfg: Settings | None = None) -> ToolRegistry:
    """按配置装配默认工具集。配置关闭的工具不会被注册到可见列表里。"""
    cfg = cfg or load_settings()
    registry = ToolRegistry()

    if cfg.enable_rag:
        registry.register("retriever", LocalRetriever(cfg.kb_dir, cfg.retrieve_top_k))
    if cfg.enable_web_search:
        registry.register("web_search", WebSearchTool(cfg.search_results))
    registry.register("calculator", SafeCalculator())

    return registry


def ensure_sample_kb(cfg: Settings | None = None) -> Path:
    """确保示例知识库存在。目录为空时写入一份占位文档，避免首次运行检索全空。"""
    cfg = cfg or load_settings()
    cfg.kb_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in cfg.kb_dir.rglob("*") if p.is_file()]
    if existing:
        return cfg.kb_dir

    sample = cfg.kb_dir / "sample-notes.md"
    sample.write_text(
        "# 示例知识库\n\n"
        "这是一个占位文档，用于让默认流程在没有真实语料时也能跑通。\n\n"
        "## 研究方法的一般建议\n\n"
        "开放式研究问题通常需要先界定边界，再比较主流技术路线，"
        "然后寻找可量化的公开指标，最后归纳仍然存在的争议。\n\n"
        "## 关于多智能体\n\n"
        "多智能体系统的优势在于角色分工带来的可审查性，"
        "代价是更高的 token 成本与更复杂的调试链路。\n",
        encoding="utf-8",
    )
    return cfg.kb_dir
