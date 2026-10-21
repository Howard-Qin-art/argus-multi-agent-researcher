"""联网搜索工具。

使用 DuckDuckGo 官方 Python 客户端，无需申请 API Key，适合作为默认外部检索源。
网络不可用或依赖未安装时**返回失败结果而不是抛异常**，
保证上层图不会因为单点工具故障而整体崩溃。
"""

from __future__ import annotations

import time
from typing import Any

from ..schemas import ToolResult

__all__ = ["WebSearchTool", "web_search"]


def _client():  # 兼容新旧包名，任一可用即可
    try:
        from ddgs import DDGS  # type: ignore

        return DDGS()
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore

            return DDGS()
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的兜底
            raise RuntimeError(
                "未安装搜索依赖，请执行 pip install ddgs（或 duckduckgo-search）"
            ) from exc


class WebSearchTool:
    """DuckDuckGo 文本搜索封装。"""

    name = "web_search"

    def __init__(self, max_results: int = 5, timeout: int = 30) -> None:
        self.max_results = max_results
        self.timeout = timeout

    def __call__(self, query: str, **kwargs: Any) -> ToolResult:
        return self.run(query)

    def run(self, query: str) -> ToolResult:
        started = time.perf_counter()
        try:
            with _client() as ddgs:
                raw = list(
                    ddgs.text(query, max_results=self.max_results)  # type: ignore
                )
        except Exception as exc:  # 网络、限流、依赖缺失统统降级为失败结果
            return ToolResult(
                tool=self.name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        lines: list[str] = []
        for idx, item in enumerate(raw, 1):
            title = item.get("title", "")
            body = item.get("body", "")
            href = item.get("href", "")
            lines.append(f"[{idx}] {title}\nURL: {href}\n摘要: {body}")

        output = "\n\n".join(lines) if lines else "未检索到结果"
        return ToolResult(
            tool=self.name,
            ok=bool(raw),
            output=output,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )


def web_search(query: str, max_results: int = 5) -> ToolResult:
    """函数式快捷入口。"""
    return WebSearchTool(max_results=max_results).run(query)
