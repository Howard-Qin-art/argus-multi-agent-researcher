"""Researcher：路由工具 → 取证 → 归纳为带出处的陈述。

这里体现两个 agent 关键能力：
1. **工具路由（tool use）** —— 由模型决定调用哪些工具、用什么关键词，
   而不是写死 pipeline；注册表负责约束与安全兜底。
2. **依据 bounded context 归纳** —— 只基于检索到的证据产出陈述，
   证据为空时返回空数组而不编造（在提示词与护栏两侧同时约束）。
"""

from __future__ import annotations

from typing import Any

from ..graph.state import ResearchState
from ..guardrails import parse_notes, truncate_evidence
from ..llm import extract_json
from ..prompts import researcher_messages, router_messages
from ..schemas import ToolResult
from .runtime import AgentRuntime

__all__ = ["build_researcher"]


def _render_evidence(results: list[ToolResult]) -> str:
    """把工具结果渲染成供模型引用的证据块，并明确标注失败的来源。"""
    blocks: list[str] = []
    for result in results:
        if result.ok and result.output:
            blocks.append(f"<{result.tool}>\n{result.output}")
        else:
            blocks.append(f"<{result.tool} 调用失败: {result.error or '无输出'}>")
    return truncate_evidence("\n\n".join(blocks))


def build_researcher(runtime: AgentRuntime):
    """构造 researcher 节点函数。"""

    def researcher(state: ResearchState) -> dict[str, Any]:
        pending: list[str] = list(state.get("pending") or [])
        done: list[str] = list(state.get("done") or [])
        existing_notes: list[dict[str, Any]] = list(state.get("notes") or [])
        tool_calls = int(state.get("tool_calls", 0) or 0)
        available_tools = runtime.tools.names

        new_notes: list[dict[str, Any]] = []
        errors: list[str] = []

        with runtime.tracer.span("researcher", pending=len(pending)) as box:
            # 单次节点处理所有待办子问题：减少图上的节点往返次数
            for subquestion in pending:
                # ---- 1. 路由：模型决定工具与关键词 ----
                route_payload, _ = runtime.call_json(
                    router_messages(subquestion, available_tools)
                )
                route: dict[str, Any] = (
                    route_payload
                    if isinstance(route_payload, dict)
                    else {"tools": [], "queries": {}}
                )

                # 护栏：模型未选工具时退化为本地检索，而不是跳过该子问题
                if not route.get("tools"):
                    route["tools"] = ["retriever"] if "retriever" in runtime.tools else []

                # ---- 2. 执行：受 max_tool_calls 约束 ----
                results = runtime.tools.run_plan(route, max_calls=runtime.cfg.max_tool_calls)
                runtime.budget.consume_tool(len(results))
                tool_calls += len(results)
                for result in results:
                    if not result.ok:
                        errors.append(f"{subquestion[:30]}: {result.tool} -> {result.error}")

                # ---- 3. 归纳：只依据证据产出陈述 ----
                evidence = _render_evidence(results)
                payload, _ = runtime.call_json(
                    researcher_messages(subquestion, evidence)
                )
                notes = parse_notes(payload)
                for note in notes:
                    # 子问题写回证据本身，Critic 才能知道这条证据填的是哪块拼图
                    new_notes.append({**note.model_dump(), "subquestion": subquestion})

                # 写入短期记忆：后续轮次的 Supervisor 才能知道这块已经查过
                if notes:
                    runtime.remember(
                        "researcher",
                        f"[{subquestion}] " + "；".join(note.claim for note in notes[:3]),
                    )

                done.append(subquestion)

            box["meta"] = {
                "tool_calls": tool_calls,
                "notes": len(new_notes),
                "errors": len(errors),
            }
            box["input"] = " | ".join(pending)[:500]
            box["output"] = str(new_notes)[:1000]

        return {
            "notes": existing_notes + new_notes,
            "pending": [],
            "done": done,
            "tool_calls": tool_calls,
            "llm_calls": runtime.budget.llm_calls,
            "errors": errors,
        }

    return researcher
