"""Supervisor：把开放式问题拆成可独立取证的子问题。

关键行为：
- 首轮：记忆上下文 + 原始问题 → 生成计划；
- 返工轮（Critic 判 revise）：带上 gaps 与已有计划**增量重规划**，
  已取过证的子问题不重复消耗预算；
- 护栏：计划为空或模型返回乱码时，用退化计划兜底，保证流水线不中断。
"""

from __future__ import annotations

from typing import Any

from ..graph.state import ResearchState
from ..guardrails import parse_plan
from ..prompts import supervisor_messages
from .runtime import AgentRuntime

__all__ = ["build_supervisor", "fallback_plan"]

_FALLBACK_TEMPLATE = [
    "{q} 的核心定义与边界是什么",
    "{q} 的主流技术路线或方案有哪些",
    "{q} 有哪些公开的量化指标或数据",
    "{q} 当前存在的主要风险与争议",
]


def fallback_plan(query: str, limit: int = 4) -> list[str]:
    """兜底计划：模型不可用或输出不可解析时的最后一道保障。"""
    return [item.format(q=query) for item in _FALLBACK_TEMPLATE[:limit]]


def build_supervisor(runtime: AgentRuntime):
    """构造 supervisor 节点函数（闭包形式，便于注入不同的运行时）。"""

    def supervisor(state: ResearchState) -> dict[str, Any]:
        query: str = state.get("query", "")
        user_id: str = state.get("user_id", "default")
        gaps: list[str] = list(state.get("gaps") or [])
        previous_plan: list[str] = list(state.get("plan") or [])
        done: list[str] = list(state.get("done") or [])

        with runtime.tracer.span("supervisor", round=state.get("critique_round", 0)) as box:
            # 上下文来自两个层级：跨会话的长期记忆 + 本会话内前几轮的短期记忆
            from_long_term = runtime.long_term(user_id).as_context(query)
            from_session = runtime.session_context()
            memory_context = "\n".join(
                part for part in (from_long_term, from_session) if part
            )
            messages = supervisor_messages(
                query=query,
                n_items=runtime.cfg.max_plan_items,
                previous_plan=previous_plan if gaps else None,
                gaps=gaps,
                memory_context=memory_context,
            )
            payload, raw_text = runtime.call_json(messages)
            plan = parse_plan(payload, limit=runtime.cfg.max_plan_items)

            if not plan.plan:
                # 退化策略：模型没给可用计划 → 用模板保证后续节点仍有输入
                plan.plan = fallback_plan(query, runtime.cfg.max_plan_items)

            # 增量重规划：已完成取证的子问题不再重复执行
            pending = [item for item in plan.plan if item not in done]
            # 缺口优先：把 Critic 指出的缺口插到队列最前
            if gaps:
                ordered = [gap for gap in gaps if gap not in done]
                pending = ordered + [item for item in pending if item not in ordered]

            runtime.remember("supervisor", "研究计划: " + "；".join(plan.plan))

            box["meta"] = {"plan_items": len(plan.plan), "pending": len(pending)}
            box["input"] = query[:500]
            box["output"] = raw_text[:1000]

        return {
            "plan": plan.plan,
            "pending": pending,
            "llm_calls": runtime.budget.llm_calls,
        }

    return supervisor
