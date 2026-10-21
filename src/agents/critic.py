"""Critic：决定是否返工。

这是整个系统里最"值钱"的一个角色 —— 没有它，系统就是一条单向 pipeline；
有了它才构成真正的**闭环自纠错**。

两个防止失控的设计：
1. 默认 pass：模型输出不可解析时不返工，避免陷入无效重试；
2. 轮数上限：接近 max_rounds 时提示词会要求只保留最关键缺口，
   且 orchestrator 会在预算耗尽时强制放行。
"""

from __future__ import annotations

from typing import Any

from ..graph.state import ResearchState, notes_from_state
from ..guardrails import parse_critique
from ..prompts import critic_messages
from .runtime import AgentRuntime

__all__ = ["build_critic"]


def build_critic(runtime: AgentRuntime):
    """构造 critic 节点函数。"""

    def critic(state: ResearchState) -> dict[str, Any]:
        query: str = state.get("query", "")
        plan: list[str] = list(state.get("plan") or [])
        round_index: int = int(state.get("critique_round", 0) or 0) + 1
        notes = notes_from_state(state)

        with runtime.tracer.span("critic", round=round_index) as box:
            note_lines = [
                f"{note.claim}（来源:{note.source or '未知'}，置信度:{note.confidence}）"
                for note in notes
            ]
            messages = critic_messages(
                query=query,
                plan=plan,
                notes=note_lines,
                round_idx=round_index,
                max_rounds=runtime.cfg.max_critic_rounds,
            )
            payload, raw_text = runtime.call_json(messages)
            critique = parse_critique(payload)

            # ---- 无需调用模型也能判定的硬性条件 ----
            has_notes = bool(notes)
            uncovered = [item for item in plan if item not in (state.get("done") or [])]

            if round_index >= runtime.cfg.max_critic_rounds:
                # 上限到达：无论模型怎么说都不再返工，防止无限循环
                critique.verdict = "pass"
                critique.gaps = []
            elif runtime.budget.exhausted:
                critique.verdict = "pass"
                critique.gaps = []
            elif not has_notes and critique.verdict == "pass":
                # 一条证据都没有却判通过，说明被告胶原：强制返工一次
                critique.verdict = "revise"
                critique.gaps = critique.gaps or uncovered or ["补一条最基础的可用证据"]

            runtime.remember(
                "critic",
                f"第{round_index}轮判定 {critique.verdict}"
                + (f"，待补证据: {'；'.join(critique.gaps[:3])}" if critique.gaps else ""),
            )

            box["meta"] = {
                "verdict": critique.verdict,
                "round": round_index,
                "notes": len(notes),
                "gaps": len(critique.gaps),
            }
            box["input"] = f"{query[:200]} | notes={len(notes)}"
            box["output"] = raw_text[:800]

        return {
            "critique": critique.model_dump(),
            "issues": critique.issues,
            "gaps": critique.gaps,
            "critique_round": round_index,
            # 实时写回预算耗尽标记，让 route_after_critic 的预算分支真正生效
            # （否则那里读到的一直是初始值 False，等于死分支）
            "budget_exhausted": runtime.budget.exhausted,
            "llm_calls": runtime.budget.llm_calls,
        }

    return critic
