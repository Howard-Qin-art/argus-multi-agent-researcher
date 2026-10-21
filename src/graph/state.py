"""LangGraph 全局状态定义。

所有字段都必须是**可序列化的**（dict / list / 标量），
因为 LangGraph 的 checkpoint 需要把状态落盘以支持断点续跑。
Pydantic 对象在进入状态时统一转成 dict，出状态时再转回来。
"""

from __future__ import annotations

from typing import Any, TypedDict

from ..schemas import Note, Reference

__all__ = ["ResearchState", "notes_from_state", "references_from_state"]


class ResearchState(TypedDict, total=False):
    """一次研究任务的共享黑板（blackboard）。"""

    # ---- 输入 ----
    query: str
    user_id: str
    trace_id: str

    # ---- 规划 ----
    plan: list[str]  # Supervisor 产出的完整计划
    pending: list[str]  # 尚未取证的子问题
    done: list[str]  # 已完成取证的子问题

    # ---- 取证 ----
    notes: list[dict[str, Any]]  # Note.model_dump()
    tool_calls: int

    # ---- 审查 ----
    critique: dict[str, Any]  # Critique.model_dump()
    issues: list[str]
    gaps: list[str]
    critique_round: int
    # 本次运行的返工上限，由 cfg 写入 state，保证路由与节点用的是同一个值
    max_critic_rounds: int

    # ---- 成文 ----
    report: str
    references: list[dict[str, Any]]  # Reference.model_dump()
    citation_issues: list[str]

    # ---- 运行元数据 ----
    llm_calls: int
    errors: list[str]
    budget_exhausted: bool
    finished: bool


def notes_from_state(state: ResearchState) -> list[Note]:
    """把状态里的 dict 还原为 Note 对象，跳过已损坏的条目。"""
    from pydantic import ValidationError

    notes: list[Note] = []
    for raw in state.get("notes", []) or []:
        if not isinstance(raw, dict):
            continue
        try:
            notes.append(Note(**raw))
        except ValidationError:
            continue
    return notes


def references_from_state(state: ResearchState) -> list[Reference]:
    """把状态里的 dict 还原为 Reference 对象。"""
    from pydantic import ValidationError

    references: list[Reference] = []
    for raw in state.get("references", []) or []:
        if not isinstance(raw, dict):
            continue
        try:
            references.append(Reference(**raw))
        except ValidationError:
            continue
    return references
