"""LangGraph 编排：把四个角色组装成带返工闭环的有向图。

拓扑：
    START → supervisor → researcher → critic ─┬─(revise 且预算充足)→ supervisor
                                              └─(pass / 到达轮数上限)→ writer → END

为什么把"返工"做成图的条件边而不是节点内部循环：
- 每一轮都留下独立的 trace 事件，事后能看清"第几轮因为什么返工"；
- 通过 checkpoint 可以在返工前挂起，等待人工补充信息（human-in-the-loop）。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from ..agents import AgentRuntime, build_critic, build_researcher, build_supervisor, build_writer
from ..config import Settings, load_settings
from ..tools.registry import ensure_sample_kb
from .state import ResearchState

__all__ = ["build_graph", "run_research", "route_after_critic"]

try:  # LangGraph 各版本的 checkpoint 入口略有差异，做兼容
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:  # pragma: no cover
    MemorySaver = None  # type: ignore[assignment]


def route_after_critic(state: ResearchState) -> str:
    """Critic 之后的路由决策。

    返工条件：verdict == revise，且既没到轮数上限也没耗尽预算。
    否则一律进入 Writer —— **宁可出一份标注了'待验证'的报告，也不无限兜圈子**。
    """
    critique = state.get("critique") or {}
    verdict = critique.get("verdict", "pass") if isinstance(critique, dict) else "pass"
    round_index = int(state.get("critique_round", 0) or 0)
    max_rounds = int(state.get("max_critic_rounds", 0) or load_settings().max_critic_rounds)

    if verdict == "revise" and round_index < max_rounds:
        if not state.get("budget_exhausted"):
            return "supervisor"
    return "writer"


def build_graph(runtime: AgentRuntime, use_checkpointer: bool = True):
    """编译并返回可执行的图。"""
    graph = StateGraph(ResearchState)

    graph.add_node("supervisor", build_supervisor(runtime))
    graph.add_node("researcher", build_researcher(runtime))
    graph.add_node("critic", build_critic(runtime))
    graph.add_node("writer", build_writer(runtime))

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"supervisor": "supervisor", "writer": "writer"},
    )
    graph.add_edge("writer", END)

    checkpointer = None
    if use_checkpointer and MemorySaver is not None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


def run_research(
    query: str,
    user_id: str = "default",
    cfg: Settings | None = None,
    runtime: AgentRuntime | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """端到端跑一次研究任务。

    返回 dict：{state, summary, report_path}
    state 是图结束时的最终状态（含 report / references / trace 元数据）。
    """
    cfg = cfg or load_settings().ensure_dirs()
    # 首次运行时若知识库为空，写入一份占位文档，避免检索工具全程 no-hit
    ensure_sample_kb(cfg)
    runtime = runtime or AgentRuntime(cfg=cfg)

    trace_id = runtime.tracer.trace_id
    app = build_graph(runtime)

    initial: ResearchState = {
        "query": query.strip(),
        "user_id": user_id,
        "trace_id": trace_id,
        "plan": [],
        "pending": [],
        "done": [],
        "notes": [],
        "tool_calls": 0,
        "critique": {},
        "issues": [],
        "gaps": [],
        "critique_round": 0,
        # 把轮数上限放进 state，保证路由判断与节点内部用的是同一个 cfg 的值
        "max_critic_rounds": cfg.max_critic_rounds,
        "report": "",
        "references": [],
        "citation_issues": [],
        "llm_calls": 0,
        "errors": [],
        "budget_exhausted": False,
        "finished": False,
    }

    config: dict[str, Any] = {"recursion_limit": 25}
    # 使用 checkpointer 时 LangGraph 强制要求 thread_id，因此这里兜底用 trace_id，
    # 既保证每次运行有独立的检查点命名空间，又让调用方不必显式传 thread。
    config["configurable"] = {"thread_id": thread_id or trace_id}

    final_state = app.invoke(initial, config=config)

    summary = runtime.summary()
    summary["query"] = query
    summary["user_id"] = user_id

    return {
        "state": final_state,
        "summary": summary,
        "report_path": str(cfg.report_dir / f"{trace_id}.md"),
    }
