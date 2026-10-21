"""端到端图测试：全程使用 MockChatModel，不联网、不需要 API Key。

这里验证的是"整条链路在没有任何真实模型的情况下也能产出一份可被审计的报告"，
以及两条防失控的边界：**轮数上限** 与 **引用编号必须真实存在**。
"""

from __future__ import annotations

import json

import pytest

# 同 conftest.py：先完成 src.graph 的初始化，避开与 src.agents 的循环导入。
from src.graph import orchestrator
from src.graph.orchestrator import build_graph, route_after_critic, run_research
from src.graph.state import ResearchState, notes_from_state, references_from_state

from src.agents.runtime import AgentRuntime
from src.agents.supervisor import fallback_plan
from src.agents.writer import build_references, strip_invalid_citations
from src.llm import MockChatModel
from src.schemas import Note, Reference

QUERY = "多智能体深度研究系统在工程落地时有哪些关键取舍？"


def _initial_state(query: str = QUERY, trace_id: str = "trace-test") -> ResearchState:
    """与 run_research 内部一致的最小初始状态。"""
    return {
        "query": query,
        "user_id": "tester",
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
        "report": "",
        "references": [],
        "citation_issues": [],
        "llm_calls": 0,
        "errors": [],
        "budget_exhausted": False,
        "finished": False,
    }


# ------------------------------------------------------------------ 端到端运行


def test_run_research_end_to_end(mock_runtime, tmp_cfg):
    result = run_research(QUERY, cfg=tmp_cfg, runtime=mock_runtime, thread_id="thread-e2e")
    state = result["state"]

    assert state["report"].strip() != ""
    assert isinstance(state["references"], list)
    assert state["references"]
    assert state["citation_issues"] == []
    assert state["critique_round"] >= 1
    assert state["plan"]
    assert state["finished"] is True

    summary = result["summary"]
    assert summary["trace_id"] == mock_runtime.tracer.trace_id
    assert summary["events"] >= 4  # supervisor / researcher / critic / writer


def test_run_research_without_explicit_runtime_uses_mock(tmp_cfg):
    """cfg.provider=mock 时即便不显式传 runtime，也走本地替身。"""
    result = run_research(QUERY, cfg=tmp_cfg, thread_id="thread-no-runtime")
    assert result["state"]["report"].strip() != ""


def test_run_research_writes_report_file(mock_runtime, tmp_cfg):
    result = run_research(QUERY, cfg=tmp_cfg, runtime=mock_runtime, thread_id="thread-file")
    report_path = tmp_cfg.report_dir / f"{mock_runtime.tracer.trace_id}.md"
    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8").strip() != ""
    assert result["report_path"] == str(report_path)


def test_run_research_writes_trace_jsonl(mock_runtime, tmp_cfg):
    run_research(QUERY, cfg=tmp_cfg, runtime=mock_runtime, thread_id="thread-trace")

    trace_files = list(tmp_cfg.trace_dir.glob("*.jsonl"))
    assert trace_files, "trace 目录下应至少有一个 .jsonl 文件"

    lines = [line for line in trace_files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "trace 文件不应为空"
    for line in lines:
        event = json.loads(line)  # 每一行都必须是合法 JSON
        assert event["trace_id"] == mock_runtime.tracer.trace_id
        assert event["node"] in {"supervisor", "researcher", "critic", "writer"}


def test_build_graph_manual_invoke(mock_runtime):
    """不经过 run_research，直接手动编译并 invoke 小图。"""
    app = build_graph(mock_runtime)
    final = app.invoke(
        _initial_state(trace_id="trace-manual"),
        config={"recursion_limit": 25, "configurable": {"thread_id": "thread-manual"}},
    )
    assert final["report"].strip() != ""
    assert final["finished"] is True
    assert final["critique_round"] >= 1


def test_runtime_is_mock(mock_runtime):
    assert mock_runtime.is_mock is True
    assert isinstance(mock_runtime.llm, MockChatModel)


def test_agent_runtime_defaults_to_mock_llm(tmp_cfg):
    """provider=mock 时 AgentRuntime 自己也能拿到替身模型。"""
    assert AgentRuntime(cfg=tmp_cfg).is_mock is True


# ------------------------------------------------------------ route_after_critic


@pytest.fixture
def fixed_settings(monkeypatch, tmp_cfg):
    """固定住 route_after_critic 读取的最大轮数，避免受环境变量影响。"""
    monkeypatch.setattr(orchestrator, "load_settings", lambda: tmp_cfg)
    return tmp_cfg


def test_route_after_critic_revise_returns_supervisor(fixed_settings):
    state = _initial_state()
    state["critique"] = {"verdict": "revise"}
    state["critique_round"] = 1
    assert route_after_critic(state) == "supervisor"


def test_route_after_critic_pass_returns_writer(fixed_settings):
    state = _initial_state()
    state["critique"] = {"verdict": "pass"}
    state["critique_round"] = 1
    assert route_after_critic(state) == "writer"


def test_route_after_critic_revise_at_round_limit_returns_writer(fixed_settings):
    """到达轮数上限后不再返工 —— 这是防无限循环的关键断言。"""
    state = _initial_state()
    state["critique"] = {"verdict": "revise"}
    state["critique_round"] = fixed_settings.max_critic_rounds
    assert route_after_critic(state) == "writer"


def test_route_after_critic_revise_beyond_round_limit_returns_writer(fixed_settings):
    state = _initial_state()
    state["critique"] = {"verdict": "revise"}
    state["critique_round"] = fixed_settings.max_critic_rounds + 5
    assert route_after_critic(state) == "writer"


def test_route_after_critic_revise_but_budget_exhausted(fixed_settings):
    state = _initial_state()
    state["critique"] = {"verdict": "revise"}
    state["critique_round"] = 1
    state["budget_exhausted"] = True
    assert route_after_critic(state) == "writer"


def test_route_after_critic_missing_critique_defaults_to_writer(fixed_settings):
    assert route_after_critic(_initial_state()) == "writer"
    assert route_after_critic({"critique": "不是 dict", "critique_round": 0}) == "writer"


# ----------------------------------------------------------------- fallback_plan


@pytest.mark.parametrize("limit", [1, 2, 3, 4])
def test_fallback_plan_length_matches_limit(limit):
    plan = fallback_plan("量子计算", limit=limit)
    assert len(plan) == limit
    assert all("量子计算" in item for item in plan)


def test_fallback_plan_cannot_exceed_template_size():
    assert len(fallback_plan("X", limit=99)) == 4


# ------------------------------------------------------------ writer 侧辅助函数


def test_build_references_dedup_and_numbering(sample_notes):
    notes = notes_from_state({"notes": sample_notes})
    references = build_references(notes)
    assert [ref.id for ref in references] == [1, 2, 3]
    assert references[0].source == "local-kb"

    duplicated = build_references(notes + notes)
    assert len(duplicated) == 3  # 相同来源不重复编号


def test_references_from_state_roundtrip():
    state = {"references": [Reference(id=1, title="A").model_dump(), "坏数据"]}
    assert [ref.id for ref in references_from_state(state)] == [1]


def test_notes_from_state_skips_broken_entries():
    broken = {"claim": ""}
    assert notes_from_state({"notes": [Note(claim="有效").model_dump(), broken, 42]}) != []
    assert len(notes_from_state({"notes": [Note(claim="有效").model_dump(), broken]})) == 1


def test_strip_invalid_citations_removes_unknown_ids():
    references = [Reference(id=1, title="A")]
    report = "结论一 [1]；结论二 [9]。"
    cleaned, issues = strip_invalid_citations(report, references)
    assert issues and "9" in issues[0]
    assert "[9]" not in cleaned
    assert "[1]" in cleaned


def test_strip_invalid_citations_keeps_report_when_valid():
    references = [Reference(id=1, title="A"), Reference(id=2, title="B")]
    report = "结论一 [1]；结论二 [2]。"
    cleaned, issues = strip_invalid_citations(report, references)
    assert issues == []
    assert cleaned == report
