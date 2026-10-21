"""护栏层单测：不依赖 LLM，全部是纯函数。

护栏的定位是"模型输出不可信"，因此这里的断言重点不是"解析正确"，
而是**解析失败时是否优雅降级**：宁可少一点信息，也不能让整条流水线崩掉，
更不能因为一个非法 verdict 就陷入无限返工。
"""

from __future__ import annotations

import json
import time

import pytest

from src.guardrails.checks import (
    Budget,
    parse_critique,
    parse_notes,
    parse_plan,
    parse_report,
    truncate_evidence,
    validate_citations,
)
from src.llm import extract_json
from src.schemas import Note, Reference

# --------------------------------------------------------------------- parse_plan


def test_parse_plan_normal_dict():
    plan = parse_plan({"plan": ["界定研究边界", "比较主流技术路线"]})
    assert plan.plan == ["界定研究边界", "比较主流技术路线"]


@pytest.mark.parametrize("raw", [None, "not a dict", 123, ["plan"], [{"plan": ["a"]}]])
def test_parse_plan_invalid_input_returns_empty(raw):
    """非 dict 输入一律退化为空计划，由上层决定是否用模板兜底。"""
    assert parse_plan(raw).plan == []


@pytest.mark.parametrize(
    "items",
    [
        "abc",  # 字符串：会被误当 Iterable，但实现里显式排除
        5,  # 非可迭代
        None,
    ],
)
def test_parse_plan_when_plan_field_not_a_list(items):
    assert parse_plan({"plan": items}).plan == []


def test_parse_plan_truncated_to_limit():
    raw = {"plan": [f"子问题-{index}" for index in range(10)]}
    assert len(parse_plan(raw, limit=3).plan) == 3


def test_parse_plan_hard_cap_even_if_limit_larger():
    """即便调用方给了很大的 limit，也有 8 条的硬上限，防止计划失控。"""
    raw = {"plan": [f"子问题-{index}" for index in range(20)]}
    assert len(parse_plan(raw, limit=100).plan) == 8


def test_parse_plan_dedup_keeps_order():
    plan = parse_plan({"plan": ["甲", "乙", "甲", "丙", "乙"]})
    assert plan.plan == ["甲", "乙", "丙"]


def test_parse_plan_strips_blank_items():
    assert parse_plan({"plan": ["甲", "  ", ""]}).plan == ["甲"]


# -------------------------------------------------------------------- parse_notes


def test_parse_notes_keeps_valid_items_and_skips_broken_ones():
    """核心降级行为：单条脏数据不应拖垮整批证据。"""
    raw = {
        "notes": [
            {"claim": "有效陈述一。", "source": "local-kb", "confidence": 0.8},
            "不是 dict",
            None,
            42,
            {"claim": "有效陈述二。", "source": "web", "url": "https://example.com"},
        ]
    }
    notes = parse_notes(raw)
    assert len(notes) == 2
    assert all(isinstance(note, Note) for note in notes)
    assert notes[0].claim == "有效陈述一。"


def test_parse_notes_drops_empty_claim():
    """claim 为空的条目无法被审计，直接丢弃。"""
    assert parse_notes({"notes": [{"source": "local-kb"}, {"claim": "   "}]}) == []


def test_parse_notes_confidence_out_of_range_is_dropped():
    """confidence 越界（>1 / <0 / 非数值）的条目被丢弃而不是被悄悄修正。"""
    raw = {
        "notes": [
            {"claim": "唯一有效的一条。", "confidence": 0.5},
            {"claim": "置信度越界。", "confidence": 5},
            {"claim": "置信度为负。", "confidence": -1},
            {"claim": "置信度不是数字。", "confidence": "high"},
        ]
    }
    notes = parse_notes(raw)
    assert len(notes) == 1
    assert notes[0].claim == "唯一有效的一条。"


@pytest.mark.parametrize("raw", [None, "x", 123, {"notes": "不是列表"}, {"notes": None}, {}])
def test_parse_notes_invalid_container(raw):
    assert parse_notes(raw) == []


# ------------------------------------------------------------------ parse_critique


@pytest.mark.parametrize("verdict", ["maybe", "REVISEx", "", "pass ", "revise-x", 123, None])
def test_parse_critique_unknown_verdict_degrades_to_pass(verdict):
    """未知 verdict 一律降级为 pass —— 防无限返工的关键行为。"""
    assert parse_critique({"verdict": verdict}).verdict == "pass"


def test_parse_critique_accepts_valid_verdicts():
    assert parse_critique({"verdict": "revise"}).verdict == "revise"
    assert parse_critique({"verdict": "PASS"}).verdict == "pass"  # 大小写与空格容错


def test_parse_critique_non_dict_input():
    critique = parse_critique("garbage")
    assert critique.verdict == "pass"
    assert critique.issues == [] and critique.gaps == []


def test_parse_critique_non_list_fields_do_not_explode():
    """issues / gaps 不是列表（但可迭代）时不炸，只是被逐字符/逐键拆开。"""
    critique = parse_critique({"verdict": "revise", "issues": "abc", "gaps": {"缺口甲": 1}})
    assert critique.verdict == "revise"
    assert isinstance(critique.issues, list)
    assert isinstance(critique.gaps, list)


def test_parse_critique_null_fields_fall_back_to_empty_lists():
    """gaps / issues 显式为 None 时必须容错为空列表。

    历史缺陷：模型的空列表经常写成 null，旧实现直接 `for item in None`，
    导致护栏本身成为崩溃点。这里固化修复后的行为。
    """
    critique = parse_critique({"verdict": "revise", "gaps": None, "issues": None})
    assert critique.verdict == "revise"
    assert critique.gaps == []
    assert critique.issues == []


def test_parse_critique_numeric_fields_are_ignored():
    """非字符串、非列表的脏字段一律忽略，而不是抛异常。"""
    critique = parse_critique({"verdict": "revise", "gaps": 42, "issues": 3.14})
    assert critique.verdict == "revise"
    assert critique.gaps == []
    assert critique.issues == []


def test_parse_critique_truncates_long_lists():
    raw = {"verdict": "revise", "issues": [f"问题{i}" for i in range(30)]}
    assert len(parse_critique(raw).issues) == 10


# ------------------------------------------------------------------- parse_report


@pytest.mark.parametrize(
    "raw",
    [None, "text", 42, {}, {"report": ""}, {"report": "   \n "}, {"title": "没有正文"}],
)
def test_parse_report_empty_body_returns_none(raw):
    assert parse_report(raw) is None


def test_parse_report_ok():
    draft = parse_report({"report": "# 标题\n\n结论 [1]", "references": [{"id": 1, "title": "A"}]})
    assert draft is not None
    assert draft.report == "# 标题\n\n结论 [1]"
    assert len(draft.references) == 1


def test_parse_report_dedup_references_by_id():
    """Writer 经常重复输出同一个 id，去重后引用编号才可能与正文对齐。"""
    draft = parse_report(
        {
            "report": "结论 [1] 与 [2]",
            "references": [
                {"id": 1, "title": "A"},
                {"id": 1, "title": "A 的重复项"},
                {"id": 2, "title": "B"},
            ],
        }
    )
    assert draft is not None
    assert [ref.id for ref in draft.references] == [1, 2]
    assert draft.references[0].title == "A"


def test_parse_report_skips_broken_reference_items():
    draft = parse_report(
        {
            "report": "正文",
            "references": [
                "不是 dict",
                {"id": "not-an-int", "title": "坏 id"},
                {"id": 0, "title": "id 必须 >= 1"},
                {"id": 3, "title": "有效"},
            ],
        }
    )
    assert draft is not None
    assert [ref.id for ref in draft.references] == [3]


# --------------------------------------------------------------- validate_citations


def test_validate_citations_all_exist():
    references = [Reference(id=1, title="A"), Reference(id=2, title="B")]
    assert validate_citations("结论 [1] 与 [2] 都有出处。", references) == []


def test_validate_citations_no_citation_at_all():
    assert validate_citations("这段没有引用任何来源。", []) == []


def test_validate_citations_out_of_range_reports_index():
    references = [Reference(id=1, title="A")]
    issues = validate_citations("结论 [1] 与 [3]、[7]。", references)
    assert len(issues) == 2
    assert "3" in issues[0] and "7" in issues[1]


def test_validate_citations_empty_report_is_safe():
    assert validate_citations("", []) == []


# --------------------------------------------------------------------- extract_json


def test_extract_json_plain():
    assert extract_json('{"plan": ["a"]}') == {"plan": ["a"]}


def test_extract_json_fenced():
    text = "好的：\n```json\n{\"verdict\": \"pass\"}\n```\n以上。"
    assert extract_json(text) == {"verdict": "pass"}


def test_extract_json_surrounded_by_noise():
    text = '这是结果，请参考：\n{"notes": [{"claim": "x"}]}\n如果还有疑问请继续提问。'
    assert extract_json(text) == {"notes": [{"claim": "x"}]}


def test_extract_json_top_level_list():
    assert extract_json('前缀 ["a", "b"] 后缀') == ["a", "b"]


@pytest.mark.parametrize("text", ["", "没有任何 JSON", None])
def test_extract_json_no_json_returns_none(text):
    assert extract_json(text) is None


def test_extract_json_roundtrip_with_real_payload():
    payload = {"report": "正文", "references": [{"id": 1}]}
    assert extract_json(json.dumps(payload, ensure_ascii=False)) == payload


# ------------------------------------------------------------------------- Budget


def test_budget_not_exhausted_initially():
    budget = Budget()
    assert budget.exhausted is False
    assert budget.llm_calls == 0 and budget.tool_calls == 0


def test_budget_exhausted_by_llm_calls():
    budget = Budget(max_llm_calls=2)
    budget.consume_llm()
    assert budget.exhausted is False
    budget.consume_llm()
    assert budget.exhausted is True


def test_budget_exhausted_by_tool_calls():
    budget = Budget(max_tool_calls=1)
    budget.consume_tool()
    assert budget.exhausted is True


def test_budget_exhausted_by_timeout():
    """手动回拨 started_at 模拟超时，避免测试真的 sleep。"""
    budget = Budget(max_seconds=10)
    budget.started_at = time.perf_counter() - 30
    assert budget.elapsed >= 10
    assert budget.exhausted is True


def test_budget_zero_seconds_exhausted_immediately():
    assert Budget(max_seconds=0).exhausted is True


def test_budget_snapshot_shape():
    snapshot = Budget(max_llm_calls=1).snapshot()
    assert set(snapshot) == {"llm_calls", "tool_calls", "elapsed_s", "exhausted"}
    assert snapshot["exhausted"] is False


# ---------------------------------------------------------------- truncate_evidence


def test_truncate_evidence_short_text_untouched():
    assert truncate_evidence("abc", limit=10) == "abc"


def test_truncate_evidence_long_text_cut():
    result = truncate_evidence("x" * 100, limit=10)
    assert len(result) < 100
    assert result.startswith("x" * 10)
    assert "截断" in result
