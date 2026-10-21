"""工具层与记忆层单测。

工具层的断言重点是"**绝不向上抛异常**"：工具失败必须表达成一个
ok=False 的 ToolResult，让 Critic 能看见失败原因，而不是让整图崩掉。
记忆层则重点验证"超预算时丢什么"与"用户输入不能穿透文件系统"。
"""

from __future__ import annotations

import json

import pytest

from src.config import load_settings
from src.memory.long_term import LongTermMemory
from src.memory.short_term import ShortTermMemory
from src.schemas import ToolResult
from src.tools.calculator import CalculationError, SafeCalculator, safe_eval
from src.tools.registry import ToolRegistry, ensure_sample_kb
from src.tools.retriever import LocalRetriever, RetrievedChunk

# ----------------------------------------------------------------------- safe_eval


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("1 + 2 * 3", 7.0),
        ("(1 + 2) * 3", 9.0),
        ("10 / 4", 2.5),
        ("7 // 2", 3.0),
        ("7 % 3", 1.0),
        ("2 ** 10", 1024.0),
        ("-3 + 1", -2.0),
        ("sqrt(16)", 4.0),
        ("abs(-5)", 5.0),
        ("round(3.14159, 2)", 3.14),
        ("min(1, 2)", 1.0),
        ("max(1, 9)", 9.0),
        ("log(1)", 0.0),
        ("pow(2, 3)", 8.0),
    ],
)
def test_safe_eval_arithmetic(expression, expected):
    assert safe_eval(expression) == pytest.approx(expected)


def test_safe_eval_constant():
    import math

    assert safe_eval("pi") == pytest.approx(math.pi)
    assert safe_eval("e") == pytest.approx(math.e)


@pytest.mark.parametrize(
    "expression",
    [
        '__import__("os").system("echo pwned")',
        "os.system('echo pwned')",
        'open("/etc/passwd").read()',
        "().__class__.__bases__[0]",
        '"".__class__',
        "(1).__class__",
        "x = 1",
        "[1, 2, 3][0]",
        "'abc'[0]",
        "1 == 1",
        "sum([1, 2])",
        "eval('1+1')",
        "exec('pass')",
        "",
        "lambda: 1",
    ],
)
def test_safe_eval_rejects_injection(expression):
    """AST 白名单拒绝一切非算术语法：属性访问、赋值、下标、比较、任意函数调用。"""
    with pytest.raises((CalculationError, ValueError)):
        safe_eval(expression)


def test_safe_eval_error_type():
    with pytest.raises(CalculationError):
        safe_eval('__import__("os")')


# ------------------------------------------------------------------ SafeCalculator


def test_safe_calculator_success():
    result = SafeCalculator().run("2 + 3")
    assert result.ok is True
    assert result.error == ""
    assert "5" in result.output


def test_safe_calculator_failure_returns_failed_result_without_raising():
    """失败必须收敛成 ok=False 的结果对象，绝不抛异常。"""
    result = SafeCalculator().run('__import__("os").system("dir")')
    assert result.ok is False
    assert result.error != ""
    assert result.output == ""


def test_safe_calculator_callable_with_query_kwarg():
    calc = SafeCalculator()
    assert calc("1 + 1").ok is True
    assert calc(query="4 * 4").ok is True
    assert calc().ok is False  # 空表达式


# ------------------------------------------------------------------ LocalRetriever


@pytest.fixture
def kb_dir(tmp_path):
    """一个只有两篇文档的小型知识库。"""
    directory = tmp_path / "kb"
    directory.mkdir()
    (directory / "vector-db.md").write_text(
        "# 向量数据库\n\n"
        "向量数据库用近似最近邻索引做召回，常见实现有 HNSW 与 IVF-PQ，"
        "取舍集中在召回率、延迟与内存占用之间。\n\n"
        "## 运维成本\n\n"
        "向量库需要独立部署与调优，索引重建会带来明显的写入放大。\n",
        encoding="utf-8",
    )
    (directory / "orchestration.md").write_text(
        "# 编排框架\n\n"
        "多智能体编排框架把一次任务拆成若干节点，节点之间用显式状态传递信息，"
        "好处是每一步都可回放与审计。\n",
        encoding="utf-8",
    )
    return directory


def test_retriever_index_reads_chunks(kb_dir):
    retriever = LocalRetriever(kb_dir)
    chunks = retriever.index()
    assert chunks
    assert all(isinstance(chunk, RetrievedChunk) for chunk in chunks)
    # 标题块也会被追加进索引，作为弱召回信号
    assert any(chunk.title == "vector-db" for chunk in chunks)


def test_retriever_index_is_cached(kb_dir):
    retriever = LocalRetriever(kb_dir)
    assert retriever.index() is retriever.index()


def test_retriever_hit_relevant_query(kb_dir):
    retriever = LocalRetriever(kb_dir, top_k=2)
    hits = retriever.retrieve("向量数据库 召回率")
    assert hits
    assert "向量" in hits[0].content
    assert all(hit.score > 0 for hit in hits)
    assert len(hits) <= 2


def test_retriever_miss_irrelevant_query(kb_dir):
    """完全无关的词不应命中任何块（分词层面无任何重叠）。"""
    assert LocalRetriever(kb_dir).retrieve("香蕉拖拉机") == []


def test_retriever_empty_query_returns_empty(kb_dir):
    assert LocalRetriever(kb_dir).retrieve("   ") == []


def test_retriever_missing_dir_is_safe(tmp_path):
    retriever = LocalRetriever(tmp_path / "not-exist")
    assert retriever.index() == []
    assert retriever.retrieve("任意关键词") == []


def test_retriever_call_returns_ok(kb_dir):
    result = LocalRetriever(kb_dir)("向量数据库")
    assert result.ok is True
    assert result.output.strip() != ""
    assert result.error == ""


def test_retriever_call_no_hit_returns_not_ok(kb_dir):
    result = LocalRetriever(kb_dir)("香蕉拖拉机")
    assert result.ok is False
    assert result.error == "no-hit"


# -------------------------------------------------------------------- ToolRegistry


def test_registry_unknown_tool():
    result = ToolRegistry().execute("no-such-tool", "query")
    assert result.ok is False
    assert "未知工具" in result.error
    assert result.tool == "no-such-tool"


def test_registry_execute_swallows_tool_exception():
    def boom(query, **kwargs):
        raise RuntimeError("工具内部炸了")

    registry = ToolRegistry({"boom": boom})
    result = registry.execute("boom", "q")
    assert result.ok is False
    assert "RuntimeError" in result.error


def test_registry_names_and_contains():
    registry = ToolRegistry({"a": lambda q, **k: ToolResult(tool="a", ok=True)})
    assert registry.names == ["a"]
    assert "a" in registry
    assert "b" not in registry


def test_registry_run_plan_respects_max_calls():
    """max_calls 是成本控制闸门：超出部分直接丢弃，不执行。"""
    calls = {"n": 0}

    def counter(query, **kwargs):
        calls["n"] += 1
        return ToolResult(tool="counter", ok=True, output=query or "ok")

    registry = ToolRegistry({"counter": counter})
    route = {"tools": ["counter", "counter", "counter", "counter"], "queries": {"counter": "q"}}
    results = registry.run_plan(route, max_calls=2)

    assert len(results) == 2
    assert calls["n"] == 2
    assert all(result.ok for result in results)


def test_registry_run_plan_skips_unknown_tools():
    registry = ToolRegistry({"counter": lambda q, **k: ToolResult(tool="counter", ok=True)})
    route = {"tools": ["counter", "ghost"], "queries": {"counter": "q"}}
    assert len(registry.run_plan(route, max_calls=5)) == 1


def test_registry_run_plan_invalid_route():
    assert ToolRegistry().run_plan("not a mapping") == []
    assert ToolRegistry().run_plan({}) == []


def test_ensure_sample_kb_creates_doc(tmp_path):
    cfg = load_settings(kb_dir=tmp_path / "empty-kb")
    kb = ensure_sample_kb(cfg)
    assert any(path.is_file() for path in kb.rglob("*"))


# ----------------------------------------------------------------- ShortTermMemory


def test_short_term_memory_compacts_over_threshold():
    memory = ShortTermMemory(threshold=200, keep_recent=2)
    # 每轮正文远长于压缩后保留的首句，才能观察到 total_chars 真实下降
    contents = [
        f"第{index}轮结论：这是一句很短的摘要。后面跟着大量细节描述用于把上下文撑大，"
        "从而触发阈值压缩并验证最近若干轮会被保留。" + "细节描述" * 20
        for index in range(10)
    ]
    for content in contents:
        memory.add("user", content)

    baseline = sum(len(content) for content in contents)
    assert memory.compacted_times >= 1
    assert memory.total_chars < baseline
    assert memory.summary != ""
    # 最近 keep_recent 条必须原样保留，只压缩更老的
    assert [turn.content for turn in memory.turns] == contents[-len(memory.turns) :]
    assert len(memory.turns) <= memory.keep_recent


def test_short_term_memory_no_compact_when_few_turns():
    """轮数不超过 keep_recent 时不压缩：没有可折叠的历史。"""
    memory = ShortTermMemory(threshold=0, keep_recent=6)
    memory.add("user", "内容")
    assert memory.compacted_times == 0
    assert len(memory.turns) == 1
    assert memory.summary == ""


def test_short_term_memory_as_text_and_history():
    memory = ShortTermMemory(threshold=10000, keep_recent=5)
    memory.add("user", "问题甲")
    memory.add("assistant", "回答甲")
    assert "问题甲" in memory.as_text()
    assert memory.history() == [
        {"role": "user", "content": "问题甲"},
        {"role": "assistant", "content": "回答甲"},
    ]


def test_turn_build_counts_chars():
    from src.memory.short_term import Turn

    assert Turn.build("user", "abcd").chars == 4


# ------------------------------------------------------------------ LongTermMemory


@pytest.fixture
def memory_cfg(tmp_path):
    return load_settings(memory_dir=tmp_path / "memory")


def test_long_term_add_dedup(memory_cfg):
    memory = LongTermMemory("u1", cfg=memory_cfg)
    memory.add("多智能体系统的可审查性来自角色分工。")
    memory.add("多智能体系统的可审查性来自角色分工。")
    assert len(memory.items) == 1


def test_long_term_add_ignores_blank(memory_cfg):
    memory = LongTermMemory("u1", cfg=memory_cfg)
    memory.add("   ")
    assert memory.items == []


def test_long_term_recall_and_hit_count(memory_cfg):
    memory = LongTermMemory("u1", cfg=memory_cfg)
    memory.add("多智能体系统的可审查性来自角色分工。")
    memory.add("向量数据库的运维成本主要来自索引重建。")

    first = memory.recall("多智能体 角色分工")
    assert first
    assert first[0].content.startswith("多智能体")
    assert first[0].hit_count == 1

    second = memory.recall("多智能体 角色分工")
    assert second[0].hit_count == 2


def test_long_term_recall_no_match(memory_cfg):
    memory = LongTermMemory("u1", cfg=memory_cfg)
    memory.add("多智能体系统的可审查性来自角色分工。")
    assert memory.recall("香蕉拖拉机") == []
    assert memory.as_context("香蕉拖拉机") == ""


def test_long_term_save_writes_file(memory_cfg):
    memory = LongTermMemory("u1", cfg=memory_cfg)
    memory.add("事实甲：这是需要跨会话记住的内容。")
    memory.save()

    assert memory.path.exists()
    payload = json.loads(memory.path.read_text(encoding="utf-8"))
    assert payload["user_id"] == "u1"
    assert len(payload["items"]) == 1


def test_long_term_reload_roundtrip(memory_cfg):
    memory = LongTermMemory("u1", cfg=memory_cfg)
    memory.add("事实甲：这是需要跨会话记住的内容。")
    memory.remember_preference("偏好：输出要结构化。")
    memory.save()

    reloaded = LongTermMemory("u1", cfg=memory_cfg)
    assert reloaded.stats() == {"total": 2, "facts": 1, "preferences": 1}


def test_long_term_corrupted_file_degrades_to_empty(memory_cfg):
    memory_cfg.memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_cfg.memory_dir / "u1.json").write_text("{坏掉的 JSON", encoding="utf-8")
    assert LongTermMemory("u1", cfg=memory_cfg).items == []


def test_long_term_safe_id_blocks_path_traversal(memory_cfg):
    """用户 id 直接来自外部输入，必须收敛成安全文件名。"""
    assert LongTermMemory._safe_id("../../evil") == "evil"
    assert LongTermMemory._safe_id("..\\..\\evil") == "evil"
    assert LongTermMemory._safe_id("...") == "default"
    assert LongTermMemory._safe_id("") == "default"

    memory = LongTermMemory("../../evil", cfg=memory_cfg)
    assert memory.user_id == "evil"
    assert memory.path.parent == memory_cfg.memory_dir
    assert ".." not in memory.path.name


def test_long_term_trim_respects_capacity(tmp_path):
    cfg = load_settings(memory_dir=tmp_path / "memory", memory_max_items=2)
    memory = LongTermMemory("u1", cfg=cfg)
    for index in range(5):
        memory.add(f"事实{index}：这是一条用于测试容量上限的记忆内容。")
    assert len(memory.items) == 2


def test_long_term_trim_keeps_preferences(tmp_path):
    cfg = load_settings(memory_dir=tmp_path / "memory", memory_max_items=2)
    memory = LongTermMemory("u1", cfg=cfg)
    memory.remember_preference("偏好：结论先行。")
    for index in range(5):
        memory.add(f"事实{index}：这是一条用于测试容量上限的记忆内容。")
    assert any(item.kind == "preference" for item in memory.items)
