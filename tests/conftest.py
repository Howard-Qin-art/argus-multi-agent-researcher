"""pytest 公共夹具。

核心约定：**所有会写盘的目录都重定向到 pytest 的 tmp_path**，
避免测试把 .memory / traces / reports / data 写到仓库里，
同时也让每个用例之间互相隔离（并行或重复跑都不会互相污染）。
"""

from __future__ import annotations

import pytest

# 注意导入顺序：src.agents 与 src.graph 互相引用（agents.critic -> graph.state，
# graph.orchestrator -> agents），必须先让 src.graph 完成初始化，
# 否则从 src.agents 进入会撞上"partially initialized module"。
import src.graph  # noqa: F401
from src.agents.runtime import AgentRuntime
from src.config import load_settings
from src.llm import MockChatModel
from src.schemas import Note

__all__ = ["tmp_cfg", "mock_runtime", "sample_notes"]


@pytest.fixture
def tmp_cfg(tmp_path):
    """一份只写临时目录的配置快照。

    关掉 web_search 是为了让"无网络"成为测试的硬保证：
    mock router 也不会路由到它，即便路由到了也没有真实实现可依赖。
    """
    return load_settings(
        provider="mock",
        enable_web_search=False,
        memory_dir=tmp_path / "memory",
        trace_dir=tmp_path / "traces",
        report_dir=tmp_path / "reports",
        kb_dir=tmp_path / "kb",
        max_plan_items=3,
        max_critic_rounds=2,
        max_tool_calls=2,
        retrieve_top_k=3,
    )


@pytest.fixture
def mock_runtime(tmp_cfg):
    """带 mock 模型的运行时。构造时会自动建好临时目录。"""
    return AgentRuntime(cfg=tmp_cfg, llm=MockChatModel())


@pytest.fixture
def sample_notes():
    """几条可直接塞进 ResearchState 的证据（Note.model_dump() 形态）。"""
    return [
        Note(
            claim="多智能体系统的优势在于角色分工带来的可审查性。",
            source="local-kb",
            url="",
            confidence=0.8,
        ).model_dump(),
        Note(
            claim="代价是更高的 token 成本与更复杂的调试链路。",
            source="local-kb",
            url="",
            confidence=0.6,
        ).model_dump(),
        Note(
            claim="公开可比数据仍然稀缺，横向评估结论差异较大。",
            source="web",
            url="https://example.com/report",
            confidence=0.5,
        ).model_dump(),
    ]
