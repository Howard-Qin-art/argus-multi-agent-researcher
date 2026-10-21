"""多智能体共享运行时。

把 LLM 客户端、工具注册表、追踪器与资源预算收敛到一个对象里，
由它在四个角色之间共享。这样做的好处：
- 每个角色都自动获得 LLM 调用计数（资源预算的关键依据）；
- trace 统一落在同一个 trace_id 下，事后能串成一条完整链路；
- 测试时替换 LLM 或工具只是替换一个字段。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from ..config import Settings, load_settings
from ..guardrails import Budget
from ..llm import MockChatModel, get_llm
from ..memory import LongTermMemory, ShortTermMemory
from ..observability import Tracer, configure_tracing, estimate_tokens
from ..tools import ToolRegistry, build_default_registry

__all__ = ["AgentRuntime"]


@dataclass
class AgentRuntime:
    """角色之间共享的执行环境。"""

    cfg: Settings = field(default_factory=load_settings)
    llm: BaseChatModel | None = None
    tools: ToolRegistry | None = None
    tracer: Tracer | None = None
    budget: Budget | None = None
    # 单次运行内跨共享，用来把前几轮的结论带给后续轮次
    session: ShortTermMemory | None = None

    def __post_init__(self) -> None:
        self.llm = self.llm if self.llm is not None else get_llm(self.cfg)
        self.tools = self.tools if self.tools is not None else build_default_registry(self.cfg)
        # 用 configure_tracing 而非直接 new Tracer，这样开启 LangSmith 时会顺带配好环境变量
        self.tracer = self.tracer if self.tracer is not None else configure_tracing(self.cfg)
        if self.budget is None:
            self.budget = Budget(
                max_llm_calls=self.cfg.max_plan_items * 4 + 6,
                max_tool_calls=self.cfg.max_plan_items * self.cfg.max_tool_calls + 2,
                max_seconds=300,
            )
        if self.session is None:
            self.session = ShortTermMemory(threshold=self.cfg.memory_summary_threshold)
        self.cfg.ensure_dirs()

    # ---------- 会话内短期记忆 ----------
    def remember(self, role: str, content: str) -> None:
        """把本轮的关键结论放进短期记忆；超阈值时内部自动压缩。"""
        if self.session is None:
            return
        self.session.add(role, content)

    def session_context(self) -> str:
        """渲染会话内已有信息，供 Supervisor/Writer 在后续轮次参考。"""
        return self.session.as_text() if self.session else ""

    # ---------- LLM ----------
    @property
    def is_mock(self) -> bool:
        return isinstance(self.llm, MockChatModel)

    def invoke_llm(self, messages: list[BaseMessage]) -> str:
        """调用 LLM 并计入预算。返回纯文本响应内容。"""
        assert self.llm is not None and self.budget is not None
        self.budget.consume_llm()
        started = time.perf_counter()
        response = self.llm.invoke(messages)
        _ = time.perf_counter() - started  # 计时在 span 粒度进行，这里只用于预算
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)

    def call_json(self, messages: list[BaseMessage]) -> tuple[Any, str]:
        """调用 LLM 并解析 JSON。返回 (payload, raw_text)。

        payload 为 None 表示解析失败，由调用方的护栏逻辑决定降级策略。
        """
        from ..llm import extract_json

        raw = self.invoke_llm(messages)
        return extract_json(raw), raw

    def tokens_of(self, text: str) -> int:
        return estimate_tokens(text)

    # ---------- 记忆 ----------
    def long_term(self, user_id: str) -> LongTermMemory:
        """按用户取长期记忆仓库（懒加载其内容）。"""
        return LongTermMemory(user_id=user_id, cfg=self.cfg)

    # ---------- 观测 ----------
    def summary(self) -> dict[str, Any]:
        """trace 汇总叠加上预算快照，供 CLI 与评测消费。"""
        base = self.tracer.summary() if self.tracer else {}
        base["budget"] = self.budget.snapshot() if self.budget else {}
        return base
