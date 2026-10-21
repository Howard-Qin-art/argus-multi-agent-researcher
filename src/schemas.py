"""数据契约（Pydantic v2）。

所有跨越节点边界的数据都用显式 schema 定义，而不是裸 dict。
这样做的好处：护栏层可以在节点之间统一做校验，评测层也能量化"格式合规率"。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "ToolCall",
    "ToolResult",
    "Note",
    "Critique",
    "Reference",
    "ReportDraft",
    "Plan",
    "RunSummary",
]


class ToolCall(BaseModel):
    """一次工具调用意图。"""

    tool: str = Field(description="工具名称")
    args: dict[str, Any] = Field(default_factory=dict, description="工具入参")


class ToolResult(BaseModel):
    """工具执行结果。即使失败也保留对象，便于 Critic 与 trace 分析失败原因。"""

    tool: str
    ok: bool
    output: str = ""
    error: str = ""
    elapsed_ms: int = 0

    @field_validator("output", "error")
    @classmethod
    def cap_length(cls, value: str) -> str:
        # 防止超长网页正文进入上下文，撑爆 token 预算
        limit = 4000
        return value[:limit] + ("…[truncated]" if len(value) > limit else "")


class Note(BaseModel):
    """Researcher 产出的一条带出处的证据。"""

    claim: str = Field(min_length=1, description="可验证的事实性陈述")
    source: str = Field(default="", description="来源标识，如 local-kb / web")
    url: str = Field(default="", description="可回溯链接，本地知识库可为空")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # 保留产出该证据的子问题，便于 Critic 判断"哪一块还没被覆盖"
    subquestion: str = Field(default="")

    @field_validator("claim")
    @classmethod
    def strip_claim(cls, value: str) -> str:
        return value.strip()


class Critique(BaseModel):
    """Critic 的审查结论。verdict=revise 时图会回边到 Researcher 补证。"""

    verdict: Literal["pass", "revise"] = "pass"
    issues: list[str] = Field(default_factory=list, description="明确指出的问题清单")
    gaps: list[str] = Field(
        default_factory=list, description="证据缺口，下一轮取证的重点"
    )


class Reference(BaseModel):
    """报告引用的外部来源。id 与正文中的 [id] 一一对应。"""

    id: int = Field(ge=1)
    title: str = ""
    url: str = ""
    source: str = ""


class ReportDraft(BaseModel):
    """Writer 的产出：正文 + 结构化引用列表。"""

    report: str = Field(min_length=1)
    references: list[Reference] = Field(default_factory=list)


class Plan(BaseModel):
    """Supervisor 的研究计划。"""

    plan: list[str] = Field(default_factory=list)

    @field_validator("plan")
    @classmethod
    def normalize(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        # 去重但保序
        seen: set[str] = set()
        result: list[str] = []
        for item in cleaned:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result


class RunSummary(BaseModel):
    """一次完整运行的观测摘要，供 CLI 打印与评测落盘。"""

    trace_id: str
    query: str
    started_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime | None = None
    plan_items: int = 0
    notes: int = 0
    tool_calls: int = 0
    critic_rounds: int = 0
    llm_calls: int = 0
    elapsed_ms: int = 0
    references: int = 0
    citation_valid: bool = True
    report_path: str = ""
