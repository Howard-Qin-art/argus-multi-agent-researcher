"""短期记忆：单次会话内的上下文管理。

为什么需要单独一层：
LangGraph 的 checkpoint 负责的是"状态持久化"（可以断点续跑），
但它不会替你决定"上下文超预算时丢什么"。这里显式实现
**保留事实摘要 + 最近若干轮原文** 的滑动窗口策略，
把"历史（可能有用）"和"上下文预算（必然有限）"的取舍落到代码里。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ShortTermMemory", "Turn"]


@dataclass(frozen=True)
class Turn:
    """一轮对话或一次节点输出。"""

    role: str
    content: str
    chars: int = 0

    @classmethod
    def build(cls, role: str, content: str) -> "Turn":
        return cls(role=role, content=content, chars=len(content))

    def as_line(self) -> str:
        return f"{self.role}: {self.content}"


@dataclass
class ShortTermMemory:
    """滑动窗口 + 摘要压缩的会话记忆。"""

    threshold: int = 4000
    keep_recent: int = 6
    summary: str = ""
    turns: list[Turn] = field(default_factory=list)
    compacted_times: int = 0

    def add(self, role: str, content: str) -> None:
        """追加一条记录，并在超预算时触发压缩。"""
        self.turns.append(Turn.build(role, content))
        if self.total_chars > self.threshold:
            self.compact()

    @property
    def total_chars(self) -> int:
        return sum(turn.chars for turn in self.turns) + len(self.summary)

    def compact(self) -> None:
        """把较老的轮次折叠进一段事实摘要，只保留最近 keep_recent 轮原文。"""
        if len(self.turns) <= self.keep_recent:
            return

        old, self.turns = self.turns[: -self.keep_recent], self.turns[-self.keep_recent :]
        digest = "；".join(turn.as_line() for turn in old)
        # 无 LLM 依赖的抽取式压缩：保留首句（通常是结论句）+ 长度上限
        condensation = []
        budget = 800
        for turn in old:
            head = turn.content.split("。")[0][:120]
            if head:
                condensation.append(f"{turn.role}: {head}")
        body = "；".join(condensation)[:budget]
        self.summary = (
            f"[历史摘要 #{self.compacted_times + 1}] {body}" if body else self.summary
        )
        self.compacted_times += 1
        _ = digest  # 保留变量以备后续接入 LLM 摘要时使用

    def as_text(self) -> str:
        """渲染成可直接塞进提示词的文本。"""
        parts: list[str] = []
        if self.summary:
            parts.append(self.summary)
        parts.extend(turn.as_line() for turn in self.turns)
        return "\n".join(parts)

    def history(self) -> list[dict[str, str]]:
        return [{"role": turn.role, "content": turn.content} for turn in self.turns]
