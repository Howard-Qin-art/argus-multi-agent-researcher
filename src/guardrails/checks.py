"""护栏：在节点边界强制执行结构与资源约束。

三道防线：
1. **结构防线** —— 模型的 JSON 输出必须能通过 Pydantic 校验，否则降级为安全默认值；
2. **事实防线** —— 报告里的引用编号必须真实存在，防止模型编造参考文献；
3. **资源防线** —— LLM 调用次数、工具次数、总耗时都设上限，防止自纠错循环演变成失控花费。

设计取舍：护栏失败时**优先返回降级结果而不是中断运行**，
因为研究任务本身允许部分信息缺失；但资源类护栏会强制终止，保护预算。
"""

from __future__ import annotations

import re
import time
from typing import Any, Iterable

from pydantic import ValidationError

from ..schemas import Critique, Note, Plan, Reference, ReportDraft

__all__ = [
    "parse_plan",
    "parse_notes",
    "parse_critique",
    "parse_report",
    "validate_citations",
    "truncate_evidence",
    "Budget",
]

_CITATION_RE = re.compile(r"\[(\d+)\]")
_MAX_PLAN_ITEMS_HARD = 8


def parse_plan(raw: Any, limit: int = 4) -> Plan:
    """解析 Supervisor 的计划。失败时返回空计划，由上层决定是否重试。"""
    if not isinstance(raw, dict):
        return Plan(plan=[])
    items = raw.get("plan")
    if not isinstance(items, Iterable) or isinstance(items, (str, bytes)):
        return Plan(plan=[])
    try:
        plan = Plan(plan=[str(item) for item in items])
    except ValidationError:
        return Plan(plan=[])
    # 即便提示词里要求不超过 n 条，模型仍可能超；这里做硬性截断
    plan.plan = plan.plan[: min(limit, _MAX_PLAN_ITEMS_HARD)]
    return plan


def parse_notes(raw: Any) -> list[Note]:
    """解析 Researcher 的证据列表，逐条校验并丢弃不合规项（保留有效部分）。"""
    if not isinstance(raw, dict):
        return []
    raw_notes = raw.get("notes")
    if not isinstance(raw_notes, list):
        return []

    notes: list[Note] = []
    for item in raw_notes:
        if not isinstance(item, dict):
            continue
        try:
            notes.append(
                Note(
                    claim=str(item.get("claim", "")).strip(),
                    source=str(item.get("source", "") or ""),
                    url=str(item.get("url", "") or ""),
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        except (ValidationError, ValueError, TypeError):
            # 单条异常不应影响其余证据
            continue
    return [note for note in notes if note.claim]


def _string_list(raw: Any, limit: int) -> list[str]:
    """把任意模型的输出安全转成字符串列表。

    模型经常把空列表写成 null，或直接给出一个字符串；
    这里统一容错，保证护栏本身永远不会成为崩溃点。
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip()][:limit] if raw.strip() else []
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item).strip() for item in raw if item and str(item).strip()][:limit]


def parse_critique(raw: Any) -> Critique:
    """解析 Critic 结论。默认 pass，避免未知输出导致无限返工。"""
    if not isinstance(raw, dict):
        return Critique(verdict="pass")
    verdict = str(raw.get("verdict", "pass")).strip().lower()
    if verdict not in {"pass", "revise"}:
        verdict = "pass"
    try:
        return Critique(
            verdict=verdict,  # type: ignore[arg-type]
            issues=_string_list(raw.get("issues"), 10),
            gaps=_string_list(raw.get("gaps"), 10),
        )
    except ValidationError:
        return Critique(verdict="pass")


def parse_report(raw: Any) -> ReportDraft | None:
    """解析 Writer 产物。report 为空时视为失败。"""
    if not isinstance(raw, dict):
        return None
    report = str(raw.get("report", "")).strip()
    if not report:
        return None

    references: list[Reference] = []
    raw_refs = raw.get("references", [])
    if isinstance(raw_refs, list):
        for index, item in enumerate(raw_refs, start=1):
            if not isinstance(item, dict):
                continue
            try:
                references.append(
                    Reference(
                        id=int(item.get("id", index)),
                        title=str(item.get("title", "") or ""),
                        url=str(item.get("url", "") or ""),
                        source=str(item.get("source", "") or ""),
                    )
                )
            except (ValidationError, ValueError, TypeError):
                continue

    # 引用 id 去重且必须连续可寻址，Writer 常 duplicate
    seen: dict[int, Reference] = {}
    for reference in references:
        seen.setdefault(reference.id, reference)

    try:
        return ReportDraft(report=report, references=list(seen.values()))
    except ValidationError:
        return None


def validate_citations(report: str, references: list[Reference]) -> list[str]:
    """检查正文引用编号是否越界。返回问题列表，空列表表示合规。"""
    cited = {int(match) for match in _CITATION_RE.findall(report or "")}
    if not cited:
        return []
    known = {reference.id for reference in references}
    missing = sorted(cited - known)
    if not missing:
        return []
    return [
        f"正文引用 [{index}] 在参考列表中不存在（疑似编造引用）"
        for index in missing
    ]


def truncate_evidence(text: str, limit: int = 6000) -> str:
    """截断超长证据，防止上下文溢出。"""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…[证据过长已截断]"


class Budget:
    """运行期资源预算。任何一项超限即为 exhausted。"""

    def __init__(
        self,
        max_llm_calls: int = 20,
        max_tool_calls: int = 12,
        max_seconds: int = 300,
    ) -> None:
        self.max_llm_calls = max_llm_calls
        self.max_tool_calls = max_tool_calls
        self.max_seconds = max_seconds
        self.llm_calls = 0
        self.tool_calls = 0
        self.started_at = time.perf_counter()

    def consume_llm(self, count: int = 1) -> None:
        self.llm_calls += count

    def consume_tool(self, count: int = 1) -> None:
        self.tool_calls += count

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def exhausted(self) -> bool:
        """预算是否已耗尽。耗尽会让图跳过返工，直接出稿。"""
        return (
            self.llm_calls >= self.max_llm_calls
            or self.tool_calls >= self.max_tool_calls
            or self.elapsed >= self.max_seconds
        )

    def snapshot(self) -> dict[str, int | float]:
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "elapsed_s": round(self.elapsed, 2),
            "exhausted": self.exhausted,
        }
