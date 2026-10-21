"""Writer：把证据组织成一份带引用、可被审计的研究报告。

三件事：
1. 把**杂乱的新闻式证据**整理成结构化 Markdown，每条结论带 [n] 引用；
2. 用护栏验证引用编号真实存在，发现编造则**自动剔除并留痕**；
3. 把置信度高的事实写入长期记忆，让下一次同主题研究可以少走一遍流程。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..graph.state import ResearchState, notes_from_state
from ..guardrails import parse_report, validate_citations
from ..prompts import writer_messages
from ..schemas import Reference
from .runtime import AgentRuntime

__all__ = ["build_writer", "build_references", "strip_invalid_citations"]

_MIN_CONFIDENCE_FOR_MEMORY = 0.7
_CITATION_RE = re.compile(r"\[(\d+)\]")


def build_references(notes: list[Any]) -> list[Reference]:
    """从证据中抽取参考来源，按去重后的顺序编号。"""
    references: list[Reference] = []
    seen: set[str] = set()

    for note in notes:
        # 无链接的本地知识库条目也允许成为引用，用文档标题做身份
        key = note.url.strip() or f"{note.source}:{note.claim[:40]}"
        if key in seen:
            continue
        seen.add(key)
        references.append(
            Reference(
                id=len(references) + 1,
                title=note.claim[:60] or (note.source or "未标注来源"),
                url=note.url,
                source=note.source or "unknown",
            )
        )
    return references


def strip_invalid_citations(report: str, references: list[Reference]) -> tuple[str, list[str]]:
    """剔除正文中不存在于参考列表的引用编号。返回 (新正文, 问题列表)。"""
    issues = validate_citations(report, references)
    if not issues:
        return report, []

    known = {reference.id for reference in references}

    def replace(match: re.Match[str]) -> str:
        return match.group(0) if int(match.group(1)) in known else ""

    cleaned = _CITATION_RE.sub(replace, report)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned, issues


def build_writer(runtime: AgentRuntime):
    """构造 writer 节点函数。"""

    def writer(state: ResearchState) -> dict[str, Any]:
        query: str = state.get("query", "")
        user_id: str = state.get("user_id", "default")
        trace_id: str = state.get("trace_id", runtime.tracer.trace_id)
        notes = notes_from_state(state)
        from_long_term = runtime.long_term(user_id).as_context(query)
        from_session = runtime.session_context()
        memory_context = "\n".join(
            part for part in (from_long_term, from_session) if part
        )

        with runtime.tracer.span("writer", notes=len(notes)) as box:
            references = build_references(notes)
            note_lines = [
                f"{note.claim}（来源:{note.source or '未知'}，置信度:{note.confidence}）"
                for note in notes
            ]
            reference_lines = [
                f"[{ref.id}] {ref.title} | {ref.url or ref.source}" for ref in references
            ]

            messages = writer_messages(query, note_lines, reference_lines, memory_context)
            payload, raw_text = runtime.call_json(messages)
            draft = parse_report(payload)

            if draft is None:
                # 兜底：模型没按格式输出时，用已确认的证据拼一份最小可用报告
                body = "\n".join(
                    f"{index}. {note.claim} [{index}]" for index, note in enumerate(notes, 1)
                )
                draft = parse_report(
                    {
                        "report": f"# {query}\n\n## 结论摘要\n\n{body or '未获取到可用证据。'}",
                        "references": [ref.model_dump() for ref in references],
                    }
                )
                assert draft is not None
                references = draft.references
            else:
                # 模型可能返回与本地编号不一致的引用列表，统一重编号后再校验
                references = references or draft.references

            report, citation_issues = strip_invalid_citations(draft.report, references)

            # ---- 落地：报告写盘，便于人工复核与放到 README 做示例 ----
            runtime.cfg.report_dir.mkdir(parents=True, exist_ok=True)
            report_path = runtime.cfg.report_dir / f"{trace_id}.md"
            rendered = report
            if references:
                rendered += "\n\n---\n\n## 参考来源\n\n"
                rendered += "\n".join(
                    f"[{ref.id}] {ref.title} — {ref.url or ref.source}"
                    for ref in references
                )
            report_path.write_text(rendered, encoding="utf-8")

            # ---- 长期记忆：仅沉淀高置信度事实 ----
            trusted = [note.claim for note in notes if note.confidence >= _MIN_CONFIDENCE_FOR_MEMORY]
            if trusted:
                memory = runtime.long_term(user_id)
                memory.remember_facts(trusted, source=trace_id)
                memory.save()

            box["meta"] = {
                "references": len(references),
                "citation_issues": len(citation_issues),
                "report_chars": len(report),
                "memory_facts": len(trusted),
            }
            box["input"] = f"{query[:200]} | notes={len(notes)}"
            box["output"] = report[:1000]

        return {
            "report": report,
            "references": [ref.model_dump() for ref in references],
            "citation_issues": citation_issues,
            "llm_calls": runtime.budget.llm_calls,
            "budget_exhausted": runtime.budget.exhausted,
            "finished": True,
        }

    return writer
