"""可观测层：让一次多智能体运行事后可解释。

为什么不直接用 print：
多智能体最难受的问题是出故障后不知道**是谁在第几步、基于什么证据** 做错了决定。
这里把每个节点的输入摘要、输出摘要、耗时、LLM 调用次数落成 JSONL，
既能用 `jq` 或 pandas 直接分析，也能无缝对接 LangSmith。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ..config import Settings, load_settings

__all__ = ["TraceEvent", "Tracer", "estimate_tokens", "configure_tracing"]


def estimate_tokens(text: str) -> int:
    """粗估 token 数。中文约 1 字 ≈ 1 token，西文约 4 字符 ≈ 1 token。"""
    if not text:
        return 0
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    others = len(text) - chinese
    return chinese + max(1, others // 4)


@dataclass
class TraceEvent:
    """一次节点执行（span）的记录。"""

    trace_id: str
    node: str
    started_at: str
    elapsed_ms: int
    ok: bool = True
    error: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Tracer:
    """线程不安全但足够轻量的运行期追踪器，按 trace_id 落一个 JSONL 文件。"""

    def __init__(self, trace_id: str | None = None, cfg: Settings | None = None) -> None:
        self.cfg = cfg or load_settings()
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.events: list[TraceEvent] = []
        self._enabled = bool(self.cfg.enable_trace)
        self._path: Path | None = None

    # ---------- 属性 ----------
    @property
    def path(self) -> Path | None:
        if not self._enabled:
            return None
        if self._path is None:
            self.cfg.trace_dir.mkdir(parents=True, exist_ok=True)
            self._path = self.cfg.trace_dir / f"{self.trace_id}.jsonl"
        return self._path

    # ---------- 记录 ----------
    @contextmanager
    def span(self, node: str, **meta: Any) -> Iterator[dict[str, Any]]:
        """记录一个节点：成功 ok=True，抛异常则记录 error 后继续向上抛。

        注意：异常**不会**被吞掉。可观测只负责留痕，不负责改变控制流，
        错误处理留给图上的重试与 Critic 返工逻辑。
        """
        box: dict[str, Any] = {"meta": meta}
        started_at = datetime.now().isoformat(timespec="seconds")
        started = time.perf_counter()
        ok, error = True, ""
        try:
            yield box
        except Exception as exc:  # 记录后原样抛出
            ok, error = False, f"{type(exc).__name__}: {exc}"
            raise
        finally:
            payload = box.get("meta", meta)
            # 注意：input / output 是由节点直接写到 box 上的，
            # 早期版本误从 meta 里取这两个键，导致 token 统计恒为 0。这里改为取 box。
            input_text = str(box.get("input", ""))
            output_text = str(box.get("output", ""))
            kept = {
                key: value
                for key, value in payload.items()
                if key
                in {
                    "tool_calls",
                    "llm_calls",
                    "notes",
                    "round",
                    "verdict",
                    "plan_items",
                    "pending",
                    "errors",
                    "gaps",
                    "references",
                    "citation_issues",
                    "report_chars",
                    "memory_facts",
                }
            }
            if self.cfg.trace_verbose:
                kept["input"] = input_text
                kept["output"] = output_text
            event = TraceEvent(
                trace_id=self.trace_id,
                node=node,
                started_at=started_at,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                ok=ok,
                error=error,
                tokens_in=estimate_tokens(input_text[:8000]),
                tokens_out=estimate_tokens(output_text[:8000]),
                meta=kept,
            )
            self.events.append(event)
            if self.path:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def record(self, node: str, **meta: Any) -> TraceEvent:
        """无需 span 的简易打点。"""
        event = TraceEvent(
            trace_id=self.trace_id,
            node=node,
            started_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_ms=0,
            meta=meta,
        )
        self.events.append(event)
        return event

    # ---------- 汇总 ----------
    def summary(self) -> dict[str, Any]:
        total_ms = sum(event.elapsed_ms for event in self.events)
        failures = [event.node for event in self.events if not event.ok]
        per_node: dict[str, dict[str, int]] = {}
        for event in self.events:
            bucket = per_node.setdefault(
                event.node, {"count": 0, "elapsed_ms": 0, "tokens_in": 0, "tokens_out": 0}
            )
            bucket["count"] += 1
            bucket["elapsed_ms"] += event.elapsed_ms
            bucket["tokens_in"] += event.tokens_in
            bucket["tokens_out"] += event.tokens_out
        return {
            "trace_id": self.trace_id,
            "elapsed_ms": total_ms,
            "events": len(self.events),
            "failed_nodes": failures,
            "tokens_in": sum(event.tokens_in for event in self.events),
            "tokens_out": sum(event.tokens_out for event in self.events),
            "per_node": per_node,
            "trace_file": str(self.path) if self.path else "",
        }


def configure_tracing(cfg: Settings | None = None) -> Tracer:
    """创建 Tracer；如开启 LangSmith，则顺带设置云端 trace 所需的环境变量。"""
    cfg = cfg or load_settings()
    if cfg.langsmith and os.getenv("LANGCHAIN_TRACING_V2") is None:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ.setdefault("LANGCHAIN_PROJECT", cfg.langsmith_project)
    return Tracer(cfg=cfg)
