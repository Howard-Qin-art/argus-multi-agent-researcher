"""评测入口：在数据集上批量跑完整流程，并把"能不能用"量化成指标。

为什么单独做一层评测而不是靠人看报告：
多智能体系统的输出是**随机且长链条**的，改一次提示词或换一个 provider，
肉眼很难判断是变好了还是只是换了一种说法。这里把每条样本跑完一次完整图，
抽出可复算的指标（出稿率、引用合规率、关键词召回、资源开销），
让"提示词 A 比 B 好"这类结论有据可依。

设计取舍：
- 指标全部从 ``run_research`` 的返回值里推导，不侵入任何节点，避免评测影响被测对象；
- 单条样本失败（异常）只记录不中断，保证一次评测能跑完全集；
- ``--provider mock`` 下不联网、不需要 API Key，CI 里也能跑。

用法：
    python -m src.eval.run_eval --provider mock
    python -m src.eval.run_eval --dataset eval/datasets/sample.jsonl --limit 2 --output runs/eval.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, Field, ValidationError

from ..config import ROOT, Settings, load_settings
from ..graph import run_research

try:  # rich 只是更好的展示，缺失时退回 print
    from rich.console import Console
    from rich.table import Table
except ImportError:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

__all__ = [
    "EvalSample",
    "load_dataset",
    "evaluate_sample",
    "run_eval",
    "DEFAULT_DATASET",
]

# 默认评测集：src/eval/run_eval.py -> 上两级是仓库根
DEFAULT_DATASET: Path = ROOT / "eval" / "datasets" / "sample.jsonl"

# 正文里的引用标记，与 guardrails/checks.py 的口径保持一致
_CITATION_RE = re.compile(r"\[(\d+)\]")

# 参与"求均值"的数值型指标
_NUMERIC_KEYS: tuple[str, ...] = (
    "report_chars",
    "notes_count",
    "tool_calls",
    "llm_calls",
    "elapsed_ms",
    "citation_count",
    "keyword_recall",
    "critique_rounds",
)

# 汇总表里展示的指标与展示名
_TABLE_ROWS: tuple[tuple[str, str], ...] = (
    ("samples", "样本数"),
    ("has_report_rate", "出稿率"),
    ("citation_valid_rate", "引用合规率"),
    ("keyword_recall", "关键词召回"),
    ("report_chars", "报告字数"),
    ("notes_count", "证据条数"),
    ("citation_count", "引用数"),
    ("tool_calls", "工具调用"),
    ("llm_calls", "LLM 调用"),
    ("critique_rounds", "Critic 轮数"),
    ("elapsed_ms", "耗时 (ms)"),
)


class EvalSample(BaseModel):
    """一条评测样本。

    刻意**不**使用 extra="forbid"：评测集由人手写，允许带备注等附加字段，
    附加字段不该让整行样本失效。
    """

    question: str = Field(description="要研究的开放式问题")
    must_include: list[str] = Field(
        default_factory=list,
        description="期望出现在报告或证据中的关键结构/关键词",
    )
    reference_objects: list[str] = Field(
        default_factory=list,
        description="可选：期望的最终产物形态描述，仅供人工复核",
    )


def load_dataset(path: Path | str) -> list[EvalSample]:
    """从 JSONL 读取评测集。文件不存在返回空列表，坏行跳过而不抛异常。"""
    dataset_path = Path(path)
    if not dataset_path.exists():
        return []

    samples: list[EvalSample] = []
    try:
        raw_lines = dataset_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            samples.append(EvalSample(**json.loads(line)))
        except (ValueError, TypeError, ValidationError):
            # 手写数据集难免有坏行：跳过比让整次评测崩掉更划算
            continue
    return samples


def _report_text(state: dict[str, Any], report_path: str = "") -> str:
    """取报告正文。状态里为空时退回落盘文件（Writer 已写入过即视为出稿）。"""
    report = str(state.get("report") or "")
    if report.strip():
        return report
    if report_path:
        try:
            return Path(report_path).read_text(encoding="utf-8")
        except OSError:
            return ""
    return ""


def _notes_claims(notes: Sequence[Any]) -> str:
    """把所有证据的 claim 拼成一段文本，供关键词命中判断。"""
    claims: list[str] = []
    for note in notes:
        if isinstance(note, dict):
            claims.append(str(note.get("claim") or ""))
        else:
            claims.append(str(getattr(note, "claim", "")))
    return "\n".join(claims)


def _keyword_recall(sample: EvalSample, report: str, notes: Sequence[Any]) -> float:
    """关键词召回：must_include 中出现在（报告 + 证据）里的比例。

    must_include 为空时返回 1.0 —— 没有期望就不应该拉低分数。
    """
    if not sample.must_include:
        return 1.0
    corpus = f"{report}\n{_notes_claims(notes)}"
    hits = sum(1 for keyword in sample.must_include if keyword and keyword in corpus)
    return round(hits / len(sample.must_include), 4)


def _mean(values: Iterable[float]) -> float:
    """均值。空列表返回 0.0，避免除零。"""
    items = list(values)
    if not items:
        return 0.0
    return round(sum(items) / len(items), 4)


def evaluate_sample(sample: EvalSample, cfg: Settings, user_id: str = "eval") -> dict[str, Any]:
    """跑一次完整流程并计算该样本的指标。"""
    metrics: dict[str, Any] = {
        "question": sample.question,
        "has_report": False,
        "report_chars": 0,
        "notes_count": 0,
        "tool_calls": 0,
        "llm_calls": 0,
        "elapsed_ms": 0,
        "citation_valid": False,
        "citation_count": 0,
        "keyword_recall": 0.0,
        "critique_rounds": 0,
        "error": "",
    }

    # 每条样本独立 thread_id：图默认带 MemorySaver，缺 thread_id 会直接报错，
    # 且独立线程号能保证样本之间不共享 checkpoint。
    try:
        result = run_research(
            query=sample.question,
            user_id=user_id,
            cfg=cfg,
            thread_id=f"eval-{uuid.uuid4().hex[:8]}",
        )
    except Exception as exc:  # 单条样本失败不影响其余样本
        metrics["error"] = f"{type(exc).__name__}: {exc}"
        return metrics

    state = result.get("state") or {}
    summary = result.get("summary") or {}
    budget = summary.get("budget") or {}

    report = _report_text(state, str(result.get("report_path") or ""))
    notes = state.get("notes") or []
    citation_issues = state.get("citation_issues") or []

    metrics["has_report"] = bool(report.strip())
    metrics["report_chars"] = len(report)
    metrics["notes_count"] = len(notes)
    metrics["tool_calls"] = int(budget.get("tool_calls", 0) or 0)
    metrics["llm_calls"] = int(budget.get("llm_calls", 0) or 0)
    metrics["elapsed_ms"] = int(summary.get("elapsed_ms", 0) or 0)
    metrics["citation_valid"] = not citation_issues
    metrics["citation_count"] = len({int(match) for match in _CITATION_RE.findall(report)})
    metrics["keyword_recall"] = _keyword_recall(sample, report, notes)
    metrics["critique_rounds"] = int(state.get("critique_round", 0) or 0)
    return metrics


def run_eval(
    dataset: Sequence[EvalSample] | Path | str,
    cfg: Settings,
    user_id: str = "eval",
) -> dict[str, Any]:
    """在数据集上跑评测并汇总。

    返回 ``{"aggregate": {...}, "samples": [每条样本的指标]}``。
    """
    samples: Sequence[EvalSample]
    if isinstance(dataset, (Path, str)):
        samples = load_dataset(Path(dataset))
    else:
        samples = dataset

    records = [evaluate_sample(sample, cfg, user_id=user_id) for sample in samples]

    aggregate: dict[str, Any] = {"samples": len(records)}
    for key in _NUMERIC_KEYS:
        aggregate[key] = _mean(float(record.get(key, 0) or 0) for record in records)
    aggregate["has_report_rate"] = _mean(1.0 if record["has_report"] else 0.0 for record in records)
    aggregate["citation_valid_rate"] = _mean(
        1.0 if record["citation_valid"] else 0.0 for record in records
    )
    return {"aggregate": aggregate, "samples": records}


def _print_plain(aggregate: dict[str, Any]) -> None:
    """无 rich 时的表格退化：一行一个指标。"""
    print("=" * 46)
    print("Argus 评测汇总")
    print("=" * 46)
    for key, label in _TABLE_ROWS:
        print(f"{label:<12}: {aggregate.get(key, 0)}")
    print("=" * 46)


def _print_rich(aggregate: dict[str, Any]) -> None:
    console = Console()
    table = Table(title="Argus 评测汇总", show_header=True, header_style="bold")
    table.add_column("指标")
    table.add_column("数值", justify="right")
    for key, label in _TABLE_ROWS:
        table.add_row(label, str(aggregate.get(key, 0)))
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argus-eval",
        description="Argus · 评测：在数据集上批量跑研究流程并汇总指标",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="评测集 JSONL 路径（默认 eval/datasets/sample.jsonl）",
    )
    parser.add_argument(
        "--provider",
        default="mock",
        help="LLM 提供方：mock / deepseek / qwen / zhipu / openai",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条样本")
    parser.add_argument("--output", default=None, help="把结果 JSON 写到该路径")
    parser.add_argument("--user", default="eval", help="用户标识，决定长期记忆归属")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cfg = load_settings(provider=args.provider).ensure_dirs()
    except ValueError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2

    samples = load_dataset(Path(args.dataset))
    if args.limit is not None and args.limit >= 0:
        samples = samples[: args.limit]
    if not samples:
        print(f"[评测中止] 评测集为空或全部无法解析：{args.dataset}", file=sys.stderr)
        return 1

    result = run_eval(samples, cfg, user_id=args.user)
    aggregate = result["aggregate"]

    if Console is not None and Table is not None:
        _print_rich(aggregate)
    else:
        _print_plain(aggregate)

    failures = [record for record in result["samples"] if record.get("error")]
    if failures:
        print(f"[告警] {len(failures)} 条样本执行失败：")
        for record in failures[:5]:
            print(f"  - {record['question'][:30]} -> {record['error']}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"结果已写入：{output_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
