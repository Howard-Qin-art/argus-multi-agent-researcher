"""命令行入口。

用法：
    python -m src.cli "帮我梳理多智能体系统的落地难点" --provider mock
    python -m src.cli "同一问题" --provider deepseek --user alice --thread session-1

--provider mock 时不需要任何 API Key，用于快速验证流程与自查架构。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import load_settings
from .graph import run_research
from .llm import PROVIDER_PRESETS

try:  # rich 只是更好的展示，缺失不影响功能
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="argus",
        description="Argus · 多智能体协同深度研究系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="要研究的开放式问题")
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM 提供方：mock / deepseek / qwen / zhipu / openai",
    )
    parser.add_argument("--model", default=None, help="覆盖默认模型名")
    parser.add_argument("--base-url", default=None, help="覆盖 OpenAI 兼容端点")
    parser.add_argument("--temperature", type=float, default=None, help="采样温度")
    parser.add_argument("--user", default="default", help="用户标识，决定长期记忆的归属")
    parser.add_argument("--thread", default=None, help="会话线程 id，用于断点续跑")
    parser.add_argument(
        "--rounds", type=int, default=None, help="Critic 最大返工轮数"
    )
    parser.add_argument("--no-trace", action="store_true", help="关闭本地 JSONL trace")
    parser.add_argument(
        "--json", dest="as_json", action="store_true", help="以 JSON 输出运行摘要"
    )
    parser.add_argument(
        "--list-providers", action="store_true", help="打印可用 provider 后退出"
    )
    return parser


def _print_plain(result: dict[str, Any]) -> None:
    state = result["state"]
    summary = result["summary"]
    print("=" * 70)
    print(state.get("report", "").strip() or "（空报告）")
    print("=" * 70)
    print(f"trace_id     : {summary.get('trace_id')}")
    print(f"耗时(ms)     : {summary.get('elapsed_ms')}")
    print(f"LLM 调用     : {summary.get('budget', {}).get('llm_calls')}")
    print(f"工具调用     : {summary.get('budget', {}).get('tool_calls')}")
    print(f"证据条数     : {len(state.get('notes') or [])}")
    print(f"Critic 轮数  : {state.get('critique_round')}")
    issues = state.get("citation_issues") or []
    print(f"引用校验     : {'合规' if not issues else issues}")
    print(f"报告文件     : {result['report_path']}")


def _print_rich(result: dict[str, Any]) -> None:
    state = result["state"]
    summary = result["summary"]
    console = Console()

    console.print(
        Panel(
            (state.get("report") or "（空报告）").strip(),
            title=f"[bold]Argus 研究报告[/bold] · trace={summary.get('trace_id')}",
            border_style="blue",
        )
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("指标")
    table.add_column("数值", justify="right")
    table.add_row("耗时 (ms)", str(summary.get("elapsed_ms")))
    table.add_row("LLM 调用", str(summary.get("budget", {}).get("llm_calls")))
    table.add_row("工具调用", str(summary.get("budget", {}).get("tool_calls")))
    table.add_row("证据条数", str(len(state.get("notes") or [])))
    table.add_row("Critic 轮数", str(state.get("critique_round")))
    refs = state.get("references") or []
    table.add_row("参考来源", str(len(refs)))
    issues = state.get("citation_issues") or []
    table.add_row("引用校验", "合规" if not issues else f"{len(issues)} 处异常")
    console.print(table)

    errors = state.get("errors") or []
    if errors:
        console.print(f"[yellow]工具告警 {len(errors)} 条：[/yellow]")
        for error in errors[:5]:
            console.print(f"  - {error}")
    if issues:
        console.print("[yellow]引用问题：[/yellow]")
        for issue in issues[:5]:
            console.print(f"  - {issue}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_providers:
        print("可用 LLM provider：")
        print("  mock      本地确定性替身，无需 API Key（默认）")
        for name, preset in PROVIDER_PRESETS.items():
            print(f"  {name:<10} {preset['base_url']}  默认模型 {preset['model']}")
        return 0

    overrides: dict[str, Any] = {}
    if args.provider:
        overrides["provider"] = args.provider
    if args.model:
        overrides["model"] = args.model
    if args.base_url:
        overrides["base_url"] = args.base_url
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.rounds is not None:
        overrides["max_critic_rounds"] = args.rounds
    if args.no_trace:
        overrides["enable_trace"] = False

    try:
        cfg = load_settings(**overrides).ensure_dirs()
        result = run_research(
            query=args.query,
            user_id=args.user,
            cfg=cfg,
            thread_id=args.thread,
        )
    except ValueError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\n已中断", file=sys.stderr)
        return 130

    if args.as_json:
        payload = {
            "summary": result["summary"],
            "report": result["state"].get("report", ""),
            "references": result["state"].get("references", []),
            "citation_issues": result["state"].get("citation_issues", []),
            "report_path": result["report_path"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif Console is not None:
        _print_rich(result)
    else:
        _print_plain(result)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
