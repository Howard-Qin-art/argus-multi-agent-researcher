"""Streamlit 可视化界面。

启动方式：
    streamlit run src/serving/app.py

设计要点：
1. streamlit 属于**可选依赖**（requirements.txt 的"服务化"段），只在 main() 里
   延迟导入，未安装时不影响 mock 模式下的 CLI 与单元测试。
2. provider 选 mock 时全程不发网络请求，可以离线跑通整条流水线，
   用于演示交互与排版；切换真实 provider 前需在环境变量里配好 API Key。
3. 侧边栏读的是长期记忆仓库的统计，user_id 变化时统计会跟着变，
   便于直观看到"不同用户拥有彼此隔离的记忆"。
"""

from __future__ import annotations

from typing import Any

try:  # 正常以包内模块被导入（如测试 import src.serving.app）
    from ..config import load_settings
    from ..graph import run_research
    from ..llm import PROVIDER_PRESETS
    from ..memory import LongTermMemory
except ImportError:  # pragma: no cover
    # `streamlit run src/serving/app.py` 会把脚本当作顶层模块执行，
    # 此时没有包上下文，相对导入不可用，退回包内绝对路径。
    from src.config import load_settings
    from src.graph import run_research
    from src.llm import PROVIDER_PRESETS
    from src.memory import LongTermMemory

__all__ = ["main", "PROVIDER_CHOICES"]


# mock 永远排在第一位，保证默认就是可离线跑通的那一项
PROVIDER_CHOICES: list[str] = ["mock"] + sorted(PROVIDER_PRESETS)


def _render_metrics(container: Any, summary: dict[str, Any], state: dict[str, Any]) -> None:
    """运行指标区：耗时 / LLM 调用 / 工具调用 / 证据数 / Critic 轮数。"""
    budget = summary.get("budget") or {}
    columns = container.columns(5)
    columns[0].metric("耗时 (ms)", summary.get("elapsed_ms", 0))
    columns[1].metric("LLM 调用", budget.get("llm_calls", 0))
    columns[2].metric("工具调用", budget.get("tool_calls", 0))
    columns[3].metric("证据数", len(state.get("notes") or []))
    columns[4].metric("Critic 轮数", state.get("critique_round", 0))


def _render_notes(container: Any, state: dict[str, Any]) -> None:
    """发掘的证据区：每条展示 claim + source + confidence。"""
    notes = state.get("notes") or []
    if not notes:
        container.info("本轮没有取到证据。")
        return
    for index, note in enumerate(notes, start=1):
        if not isinstance(note, dict):
            continue
        container.markdown(
            f"**{index}. {note.get('claim', '')}**\n\n"
            f"- 来源 source：`{note.get('source', '') or '未标注'}`\n"
            f"- 置信度 confidence：`{note.get('confidence', 0)}`"
        )
        url = note.get("url") or ""
        if url:
            container.markdown(f"- 链接 url：{url}")


def _render_references(container: Any, state: dict[str, Any]) -> None:
    """参考来源区：id 与正文里的 [id] 标注一一对应。"""
    references = state.get("references") or []
    if not references:
        container.info("报告未引用外部来源。")
        return
    for reference in references:
        if not isinstance(reference, dict):
            continue
        title = reference.get("title") or "（无标题）"
        url = reference.get("url") or ""
        source = reference.get("source") or ""
        line = f"- [{reference.get('id', '-')}] {title}"
        if url:
            line += f" — {url}"
        if source:
            line += f"（来源：{source}）"
        container.markdown(line)


def _render_citations(container: Any, state: dict[str, Any]) -> None:
    """引用校验区：合规给出通过提示，否则逐条列出问题。"""
    issues = state.get("citation_issues") or []
    if not issues:
        container.success("引用校验合规：正文标注与参考列表一一对应。")
        return
    container.warning(f"引用校验发现 {len(issues)} 处问题：")
    for issue in issues:
        container.markdown(f"- {issue}")


def main() -> None:
    """界面入口：侧边栏放配置，主区放输入与结果。"""
    # 延迟导入：streamlit 是可选依赖
    import streamlit as st

    st.set_page_config(page_title="Argus · 深度研究", layout="wide")
    st.title("Argus · 多智能体协同深度研究系统")
    st.caption("Supervisor 规划 → Researcher 取证 → Critic 审查 → Writer 成文")

    # ---------------- 侧边栏 ----------------
    st.sidebar.header("运行配置")
    provider = st.sidebar.selectbox(
        "LLM provider",
        options=PROVIDER_CHOICES,
        index=0,
        help="mock 为本地确定性替身，无需 API Key，也不发网络请求",
    )
    user_id = st.sidebar.text_input(
        "用户标识 user_id", value="default", help="决定长期记忆的归属，不同用户互相隔离"
    )
    thread_id = st.sidebar.text_input(
        "会话线程 thread_id", value="", help="留空则每次运行自动生成，填入可断点续跑"
    )
    max_critic_rounds = st.sidebar.slider(
        "Critic 最大返工轮数", min_value=0, max_value=3, value=2
    )

    st.sidebar.subheader("系统 memory 统计")
    cfg = load_settings().ensure_dirs()
    memory_stats = LongTermMemory(user_id or "default", cfg).stats()
    st.sidebar.metric("记忆总数 total", memory_stats.get("total", 0))
    st.sidebar.metric("事实 facts", memory_stats.get("facts", 0))
    st.sidebar.metric("偏好 preferences", memory_stats.get("preferences", 0))
    st.sidebar.json(memory_stats)

    # ---------------- 主区 ----------------
    query = st.text_area(
        "要研究的问题",
        value="",
        height=100,
        placeholder="例如：帮我梳理多智能体系统落地的主要难点",
    )
    start = st.button("开始研究", type="primary")

    if not start:
        return

    if not query or not query.strip():
        st.warning("请先输入要研究的问题。")
        return

    overrides: dict[str, Any] = {
        "provider": provider,
        "max_critic_rounds": max_critic_rounds,
    }
    try:
        run_cfg = load_settings(**overrides).ensure_dirs()
        with st.spinner("多智能体协同研究中（Supervisor → Researcher → Critic → Writer），请稍候..."):
            result = run_research(
                query=query.strip(),
                user_id=user_id or "default",
                cfg=run_cfg,
                thread_id=thread_id.strip() or None,
            )
    except ValueError as exc:
        # 配置类错误：未知 provider、缺少 API Key 等
        st.error(f"配置错误：{exc}")
        return
    except Exception as exc:
        st.error(f"研究流程执行失败：{type(exc).__name__}: {exc}")
        return

    state = result.get("state") or {}
    summary = result.get("summary") or {}

    st.caption(
        f"trace_id：`{summary.get('trace_id', '')}`　报告文件：`{result.get('report_path', '')}`"
    )

    with st.expander("① 研究报告", expanded=True):
        st.markdown(state.get("report") or "（本次未生成报告正文）")

    with st.expander("② 参考来源", expanded=True):
        _render_references(st, state)

    with st.expander("③ 运行指标", expanded=True):
        _render_metrics(st, summary, state)

    with st.expander("④ 发掘的证据", expanded=False):
        _render_notes(st, state)

    with st.expander("⑤ 引用校验结果", expanded=False):
        _render_citations(st, state)

    errors = state.get("errors") or []
    if errors:
        with st.expander("工具告警", expanded=False):
            for error in errors[:10]:
                st.markdown(f"- {error}")


if __name__ == "__main__":  # pragma: no cover
    main()
