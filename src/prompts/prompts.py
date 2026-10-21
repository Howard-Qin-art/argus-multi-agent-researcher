"""各角色的提示词。集中管理，便于迭代与做提示词层面的 A/B。

约定：每条 system 消息都以 ``[role=xxx]`` 开头。
- 真实模型：这个标记无害，还能帮助后处理判断消息来源；
- mock 模型：依赖它决定返回哪种占位结构化输出。
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

__all__ = [
    "supervisor_messages",
    "router_messages",
    "researcher_messages",
    "critic_messages",
    "writer_messages",
]

_JSON_RULE = (
    "严格只输出 JSON，不要输出解释文字、不要使用 Markdown 代码块。"
    "字段名必须使用英文且与给定示例完全一致。"
)


def supervisor_messages(
    query: str,
    n_items: int = 4,
    previous_plan: list[str] | None = None,
    gaps: list[str] | None = None,
    memory_context: str = "",
) -> list[BaseMessage]:
    """Supervisor：把开放式问题拆成互不重叠、可独立取证的子问题。"""
    revision = ""
    if gaps:
        revision = (
            "\n\n上一轮审查发现以下证据缺口，请在修订计划时**优先覆盖**它们，"
            "并保留已经获得充分证据的子问题：\n"
            + "\n".join(f"- {gap}" for gap in gaps)
        )
    if previous_plan:
        revision += "\n\n上一轮计划为：\n" + "\n".join(
            f"- {item}" for item in previous_plan
        )

    system = SystemMessage(
        content=(
            f"[role=supervisor]\n你是 Argus 研究系统的规划器。"
            f"你的职责是把一个开放式问题拆解为 {n_items} 个以内、"
            "彼此尽量不重叠、且每个都能独立取证的子问题。\n"
            "要求：\n"
            "1. 子问题要具体，避免'介绍一下X'这类无法验证的问法；\n"
            "2. 覆盖'定义边界 / 主流方案 / 可量化数据 / 风险与争议'中的若干维度；\n"
            "3. 不要包含需要主观臆断、无法从公开资料取证的子问题。\n\n"
            f"输出 JSON：{{\"plan\": [\"子问题1\", \"子问题2\", ...]}}\n{_JSON_RULE}"
        )
    )
    human = HumanMessage(
        content=(
            f"研究问题：{query}\n"
            f"{'用户历史偏好与已知结论：' + memory_context if memory_context else ''}"
            f"{revision}"
        )
    )
    return [system, human]


def router_messages(subquestion: str, available_tools: list[str]) -> list[BaseMessage]:
    """Router（隶属 Researcher）：决定用哪些工具、以什么关键词去取证。"""
    descriptions = {
        "retriever": "本地知识库检索，适合已有沉淀文档、术语定义、内部资料",
        "web_search": "联网搜索，适合时效性强、外部公开信息",
        "calculator": "数值计算，适合同比/环比/占比等需要算的场景",
    }
    tool_lines = "\n".join(
        f"- {name}: {descriptions.get(name, '')}" for name in available_tools
    )
    system = SystemMessage(
        content=(
            "[role=router]\n你是工具路由决策器。"
            "给定一个子问题，判断应该调用哪些工具，并给出每个工具的检索关键词。\n"
            "原则：\n"
            "1. 能用本地知识库解决的优先 retriever，成本低且可控；\n"
            "2. 只有需要时效或外部资料时才用 web_search；\n"
            "3. 涉及数值推导才用 calculator；\n"
            "4. 关键词要短、命中率高，避免整句。\n\n"
            '输出 JSON：{"tools": ["retriever", ...], '
            '"queries": {"retriever": "关键词", "web_search": "关键词"}}\n'
            f"{_JSON_RULE}"
        )
    )
    human = HumanMessage(
        content=f"子问题：{subquestion}\n\n可用工具：\n{tool_lines}"
    )
    return [system, human]


def researcher_messages(subquestion: str, evidence: str) -> list[BaseMessage]:
    """Researcher：只依据检索到的证据产出不少于一条、带出处的陈述。"""
    system = SystemMessage(
        content=(
            "[role=researcher]\n你是取证研究员。"
            "你只能依据下方【检索证据】陈述事实，不得凭先验知识补充未被证据支持的内容。"
            "每条陈述必须：\n"
            "1. 是可被第三方核对的事实，而非评价；\n"
            "2. 标注来源 source（local-kb / web）与 url（无则留空）；\n"
            "3. 给出 0~1 的 confidence：证据直接支持取 0.8 以上，"
            "只有间接线索则降到 0.4 以下。\n\n"
            "若证据为空或与子问题无关，请返回空的 notes 数组，绝不允许编造。\n\n"
            '输出 JSON：{"notes": [{"claim": "事实陈述", "source": "", '
            '"url": "", "confidence": 0.7}]}\n'
            f"{_JSON_RULE}"
        )
    )
    human = HumanMessage(
        content=(
            f"子问题：{subquestion}\n\n【检索证据】\n"
            f"{evidence if evidence.strip() else '（无）'}"
        )
    )
    return [system, human]


def critic_messages(
    query: str, plan: list[str], notes: list[str], round_idx: int, max_rounds: int
) -> list[BaseMessage]:
    """Critic：独立审查证据是否足以支撑结论，决定是否返工。"""
    notes_text = "\n".join(f"{i}. {note}" for i, note in enumerate(notes, 1))
    plan_text = "\n".join(f"- {item}" for item in plan)
    system = SystemMessage(
        content=(
            "[role=critic]\n你是独立的审查者，默认怀疑。"
            "你要判断现有证据是否足以回答原始问题。\n"
            "判定规则：\n"
            "1. 只要存在未被覆盖的子问题，或关键结论没有来源支撑，就判 revise；\n"
            "2. issues 列出具体缺陷（如'第2条结论缺少来源'）；\n"
            "3. gaps 给出下一轮必须补到的信息，**可直接被执行**；\n"
            f"4. 当前是第 {round_idx}/{max_rounds} 轮，接近上限时只允许保留最关键的缺口，"
            "避免在最后一轮仍要求返工导致资源浪费。\n\n"
            '输出 JSON：{"verdict": "pass 或 revise", "issues": [...], '
            '"gaps": [...]}\n'
            f"{_JSON_RULE}"
        )
    )
    human = HumanMessage(
        content=(
            f"原始问题：{query}\n\n【研究计划】\n{plan_text}\n\n"
            f"【已获得的证据】\n{notes_text or '（尚无）'}"
        )
    )
    return [system, human]


def writer_messages(
    query: str, notes: list[str], references: list[str], memory_context: str = ""
) -> list[BaseMessage]:
    """Writer：把证据组织成带引用标记的结构化报告。"""
    notes_text = "\n".join(f"{i}. {note}" for i, note in enumerate(notes, 1))
    refs_text = "\n".join(references) if references else "（无）"
    system = SystemMessage(
        content=(
            "[role=writer]\n你是研究报告撰写者。\n"
            "硬性要求：\n"
            "1. 报告中的每一条关键结论都必须用 [n] 标注引用，n 必须出现在参考来源的 id 中；\n"
            "2. 不得出现参考来源中不存在的引用编号；\n"
            "3. 结构包含：结论摘要（编号列表）、分点论述、风险与待验证；\n"
            "4. 不确定的内容要明确写成'待验证'，不得写成确定结论。\n\n"
            '输出 JSON：{"report": "Markdown 正文", "references": '
            '[{"id": 1, "title": "", "url": "", "source": ""}]}\n'
            f"{_JSON_RULE}"
        )
    )
    human = HumanMessage(
        content=(
            f"研究问题：{query}\n\n【证据】\n{notes_text or '（无）'}\n\n"
            f"【可用参考来源】\n{refs_text}\n"
            f"{'【用户偏好】' + memory_context if memory_context else ''}"
        )
    )
    return [system, human]
