"""LLM 接入层：统一抽象 + 可插拔 provider + 本地确定性替身（mock）。

为什么这么设计：
1. 国产模型（DeepSeek / 通义千问 / 智谱）与 OpenAI 都提供兼容的 Chat Completions 协议，
   因此用同一套 ChatOpenAI 客户端 + 不同 base_url 即可切换，避免为每个厂商写一套调用。
2. mock provider 让整张图在**没有 API Key** 的情况下也能端到端跑通，
   这既是 CI 里的确定性单测依赖，也让面试官可以 clone 后直接 `pytest` 看到流程。
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import Settings, load_settings

__all__ = ["PROVIDER_PRESETS", "MockChatModel", "get_llm", "extract_json"]


# 各 provider 的官方 OpenAI 兼容端点与推荐默认模型。
# base_url / model 留空时取这里的值，也可由环境变量覆盖。
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "qwen": {
        # 阿里云百炼的兼容模式端点
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-plus",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
}

_ROLE_MARKER = re.compile(r"\[role=(\w+)\]", re.IGNORECASE)

# mock 模式下各角色的占位输出。刻意保持结构完整（尤其是键名），
# 这样下游解析器、护栏与评测逻辑在 mock 下也能被真实地覆盖到。
_MOCK_REPLIES: dict[str, str] = {
    "supervisor": json.dumps(
        {
            "plan": [
                "明确研究对象与边界定义",
                "梳理主流技术路线与代表方案",
                "收集可量化的指标或公开数据",
                "归纳主要风险与尚未解决的问题",
            ]
        },
        ensure_ascii=False,
    ),
    # 检索词刻意贴合内置知识库的主题，这样 mock 模式下也能真实演练
    # "模型决策 → 工具调用 → 命中本地证据 → 归纳成 claim" 这条链路。
    "router": json.dumps(
        {
            "tools": ["retriever"],
            "queries": {
                "retriever": "多智能体 角色分工 落地难点",
                "web_search": "最新进展",
            },
        },
        ensure_ascii=False,
    ),
    "researcher": json.dumps(
        {
            "notes": [
                {
                    "claim": "该领域近两年关注度显著提升，公开资料数量增长明显。",
                    "source": "local-kb",
                    "url": "",
                    "confidence": 0.6,
                },
                {
                    "claim": "主流路线可分为端到端与分阶段两类，取舍在于可控性与效果的平衡。",
                    "source": "local-kb",
                    "url": "",
                    "confidence": 0.6,
                },
            ]
        },
        ensure_ascii=False,
    ),
    "critic": json.dumps(
        {"verdict": "pass", "issues": [], "gaps": []},
        ensure_ascii=False,
    ),
    "writer": json.dumps(
        {
            "report": (
                "# 研究报告（mock 模式占位输出）\n\n"
                "> 当前运行在 mock provider 下，以下内容为占位结果；"
                "配置真实模型后可得到实际分析。\n\n"
                "## 结论摘要\n\n1. 该问题存在多条技术路线，取舍集中在可控性与效果之间。\n"
                "2. 公开可比数据仍然稀缺，横向评估报告结论差异较大。\n\n"
                "## 风险与待验证\n\n"
                "1. 评测口径不统一导致结论难以直接比较 [1]。\n"
            ),
            "references": [
                {
                    "id": 1,
                    "title": "示例参考来源（请替换为真实检索结果）",
                    "url": "https://example.com",
                    "source": "local-kb",
                }
            ],
        },
        ensure_ascii=False,
    ),
}


class MockChatModel(BaseChatModel):
    """无外部依赖的确定性替身模型。

    依据提示中的 ``[role=xxx]`` 标记返回对应角色的结构化占位响应，
    使得规划 / 取证 / 审查 / 成文四个环节以及图上的所有条件边都能被真实执行。
    """

    @property
    def _llm_type(self) -> str:  # LangChain 约定属性
        return "argus-mock-chat-model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"provider": "mock"}

    def bind(self, **kwargs: Any) -> "MockChatModel":  # 兼容偶尔的 bind 调用
        return self

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        role = "supervisor"
        for message in reversed(list(messages)):
            found = _ROLE_MARKER.search(str(getattr(message, "content", "")))
            if found:
                role = found.group(1).lower()
                break
        content = _MOCK_REPLIES.get(role, _MOCK_REPLIES["writer"])
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )


def get_llm(cfg: Settings | None = None) -> BaseChatModel:
    """按配置返回 LLM 客户端。provider=mock 时返回本地替身，不发任何网络请求。"""
    cfg = cfg or load_settings()

    if cfg.provider == "mock":
        return MockChatModel()

    if cfg.provider not in PROVIDER_PRESETS:
        raise ValueError(
            f"未知的 LLM provider: {cfg.provider!r}，可选："
            f"{sorted(list(PROVIDER_PRESETS) + ['mock'])}"
        )

    # 延迟导入：mock 模式下不必依赖 langchain_openai 是否已安装
    from langchain_openai import ChatOpenAI

    preset = PROVIDER_PRESETS[cfg.provider]
    api_key = cfg.api_key or ""
    if not api_key:
        raise ValueError(
            f"provider={cfg.provider!r} 需要 API Key，请设置环境变量 ARGUS_LLM_API_KEY"
        )

    return ChatOpenAI(
        api_key=api_key,
        base_url=cfg.base_url or preset["base_url"],
        model=cfg.model or preset["model"],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.request_timeout,
    )


def extract_json(text: str) -> dict | list | None:
    """从模型输出中稳健地取出 JSON。

    真实模型常把 JSON 包在 ```json 代码块里，或直接多说两句话，
    因此逐层尝试：直接解析 → 去代码块 → 截取首个 {...} / [...] 片段。
    """
    if text is None:
        return None

    cleaned = text.strip()

    candidates = [cleaned]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    brace = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if brace:
        candidates.append(brace.group(1).strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    return None
