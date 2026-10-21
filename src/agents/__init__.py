"""四个协作角色。

角色分工（刻意保持职责单一，便于单独替换与单测）：
    supervisor  规划：问题 → 可取证的子问题
    researcher  取证：工具路由 + 证据归纳
    critic      审查：证据是否充分，决定是否返工
    writer      成文：带引用的结构化报告 + 写入长期记忆
"""

from .critic import build_critic
from .researcher import build_researcher
from .runtime import AgentRuntime
from .supervisor import build_supervisor
from .writer import build_writer

__all__ = [
    "AgentRuntime",
    "build_supervisor",
    "build_researcher",
    "build_critic",
    "build_writer",
]
