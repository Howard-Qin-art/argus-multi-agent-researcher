"""提示词模块入口。"""

from .prompts import (
    critic_messages,
    researcher_messages,
    router_messages,
    supervisor_messages,
    writer_messages,
)

__all__ = [
    "supervisor_messages",
    "router_messages",
    "researcher_messages",
    "critic_messages",
    "writer_messages",
]
