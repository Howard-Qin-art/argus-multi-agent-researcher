"""长期记忆：跨会话持久化。

短期记忆只在一次会话里有效，进程结束即消失（除非依赖 checkpoint）。
长期记忆解决的是"同一个用户隔几天再来问，系统是否记得他的偏好和上次结论"。

存储策略：每个用户一个 JSON 文件，写入采用 **先落临时文件再原子替换**，
避免进程被中断时写坏历史数据。
持久化两类信息：
- preferences：用户偏好（写作风格、关注维度等）
- facts：经过取证且被 Critic 放行的事实，可跨会话复用，减少重复检索
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, load_settings
from ..utils import text_tokens

__all__ = ["LongTermMemory", "MemoryItem"]


@dataclass
class MemoryItem:
    """一条长期记忆。"""

    content: str
    kind: str = "fact"  # fact | preference
    source: str = ""
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "kind": self.kind,
            "source": self.source,
            "created_at": self.created_at,
            "hit_count": self.hit_count,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MemoryItem":
        return cls(
            content=str(payload.get("content", "")),
            kind=str(payload.get("kind", "fact")),
            source=str(payload.get("source", "")),
            created_at=float(payload.get("created_at", time.time())),
            hit_count=int(payload.get("hit_count", 0)),
        )


def _keywords(text: str) -> set[str]:
    """关键词集合。与检索侧共用同一套分词，保证记忆召回与 RAG 口径一致。

    过去这里有一段自己手写的分词逻辑，遇到中文就 break，导致英文部分被整体跳过；
    现在统一收敛到 utils.text_tokens，消除了这处隐患。
    """
    return text_tokens(text)


class LongTermMemory:
    """按用户维度持久化的记忆仓库。"""

    def __init__(self, user_id: str = "default", cfg: Settings | None = None) -> None:
        self.cfg = cfg or load_settings()
        self.user_id = self._safe_id(user_id)
        self.path = self.cfg.memory_dir / f"{self.user_id}.json"
        self._items: list[MemoryItem] | None = None

    # ---------- 基础 ----------
    @staticmethod
    def _safe_id(user_id: str) -> str:
        """防止路径穿越，把用户 id 收敛成安全文件名。"""
        safe = "".join(ch for ch in user_id if ch.isalnum() or ch in "-_")
        return safe or "default"

    @property
    def items(self) -> list[MemoryItem]:
        if self._items is None:
            self._items = self._load()
        return self._items

    def _load(self) -> list[MemoryItem]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # 文件损坏时不应让整条流水线失败，退化为空记忆并记录一次
            return []
        raw_items = payload.get("items", []) if isinstance(payload, dict) else []
        return [MemoryItem.from_dict(item) for item in raw_items if isinstance(item, dict)]

    def save(self) -> None:
        """原子写入。"""
        self.cfg.memory_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "user_id": self.user_id,
            "updated_at": time.time(),
            "items": [item.to_dict() for item in self.items],
        }
        directory = str(self.path.parent)
        fd, temp_path = tempfile.mkstemp(prefix=".mem-", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.path)
        finally:
            if os.path.exists(temp_path):  # pragma: no cover - 竞态保护
                os.unlink(temp_path)

    # ---------- 写入 ----------
    def add(self, content: str, kind: str = "fact", source: str = "") -> None:
        """新增一条记忆；重复内容只会刷新命中，不产生堆积。"""
        content = content.strip()
        if not content:
            return
        for existing in self.items:
            if existing.content == content:
                existing.source = existing.source or source
                return
        self.items.append(MemoryItem(content=content, kind=kind, source=source))
        self._trim()

    def remember_facts(self, claims: list[str], source: str = "") -> None:
        for claim in claims:
            self.add(claim, kind="fact", source=source)

    def remember_preference(self, content: str) -> None:
        self.add(content, kind="preference")

    def _trim(self) -> None:
        """超出容量时淘汰：优先保留偏好，其次按（命中次数, 时间）淘汰事实。"""
        limit = self.cfg.memory_max_items
        if len(self.items) <= limit:
            return
        preferences = [item for item in self.items if item.kind == "preference"]
        facts = [item for item in self.items if item.kind != "preference"]
        facts.sort(key=lambda item: (item.hit_count, item.created_at), reverse=True)
        kept = preferences + facts
        self._items = kept[:limit]

    # ---------- 回忆 ----------
    def recall(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """按关键词重叠度召回相关记忆，并累加命中次数。"""
        query_tokens = _keywords(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, MemoryItem]] = []
        for item in self.items:
            item_tokens = _keywords(item.content)
            overlap = len(query_tokens & item_tokens)
            if overlap == 0:
                continue
            score = overlap / max(1, len(query_tokens))
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        recalled = [item for _, item in scored[:top_k]]
        for item in recalled:
            item.hit_count += 1
        return recalled

    def as_context(self, query: str, top_k: int = 5) -> str:
        """渲染成可直接注入提示词的记忆上下文。"""
        recalled = self.recall(query, top_k)
        if not recalled:
            return ""
        return "\n".join(
            f"- ({item.kind}) {item.content}" for item in recalled
        )

    def stats(self) -> dict[str, int]:
        return {
            "total": len(self.items),
            "facts": sum(1 for item in self.items if item.kind == "fact"),
            "preferences": sum(1 for item in self.items if item.kind == "preference"),
        }
