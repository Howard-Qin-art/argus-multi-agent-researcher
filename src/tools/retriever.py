"""本地知识库检索（轻量 RAG）。

为什么不直接用向量库：
- 默认要把项目"clone 即用"，零额外依赖、零 embedding 服务是关键约束；
- 关键词 + 中文二元切分（bigram）在小规模知识库上召回质量足够，
  且结果可解释、可复现，便于测试断言。

需要更高质量召回时，只需替换本文件的 scoring 实现为向量相似度，
上层调用方式不变（面向 `retrieve() -> ToolResult` 编程）。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from ..schemas import ToolResult
from ..utils import text_tokens

__all__ = ["LocalRetriever", "RetrievedChunk"]

_TEXT_SUFFIX = {".md", ".txt", ".markdown"}


@dataclass(frozen=True)
class RetrievedChunk:
    """一个检索命中的文本块。"""

    path: str
    title: str
    content: str
    score: float

    def as_context(self, index: int) -> str:
        return (
            f"[{index}] {self.title}（来源:{Path(self.path).name}）\n"
            f"{self.content}\n"
        )


def _tokenize(text: str) -> list[str]:
    """统一分词：英文按词、中文按二元字组。具体实现见 src/utils/text.py。"""
    return list(text_tokens(text))


class LocalRetriever:
    """扫描目录下的纯文本文件，按段落切块后做关键词打分检索。"""

    name = "retriever"

    def __init__(self, kb_dir: Path, top_k: int = 4, chunk_size: int = 500) -> None:
        self.kb_dir = Path(kb_dir)
        self.top_k = top_k
        self.chunk_size = chunk_size
        self._chunks: list[RetrievedChunk] | None = None

    # ---------- 索引 ----------
    def index(self) -> list[RetrievedChunk]:
        """加载并缓存所有文本块。目录为空时返回空列表，不抛异常。"""
        if self._chunks is not None:
            return self._chunks

        chunks: list[RetrievedChunk] = []
        if self.kb_dir.exists():
            for path in sorted(self.kb_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in _TEXT_SUFFIX:
                    chunks.extend(self._split_file(path))
        self._chunks = chunks
        return chunks

    def _split_file(self, path: Path) -> list[RetrievedChunk]:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []

        title = path.stem
        chunks: list[RetrievedChunk] = []
        for block in re.split(r"\n\s*\n", raw):
            block = block.strip()
            if not block:
                continue
            for start in range(0, len(block), self.chunk_size):
                piece = block[start : start + self.chunk_size].strip()
                if piece:
                    chunks.append(
                        RetrievedChunk(
                            path=str(path), title=title, content=piece, score=0.0
                        )
                    )
        # 标题命中作为弱信号，帮助召回"文章级"相关而非"段落级"
        chunks.append(
            RetrievedChunk(path=str(path), title=title, content=title, score=0.0)
        )
        return chunks

    # ---------- 检索 ----------
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """返回按相关度降序的文本块，最多 top_k 个。"""
        chunks = self.index()
        if not chunks:
            return []
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        scored: list[RetrievedChunk] = []
        for chunk in chunks:
            content_tokens = set(_tokenize(chunk.content))
            overlap = len(query_tokens & content_tokens)
            if overlap == 0:
                continue
            # 归一化：除以查询词数，避免长查询天然占优
            score = overlap / max(1, len(query_tokens))
            # 轻微偏好长文本块，避免总召回到标题这种过短片段
            score *= min(1.5, 1 + len(chunk.content) / 2000)
            scored.append(
                RetrievedChunk(
                    path=chunk.path,
                    title=chunk.title,
                    content=chunk.content,
                    score=round(score, 4),
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: (top_k or self.top_k)]

    def __call__(self, query: str, **kwargs) -> ToolResult:
        """以工具协议暴露，便于注册表统一调度。"""
        started = time.perf_counter()
        hits = self.retrieve(query, kwargs.get("top_k"))
        output = "\n".join(chunk.as_context(i) for i, chunk in enumerate(hits, 1))
        return ToolResult(
            tool=self.name,
            ok=bool(hits),
            output=output or "本地知识库未命中",
            error="" if hits else "no-hit",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
