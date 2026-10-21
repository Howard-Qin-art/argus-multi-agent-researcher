"""文本工具：统一分词。

放在独立模块的目的很简单：RAG 检索短暂存在。

设计要点：
- 英文/数字按连续串切词；
- 中文没有空格，改用**二元字组（bigram）**，这是中文检索里性价比最高的近似，
  不需要分词表，且对"智能体/代理"这类同义表述有一定容错；
- 只返回长度 > 1 的 token，避免单字带来的大量噪声命中。
"""

from __future__ import annotations

import re

__all__ = ["text_tokens"]

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def text_tokens(text: str) -> set[str]:
    """把一段文本转成用于匹配与打分的 token 集合。中英混排均可。"""
    if not text:
        return set()

    lowered = text.lower()
    tokens: set[str] = {match for match in _WORD_RE.findall(lowered) if len(match) > 1}

    for run in _CJK_RUN_RE.findall(lowered):
        if len(run) == 1:
            tokens.add(run)
            continue
        tokens.update(run[index : index + 2] for index in range(len(run) - 1))

    return tokens
