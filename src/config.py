"""全局配置。

所有可调项都通过环境变量覆盖（前缀 ARGUS_），避免把密钥或本机路径写进仓库。
配置在导入时快照一次，运行期只读；如需在测试中临时替换，用 load_settings(**override)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 仓库根目录：src/config.py -> 上两级
ROOT = Path(__file__).resolve().parents[1]

ENV_PREFIX = "ARGUS_"


def _str(name: str, default: str = "") -> str:
    value = os.getenv(ENV_PREFIX + name)
    return default if value is None or value.strip() == "" else value.strip()


def _int(name: str, default: int) -> int:
    raw = os.getenv(ENV_PREFIX + name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    raw = os.getenv(ENV_PREFIX + name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(ENV_PREFIX + name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    raw = os.getenv(ENV_PREFIX + name)
    if raw is None or raw.strip() == "":
        return default
    candidate = Path(raw)
    # 相对路径一律相对仓库根，避免受当前工作目录影响
    return candidate if candidate.is_absolute() else (ROOT / candidate)


@dataclass(frozen=True)
class Settings:
    """运行期不可变配置快照。"""

    # ---- LLM ----
    provider: str = "mock"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048
    request_timeout: int = 60

    # ---- 编排预算（防止无限循环 / 成本失控）----
    max_plan_items: int = 4
    max_critic_rounds: int = 2
    max_tool_calls: int = 3

    # ---- 工具 ----
    enable_web_search: bool = True
    enable_rag: bool = True
    search_results: int = 5
    retrieve_top_k: int = 4
    kb_dir: Path = field(default_factory=lambda: ROOT / "data" / "sample_kb")

    # ---- 记忆 ----
    memory_dir: Path = field(default_factory=lambda: ROOT / ".memory")
    memory_summary_threshold: int = 4000
    memory_max_items: int = 200

    # ---- 可观测 ----
    trace_dir: Path = field(default_factory=lambda: ROOT / "traces")
    enable_trace: bool = True
    trace_verbose: bool = False
    langsmith: bool = False
    langsmith_project: str = "argus-research"

    # ---- 输出 ----
    report_dir: Path = field(default_factory=lambda: ROOT / "reports")

    def ensure_dirs(self) -> "Settings":
        """确保运行时需要的目录存在。幂等，可重复调用。"""
        for directory in (
            self.kb_dir,
            self.memory_dir,
            self.trace_dir,
            self.report_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def load_settings(**overrides) -> Settings:
    """读取环境变量构造 Settings；overrides 优先级最高（便于测试注入）。"""
    defaults: dict = {
        "provider": _str("LLM_PROVIDER", "mock").lower(),
        "api_key": _str("LLM_API_KEY", ""),
        "base_url": _str("LLM_BASE_URL", ""),
        "model": _str("LLM_MODEL", ""),
        "temperature": _float("TEMPERATURE", 0.2),
        "max_tokens": _int("MAX_TOKENS", 2048),
        "request_timeout": _int("REQUEST_TIMEOUT", 60),
        "max_plan_items": _int("MAX_PLAN_ITEMS", 4),
        "max_critic_rounds": _int("MAX_CRITIC_ROUNDS", 2),
        "max_tool_calls": _int("MAX_TOOL_CALLS", 3),
        "enable_web_search": _bool("ENABLE_WEB_SEARCH", True),
        "enable_rag": _bool("ENABLE_RAG", True),
        "search_results": _int("SEARCH_RESULTS", 5),
        "retrieve_top_k": _int("RETRIEVE_TOP_K", 4),
        "kb_dir": _path("KB_DIR", ROOT / "data" / "sample_kb"),
        "memory_dir": _path("MEMORY_DIR", ROOT / ".memory"),
        "memory_summary_threshold": _int("MEMORY_SUMMARY_THRESHOLD", 4000),
        "memory_max_items": _int("MEMORY_MAX_ITEMS", 200),
        "trace_dir": _path("TRACE_DIR", ROOT / "traces"),
        "enable_trace": _bool("ENABLE_TRACE", True),
        "trace_verbose": _bool("TRACE_VERBOSE", False),
        "langsmith": _bool("LANGSMITH", False),
        "langsmith_project": _str("LANGSMITH_PROJECT", "argus-research"),
        "report_dir": _path("REPORT_DIR", ROOT / "reports"),
    }
    defaults.update(overrides)
    return Settings(**defaults)


# 模块级默认配置。测试或 CLI 可通过 load_settings(...) 得到新快照后自行传递。
settings: Settings = load_settings()
