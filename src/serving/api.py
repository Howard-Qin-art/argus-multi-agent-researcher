"""服务化入口：把 `run_research` 暴露为 HTTP 接口。

启动方式：
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
    python -m src.serving.api

设计要点：
1. FastAPI / uvicorn 属于**可选依赖**（见 requirements.txt 的"服务化"段），
   因此一律延迟导入，缺失时给出中文提示，不影响 mock 模式下 CLI 与单测的运行。
2. 一次研究要跑完整张图（规划 / 取证 / 审查 / 成文），通常是几十秒量级，
   直接跑在事件循环里会把整个服务阻塞住，所以 `/research` 用
   `run_in_threadpool` 丢进线程池执行。
3. 异常分层：ValueError 视为配置或参数问题 → 400；其余一律 → 500，
   且只回中文摘要，绝不把堆栈泄漏给调用方。
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import Settings, load_settings
from ..graph import run_research

__all__ = ["ResearchRequest", "ResearchResponse", "create_app", "app"]


# `/traces` 最多返回的条目数，避免目录长期积累后一次拉回几千条
MAX_TRACES = 50

# trace 文件名只允许这些字符，既便于校验也顺手挡住路径穿越
_TRACE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


class ResearchRequest(BaseModel):
    """研究请求。

    未给出的字段沿用服务端默认配置（环境变量 / Settings 默认值），
    这样同一份部署既能用 mock 演示，也能按请求临时切换到真实模型。
    """

    query: str = Field(min_length=1, description="要研究的开放式问题")
    user_id: str = Field(default="default", description="用户标识，决定长期记忆归属")
    thread_id: str | None = Field(default=None, description="会话线程 id，用于断点续跑")
    provider: str | None = Field(
        default=None, description="LLM 提供方：mock / deepseek / qwen / zhipu / openai"
    )
    max_critic_rounds: int | None = Field(
        default=None, ge=0, le=5, description="Critic 最大返工轮数"
    )


class ResearchResponse(BaseModel):
    """研究响应：一次运行的产物与运行指标。"""

    trace_id: str = Field(default="", description="本次运行的 trace 标识")
    query: str = Field(default="", description="回显的研究问题")
    report: str = Field(default="", description="Markdown 研究报告正文")
    references: list[dict] = Field(default_factory=list, description="参考来源列表")
    citation_issues: list[str] = Field(
        default_factory=list, description="引用校验发现的问题，空列表表示合规"
    )
    plan: list[str] = Field(default_factory=list, description="Supervisor 的研究计划")
    notes_count: int = Field(default=0, description="取证得到的证据条数")
    critique_rounds: int = Field(default=0, description="Critic 实际执行的轮数")
    summary: dict = Field(default_factory=dict, description="trace 汇总与预算快照")
    report_path: str = Field(default="", description="报告落盘路径")


def _to_response(payload: ResearchRequest, result: dict[str, Any]) -> ResearchResponse:
    """把 `run_research` 的返回值映射成响应模型。

    state 里的键可能因为节点提前返回而缺失，这里统一取默认值，
    保证接口不会因为某条链路没跑到就 500。
    """
    state = result.get("state") or {}
    summary = result.get("summary") or {}
    return ResearchResponse(
        trace_id=str(state.get("trace_id") or summary.get("trace_id") or ""),
        query=str(state.get("query") or payload.query),
        report=str(state.get("report") or ""),
        references=list(state.get("references") or []),
        citation_issues=list(state.get("citation_issues") or []),
        plan=list(state.get("plan") or []),
        notes_count=len(state.get("notes") or []),
        critique_rounds=int(state.get("critique_round") or 0),
        summary=summary,
        report_path=str(result.get("report_path") or ""),
    )


def _resolve_trace_path(trace_id: str, cfg: Settings) -> Path | None:
    """把 trace_id 收敛成安全的 jsonl 路径。

    非法 id 返回 None（调用方按 404 处理），同时天然挡住 ../ 这类路径穿越。
    """
    name = (trace_id or "").strip()
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    if not _TRACE_ID_PATTERN.fullmatch(name):
        return None
    return Path(cfg.trace_dir) / f"{name}.jsonl"


def create_app() -> "FastAPI":  # noqa: F821 - 仅作类型标注，FastAPI 延迟导入
    """构造 FastAPI 应用。

    用工厂而不是模块级单例，方便测试里用不同配置各起一份实例。
    """
    # 延迟导入：未安装服务化依赖时，导入本模块不该炸在这里之外的地方
    from fastapi import FastAPI, HTTPException
    from starlette.concurrency import run_in_threadpool

    application = FastAPI(
        title="Argus Research API",
        version="0.1.0",
        description="Argus · 多智能体协同深度研究系统：Supervisor / Researcher / Critic / Writer",
    )

    @application.get("/health")
    async def health() -> dict[str, Any]:
        """存活探针，顺带报告当前默认 provider，便于确认是否跑在 mock 下。"""
        cfg = load_settings()
        return {
            "status": "ok",
            "provider": cfg.provider,
            "mock": cfg.provider == "mock",
        }

    @application.post("/research", response_model=ResearchResponse)
    async def research(payload: ResearchRequest) -> ResearchResponse:
        """跑一次完整研究。

        整条流水线是同步阻塞的（LangGraph 的 invoke），放进线程池执行，
        避免把事件循环占满导致 /health 一起被拖住。
        """
        overrides: dict[str, Any] = {}
        if payload.provider:
            overrides["provider"] = payload.provider.strip().lower()
        if payload.max_critic_rounds is not None:
            overrides["max_critic_rounds"] = payload.max_critic_rounds

        cfg = load_settings(**overrides).ensure_dirs()

        try:
            result = await run_in_threadpool(
                functools.partial(
                    run_research,
                    query=payload.query,
                    user_id=payload.user_id,
                    cfg=cfg,
                    thread_id=payload.thread_id,
                )
            )
        except ValueError as exc:
            # 配置类错误（未知 provider、缺少 API Key 等）：客户端可自行修正
            raise HTTPException(status_code=400, detail=f"配置错误：{exc}") from exc
        except Exception as exc:  # 其余一律视为服务端故障，不泄漏堆栈
            raise HTTPException(
                status_code=500, detail=f"研究流程执行失败：{type(exc).__name__}"
            ) from exc

        return _to_response(payload, result)

    @application.get("/traces")
    async def list_traces() -> dict[str, Any]:
        """列出最近的 trace 文件（按修改时间倒序，最多 50 个）。"""
        cfg = load_settings().ensure_dirs()
        trace_dir = Path(cfg.trace_dir)
        if not trace_dir.exists():
            return {"count": 0, "traces": []}

        files = [path for path in trace_dir.glob("*.jsonl") if path.is_file()]
        files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        names = [path.name for path in files[:MAX_TRACES]]
        return {"count": len(names), "traces": names}

    @application.get("/traces/{trace_id}")
    async def get_trace(trace_id: str) -> dict[str, Any]:
        """读取某个 trace 的全部事件（每行一个 JSON 对象）。"""
        cfg = load_settings().ensure_dirs()
        path = _resolve_trace_path(trace_id, cfg)
        if path is None or not path.exists():
            raise HTTPException(status_code=404, detail=f"未找到 trace：{trace_id}")

        events: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # 进程被中断时可能留下半行，跳过而不是让整个接口失败
                    continue

        return {"trace_id": path.stem, "count": len(events), "events": events}

    return application


# uvicorn / 测试直接引用这个模块级实例
app = create_app()


if __name__ == "__main__":  # pragma: no cover
    try:
        import uvicorn
    except ImportError:  # 可选依赖缺失时给中文提示，而不是抛一长串堆栈
        raise SystemExit(
            "缺少 uvicorn，请先安装服务化依赖：pip install -r requirements.txt "
            "后执行 pip install '.[serve]'（或单独 pip install uvicorn）"
        )
    uvicorn.run(app, host="0.0.0.0", port=8000)
