# Argus · 多智能体协同深度研究系统 —— 服务化镜像
#
# 构建： docker build -t argus:0.1.0 .
# 运行： docker run --rm -p 8000:8000 argus:0.1.0
# 自检： curl http://localhost:8000/health
#        curl -X POST http://localhost:8000/research -H "Content-Type: application/json" \
#             -d '{"query":"多智能体系统落地的主要难点","provider":"mock"}'
#
# 依赖说明：requirements.txt 只含核心依赖，服务化相关（fastapi / uvicorn / streamlit）
# 走 pyproject 的 [serve] extra，即下面的 `pip install -e ".[serve]"`。
# 注意：仓库根目录还应放一份 .dockerignore，排除 .git、traces/、reports/、.memory/、
# __pycache__/、*.egg-info/、tests/ 等不需要进镜像的内容（Dockerfile 里无法声明）。

FROM python:3.11-slim

# 让容器日志实时输出，且不生成 .pyc 污染镜像层
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 先拷依赖清单再装依赖：源码变动时不会重装依赖，充分利用 Docker 层缓存。
# pyproject.toml 是 `pip install -e .` 与 [serve] extra 的来源，必须一起拷入。
COPY requirements.txt pyproject.toml README.md ./
COPY src/ ./src/
COPY data/ ./data/

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir -e ".[serve]"

# 运行期配置：默认 mock provider，数据目录全部落在 /app 下（容器内可写）
ENV ARGUS_LLM_PROVIDER=mock \
    ARGUS_KB_DIR=/app/data/sample_kb \
    ARGUS_TRACE_DIR=/app/traces \
    ARGUS_MEMORY_DIR=/app/.memory \
    ARGUS_REPORT_DIR=/app/reports \
    PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
