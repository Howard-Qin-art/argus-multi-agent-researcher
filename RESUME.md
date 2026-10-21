# 项目二 · 简历内容包

> 所有内容均对照 `E:\项目二` 的实际源码撰写，每句话都能被追问到具体文件。
> **两个务必**：① 带 `【待补】` 的数字必须替换成你自己跑真实模型的实测值；② 最后一节「不要写进简历的能力」请务必看一眼。

---

## 〇、自然手写版（去 AI 味，推荐直接用）

**Argus · 多智能体协同深度研究系统**（个人项目）
`Python · LangGraph · LangChain · Pydantic v2 · FastAPI · Streamlit · pytest · Docker`

让四个 agent 分工做一件小事：输入一个开放式问题，输出一份每条结论都能溯源的报告。规划的那个负责把大问题拆成能查的小问题，取证的那个去真的查，第三个专门挑刺，最后一个把查到的东西写成人能读的报告。

- 流程用 LangGraph 的状态图串起来。挑刺这个角色觉得证据不够时就打回上一层重来，一开始没加限制，有个问题卡在里面反复重写出不来，后来加了轮次上限和调用预算两条线，才算兜住。
- 调哪个工具不是写死的，让模型看着子问题自己定该用哪个工具、用什么样的关键词去查，再交给一个统一的注册表执行。注册表里做了白名单、异常收敛和调用次数限制，单个工具挂掉不会连累整条流程。
- 结构化输出这块吃过亏。模型隔三差五不返回 JSON，或者格式歪一下，后来每个环节的产出都用 Pydantic 过一遍，过不了就降级，实在不行用兜底方案把流程走完，不让它直接崩。
- 报告要求每条结论都标引用编号。怕它编参考文献，就加了一道校验，正文里出现但参考列表里找不到的编号会被自动删掉，改动会记进 trace 而不是悄悄抹掉。
- 工具里有个计算器。eval 太危险，改成先把表达式解析成 AST，只放行白名单里的运算符和几个数学函数，赋值和属性访问一律拒掉。
- 调试多智能体最麻烦的是不知道错在第几步，所以每个节点都留了记录，耗时、token、有没有报错都写进去，靠 trace_id 能串成一条完整链路，需要的时候也可以切到 LangSmith 上看。
- 顺手补了 151 个单测。为了能离线跑，内置了一个 mock 模型替身，没网没 key 的情况下 CI 里 12 秒跑完。

---

## 一、一句话版（用于项目列表 / 自我介绍）

> 基于 LangGraph 的多智能体研究系统：4 个角色协同完成「规划 → 取证 → 审查返工 → 带引用成文」，并通过分层记忆、三层护栏与全链路 trace 解决多智能体的可靠性与可复盘问题。

---

## 二、简历正文（推荐版，6 条）

**Argus · 多智能体协同深度研究系统**（个人项目 / GitHub 可附链接）

`Python · LangGraph · LangChain · Pydantic v2 · FastAPI · Streamlit · pytest · Docker`

> 面向开放式研究问题的多智能体系统。用户输入自然语言问题，系统自动完成研究计划拆解、多工具取证、证据交叉审查与返工、以及带引用溯源的结构化报告生成。

- **设计 4 角色协同架构**（Supervisor 规划 / Researcher 取证 / Critic 审查 / Writer 成文），基于 LangGraph 状态图编排；以 Critic → Supervisor 条件回边实现「证据不足自动返工」，并设置**轮数上限、预算耗尽、审查结论**三重终止条件，防止自纠错循环退化为无限重试与成本失控。
- **实现模型决策式工具调用**：由 Router 角色根据子问题判断该调用哪些工具与检索关键词，经统一工具注册表执行名称白名单校验、异常收敛与调用次数限流，使非法工具调用与单点工具故障**不会中断整条流水线**。
- **设计分层记忆机制**：会话内用滑动窗口 + 摘要压缩抑制上下文膨胀；跨会话按用户维度持久化已验证事实与偏好（临时文件 + 原子替换写入、容量淘汰策略），并借 LangGraph checkpoint 支持按 thread 断点续跑。
- **构建结构 / 事实 / 资源三层护栏**：节点间数据全部由 Pydantic v2 显式 schema 校验并在失败时降级；自动剔除正文中不存在的引用编号以**杜绝编造参考文献**；计算器工具基于 AST 白名单求值，拒绝变量赋值、属性访问与函数调用。
- **实现全链路可观测**：按 trace_id 串联各角色，落地含耗时、token、成功标记、失败原因的 JSONL trace，支持通过环境变量一键切换 LangSmith 云端追踪。
- **建立可量化的评测与回归能力**：以出稿率、引用合规率、关键词召回、LLM 与工具调用成本四类指标衡量每次改动；内置 mock 模型使全部 **151 个单元测试在离线、无密钥环境下 12 秒内跑通**，作为 CI 基线。

---

## 三、精简版（简历空间紧张时用，收敛为 4 条）

- 设计 Supervisor / Researcher / Critic / Writer 四角色多智能体架构，基于 LangGraph 状态图编排，以条件回边实现证据不足自动返工，并用三重终止条件防止无限循环。
- 实现模型决策式工具调用与统一工具注册表，提供白名单校验、异常收敛与调用限流；计算器基于 AST 白名单求值，杜绝任意代码执行。
- 设计会话内 / 跨会话分层记忆；以 Pydantic 显式 schema 校验与引用编号真实性校验构成事实护栏，自动剔除编造引用。
- 实现按 trace_id 串联的 JSONL 全链路追踪；沉淀出稿率、引用合规率、关键词召回等评测指标，151 个单测离线 12 秒内跑通。

---

## 四、English Version（部分 JD 需要英文简历）

**Argus — Multi-Agent Deep Research System** (Personal Project)
*Python · LangGraph · LangChain · Pydantic v2 · FastAPI · Streamlit · Docker*

A multi-agent system that turns open-ended questions into evidence-grounded reports through four specialized agents: planning, evidence retrieval, critical review with revision loops, and cited report writing.

- Architected a 4-agent collaboration pipeline (Supervisor / Researcher / Critic / Writer) orchestrated as a LangGraph state graph; a conditional back-edge from Critic to Supervisor enables automatic revision when evidence is insufficient, guarded by round limits, budget exhaustion, and verdict checks to prevent runaway loops.
- Implemented model-driven tool orchestration: a Router agent selects tools and queries at runtime, while a central registry enforces name whitelisting, exception containment, and per-call rate limits, so tool failures never break the pipeline.
- Built layered memory: sliding-window summarization within a session, per-user persistent storage of verified facts (atomic writes, capacity eviction) across sessions, and LangGraph checkpoints for resume-from-breakpoint.
- Enforced three guardrail layers — Pydantic v2 schema validation with graceful degradation, citation authenticity checks that strip fabricated references, and an AST-whitelist calculator rejecting code injection.
- Delivered full observability via trace_id-linked JSONL traces (latency, tokens, failure reasons), switchable to LangSmith; defined evaluation metrics and shipped 151 offline unit tests running in 12s.

---

## 五、每句话背后的技术点（面试展开用）

| 简历里的表述 | 被追问时可以展开的技术点 | 代码位置 |
|---|---|---|
| 条件回边 / 三重终止 | 为什么回边指向 Supervisor 而不是 Researcher（重规划 vs 重取证）；不足 evidence 却判 pass 时强制返工一次 | `src/graph/orchestrator.py`、`src/agents/critic.py` |
| 模型决策式工具调用 | 与硬编码 pipeline 的对比取舍；Router 多花一次 LLM 调用换来什么 | `src/agents/researcher.py`、`src/tools/registry.py` |
| 工具注册表 / 限流 | 未知工具返回失败结果而非抛异常；max_calls 截断 | `src/tools/registry.py` |
| 分层记忆 | 短期 vs 长期 vs LangGraph checkpoint 三者本质区别（高频误区） | `src/memory/short_term.py`、`src/memory/long_term.py` |
| 引用真实性校验 | 先用 Pydantic 解析，再剔除越界 `[n]`，同时留痕而非静默修改 | `src/guardrails/checks.py`、`src/agents/writer.py` |
| AST 白名单 | 为什么不用 eval；白名单允许的运算符与函数范围 | `src/tools/calculator.py` |
| 151 个单测离线跑通 | mock provider 的设计动机：既服务 CI，也让系统可离线演示 | `src/llm.py`、`tests/` |
| 结构化降级 | 模型不返回 JSON 时的兜底计划与最小可用报告 | `src/agents/supervisor.py`、`src/agents/writer.py` |

---

## 六、必须自己补上的数字（不要照抄占位值）

以下占位内容来自 **mock 占位模型**，**不能**作为模型效果写进简历：

- 单次任务 11 次 LLM 调用、耗时 12ms —— 这是确定性替身模型的产物
- 证据 8 条、参考来源 2 个 —— 同上

**推荐你跑真实模型后替换成这样一句**（放在第一条或最后一条 bullet 末尾）：

> 接入 DeepSeek 后，在自建 N 题评测集上：出稿率 【待补】、引用合规率 【待补】、端到端平均时延 【待补】s、单次任务平均成本约 ¥【待补】。

跑法：

```bash
export ARGUS_LLM_PROVIDER=deepseek ARGUS_LLM_API_KEY=sk-xxxxx
python -m src.eval.run_eval --provider deepseek --output result.json
```

**可以直接安全使用的真实数字**（与模型无关的结构性事实）：

- 151 个单元测试、约 12 秒、离线无密钥可跑
- 4 个协作角色、1 条自纠错条件回边
- 本地知识库 76 个可检索文本块
- 代码规模约 4665 行（src + tests）

---

## 七、不要写进简历的能力（写了会被问穿）

以下都是本项目**尚未实现**的，切勿暗示：

- ❌ **向量检索 / embedding 召回** —— 当前是关键词 + 中文 bigram，不是语义向量
- ❌ **子问题并行处理** —— Researcher 是串行逐个处理
- ❌ **LLM-as-judge 自动评分** —— 评测是关键词召回类启发式指标
- ❌ **流式输出 / 中间过程实时展示** —— Streamlit 是整次跑完后渲染
- ❌ **人工打断介入（interrupt）** —— 只支持按 thread 断点续跑，没有真正的 human-in-the-loop 中断
- ❌ **真实 token 精确计数** —— 是按中文 1 字 ≈ 1 token 的规则估算
- ❌ **多智能体辩论（Peer Debate）** —— 只有单一 Critic

如果面试官问「后续怎么改进」，这些反而是最好的答案素材（README 第七节列出了完整清单）。
