# Argus 项目 · 技术栈清单与学习路线

这份文档回答两个问题：**这个项目用到了哪些技术**，以及**你要学到什么程度才敢把它写进简历**。

写作前提：项目代码已经能跑、测试全绿。但代码是我搭的，知识得是你自己的。面试时没人会考你"这个项目第几行写了什么"，考的是"为什么这么设计"和"换你你会怎么改"。

---

## 零、先分清三档水平

| 档位 | 表现 | 面试结果 |
|---|---|---|
| 一档 · 跑得通 | 照 README 跑起来，能改改 prompt，能复述架构图 | 会被问穿。三句话之后答不上，反而扣分 |
| 二档 · 改得动 | 每行为什么这么写、换个写法会怎样，都能说；能自己加一个节点 | 及格线。**写进简历的最低要求** |
| 三档 · 说得出取舍 | 能主动说"这里我选了 A 而不是 B，代价是 C"，能讲"我故意没做 X" | 加分。这一档才轮到项目和别人拉开差距 |

三档的关键区别不在知识量，而在**能不能讲失败和代价**。二档和三档的分水岭，是你会不会主动说这个项目的毛病。

---

## 一、技术清单（按模块拆）

### 1. Python 工程基本功

| 要掌握 | 为什么这个项目需要 | 看哪里 |
|---|---|---|
| 类型注解、`TypedDict` | `ResearchState` 用 TypedDict 定义，图的每个节点靠它做契约 | `src/graph/state.py` |
| `dataclass` + `field(default_factory=...)` | `AgentRuntime` 的延迟初始化 | `src/agents/runtime.py` |
| `pathlib`、文件原子写（写临时文件再 `os.replace`） | 长期记忆落盘，中途崩了不能写坏文件 | `src/memory/long_term.py` |
| 相对导入与 `__init__.py` 的延迟导出 | 这个项目真实踩过 `src.agents` 与 `src.graph` 循环导入 | `src/graph/__init__.py` |
| 上下文管理器 `__enter__/__exit__` | `tracer.span(...)` 用 `with` 包住每个节点，异常也要留痕 | `src/observability/tracer.py` |
| `ast` 模块（抽象语法树） | 计算器的白名单求值是靠遍历 AST 实现的 | `src/tools/calculator.py` |
| 异常分层与降级 | 工具调用失败不往上抛，而是收敛成 `ToolResult(ok=False)` | 各 `src/tools/*.py` |
| `asyncio` 基础 | 知道为什么 `/research` 用 `run_in_threadpool` 而不是 `async def` | `src/serving/api.py` |

过关标准：能不看现有代码，自己写出一个同等质量的模块（比如新增一个"网页正文抓取"工具）。

### 2. Pydantic v2

- `BaseModel`、`Field`（`min_length`、`ge/le`、`description`）
- `field_validator`、`model_validator`
- `ConfigDict(extra="forbid")` —— 模型多返回字段时立刻暴露，而不是静默吞掉
- `model_dump()` / `model_validate()` / 捕获 `ValidationError`
- **关键认知**：为什么数据契约用 Pydantic，而图状态用 TypedDict？因为状态要能被 LangGraph 序列化和做增量合并，契约要能严格校验模型输出。这两个东西的分工本身就是考点。

过关标准：能设计出一个"模型怎么乱输出都不会让流程崩"的 schema，并说清校验失败时是抛错还是降级。

### 3. LLM 接口层

- OpenAI 兼容协议：`/chat/completions` 的请求结构、`messages` 三种角色、`temperature`、`max_tokens`、`response_format`
- 各家国产模型的差异：`base_url`、模型名、是否支持 JSON mode、是否支持 function calling
- 超时、重试、错误分类（哪些错该重试、哪些该直接放弃）
- **结构化输出的三种做法**，以及本项目为什么选了最笨的那种：

| 做法 | 优点 | 缺点 |
|---|---|---|
| prompt 里约定 JSON + 自己解析 | 所有模型都能用，无依赖 | 模型经常不守规矩，要写兜底 |
| `response_format: {"type": "json_object"}` | 稳 | 不是所有国产模型都支持 |
| function calling / tool calling | 最稳，且天然支持工具调用 | 各家实现有差异，参数结构复杂 |

本项目选第一种是为了**provider 可插拔**。这是一个真实的取舍，面试讲出来很加分。

- token 估算（按字符数粗略折算）和 `tiktoken` 真实计数的差别。本项目是估算，这是已知的局限之一。

过关标准：能自己写一个带重试和降级的 LLM 封装，并说清每种异常的处理策略。

### 4. LangChain 核心抽象

- `BaseChatModel` —— 统一接口，换模型不改业务代码
- `invoke` / `batch` / `stream`
- `SystemMessage` / `HumanMessage` / `AIMessage`
- 本项目**只用 `langchain-core` 和 `langchain-openai`，没用 chain / agent 那一套**。要知道为什么：那一层封装会让控制流变得不透明，而你要的是"每个决策都可观测"。这也是个能讲的取舍。

### 5. LangGraph（项目骨架，重点）

- `StateGraph`、节点函数签名（收 state、返回增量 dict）
- `add_edge` / `add_conditional_edge` / `END`
- **状态合并与 reducer**（`Annotated[list, add]`）—— 为什么列表字段要 reducer，不然会被覆盖
- `checkpoint` + `thread_id`：一次运行的快照，支持断点续跑
- `interrupt`：human-in-the-loop 挂起
- `recursion_limit`：图层面的循环保险丝
- 进阶（本项目没用但你要知道）：`Send` 做 fan-out 并行、子图 `subgraph`

过关标准：能**手画**出这张状态图，说出每条边在什么条件下走，并能回答"如果我要让 Researcher 的四个子问题并行，图要怎么改"。

### 6. Agent 核心机制（理论，最重要的一块）

这一块不是框架知识，是**领域知识**。框架会过时，这些不会。

- **ReAct**：思考 → 行动 → 观察 的循环
- **Plan-and-Execute**：先规划再执行，本项目属于这一类
- **Reflection / 自纠错**：让模型审查自己的产出。Critic 角色就是这个
- **Self-consistency**：同一问题跑多次投票。你在 NL2SQL 里用结果集投票，这里用的是单次 Critic
- **工具调用（function calling）**：`tools` 参数、`tool_choice`、模型返回 tool call、执行后把结果回喂
- **多智能体拓扑**：supervisor（本项目）、peer debate、assembly line、hierarchical，各自适合什么
- **多智能体的代价**：LLM 调用次数翻倍、延迟叠加、错误在角色间传播。什么时候**不该**用多智能体 —— 这道题大概率会被问
- **上下文管理**：膨胀、压缩、中间内容被忽略（lost in the middle）

过关标准：能讲清楚"为什么是四个角色而不是一个"，以及"什么情况下四个角色是浪费"。

### 7. 记忆

- **短期记忆**：滑动窗口、超阈值摘要压缩、token 预算
- **长期记忆**：跨会话持久化、召回、去重、容量淘汰
- **checkpoint**：单次运行的断点快照

三者区别是**高频考点**，很多人会混为一谈：

| | 作用域 | 生命周期 | 本项目实现 |
|---|---|---|---|
| 短期记忆 | 单次会话内 | 会话结束即弃 | `ShortTermMemory`，超阈值触发压缩 |
| 长期记忆 | 跨会话、按用户 | 永久持久化 | `LongTermMemory`，JSON 落盘 + 关键词召回 |
| checkpoint | 单次图运行 | 运行结束可弃 | LangGraph 的 `thread_id` |

过关标准：被问"你的长期记忆是怎么召回的"时，能说出"关键词 + 字面匹配，不是语义检索，这是我的局限，补法是用 embedding"。

### 8. 检索 / RAG

- 切分（chunking）策略、为什么要重叠、块大小怎么定
- 中文分词：为什么本项目用**二元字组（bigram）**而不是单字或词
- 打分检索：词频、TF-IDF、BM25 的原理与差别
- 向量召回：embedding 模型、faiss、余弦相似度
- 混合检索（关键词 + 向量）与 rerank
- 检索评测：召回率、MRR、NDCG（知道名词和含义即可）

本项目**只做了关键词 + bigram，没做向量**。这是写在 ARCHITECTURE.md 已知局限里的第一条。你要做的不是掩饰它，而是知道怎么补：加 embedding、建 faiss 索引、做混合召回。

过关标准：能讲出"关键词检索漏的是什么"（同义改写、语义相关但字面不重叠），并能说出补向量召回的大致步骤。

### 9. 护栏与安全（项目的差异化亮点）

- **结构层**：Pydantic 校验失败时的降级策略 —— 是抛错、是重试、还是用兜底产出继续走
- **事实层**：引用真实性校验。正文里出现但参考列表里找不到的编号，自动剔除并留痕（防编造参考文献）
- **资源层**：调用次数预算、超时、最大返工轮数
- **代码执行安全**：为什么 `eval` 危险、AST 白名单怎么拦住 `__import__`、`open`、属性访问、赋值
- **提示注入（prompt injection）**：联网搜到的内容里可能藏着指令，怎么防（本项目防得有限，知道这是问题就比不知道强）

过关标准：能说出"如果模型连续三次不返回 JSON，我的系统会怎样"，并且答案是"降级继续出稿"而不是"崩溃"。

### 10. 可观测性

- trace / span 的数据结构设计：要记录哪些字段才有诊断价值
- 按 `trace_id` 串联多节点、多轮次
- 耗时、token、失败原因的分层记录
- 日志 vs trace 的区别：日志是"发生了什么"，trace 是"一次完整请求里各步骤的关系"
- LangSmith / Langfuse 接入原理（环境变量一开就生效，本项目就是这么做的）

过关标准：能说出你的 trace 里有哪些字段，以及"如果线上一个请求慢了 30 秒，你怎么定位是哪一步"。

### 11. 评测

- 指标设计：出稿率、引用合规率、关键词召回、成本、时延 —— 每一指标对应哪个失败模式
- 评测集怎么建、为什么要分层（能力维度、难度）
- **LLM-as-judge**：rubric 设计、位置偏差、长度偏好、评分漂移，怎么缓解
- 回归评测接 CI
- 硬指标 vs 软指标：对比你 NL2SQL 项目的执行准确率（EX），理解为什么"能自动判分"的指标更值钱

本项目只有自建小样本 + 关键词召回，**没有 LLM-as-judge**。这是第二个已知局限，也是你最值得补的一块。

### 12. 服务化与工程质量

- FastAPI：路由、请求响应模型、`run_in_threadpool`、异常映射成 HTTP 状态码、不泄漏堆栈
- Streamlit 基础（会跑会改即可，不必深挖）
- Docker：镜像分层、`PYTHONPATH`、环境变量注入、`.dockerignore`
- pytest：`fixture`、`tmp_path`、`monkeypatch`、`parametrize`、mock
- **离线可测设计**：mock provider 的意义 —— 让 CI 不依赖网络和 API Key。这是一个工程成熟度信号，面试提一句很有效
- 配置管理：环境变量 + `.env` + 自定义 `Settings`

---

## 二、学习顺序（四阶段，每阶段有验收）

### 阶段一：把项目变成自己的（3–5 天）

1. 跑通 CLI、Streamlit、评测，各一次
2. 按模块读源码，顺序：`config` → `schemas` → `graph/state` → `agents/*` → `graph/orchestrator` → `tools` → `memory` → `guardrails` → `observability`
3. 自己手画一遍架构图，再和 README 里的对照
4. **动手改一处**：比如给 Researcher 加一个"证据条数不足就强制再查一轮"的判断

验收：不看任何文档，能把数据从输入到报告的完整流向讲一遍，包括 state 每个字段是谁写的。

### 阶段二：补原理地基（1–2 周）

- LangGraph 官方教程，重点看状态、条件边、checkpoint、persistence
- Pydantic v2 官方文档，把校验器那章看完
- OpenAI API 文档的 chat completions 部分，把每个参数的作用过一遍
- **独立写一个最小 ReAct agent，不用 LangGraph**，只用裸 API。这一步最重要 —— 做完你才算真的懂 agent，而不是懂框架

验收：能手写一个 100 行以内、带工具调用和自纠错的 agent。

### 阶段三：补齐项目没做的（1–2 周，选做但强烈建议）

按性价比排：

1. **真实 token 计数**（半天）用 tiktoken 替换估算，成本数据立刻可信
2. **向量召回**（2–3 天）embedding + faiss，与关键词做混合检索
3. **LLM-as-judge 评测**（1–2 天）写一个带 rubric 的评判器，补上软指标
4. **Researcher 子问题并行**（1–2 天）用 LangGraph 的 `Send` 做 fan-out

验收：README 里"已知局限"那一节能划掉三条，并且有前后对比数字。

### 阶段四：面试演练（3–5 天）

- 拿 `ARCHITECTURE.md` 最后一节那 9 道题，每题**口述 2 分钟**并录音回听
- 找人模拟追问，重点练"为什么"和"代价是什么"两类问题
- 准备一个 30 秒版本和一个 3 分钟版本的项目介绍

---

## 三、考点分级

**必会（答不上直接扣分）**
- LangGraph 的状态图和条件边怎么工作
- 短期记忆 / 长期记忆 / checkpoint 的区别
- 为什么 Critic 之后回边到 Supervisor 而不是 Researcher
- 怎么防止自纠错变成无限循环
- 工具调用失败时系统怎么处理

**应该会（答不上不算硬伤，但会削弱可信度）**
- 为什么用 TypedDict 而不是 Pydantic 做状态
- 结构化输出为什么选 prompt 约定而不是 function calling
- 检索为什么用 bigram，漏了什么
- trace 里记录了哪些字段，怎么用它定位问题
- mock provider 除了省 Key 还有什么价值

**加分项（答上来能拉开差距）**
- 多智能体比单 agent 贵在哪，什么情况下不该用
- 你的评测指标各对应哪个失败模式，局限在哪
- prompt injection 在你的链路里可能从哪里进来
- 如果要上生产，你还缺什么（流式输出、并发限流、成本控制、权限）
- 这个项目和你 NL2SQL 项目在 agent 能力上的互补关系

---

## 四、自测题（能答上来算过关）

**编排**：如果 Critic 判 pass 但证据数为 0，会发生什么？该怎么改？

**状态**：`notes` 字段在 researcher 里是追加还是覆盖？如果写成覆盖会出什么问题？

**终止**：轮数上限、预算耗尽、verdict 三重条件，哪个优先级最高？是"与"还是"或"的关系？

**记忆**：两个不同 user_id 的长期记忆会不会互相污染？怎么隔离的？

**工具**：注册表收到一个不存在的工具名，会抛异常还是返回失败？为什么这么选？

**检索**：知识库里有一句话和查询语义相同但用词完全不同，能召回吗？为什么？

**安全**：给计算器传 `__import__("os").system("rm -rf /")`，在哪一步被拦住？

**引用**：正文写了 `[7]` 但参考列表只有 5 条，系统怎么处理？改动会留痕吗？

**可观测**：一次运行产生了几个 span？怎么用一个 id 把它们串起来？

**评测**："关键词召回"这个指标，在什么情况下会给出虚高的分数？

**成本**：四个角色各调几次 LLM？一次完整研究大概多少调用？如果子问题数翻倍，成本怎么变？

**取舍**：如果让你砍掉一个角色，你砍哪个？为什么？

---

## 五、学习资源

以官方文档为主，中文博客只用来辅助理解概念，不要拿来学 API（版本差异太大）。

| 主题 | 资源 |
|---|---|
| LangGraph | https://langchain-ai.github.io/langgraph/ |
| LangChain Python | https://python.langchain.com/docs/introduction/ |
| Pydantic v2 | https://docs.pydantic.dev/latest/ |
| FastAPI | https://fastapi.tiangolo.com/ |
| pytest | https://docs.pytest.org/ |
| OpenAI Chat API | https://platform.openai.com/docs/api-reference/chat |
| tiktoken | https://github.com/openai/tiktoken |
| LangSmith | https://docs.smith.langchain.com/ |
| DeepSeek API | https://api-docs.deepseek.com/ |

项目内部三份文档，按这个顺序读：`README.md`（全貌）→ `ARCHITECTURE.md`（设计理由与失败模式）→ `RESUME.md`（怎么讲）。

---

## 六、四个常见误区

**误区一：跑通了就等于会了。**
跑通只值一档。面试官问的是"为什么"，不是"是什么"。

**误区二：把时间都花在背框架 API 上。**
LangGraph 的 API 半年一变，agent 的机制不会变。阶段二那个"不用框架手写 ReAct"的练习，价值高于读十篇框架文档。

**误区三：想把没做的都补上。**
与其堆功能，不如把已有的一条链路讲透。向量召回、并行、judge 三件事里挑一到两件做深，剩下的诚实说"没做，因为 ……"，反而更像真做过项目的人。

**误区四：背答案。**
面试官追问两层就能听出是不是背的。真正的准备方式是阶段四的口述练习，答的时候要能随时拐到"我当时还试过另一种做法，但放弃了，因为 ……"。

---

## 七、一句话总结

这个项目要求你掌握的，表面上是 LangGraph + Pydantic + FastAPI 这套技术栈，实际上考的是**你有没有把一个不可靠的组件（LLM）组装成可靠系统的工程能力**。前者两周能学完，后者只能靠真的踩过坑。
