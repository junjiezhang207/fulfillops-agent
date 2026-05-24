# 全域电商供应链智能履约多 Agent 平台

> 基于 LangChain / LangGraph / LlamaIndex 的生产级多 Agent 系统，覆盖订单履约决策的完整 AI 工程链路。

---

## 架构总览

```mermaid
graph TD
    User(["用户 / 客户端"])

    subgraph API["API 层（FastAPI）"]
        Guard["Guardrails 双层防护\nLayer1: 直接注入检测（用户输入）\nLayer2: 间接注入净化（工具输出）"]
        RL["Rate Limiter\n固定窗口限流"]
        Log["结构化 JSON 日志\n全链路 X-Trace-Id"]
        Prom["Prometheus /metrics"]
    end

    subgraph Router["HybridService 智能路由"]
        Hybrid["意图识别\nembedding 相似度判断\nSIMPLE / MEDIUM / COMPLEX"]
    end

    subgraph WF["LangGraph Workflow（简单查询）"]
        direction LR
        N1["dispatch"] --> N2["order_analysis\nCommand 错误短路"]
        N2 --> N3["inventory_analysis\n风险多规则评估"]
        N3 -->|库存充足| N5["finalize\nLLM 结论生成"]
        N3 -->|缺货| N4["knowledge_retrieval"]
        N4 --> N5
        N3 -->|HIGH RISK| HITL["HITL\ninterrupt + Command resume"]
        HITL --> N5
    end

    subgraph Agent["ReAct Agent（中等复杂）"]
        direction TB
        LLM["BaseChatModel\n多 Provider 统一"] --> Tools["6+1 工具\n结构化 JSON 输出"]
        Tools --> TCache["目录数据工具缓存\n规则/替代关系 TTL 5~10min\n实时数据工具不缓存"]
        TCache --> CB["Circuit Breaker\n+ Retry + Timeout"]
        LLM --> Reflect["自反思 0.6 阈值\n最多重试 2 次"]
    end

    subgraph MA["Multi-Agent Supervisor（多域问题）"]
        direction LR
        Sup["Supervisor LLM\nwith_structured_output"] -->|"Send API 并行"| A1["库存专家"]
        Sup --> A2["履约专家"]
        Sup --> A3["风险专家"]
        A1 & A2 & A3 --> Bar["fan-in barrier"]
        Bar --> Sup
        Sup --> Syn["Synthesizer"]
    end

    subgraph RAG["RAG 知识检索"]
        direction LR
        Ctx["订单 + 库存上下文\n意图识别"] --> QR["LLM 改写 + 规则扩展\n多查询召回"]
        QR --> Fusion["BM25 + 向量\nReciprocal Rank Fusion"]
        Fusion --> BR["业务意图重排\ncategory-intent boost"]
        BR --> CE["Reranker 精排\nqwen3-rerank / BGE / Jina"]
        CE --> Cmp["上下文压缩\nLLM 去噪"]
    end

    subgraph Eval["评测与可观测"]
        LF["Langfuse\n链路追踪 + LLM-as-judge"]
        RG["Ragas\nfaithfulness\nanswer_relevancy"]
        DE["DeepEval\nCI 回归测试\n无 LLM 单测覆盖"]
    end

    User --> Guard --> RL --> Hybrid
    Hybrid -->|SIMPLE| WF
    Hybrid -->|MEDIUM| Agent
    Hybrid -->|COMPLEX| MA
    Tools --> RAG
    Agent --> LF
    WF --> LF
    LF --> RG
    DE --> Agent
```

---

## 核心特性

### Agent 层
| 特性 | 实现方案 |
|---|---|
| ReAct Agent | LangChain `create_agent` + LangGraph `RedisSaver` checkpointer（对话历史持久化到 Redis，服务重启/多实例均可恢复；Redis 不可用时自动降级 MemorySaver）；提供普通接口和 NDJSON 流式接口 |
| 长短期记忆 | 短期记忆用 LangGraph checkpointer 保存会话消息；长期记忆默认用 `SQLiteLongTermMemoryStore` 本地落盘，生产切换 `MySQLMilvusLongTermMemoryStore`，由 MySQL 保存结构化记忆、Milvus 提供语义召回 |
| 自反思评估 | 三维质量评分（工具调用 40% + 相关性 30% + 具体性 30%），低于阈值自动重试；流式接口优先首字响应，跳过前置反思 |
| Plan-and-Execute Agent | LangGraph 四节点图（Planner → Executor → Replanner → Synthesizer）；先全局规划步骤，再按序执行，适合步骤有前后依赖的复杂分析 |
| Multi-Agent Supervisor | LangGraph StateGraph + `with_structured_output(SupervisorDecision)` + Send API 并行 |
| 并行工具执行 | `ParallelToolRunner`（asyncio.gather）+ `parallel_query` 元工具 |
| 工具结果缓存 | 按数据性质区分：目录数据（`retrieve_knowledge` 10min / `find_substitute_sku` 5min）可缓存；实时数据（库存 / 订单 / 履约方案）禁用缓存，每次取最新 |
| 工具弹性 | Circuit Breaker + `RunnableLambda.with_retry()` + ThreadPoolExecutor 超时 |
| 结构化输出 | 所有工具返回 `{status, data, summary}` JSON，回复提取 `FulfillmentDecision` |
| Prompt Injection 防护 | **双层防护**：Layer 1 `InputGuardrails` 拦截用户直接注入（指令覆盖 / 定界符 / 越狱 / 系统 Prompt 提取）；Layer 2 `ToolOutputSanitizer` 净化工具返回内容中的间接注入（订单备注/知识库文档被投毒），注入内容传给 LLM 前自动替换为占位符 |
| ReAct 思考轨迹 | `ReActCycle`（Think → Act → Observe）序列记录，完整可审计 |

### RAG 层
| 特性 | 实现方案 |
|---|---|
| 业务上下文 | 检索前分析订单库存，识别 `QueryIntent`，把缺货 SKU、履约状态、订单摘要拼入查询上下文 |
| 多查询扩展 | LLM Query Rewriter 和规则 query 合并去重，保留缺货、跨仓、优先级等业务关键词 |
| 混合召回 | LlamaIndex `QueryFusionRetriever`：BM25 处理精确术语，向量检索处理语义相近表达，使用 RRF 融合 |
| 业务重排 | `BusinessRulePostprocessor` 按 `category-intent` 对齐做轻量加权 |
| Reranker 精排 | 默认接入阿里云百炼 `qwen3-rerank`；可切换本地 `BAAI/bge-reranker-base` 或 Jina，异步路径用 `asyncio.to_thread()` 避免阻塞事件循环 |
| 上下文压缩 | `ContextualCompressor` 抽取与问题直接相关的句子，减少下游 prompt 噪声 |
| 异步检索 | `aretrieve()` 并发执行多个扩展 query，供 async API 调用链使用 |
| 文档摄取 | LlamaIndex `IngestionPipeline`（`SentenceSplitter` + `BusinessMetadataEnricher`），为 chunk 注入 `category / chunk_id / source_path` |
| 过滤与去重 | 向量检索使用 `MetadataFilters`；融合后再统一过滤，覆盖 BM25 召回结果 |
| 索引缓存 | 文档指纹包含知识文件、Embedding 配置、向量后端、chunk 参数 |
| 实体 Embedding | 默认通过模型网关使用阿里云百炼 `text-embedding-v4`（1024 维）；也可切回本地 `BAAI/bge-small-zh-v1.5` 或 OpenAI-compatible Embedding |
| Reranker | 默认使用阿里云百炼 `qwen3-rerank`；也可切回本地 `BAAI/bge-reranker-base` 或 Jina Reranker |
| 向量存储后端 | RAG 默认使用 Milvus；`VectorStoreFactory` 使用 `pymilvus` 管理连接、database、collection 加载，本地开发可临时切回 `VECTOR_STORE_TYPE=local` |
| 知识库热更新 | `POST /knowledge-mgmt/rebuild` 触发增量重建，无需重启服务 |

### Workflow 层
| 特性 | 实现方案 |
|---|---|
| 状态图 | LangGraph `StateGraph` + `Annotated[list, add]` reducer |
| 条件边 | `route_after_inventory` 分支（fulfillable / stockout） |
| 并行执行 | Stage-1 fan-out（订单 + 库存同时查询）+ Stage-2 Send API 条件并行 |
| HITL | `RiskEvaluator` 多规则引擎（CRITICAL/HIGH/MEDIUM/LOW）+ `interrupt()` + `Command(resume=...)` |
| 错误恢复 | `Command(goto="finalize")` 短路不可恢复错误，跳过中间节点 |
| 异步流式 | `graph.astream(stream_mode="updates")` + SSE，每节点完成即推送 |
| 超时保护 | `asyncio.timeout()` 覆盖 async generator（Python 3.11+） |
| 幂等性 | SHA-256 请求指纹，5 分钟内相同请求直接返回缓存 |

### Memory 层
| 特性 | 实现方案 |
|---|---|
| 短期记忆 | LangGraph checkpointer，优先 `RedisSaver(redis_url=...)`，支持 TTL；Redis 不可用自动降级 `MemorySaver` |
| 上下文窗口 | `ContextWindowManager` 使用 LangChain `trim_messages()` 按 token 裁剪，避免多轮对话无限膨胀 |
| 长期记忆 | 默认 `SQLiteLongTermMemoryStore` 本地落盘；配置 `LONG_TERM_MEMORY_BACKEND=mysql_milvus` 后使用 MySQL + Milvus |
| 记忆召回 | Agent 每轮执行前按 session/order namespace 检索相关长期记忆，并作为显式上下文注入当前问题 |
| 记忆沉淀 | Agent 每轮结束后写入会话摘要、工具调用、订单维度记录；显式偏好（如不接受替代 SKU、时效优先）单独高权重保存 |
| 检索排序 | SQLite 使用 namespace 前缀 + filter + TTL + 关键词相似度 + importance + access_count；MySQL + Milvus 使用语义相似度并融合 importance/access_count |
| 生产演进 | 保持 LangGraph `BaseStore` 接口不变，本地 SQLite、生产 MySQL + Milvus 可通过 `.env` 切换 |

### 工程层
| 特性 | 实现方案 |
|---|---|
| 模型网关 | 后端 `ModelGateway` 统一管理 Chat / Embedding / Reranker，覆盖云端大模型、云端小模型、本地开源模型；按 `model_type + use_case` 路由，前端只选择模型 ID，不接触 API Key；支持 `extra_body` 和 `streaming` 等厂商扩展参数 |
| Prompt 管理 | `PromptRegistry` 三层降级：**Langfuse Prompt Management**（在线，trace 自动关联版本）→ YAML 文件（本地热重载）→ 内置 hardcoded（兜底），所有 Agent Prompt 均已脱硬编码 |
| 结构化日志 | JSON 单行格式 + ContextVar trace_id + `TraceIdMiddleware` 全链路透传 |
| 可观测性 | Langfuse 链路追踪 + 结构化日志 + Prometheus `/metrics` 端点 |
| 评测闭环 | Langfuse（在线）+ Ragas（批量离线）+ DeepEval（CI 回归）|
| 限流保护 | 固定窗口计数器（Agent 20次/分钟，Workflow 10次/分钟） |
| 单元测试 | 覆盖 RAG / Workflow nodes / risk_evaluator / tools / guardrails（含双层注入防护），核心单测无 LLM 依赖 |

---

## 快速启动

### 环境要求
- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

```bash
git clone <repo-url>
cd multiship-agent
uv sync
cp .env.example .env
```

### 配置模型网关

```bash
# 模型清单在 config/model_gateway.yaml
MODEL_GATEWAY_CONFIG_PATH=config/model_gateway.yaml
DEFAULT_LLM_MODEL_ID=deepseek-v4-pro

# API Key 只留在后端环境变量里
LLM_API_KEY=sk-xxx
DASHSCOPE_API_KEY=sk-xxx

# 启用其他模型时，按 config/model_gateway.yaml 里的 api_key_env 配置
# OPENAI_API_KEY=sk-xxx
# QWEN_API_KEY=sk-xxx
# KIMI_API_KEY=sk-xxx
# JINA_API_KEY=jina_xxx
# ANTHROPIC_API_KEY=sk-ant-xxx
```

前端可通过 `GET /api/v1/models` 查看可选模型，通过 `GET /api/v1/models/active` 查看当前默认模型。接口只返回模型元数据，不返回密钥。调用 `/api/v1/agent/chat` 或 `/api/v1/agent/plan-execute` 时可传 `model_id`，后端会通过模型网关按用途构造对应模型。

`config/model_gateway.yaml` 支持按任务配置默认模型，例如：

```yaml
default_models:
  agent: deepseek-v4-pro              # 复杂 Agent 决策，强模型
  workflow: deepseek-v4-pro           # Workflow 结论生成，强模型
  plan_execute: deepseek-v4-pro       # 规划/重规划，强模型
  supervisor: deepseek-v4-pro         # Multi-Agent 调度，强模型
  judge: deepseek-v4-pro              # 评测，强模型
  workflow_finalize: deepseek-v4-flash
  structured_extract: deepseek-v4-flash
  rag_rewrite: deepseek-v4-flash
  rag_compress: deepseek-v4-flash
  embedding: aliyun-text-embedding-v4 # 阿里云百炼 text-embedding-v4，1024 维
  reranker: aliyun-qwen3-rerank       # 阿里云百炼 qwen3-rerank
```

模型类型通过 `model_type` 区分：`chat`、`embedding`、`reranker`。本地开源 Chat 模型可通过 vLLM / Ollama / LM Studio 暴露 OpenAI-compatible 接口，配置 `requires_api_key: false` 即可接入；本地 Embedding / Reranker 可直接使用 HuggingFace / sentence-transformers。生产环境可以把 YAML 替换成配置中心或数据库。

DeepSeek V4 系列默认关闭思考模式并开启流式输出：

```yaml
extra_body:
  thinking:
    type: disabled
streaming: true
```

这样可以避免 thinking mode 多轮对话中 `reasoning_content` 未回传导致 400，同时让 Agent 流式接口有更快首字响应。

> 不配置模型 Key 时服务仍可启动，Agent/Workflow 接口返回 503，其余接口正常。旧版 `LLM_PROVIDER/LLM_MODEL/LLM_BASE_URL` 仍作为本地 Demo fallback 保留。

### 可选：启动 MySQL + Milvus 长期记忆

默认长期记忆使用 SQLite，本地无需额外服务。生产推荐 `mysql_milvus`：MySQL 保存结构化记忆、TTL、importance、访问次数和审计记录，Milvus 只保存可重建的语义向量索引。

```bash
docker compose -f docker-compose.milvus.yml up -d
```

该 compose 会同时启动 MySQL 8.4 和 Milvus standalone。然后在 `.env` 中配置：

```bash
LONG_TERM_MEMORY_BACKEND=mysql_milvus
MYSQL_URL=mysql+pymysql://root:root@localhost:3306/multiship_agent
# 可选：长期记忆单独使用不同 MySQL 连接
# LONG_TERM_MEMORY_MYSQL_URL=mysql+pymysql://root:root@localhost:3306/multiship_agent
LONG_TERM_MEMORY_MILVUS_COLLECTION=long_term_memory_vectors_1024
LONG_TERM_MEMORY_VECTOR_DIMENSION=1024
MILVUS_URI=http://localhost:19530
```

`LONG_TERM_MEMORY_VECTOR_DIMENSION` 需要和模型网关里的 Embedding 模型维度一致。当前默认 `aliyun-text-embedding-v4` 是 1024 维；如果切回 `bge-small-zh`，需要改成 512，并建议换一个新的 `LONG_TERM_MEMORY_MILVUS_COLLECTION`。

### 启动 Milvus RAG 向量库

RAG 向量数据库默认使用 Milvus：

```bash
docker compose -f docker-compose.milvus.yml up -d
```

`.env` 保持：

```bash
VECTOR_STORE_TYPE=milvus
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=knowledge_base_1024
MILVUS_DIM=1024
MILVUS_UPSERT_MODE=true
```

`MILVUS_DIM` 要和 Embedding 维度一致：当前默认 `aliyun-text-embedding-v4` 是 1024，`bge-small-zh` 是 512。切换 Embedding 维度后建议换一个新的 `MILVUS_COLLECTION` 或设置 `MILVUS_OVERWRITE=true` 重建索引，否则旧 collection 会因为向量维度不同而写入失败。开发机没启动 Milvus 时，默认会回退本地向量存储；生产环境建议设置 `VECTOR_STORE_FALLBACK_TO_LOCAL=false`，避免静默降级。

### 启动后端

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# API 文档：http://localhost:8000/docs
# 指标监控：http://localhost:8000/metrics
```

`--reload` 会监听代码变化并自动重启，适合本地开发和面试演示。生产环境建议去掉 `--reload`，交给进程管理器或容器编排系统托管。

### 启动 MCP

本地 IDE / Claude Desktop / Cursor 调试用 stdio：

```bash
uv run python app/mcp/server.py
uv run python app/mcp/filesystem_server.py
```

企业内网共享 MCP 能力用 Streamable HTTP：

```bash
uv run python app/mcp/server.py --transport streamable-http --host 0.0.0.0 --port 9000
uv run python app/mcp/filesystem_server.py --transport streamable-http --host 0.0.0.0 --port 9001
```

默认 endpoint path 是 `/mcp`。生产部署建议放到内部网关后面，再统一加鉴权、限流和审计日志。

### 启动 React 前端（推荐）

当前主前端已升级为 React + TypeScript + Vite，面向运营人员提供订单处理台、人工审查队列和独立审查页。

```bash
cd frontend
npm install
npm run dev
# 前端地址：http://localhost:5173
```

开发环境会通过 Vite proxy 把 `/api` 转发到 `http://localhost:8000`。如果后端端口不同，在 `frontend/.env.local` 中配置：

```bash
VITE_API_TARGET=http://localhost:8001
VITE_API_BASE=/api/v1
```

生产部署：

```bash
cd frontend
npm run build
```

构建产物会输出到 `frontend/dist`。FastAPI 启动时如果发现该目录，会自动托管静态前端。

### 运行测试

```bash
# 单元测试（无 LLM，< 10s）
uv run pytest tests/unit/ -v

# Eval 测试（需要 LLM，Layer 1 无 LLM 断言）
uv run pytest tests/eval/ -m "not slow"

# 完整 DeepEval 评测（LLM-as-judge）
uv run pytest tests/eval/ --slow

# Ragas 批量离线评测
uv run python scripts/eval_ragas.py --limit 10
```

---

## API 端点速览

### Agent
| 方法 | 路径 | 描述 |
|---|---|---|
| POST | `/api/v1/agent/chat` | ReAct Agent 对话（async + Guardrails） |
| POST | `/api/v1/agent/chat/stream` | ReAct Agent 流式对话（NDJSON，实时返回 token / 工具调用 / 工具结果 / done） |
| POST | `/api/v1/agent/plan-execute` | Plan-and-Execute Agent（先规划后执行，适合步骤有依赖的复杂分析） |
| GET | `/api/v1/agent/cache/stats` | 工具缓存命中率统计 |
| DELETE | `/api/v1/agent/cache` | 手动失效缓存 |

### Workflow
| 方法 | 路径 | 描述 |
|---|---|---|
| POST | `/api/v1/workflow/run` | 同步执行（含幂等性，自动批准 HITL） |
| POST | `/api/v1/workflow/run/stream` | SSE 流式（每节点完成实时推送） |
| POST | `/api/v1/workflow/run/timeout` | 带超时保护的异步执行，超时返回 408 |

### 知识库管理
| 方法 | 路径 | 描述 |
|---|---|---|
| GET | `/api/v1/knowledge-mgmt/documents` | 列出所有知识文档 |
| GET | `/api/v1/knowledge-mgmt/documents/{id}` | 查看文档内容 |
| POST | `/api/v1/knowledge-mgmt/rebuild` | 后台异步重建索引（立即返回，BackgroundTasks 执行） |
| GET | `/api/v1/knowledge-mgmt/status` | 索引状态 + 后台重建进度查询 |

### 运维
| 方法 | 路径 | 描述 |
|---|---|---|
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/models` | 查看模型网关可选模型（不返回密钥） |
| GET | `/api/v1/models/active` | 查看某个 use_case 当前生效模型 |
| GET | `/metrics` | Prometheus 格式指标 |

---

## 项目结构

```
multiship-agent/
├── app/
│   ├── agent/                      # Agent 层
│   │   ├── agent.py                # ReAct + ReflectiveAgentRunner
│   │   ├── plan_execute.py         # Plan-and-Execute Agent（Planner/Executor/Replanner/Synthesizer）
│   │   ├── multi_agent.py          # Supervisor + 专家 Agent + Send API
│   │   ├── tools.py                # 6 个结构化 JSON 输出工具
│   │   ├── parallel_tools.py       # 并行工具执行（asyncio.gather）
│   │   ├── tool_wrapper.py         # 弹性层（缓存 + 熔断 + 重试）
│   │   ├── tool_cache.py           # 目录数据工具 TTL 缓存（规则/替代关系）
│   │   ├── context_manager.py      # Context window 管理
│   │   ├── guardrails.py           # 双层注入防护：InputGuardrails（直接注入）+ ToolOutputSanitizer（间接注入）
│   │   └── evaluation/
│   │       ├── langfuse_tracer.py  # Langfuse 链路追踪
│   │       ├── ragas_evaluator.py  # RAG 质量评测
│   │       └── golden_dataset.py   # 10 个标注测试用例
│   │
│   ├── graph/                      # Workflow 层（LangGraph）
│   │   ├── workflow.py             # 主工作流（StateGraph）
│   │   ├── parallel_workflow.py    # 并行工作流（Send API）
│   │   ├── nodes.py                # 5 节点（含 Command 错误恢复）
│   │   ├── state.py                # GraphState TypedDict
│   │   ├── router.py               # 条件边路由
│   │   ├── risk_evaluator.py       # HITL 多规则风险引擎
│   │   ├── llm_adapter.py          # LLMFactory 兼容入口，内部委托 ModelGateway
│   │   └── embed_adapter.py        # Embedding 工厂（模型网关优先，支持本地 / OpenAI-compatible）
│   │
│   ├── services/                   # 业务服务层
│   │   ├── agent_service.py        # Agent 外观（async chat + NDJSON 流式输出）
│   │   ├── workflow_service.py     # Workflow 外观（SSE + 超时 + 幂等）
│   │   ├── model_gateway.py        # Chat / Embedding / Reranker 模型网关
│   │   ├── hybrid_service.py       # 智能路由
│   ├── rag/                        # RAG 能力模块（检索规划、召回、重排、答案组装）
│   │   ├── knowledge_retrieval_service.py  # RAG 主服务（意图、混合召回、重排、摘要）
│   │   ├── rag_query_planner.py    # RAG 意图识别 + query 扩展
│   │   ├── rag_answer_builder.py   # RAG 答案摘要生成
│   │   └── reranker.py             # 本地 / Jina / DashScope Reranker + 上下文压缩
│   │
│   ├── core/
│   │   ├── config.py               # 统一配置（pydantic-settings）
│   │   ├── logging.py              # JSON 日志 + TraceIdMiddleware
│   │   ├── prompt_registry.py      # Prompt 三层降级注册表（Langfuse → YAML → 内置）
│   │   └── rate_limiter.py         # 固定窗口限流器
│   │
│   └── api/routes/                 # FastAPI 路由层
│
├── tests/
│   ├── unit/                       # 无 LLM 单元测试（覆盖 RAG / Workflow / Guardrails / Tools）
│   │   ├── test_guardrails.py
│   │   ├── test_knowledge_retrieval_service.py
│   │   ├── test_risk_evaluator.py
│   │   ├── test_tools_output.py
│   │   └── test_workflow_nodes.py
│   └── eval/                       # Eval 套件（DeepEval + Ragas）
│
├── prompts/                        # Prompt YAML 版本文件（Langfuse 未配置时的本地降级）
│   ├── fulfillment_agent_v1.yaml   # ReAct Agent system prompt
│   ├── supervisor_agent_v1.yaml    # Multi-Agent Supervisor prompt
│   └── synthesizer_agent_v1.yaml  # Synthesizer prompt
│
├── scripts/
│   └── eval_ragas.py               # Ragas 批量离线评测
│
└── docs/                           # 架构设计文档（开发过程记录）
```

---

## 技术栈

| 层次 | 技术选型 | 核心用途 |
|---|---|---|
| **Agent 框架** | LangChain 1.0+ | ReAct Agent、LCEL 链、Callback、工具 |
| **Workflow** | LangGraph 1.0+ | StateGraph、条件边、Send API 并行、HITL |
| **RAG** | LlamaIndex 0.12+ | 文档摄取、BM25、向量索引、QueryFusionRetriever；向量后端默认 Milvus，本地文件作为开发兜底 |
| **Embedding / Reranker** | 阿里云百炼 + sentence-transformers + Jina | 默认 `text-embedding-v4` + `qwen3-rerank`；可切本地 BGE 或云端 Jina |
| **API** | FastAPI + Uvicorn | 异步 HTTP、SSE / NDJSON 流式、自动 OpenAPI |
| **数据校验** | Pydantic v2 | Schema 验证、结构化 LLM 输出 |
| **配置** | pydantic-settings | 多 Provider 环境变量统一管理 |
| **可观测** | Langfuse 3.x | LLM 链路追踪、在线评分 |
| **评测** | Ragas 0.4 + DeepEval 3.x | RAG 质量指标 + CI 回归测试 |
| **指标** | prometheus-client | Prometheus 格式，接 Grafana 大盘 |
| **日志** | Python logging + JSON | 结构化日志 + 全链路 trace_id |

---

## 设计亮点

### 四种 Agent 执行模式

| 模式 | 路径 | 适合场景 |
|---|---|---|
| **Workflow** | `/workflow/run` | 固定业务流程，延迟最低，适合标准查询 |
| **ReAct Agent** | `/agent/chat`、`/agent/chat/stream` | 每步局部决策，适合动态单域问题；流式接口改善首字响应 |
| **Plan-and-Execute** | `/agent/plan-execute` | 先全局规划，再按序执行，适合步骤有依赖的复杂分析 |
| **Multi-Agent** | `/hybrid/process` | 并行专家协作，适合多维度综合分析 |

### 两阶段工具并行
- 粗粒度：Agent 主动调用 `parallel_query` 元工具，同时触发多个独立工具
- 细粒度：`ParallelToolRunner`（asyncio.gather）内部并发，节省 ~33% 延迟

### RAG 检索流水线
这一层主要解决履约规则检索的稳定性问题：
1. **业务上下文构造**：先分析订单库存，把缺货 SKU、履约状态、订单摘要写入检索上下文。
2. **多查询扩展**：LLM 改写补充问法，规则 query 保留固定业务词，两者合并去重。
3. **BM25 + 向量混合召回**：BM25 处理 SKU、规则名等精确词，向量检索处理语义表达，RRF 做融合。
4. **业务重排**：按 `QueryIntent` 给对应知识域小幅加权。
5. **精排和压缩**：使用 Reranker 对候选片段二次排序，再抽取与问题直接相关的句子。

RAG 返回 `intent`、`expanded_queries`、`matched_categories`、`score_detail`、`source_path` 和 `answer_summary`，方便 Agent 或 Workflow 继续使用。

### HITL 分级审批
`RiskEvaluator` 按业务规则评估风险等级，只有 HIGH/CRITICAL 才触发人工审批：
- CRITICAL：VIP + 50 万 + 2h 截止，5 分钟超时自动上报
- HIGH：高价值订单 / VIP 临近截止 / 严重缺货，30 分钟人工审批
- MEDIUM：拆单 / 冷链 / 跨区，自动放行 + 打标签
- LOW：常规订单，全自动通过

### 可观测评测闭环
```
生产请求 → Langfuse 链路追踪 → tool_grounding 自动评分
         → Ragas 批量离线评测 → faithfulness / answer_relevancy
         → DeepEval CI 回归  → 每次 PR 自动运行，防止质量退化
```
---

## 企业数据接入

项目现在不再只能查询内置 RAG 文档和 demo 订单。后台新增了企业数据接入层：

- `EnterpriseDataRepository`：保存企业导入的订单和库存快照，默认落盘到 `storage/enterprise_data`。
- `CompositeOrderRepository` / `CompositeInventoryRepository`：查询时先读企业数据，企业数据没有命中时再回退到原来的 demo 数据。
- Agent、Workflow、订单分析、库存分析、RAG 检索上下文复用同一套共享服务，所以后台导入的数据会自然进入业务分析链路。

配置项：

```bash
ENTERPRISE_DATA_DIR=storage/enterprise_data
```

前端导入：

1. 启动 FastAPI 和 React 前端后，打开 `http://localhost:5173`。
2. 进入订单处理台或系统状态页，确认企业数据统计可见。
3. 上传已经清洗好的标准 CSV / JSON 文件，预览无误后点击确认导入。
4. 如需覆盖同一来源的旧数据，勾选“覆盖该来源已有数据”。

订单 CSV 字段：

```csv
order_id,platform,order_time,order_status,region,priority,sku_id,product_name,quantity,unit_price
SO-ENT-001,ERP,2026-05-11T09:30:00,待履约,华东-上海,高,SKU-ENT-001,企业导入商品,3,99.0
```

库存 CSV 字段：

```csv
warehouse_id,warehouse_name,region,sku_id,available_stock,locked_stock,updated_at
WH-ENT-001,企业上海仓,华东-上海,SKU-ENT-001,8,1,2026-05-11T10:00:00
```

JSON 可以直接上传数组，也可以上传与接口一致的对象，例如 `{"source_id":"erp-main","orders":[...]}` 或 `{"source_id":"erp-main","records":[...]}`。

常用接口：

```bash
# 1. 登记企业数据源
curl -X POST http://localhost:8000/api/v1/enterprise-data/sources \
  -H "Content-Type: application/json" \
  -d '{"source_id":"erp-main","name":"ERP 主数据源","source_type":"api","config":{"base_url":"https://erp.example.com"}}'

# 2. 导入企业订单
curl -X POST http://localhost:8000/api/v1/enterprise-data/orders/import \
  -H "Content-Type: application/json" \
  -d '{"source_id":"erp-main","orders":[{"order_id":"SO-ENT-001","platform":"ERP","order_time":"2026-05-11T09:30:00","order_status":"待履约","region":"华东-上海","priority":"高","items":[{"sku_id":"SKU-ENT-001","product_name":"企业导入商品","quantity":3,"unit_price":99.0}]}]}'

# 3. 导入企业库存
curl -X POST http://localhost:8000/api/v1/enterprise-data/inventory/import \
  -H "Content-Type: application/json" \
  -d '{"source_id":"erp-main","records":[{"warehouse_id":"WH-ENT-001","warehouse_name":"企业上海仓","region":"华东-上海","sku_id":"SKU-ENT-001","available_stock":8,"locked_stock":1,"updated_at":"2026-05-11T10:00:00"}]}'

# 4. 导入后，原有分析接口会优先使用企业数据
curl -X POST http://localhost:8000/api/v1/inventory/analyze \
  -H "Content-Type: application/json" \
  -d '{"order_id":"SO-ENT-001"}'
```

设计上这里故意没有把企业订单、库存塞进 RAG，因为订单和库存是强结构化、实时性更高的业务数据；RAG 更适合承载履约规则、售后政策、仓配说明这类非结构化知识。这个边界清楚，后续接 MySQL、ERP API、消息队列同步任务都会更自然。
