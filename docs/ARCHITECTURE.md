# 系统架构

## 1. 架构目标

Multiship Agent 的目标是把供应链履约场景拆成三条可对比链路：

- 固定 Workflow：适合规则稳定、路径清晰、需要可解释的履约决策。
- ReAct Agent：适合开放式问题，由模型自主选择工具并生成分析。
- RAG：适合从企业制度、售后规则、调拨规则和知识文档中检索依据。

项目不是把所有逻辑都塞进大模型，而是让结构化业务数据、规则引擎、知识库和模型各做自己擅长的事。

## 2. 分层结构

```text
React / TypeScript 前端
  |
FastAPI API 层
  |
Service 应用服务层
  |
Agent / Workflow / RAG / Memory
  |
Repository / Vector Store / 企业数据
  |
模型网关 / Milvus / MySQL / Redis / Langfuse
```

## 3. 主要目录

| 路径 | 职责 |
| --- | --- |
| `app/api/` | FastAPI 路由，负责请求入口和响应模型 |
| `app/services/` | 业务服务层，放订单分析、库存分析、RAG、Agent 服务 |
| `app/agents/` | Agent 编排、运行时、工具、安全防护、质量评测与可观测 |
| `app/workflows/` | 履约 Workflow、状态、节点、路由、风险判断和 Trace |
| `app/infrastructure/llm/` | ChatModel 与 Embedding 基础设施适配 |
| `app/repositories/` | 企业数据、知识库、向量库访问 |
| `app/memory/` | 短期记忆、长期记忆、检查点 |
| `config/` | 模型网关、运行时配置 |
| `app/data/` | 演示订单、库存、知识库文档 |
| `tests/` | 单元测试和评测相关测试 |
| `frontend/` | React + TypeScript 运营操作台、人工审查页面和生产构建配置 |

## 4. 三条核心链路

### Workflow 链路

1. 前端提交订单号和业务问题。
2. 后端读取订单、库存、业务规则。
3. LangGraph 按固定节点执行订单分析、库存分析、履约方案生成和风险判断。
4. 返回结构化决策结果、命中规则和建议动作。

Workflow 的优势是稳定、可解释、容易测试。

### Agent 链路

1. 前端提交自然语言问题和会话 ID。
2. Agent 根据问题选择工具：查订单、查库存、查知识库、生成履约方案等。
3. 工具结果回到模型，由模型整理答案。
4. 支持流式输出，让前端更快看到首字响应。

Agent 的优势是灵活，适合复杂追问和跨工具分析。

### RAG 链路

1. Query Planner 判断意图并扩展查询。
2. 检索服务执行向量检索、关键词检索、融合和过滤。
3. Reranker 对候选文档重新排序。
4. Answer Builder 生成有依据的摘要答案。

RAG 的优势是让模型回答有企业知识依据，而不是凭空生成。

## 5. 企业级改造点

当前项目已经从简单 Demo 向企业级结构靠近：

- 模型网关：按用途区分大模型、小模型、Embedding、Reranker。
- RAG 向量库：Milvus，Embedding 维度统一为 1024。
- 长期记忆：MySQL + Milvus。
- 短期记忆：优先 Redis，失败后降级 MemorySaver。
- 可观测：Langfuse + 结构化日志。
- Agent 防护：Prompt Injection 检测、工具输出隔离、PII 脱敏。
- 前端体验：React 操作台支持订单处理、人工审查队列、独立审查页和系统状态查看。
