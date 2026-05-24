# Multiship Agent 项目文档

这套文档描述当前项目的真实实现，而不是早期阶段规划。

Multiship Agent 是一个面向全域电商供应链履约的智能决策演示系统。项目用 FastAPI 提供后端能力，用 React + TypeScript + Vite 提供前端操作台，核心能力包括固定 Workflow、ReAct Agent、RAG 知识检索、模型网关、企业数据导入、长期/短期记忆、Langfuse 可观测和评测体系。

## 文档入口

| 文档 | 内容 |
| --- | --- |
| [SETUP.md](SETUP.md) | 本地启动、环境变量、Milvus、MySQL、常见问题 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 当前系统架构、模块边界、调用链路 |
| [MODEL-GATEWAY.md](MODEL-GATEWAY.md) | 多模型网关、DeepSeek、阿里云 Embedding / Reranker 配置 |
| [RAG.md](RAG.md) | 知识库内容、索引、检索、重排、答案生成 |
| [AGENT-WORKFLOW.md](AGENT-WORKFLOW.md) | Workflow、Agent、工具、流式输出和反思机制 |
| [DATA-AND-MEMORY.md](DATA-AND-MEMORY.md) | 企业数据、短期记忆、长期记忆、MySQL + Milvus |
| [OBSERVABILITY-EVAL.md](OBSERVABILITY-EVAL.md) | Langfuse、日志、评测、面试可讲指标 |
| [INTERVIEW-GUIDE.md](INTERVIEW-GUIDE.md) | 面试时推荐讲法和容易被追问的点 |

## 当前主链路

1. 前端在 React 订单处理台中发起履约分析，并在高风险时进入人工审查页。
2. FastAPI 接收请求，完成参数校验、会话管理和业务编排。
3. 模型网关根据用途选择大模型、小模型、Embedding 或 Reranker。
4. RAG 使用知识库文档和 Milvus / 本地索引完成召回、融合、精排和摘要。
5. Agent 使用工具层查询订单、库存、知识库和履约建议，并支持流式输出。
6. 关键调用进入日志、Langfuse Trace、评测数据和会话记忆。

## 项目定位

这个项目不是单纯聊天机器人，而是一个“供应链履约决策智能体”：

- Workflow 适合路径稳定、可解释、可控的履约判断。
- Agent 适合问题更开放、需要自主选择工具的分析。
- RAG 负责把企业规则、售后政策、库存调拨规则等知识接入回答。
- 模型网关负责把不同模型按成本、速度、能力分层使用。
