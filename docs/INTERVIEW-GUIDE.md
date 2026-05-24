# 面试讲解指南

## 1. 项目一句话

这是一个全域电商供应链履约智能体项目，用 Workflow 保证关键链路可控，用 Agent 处理开放式问题，用 RAG 接入企业规则知识，用模型网关实现多模型分层调用。

## 2. 推荐讲解顺序

1. 先讲业务：订单履约、缺货、跨仓调拨、售后规则。
2. 再讲架构：前端、FastAPI、Service、Agent / Workflow / RAG、数据层。
3. 再讲 RAG：知识库放规则和 SOP，不放实时订单库存。
4. 再讲 Agent：工具调用、反思、Guardrails、流式输出。
5. 再讲企业级：模型网关、Milvus、MySQL 长期记忆、Langfuse、评测。

## 3. 面试官可能追问

### 为什么同时有 Workflow 和 Agent？

因为企业关键链路需要可控。固定履约流程用 Workflow 更稳定，开放问题用 Agent 更灵活。

### 为什么订单库存不直接放 RAG？

订单库存是高频变化的结构化数据，应该查数据库或业务服务。RAG 更适合制度、规则、SOP 这种知识。

### 为什么需要 Reranker？

向量召回负责“找可能相关”，Reranker 负责“重新判断最相关”。它能提升 Top-K 结果质量，减少答案引用错误文档。

### 为什么用模型网关？

企业会同时用云端大模型、小模型、Embedding、Reranker 和本地模型。模型网关让业务代码按用途调用，底层模型可以配置替换。

### 为什么长期记忆用 MySQL + Milvus？

长期记忆既要保存结构化元数据，又要支持语义检索。MySQL 保存 namespace、TTL、importance、访问次数和审计记录；Milvus 保存可重建的语义向量索引。这样结构化业务数据和向量检索职责分离，更贴近电商 OMS/WMS 系统的生产架构。

### 如何证明项目不是简单套壳？

可以展示：

- RAG 有 Query Planner、混合召回、Reranker、Answer Builder。
- Agent 有工具层、Guardrails、上下文管理、反思触发。
- Workflow 有固定节点和业务规则。
- 模型通过网关分层，不直接写死。
- Langfuse 和评测可以观察质量。

## 4. 代码阅读优先级

如果时间有限，不需要挨个看每一行。建议按这条路线看：

1. `app/main.py`
2. `app/api/routes/agent.py`
3. `app/agents/runtime/agent_service.py`
4. `app/agents/orchestration/react_agent.py`
5. `app/rag/knowledge_retrieval_service.py`
6. `app/rag/rag_query_planner.py`
7. `app/rag/reranker.py`
8. `app/services/model_gateway.py`
9. `app/rag/vector_store_factory.py`
10. `frontend/`

先掌握主链路，再看细节实现。
