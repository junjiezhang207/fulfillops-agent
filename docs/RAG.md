# RAG 知识检索

## 1. 知识库作用

RAG 负责把企业知识接入履约决策，避免模型只凭参数记忆回答。

适合放入知识库的内容：

- 履约规则：缺货、拆单、跨仓调拨、优先级策略。
- 售后规则：取消、退款、补发、异常处理。
- 运营制度：大促优先级、VIP 客户策略、区域限制。
- SOP 文档：客服处理流程、仓库操作流程。
- 风险规则：超时、库存不足、异常订单识别。

不适合放入知识库的内容：

- 实时订单状态。
- 实时库存数量。
- 价格、运费、促销活动等高频变化数据。

这些结构化数据应该走数据库或业务服务，而不是塞进 RAG。

## 2. 当前核心文件

| 文件 | 职责 |
| --- | --- |
| `app/rag/knowledge_retrieval_service.py` | RAG 主服务，负责索引、召回、融合、重排和结果组装 |
| `app/rag/rag_query_planner.py` | 意图识别、查询扩展、过滤条件生成 |
| `app/rag/rag_answer_builder.py` | 根据命中文档生成摘要答案 |
| `app/rag/reranker.py` | Reranker 接入和候选文档精排 |
| `app/repositories/knowledge_repository.py` | 知识库文档读取和管理 |
| `app/rag/vector_store_factory.py` | Milvus / 本地向量库创建 |
| `app/infrastructure/llm/embedding_adapter.py` | Embedding 模型适配 |
| `app/schemas/knowledge.py` | RAG 请求和响应模型 |

## 3. 检索流程

```text
用户问题
  |
Query Planner
  |-- 识别意图
  |-- 扩展查询
  |-- 生成类别过滤
  |
Hybrid Retrieval
  |-- 向量召回
  |-- 关键词召回
  |-- RRF 融合
  |
Reranker
  |-- 对候选片段重新排序
  |
Answer Builder
  |-- 生成摘要
  |-- 返回命中片段和依据
```

## 4. 向量库

生产级 RAG 使用 Milvus。

当前配置：

```env
VECTOR_STORE_TYPE=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION=knowledge_base_1024
MILVUS_DIM=1024
```

Embedding 使用阿里云 `text-embedding-v4`，维度是 1024。因此 Milvus collection 也必须是 1024 维。

如果从旧的 256 / 512 维模型切换过来，不要复用旧 collection，建议新建 collection，例如 `knowledge_base_1024`。

## 5. 降级策略

开发环境下，如果 Milvus 不可用，并且：

```env
VECTOR_STORE_FALLBACK_TO_LOCAL=true
```

系统会降级为本地索引，保证前端演示和单元测试还能跑。

如果企业生产环境使用，建议关闭自动降级，让问题尽早暴露：

```env
VECTOR_STORE_FALLBACK_TO_LOCAL=false
```

## 6. RAG 和订单库存的边界

RAG 只回答“规则依据是什么”。订单和库存仍然由结构化服务读取。

例如：

- “SO202502140002 当前库存够不够？”应该查订单和库存服务。
- “缺货时是否允许跨仓调拨？”应该查 RAG 知识库。
- “这个订单缺货时怎么处理？”应该结合订单、库存和 RAG。

这个边界很重要，面试时可以强调：RAG 不是数据库替代品。

