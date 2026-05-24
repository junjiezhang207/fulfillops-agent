# 模型网关

## 1. 为什么需要模型网关

企业项目不会只接一个模型。真实使用中通常会按任务分层：

- 大模型：复杂推理、最终答案、监督判断。
- 小模型：意图识别、查询改写、分类、摘要草稿。
- Embedding：知识库向量化。
- Reranker：候选文档精排。

模型网关把这些模型统一配置在 `config/model_gateway.yaml`，业务代码只按用途调用，不直接绑定某个厂商。

## 2. 配置文件职责

| 文件 | 职责 |
| --- | --- |
| `.env` | 保存 API Key、基础开关、连接地址 |
| `config/model_gateway.yaml` | 保存模型列表、用途、能力、路由、是否启用 |
| `app/infrastructure/llm/model_gateway.py` | 读取配置并创建 LangChain ChatModel |
| `app/infrastructure/llm/embedding_adapter.py` | 创建 Embedding 模型 |
| `app/rag/reranker.py` | 创建 Reranker 模型 |

## 3. 当前模型分层

| 用途 | 当前推荐模型 |
| --- | --- |
| Agent / Workflow 复杂推理 | `deepseek-v4-pro` |
| 快速任务 / 低成本任务 | `deepseek-v4-flash` |
| Embedding | 阿里云百炼 `text-embedding-v4`，1024 维 |
| Reranker | 阿里云百炼 `qwen3-rerank` |

## 4. DeepSeek 配置

`.env` 示例：

```env
LLM_PROVIDER=openai_compatible
LLM_MODEL=deepseek-v4-pro
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-你的DeepSeekKey
```

`config/model_gateway.yaml` 中应存在两个模型：

```yaml
models:
  deepseek-v4-pro:
    enabled: true
    provider: openai_compatible
    model: deepseek-v4-pro
    base_url: https://api.deepseek.com/v1
    api_key_env: LLM_API_KEY
    tier: strong
    location: cloud
    streaming: true
    extra_body:
      thinking:
        type: disabled

  deepseek-v4-flash:
    enabled: true
    provider: openai_compatible
    model: deepseek-v4-flash
    base_url: https://api.deepseek.com/v1
    api_key_env: LLM_API_KEY
    tier: fast
    location: cloud
    streaming: true
    extra_body:
      thinking:
        type: disabled
```

`thinking.type: disabled` 用来关闭 DeepSeek 思考模式，避免接口要求回传 `reasoning_content`。

## 5. 阿里云 Embedding 配置

`.env`：

```env
DASHSCOPE_API_KEY=sk-你的阿里云百炼Key
```

`config/model_gateway.yaml`：

```yaml
embedding:
  default: aliyun-text-embedding-v4
  models:
    aliyun-text-embedding-v4:
      enabled: true
      provider: openai_compatible
      model: text-embedding-v4
      base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
      api_key_env: DASHSCOPE_API_KEY
      dimension: 1024
```

项目使用自定义 OpenAI-compatible Embedding 适配器调用阿里云 `/embeddings` 接口，避免 LlamaIndex 内置枚举不认识 `text-embedding-v4`。

## 6. 阿里云 Reranker 配置

`config/model_gateway.yaml`：

```yaml
reranker:
  default: aliyun-qwen3-rerank
  models:
    aliyun-qwen3-rerank:
      enabled: true
      provider: dashscope
      model: qwen3-rerank
      base_url: https://dashscope.aliyuncs.com/compatible-api/v1/reranks
      api_key_env: DASHSCOPE_API_KEY
```

RAG 检索会先召回候选文档，再用 Reranker 对候选文档按语义相关性重新排序。

## 7. 面试讲法

可以这样解释：

> 我没有把模型写死在业务代码里，而是做了模型网关。业务只声明用途，比如 agent、embedding、reranker。实际用 DeepSeek、阿里云还是本地模型，由配置决定。这样方便降本、灰度、替换厂商，也更符合企业里多模型并存的情况。

