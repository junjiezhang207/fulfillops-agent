# 可观测与评测

## 1. 为什么不只靠日志

普通日志只能看到接口是否报错，但看不到一次 Agent 调用了哪些工具、RAG 命中了哪些文档、答案质量是否稳定。

所以项目同时使用：

- 结构化日志：定位后端错误和运行状态。
- Langfuse：观察 LLM 调用、Trace、工具调用链和评分。
- 单元测试：保证核心逻辑不被改坏。
- RAG / Agent 评测：观察答案质量。

## 2. 当前核心文件

| 文件 | 职责 |
| --- | --- |
| `app/agent/evaluation/langfuse_tracer.py` | Langfuse Trace 和评分上报 |
| `app/core/logging.py` | 结构化日志和 trace_id |
| `app/api/routes/metrics.py` | Prometheus `/metrics` 指标 |
| `app/agent/evaluation/ragas_evaluator.py` | RAG 相关评测 |
| `app/agent/evaluation/golden_dataset.py` | Golden Dataset 样例 |
| `tests/` | 单元测试 |

## 3. Langfuse 适合看什么

- 单次请求完整链路。
- LLM 输入输出。
- 工具调用次数和耗时。
- RAG 命中文档。
- 人工或自动评分。

## 4. 评分原则

评分别追求“看起来很细”，而是要能反映质量。

推荐关注：

- 答案是否命中业务规则。
- 是否引用了正确知识依据。
- 是否使用了正确工具。
- 是否出现幻觉。
- 是否违反安全规则。
- 是否在可接受延迟内完成。

## 5. 面试讲法

可以这样说：

> 我没有只做接口日志，而是把 LLM 调用、工具调用、RAG 命中和评分都纳入可观测。这样面试官追问“怎么证明 Agent 有效”时，可以回答：看 Trace、看 Golden Dataset、看工具命中和 RAG 引用，而不是只看一次演示结果。
