"""Prometheus 指标端点 — 暴露 AI 业务指标供监控大盘使用。

端点：GET /metrics（Prometheus 标准格式）

指标分类：
  agent_*    — Agent 对话质量和性能
  rag_*      — RAG 检索质量
  tool_*     — 工具调用成功率
  workflow_* — Workflow 执行统计
  cache_*    — 工具缓存命中率

使用方式：
  Prometheus 抓取配置：
    scrape_configs:
      - job_name: multiship_agent
        static_configs:
          - targets: ['localhost:8000']
        metrics_path: /metrics

  Grafana 大盘推荐看板：
    - agent_llm_tokens_total / agent_request_duration_seconds
    - rag_cache_hit_ratio（Ragas faithfulness 趋势）
    - tool_success_rate（按工具名分组）

学习重点：
1. 业务 API 返回 JSON，但 Prometheus 需要标准文本格式，所以这里直接返回 ``Response``。
2. Counter 只能递增，适合请求数、token 数、错误数。
3. Gauge 可以上下变化，适合当前缓存命中率、队列长度、在线会话数。
4. Histogram 适合耗时分布，比如 P50/P95/P99 延迟。

面试官可能问：为什么 Agent 项目要做指标？
回答：LLM 应用上线后最怕“慢、贵、不稳定”。指标可以持续观察请求量、耗时、工具失败、
缓存命中、token 消耗和质量分数，帮助定位是模型慢、工具慢、RAG 差，还是限流/缓存策略有问题。
"""

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

router = APIRouter()

# ── Agent 指标 ────────────────────────────────────────────────────────────────
# 这组指标关注“用户和 Agent 对话”的整体情况：请求是否成功、用了多少 token、
# 端到端耗时如何、反思质量分是否稳定。

agent_requests_total = Counter(
    "agent_requests_total",
    "Agent 接口总请求数",
    ["status"],  # success / error / rate_limited
)

agent_llm_tokens_total = Counter(
    "agent_llm_tokens_total",
    "Agent 累计消耗 token 数",
    ["type"],  # prompt / completion
)

agent_llm_cost_usd_total = Counter(
    "agent_llm_cost_usd_total",
    "Agent 累计 LLM 费用（美元）",
)

agent_request_duration_seconds = Histogram(
    "agent_request_duration_seconds",
    "Agent 单次请求端到端耗时（秒）",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

agent_reflection_score = Histogram(
    "agent_reflection_score",
    "Agent 自反思质量分分布",
    buckets=[0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── 工具指标 ──────────────────────────────────────────────────────────────────
# Agent 的真实能力很大程度来自工具。如果工具失败率高，LLM 回答就容易变差。
# 所以工具调用次数、失败类型、缓存命中率都应该单独观测。

tool_calls_total = Counter(
    "tool_calls_total",
    "工具调用总次数",
    ["tool_name", "status"],  # status: success / error / timeout / circuit_open
)

tool_cache_hits_total = Counter(
    "tool_cache_hits_total",
    "工具缓存命中总次数",
)

tool_cache_misses_total = Counter(
    "tool_cache_misses_total",
    "工具缓存未命中总次数",
)

tool_cache_hit_ratio = Gauge(
    "tool_cache_hit_ratio",
    "工具缓存当前命中率（0-1）",
)

# ── Workflow 指标 ─────────────────────────────────────────────────────────────
# Workflow 是可审计链路，指标重点是完成/中断/超时/错误，以及是否触发 HITL。

workflow_executions_total = Counter(
    "workflow_executions_total",
    "Workflow 执行总次数",
    ["status"],  # completed / interrupted / timeout / error
)

workflow_duration_seconds = Histogram(
    "workflow_duration_seconds",
    "Workflow 执行耗时（秒）",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0],
)

workflow_hitl_total = Counter(
    "workflow_hitl_total",
    "HITL 人工审批触发总次数",
    ["decision"],  # approved / rejected / escalate
)

# ── RAG 指标 ──────────────────────────────────────────────────────────────────
# RAG 指标用于判断知识检索是否真的被使用，以及 query rewrite 是否过度膨胀。

rag_retrievals_total = Counter(
    "rag_retrievals_total",
    "RAG 检索总次数",
)

rag_rewrite_variants_histogram = Histogram(
    "rag_rewrite_query_count",
    "每次检索生成的改写查询数",
    buckets=[1, 2, 3, 4, 5],
)


# ── 端点 ──────────────────────────────────────────────────────────────────────

@router.get("/metrics")
def metrics_endpoint() -> Response:
    """暴露 Prometheus 格式指标。

    Prometheus 配置示例：
        scrape_configs:
          - job_name: multiship_agent
            static_configs:
              - targets: ['localhost:8000']
            metrics_path: /metrics
    """
    # 同步更新工具缓存命中率（Gauge 需要手动刷新）
    try:
        from app.agents.tools.cache import get_tool_cache

        # 注意这里访问了 prometheus_client 的内部 _value，是为了把缓存对象里的累计统计
        # 同步到 Counter。更严格的生产写法可以在缓存命中/未命中发生时直接 inc()。
        stats = get_tool_cache().stats
        total = stats["total_hits"] + stats["total_misses"]
        if total > 0:
            tool_cache_hit_ratio.set(stats["hit_rate"])
        tool_cache_hits_total._value.set(stats["total_hits"])
        tool_cache_misses_total._value.set(stats["total_misses"])
    except Exception:
        pass

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
