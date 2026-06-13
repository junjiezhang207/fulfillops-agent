"""Prometheus 指标端点。

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

业务 API 返回 JSON，但 Prometheus 需要标准文本格式，所以这里直接返回 ``Response``。
指标用于观察请求量、耗时、工具失败、缓存命中、token 消耗和质量分数。
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

tool_avg_latency_ms = Gauge(
    "tool_avg_latency_ms",
    "工具平均调用耗时（毫秒）",
    ["tool_name"],
)

tool_max_latency_ms = Gauge(
    "tool_max_latency_ms",
    "工具最大调用耗时（毫秒）",
    ["tool_name"],
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

business_request_total = Counter(
    "business_request_total",
    "按路由和状态统计的业务决策请求数",
    ["route", "status"],
)
business_request_duration_seconds = Histogram(
    "business_request_duration_seconds",
    "业务请求端到端耗时（秒）",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)
business_trace_step_duration_seconds = Histogram(
    "business_trace_step_duration_seconds",
    "业务 trace 步骤耗时（秒）",
    ["step_type", "name", "status"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)
tool_call_total = Counter(
    "tool_call_total",
    "业务 trace 记录到的工具调用数",
    ["tool_name", "status"],
)
rag_retrieval_total = Counter(
    "rag_retrieval_total",
    "业务 trace 记录到的 RAG 检索次数",
    ["status"],
)
rag_empty_result_total = Counter("rag_empty_result_total", "没有可用证据的 RAG 检索次数")
rag_index_rebuild_total = Counter("rag_index_rebuild_total", "RAG 索引重建次数", ["status"])
rag_index_rebuild_duration_seconds = Histogram(
    "rag_index_rebuild_duration_seconds",
    "RAG 索引重建耗时（秒）",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
)
model_gateway_calls_total = Counter("model_gateway_calls_total", "模型网关调用次数", ["provider", "model", "status"])
model_gateway_fallback_total = Counter("model_gateway_fallback_total", "模型网关降级次数", ["from_provider", "to_provider"])
hitl_trigger_total = Counter("hitl_trigger_total", "HITL 人工介入触发次数", ["status"])
workflow_interrupted_total = Counter("workflow_interrupted_total", "Workflow 中断次数", ["workflow_name"])
memory_write_total = Counter("memory_write_total", "记忆写入次数", ["backend", "status"])
guardrail_block_total = Counter("guardrail_block_total", "安全护栏拦截次数", ["rule"])
prompt_injection_detected_total = Counter("prompt_injection_detected_total", "Prompt injection 命中次数", ["source"])
data_freshness_lag_seconds = Gauge("data_freshness_lag_seconds", "数据新鲜度延迟（秒）", ["source"])
llm_cost_total = Counter("llm_cost_total", "LLM 累计成本", ["provider", "model"])


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

        from app.agents.tools.telemetry import get_tool_telemetry

        for tool_name, metric in get_tool_telemetry().snapshot.items():
            tool_calls_total.labels(tool_name=tool_name, status="success")._value.set(metric["success"])
            tool_calls_total.labels(tool_name=tool_name, status="error")._value.set(metric["error"])
            tool_calls_total.labels(tool_name=tool_name, status="timeout")._value.set(metric["timeout"])
            tool_calls_total.labels(tool_name=tool_name, status="circuit_open")._value.set(metric["circuit_open"])
            tool_calls_total.labels(tool_name=tool_name, status="permission_denied")._value.set(metric["permission_denied"])
            tool_avg_latency_ms.labels(tool_name=tool_name).set(metric["avg_latency_ms"])
            tool_max_latency_ms.labels(tool_name=tool_name).set(metric["max_latency_ms"])
    except Exception:
        pass

    try:
        from app.observability.business_trace import get_observability_metrics_snapshot

        snapshot = get_observability_metrics_snapshot()
        # 自研 trace 的实时计数先保存在进程内快照里。这里同步到 prometheus_client，
        # 好处是业务模块不需要直接 import 指标对象，后续替换为队列/数据库也更容易。
        for (route, status), value in snapshot.get("business_request_total", {}).items():
            business_request_total.labels(route=route, status=status)._value.set(value)
        for (tool_name, status), value in snapshot.get("tool_call_total", {}).items():
            tool_call_total.labels(tool_name=tool_name, status=status)._value.set(value)
        for (status,), value in snapshot.get("rag_retrieval_total", {}).items():
            rag_retrieval_total.labels(status=status)._value.set(value)
        rag_empty_result_total._value.set(snapshot.get("rag_empty_result_total", 0))
        for (workflow_name,), value in snapshot.get("workflow_interrupted_total", {}).items():
            workflow_interrupted_total.labels(workflow_name=workflow_name)._value.set(value)
        for (status,), value in snapshot.get("hitl_trigger_total", {}).items():
            hitl_trigger_total.labels(status=status)._value.set(value)
        for (rule,), value in snapshot.get("guardrail_block_total", {}).items():
            guardrail_block_total.labels(rule=rule)._value.set(value)
    except Exception:
        pass

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
