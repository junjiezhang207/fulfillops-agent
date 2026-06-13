"""自研业务可观测系统的轻量回归测试。

这些测试不追求跑完整模型链路，而是锁住企业监控最容易退化的契约：
trace_id 能透传、工具/RAG 会自动写 step、审计事件会脱敏、Prometheus 能暴露指标。
"""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.tools import StructuredTool

from app.agents.tools.wrapper import wrap_tool_with_resilience
from app.api.routes.metrics import router as metrics_router
from app.observability.audit_log import record_audit_event
from app.observability.business_trace import (
    finish_trace,
    reset_trace,
    snapshot_current_trace,
    start_trace,
)
from app.observability.trace_context import BusinessTraceMiddleware
from app.observability.trace_store import SQLiteTraceStore
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService


def test_trace_context_header_is_generated_and_returned():
    app = FastAPI()
    app.add_middleware(BusinessTraceMiddleware)

    @app.get("/demo")
    def demo():
        return {"ok": True}

    response = TestClient(app).get("/demo", headers={"X-Trace-Id": "trace-from-client"})

    assert response.status_code == 200
    assert response.headers["X-Trace-Id"] == "trace-from-client"


def test_observability_routes_are_not_business_traced():
    app = FastAPI()
    app.add_middleware(BusinessTraceMiddleware)

    @app.get("/api/v1/observability/traces")
    def trace_center():
        return {"trace": snapshot_current_trace()}

    response = TestClient(app).get(
        "/api/v1/observability/traces",
        headers={"X-Trace-Id": "trace-center-query"},
    )

    assert response.status_code == 200
    assert response.json()["trace"] is None


def test_trace_store_preserves_empty_evidence_and_filters_internal_routes(tmp_path):
    store = SQLiteTraceStore(tmp_path / "traces.db")
    store.save_trace(
        {
            "trace_id": "trace-center-query",
            "request_id": "trace-center-query",
            "route": "/api/v1/observability/traces",
            "status": "success",
            "steps": [],
        }
    )
    store.save_trace(
        {
            "trace_id": "business-trace",
            "request_id": "business-trace",
            "order_id": "SO-1",
            "route": "hybrid:multi_agent",
            "status": "completed",
            "steps": [
                {
                    "id": "step-1",
                    "type": "router",
                    "name": "hybrid_router",
                    "status": "success",
                    "evidence": [],
                }
            ],
        }
    )

    assert [trace["trace_id"] for trace in store.list_traces()] == ["business-trace"]
    assert {trace["trace_id"] for trace in store.list_traces(include_internal=True)} == {
        "business-trace",
        "trace-center-query",
    }
    assert store.get_trace("business-trace")["steps"][0]["evidence"] == []


def test_tool_wrapper_records_business_trace_step():
    token = start_trace(trace_id="trace-tool-test", route="unit-test")
    try:
        tool = StructuredTool.from_function(
            func=lambda order_id: f"订单 {order_id} 可发货",
            name="check_inventory",
            description="检查库存",
        )
        wrapped = wrap_tool_with_resilience(tool, enable_cache=False)

        result = wrapped.invoke({"order_id": "SO-1"})
        trace = snapshot_current_trace()

        assert "可发货" in result
        assert trace is not None
        assert trace["steps"][0]["type"] == "tool"
        assert trace["steps"][0]["name"] == "check_inventory"
        assert trace["steps"][0]["status"] == "success"
    finally:
        finish_trace(persist=False)
        reset_trace(token)


def test_rag_retrieve_records_evidence_in_trace():
    token = start_trace(trace_id="trace-rag-test", route="unit-test")
    try:
        service = KnowledgeRetrievalService.__new__(KnowledgeRetrievalService)
        service._cross_encoder = None
        service._query_planner = SimpleNamespace(
            prepare=lambda order_id, question, filters: SimpleNamespace(intent="stockout", active_filters=filters or [])
        )
        service._retrieve_nodes = lambda question, prepared: SimpleNamespace(queries=[question, "缺货规则"], nodes=[])
        hit = SimpleNamespace(
            score=0.87,
            metadata=SimpleNamespace(
                category="stockout_rule",
                source_file="stockout_rules.md",
                chunk_id="stockout_rules::chunk-0002",
                title="缺货处理规则",
            ),
        )
        service._rank_and_build_hits = lambda question, retrieved, intent: [hit]
        service._build_result = lambda order_id, question, prepared, queries, hits: SimpleNamespace(hits=hits)

        service.retrieve("SO-1", "是否缺货", ["stockout_rule"])
        trace = snapshot_current_trace()

        rag_step = next(step for step in trace["steps"] if step["type"] == "rag")
        assert rag_step["metadata"]["matched_categories"] == ["stockout_rule"]
        assert rag_step["evidence"][0]["source_file"] == "stockout_rules.md"
        assert rag_step["evidence"][0]["score"] == 0.87
    finally:
        finish_trace(persist=False)
        reset_trace(token)


def test_audit_event_sanitizes_sensitive_metadata():
    token = start_trace(trace_id="trace-audit-test", order_id="SO-1", route="unit-test")
    try:
        event = record_audit_event(
            event_type="hitl_decision",
            action="approved",
            order_id="SO-1",
            summary="审批通过",
            metadata={"phone": "13800138000", "comment": "联系 test@example.com"},
        )

        assert event.metadata["phone"] == "<redacted>"
        assert "<redacted>" in event.metadata["comment"]
    finally:
        finish_trace(persist=False)
        reset_trace(token)


def test_metrics_endpoint_exposes_business_observability_metrics():
    token = start_trace(trace_id="trace-metrics-test", route="unit-test")
    try:
        finish_trace(status="success", persist=False)
    finally:
        reset_trace(token)

    app = FastAPI()
    app.include_router(metrics_router)
    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "business_request_total" in body
    assert "tool_call_total" in body
    assert "rag_retrieval_total" in body
