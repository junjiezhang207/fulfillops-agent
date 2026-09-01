"""自研业务可观测系统的轻量回归测试。

这些测试不追求跑完整模型链路，而是锁住企业监控最容易退化的契约：
trace_id 能透传、工具/RAG 会自动写 step、审计事件会脱敏、Prometheus 能暴露指标。
"""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.tools import StructuredTool

from app.api.routes import observability as observability_routes
from app.agents.tools.wrapper import wrap_tool_with_resilience
from app.api.routes.metrics import router as metrics_router
from app.observability import audit_log
from app.observability.audit_log import record_audit_event
from app.observability.business_trace import (
    add_trace_step,
    finish_trace,
    record_prompt_injection_detected,
    reset_trace,
    snapshot_current_trace,
    snapshot_fulfillops_trace,
    start_trace,
)
from app.observability.trace_context import BusinessTraceMiddleware
from app.observability.trace_store import INTERNAL_ROUTE_PREFIXES, PostgreSQLTraceStore
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


def test_trace_store_preserves_empty_evidence_and_filters_internal_routes():
    store = PostgreSQLTraceStore.__new__(PostgreSQLTraceStore)
    trace = store._hydrate_trace(
        {
            "trace_id": "business-trace",
            "request_id": "business-trace",
            "session_id": None,
            "tenant_id": "default",
            "user_id": None,
            "order_id": "SO-1",
            "route": "hybrid:multi_agent",
            "status": "completed",
            "error_code": None,
            "error_message": None,
            "started_at": "2026-08-30T00:00:00Z",
            "ended_at": "2026-08-30T00:00:01Z",
            "duration_ms": 1000,
            "metadata_json": "{}",
        },
        [
            {
                "id": "step-1",
                "parent_id": None,
                "type": "router",
                "name": "hybrid_router",
                "status": "success",
                "started_at": "2026-08-30T00:00:00Z",
                "ended_at": "2026-08-30T00:00:01Z",
                "duration_ms": 1000,
                "summary": "",
                "error_code": None,
                "error_message": None,
                "input_summary": None,
                "output_summary": None,
                "metadata_json": "{}",
                "evidence_json": "[]",
            }
        ],
    )

    assert "/api/v1/observability" in INTERNAL_ROUTE_PREFIXES
    assert trace["steps"][0]["evidence"] == []


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
            prepare=lambda order_id, question, filters, session_memory=None: SimpleNamespace(
                intent="stockout",
                active_filters=filters or [],
            )
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


def test_audit_event_sanitizes_sensitive_metadata(monkeypatch):
    saved_events = []

    class FakeTraceStore:
        def save_audit_event(self, event):
            saved_events.append(event)

    monkeypatch.setattr(audit_log, "get_trace_store", lambda: FakeTraceStore())
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
        assert saved_events and saved_events[0]["metadata"]["phone"] == "<redacted>"
    finally:
        finish_trace(persist=False)
        reset_trace(token)


def test_metrics_endpoint_exposes_business_observability_metrics():
    token = start_trace(trace_id="trace-metrics-test", route="unit-test")
    try:
        finish_trace(status="success", persist=False)
    finally:
        reset_trace(token)
    record_prompt_injection_detected("unit_test")

    app = FastAPI()
    app.include_router(metrics_router)
    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    body = response.text
    assert "business_request_total" in body
    assert "tool_call_total" in body
    assert "rag_retrieval_total" in body
    assert 'prompt_injection_detected_total{source="unit_test"}' in body


def test_fulfillops_trace_document_exposes_governance_contract():
    token = start_trace(
        trace_id="trace-governance-test",
        request_id="req-governance-test",
        order_id="SO-GOV-1",
        route="ops_case",
        metadata={
            "case_id": "CASE-GOV-1",
            "plan_version": 2,
            "context_version": 5,
            "workflow_version": "2026-08-30",
            "source_systems": ["OMS", "WMS", "TMS", "ERP", "PIM", "CRM"],
        },
    )
    try:
        add_trace_step(
            step_type="memory",
            name="short_term_memory_extract",
            summary="抽取短期记忆补丁",
            metadata={"use_case": "memory_extraction"},
        )
        add_trace_step(
            step_type="llm",
            name="model_gateway.invoke",
            summary="生成 Action DAG",
            metadata={"use_case": "planner", "total_tokens": 42, "cost_usd": 0.003},
        )
        add_trace_step(
            step_type="tool_gateway",
            name="create_logistics_request",
            status="pending_human",
            summary="进入 WAITING，等待外部任务回调",
            metadata={"retry_count": 1, "event_type": "ENTER_WAITING"},
        )
        add_trace_step(
            step_type="workflow",
            name="webhook_received",
            summary="外部物流申请完成，恢复 workflow",
            metadata={"event_type": "WEBHOOK_RECEIVED", "case_status": "VERIFYING"},
        )
        add_trace_step(
            step_type="verification",
            name="verify_external_task",
            status="error",
            summary="校验外部任务结果",
            error_code="VERIFY_FAILED",
            error_message="缺少 TMS ETA",
        )

        document = snapshot_fulfillops_trace()

        assert set(document) == {"trace_meta", "execution_context", "summary", "modules", "events", "errors"}
        assert document["trace_meta"]["schema_version"] == "fulfillops.trace.v1"
        assert document["trace_meta"]["case_id"] == "CASE-GOV-1"
        assert document["trace_meta"]["order_id"] == "SO-GOV-1"
        assert document["execution_context"]["source_systems"] == ["OMS", "WMS", "TMS", "ERP", "PIM", "CRM"]
        assert set(document["modules"]) == {
            "memory",
            "context",
            "rag",
            "model_gateway",
            "planner",
            "tool_gateway",
            "verification",
        }
        assert document["modules"]["model_gateway"][0]["metadata"]["use_case"] == "planner"
        assert document["modules"]["tool_gateway"][0]["status"] == "pending_human"
        assert {event["event_type"] for event in document["events"]} >= {"ENTER_WAITING", "WEBHOOK_RECEIVED"}
        assert document["errors"][0]["module"] == "verification"
        assert document["summary"]["model_call_count"] == 1
        assert document["summary"]["tool_call_count"] == 1
        assert document["summary"]["retry_count"] == 1
        assert document["summary"]["total_tokens"] == 42
        assert document["summary"]["error_count"] == 1
    finally:
        finish_trace(persist=False)
        reset_trace(token)


def test_observability_api_returns_fulfillops_trace_document(monkeypatch):
    class FakeTraceStore:
        def get_trace(self, trace_id: str):
            if trace_id != "trace-api-test":
                return None
            return {
                "trace_id": "trace-api-test",
                "request_id": "trace-api-test",
                "order_id": "SO-API-1",
                "tenant_id": "default",
                "route": "ops_case",
                "status": "success",
                "started_at": "2026-08-30T00:00:00Z",
                "ended_at": "2026-08-30T00:00:01Z",
                "duration_ms": 1000,
                "metadata": {"case_id": "CASE-API-1", "source_systems": ["OMS", "WMS"]},
                "steps": [
                    {
                        "id": "step-api-1",
                        "type": "rag",
                        "name": "retrieve_sop_and_cases",
                        "status": "success",
                        "started_at": "2026-08-30T00:00:00Z",
                        "ended_at": "2026-08-30T00:00:01Z",
                        "summary": "召回 SOP 与优秀案例",
                        "metadata": {"sop_collection": "sop_collection"},
                        "evidence": [],
                    }
                ],
            }

    monkeypatch.setattr(observability_routes, "get_trace_store", lambda: FakeTraceStore())
    app = FastAPI()
    app.include_router(observability_routes.router)

    response = TestClient(app).get("/observability/traces/trace-api-test/fulfillops")

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["trace_meta"]["case_id"] == "CASE-API-1"
    assert payload["data"]["modules"]["rag"][0]["name"] == "retrieve_sop_and_cases"
    assert payload["data"]["summary"]["step_count"] == 1


def test_observability_api_fulfillops_trace_returns_404(monkeypatch):
    class FakeTraceStore:
        def get_trace(self, trace_id: str):
            return None

    monkeypatch.setattr(observability_routes, "get_trace_store", lambda: FakeTraceStore())
    app = FastAPI()
    app.include_router(observability_routes.router)

    response = TestClient(app).get("/observability/traces/missing/fulfillops")

    assert response.status_code == 404
