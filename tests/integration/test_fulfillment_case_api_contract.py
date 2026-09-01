"""Fulfillment Case API closed-loop contract tests."""

from datetime import datetime
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agents.tools.contracts import ToolRuntimeContext, tool_runtime_context
from app.api.routes import fulfillment_cases as fulfillment_case_routes
from app.application.routing.fulfillment_case_service import (
    FulfillmentCaseService,
    InMemoryFulfillmentCaseStore,
)
from app.domain.inventory.analysis import InventoryAnalysisService
from app.domain.orders.analysis import OrderAnalysisService
from app.schemas.inventory import InventoryRecord
from app.schemas.orders import OrderItem, OrderRecord
from app.workflows.fulfillment.nodes import WorkflowNodes


class FakeOrderRepository:
    def __init__(self, orders):
        self.orders = orders

    def get_order_by_id(self, order_id):
        return self.orders.get(order_id)


class FakeInventoryRepository:
    def __init__(self, records_by_sku):
        self.records_by_sku = records_by_sku

    def list_inventory_by_sku(self, sku_id):
        return self.records_by_sku.get(sku_id, [])


class FakeKnowledgeService:
    def retrieve(self, **kwargs):
        return SimpleNamespace(
            order_id=kwargs["order_id"],
            answer_summary=SimpleNamespace(
                conclusion="按缺货 SOP 处理，优先跨仓拆单并保留人工审批。",
                key_rules=[
                    "缺货订单先检查同区域仓，再评估跨仓调拨。",
                    "拆单发货必须保留客户确认和二次校验记录。",
                    "执行前必须重新校验库存、订单状态和物流报价。",
                ],
                suggested_actions=["创建 WMS/TMS 外部任务后等待回调。"],
            ),
            hits=[
                SimpleNamespace(source_file="缺货订单处理-sop.md", category="stockout_rule", score=0.91),
                SimpleNamespace(source_file="excellent-case-001.md", category="excellent_case", score=0.82),
            ],
        )


def _client_for_fulfillment_case_router() -> TestClient:
    app = FastAPI()
    app.include_router(fulfillment_case_routes.router, prefix="/api/v1")
    return TestClient(app)


def _authorized_hitl_context():
    return tool_runtime_context(
        ToolRuntimeContext(
            tenant_id="tenant-test",
            user_id="ops-1",
            roles=["ops_approver"],
            permissions=["wms:write", "tms:write", "erp:write", "crm:write"],
            request_id="api-contract-test",
        )
    )


def _inventory(warehouse_id: str, sku_id: str, available_stock: int) -> InventoryRecord:
    return InventoryRecord(
        warehouse_id=warehouse_id,
        warehouse_name=f"{warehouse_id} 仓",
        region="华东",
        sku_id=sku_id,
        available_stock=available_stock,
        locked_stock=0,
        updated_at=datetime(2026, 8, 22, 9, 0, 0),
    )


def _business_result_for_task_payload(task: dict) -> dict:
    base = {"business_state_verified": True}
    if task["target_system"] == "WMS" and task["action_type"] == "inventory_transfer":
        return {**base, "transfer_order_id": f"TR-{task['task_id'][-8:]}", "transfer_status": "CONFIRMED"}
    if task["target_system"] == "WMS":
        return {**base, "fulfillment_task_id": f"WMS-{task['task_id'][-8:]}", "fulfillment_task_created": True}
    if task["target_system"] == "TMS":
        return {**base, "quote_confirmed": True, "serviceable": True, "channel_status": "confirmed"}
    if task["target_system"] == "ERP":
        return {**base, "replenishment_request_id": f"ERP-{task['task_id'][-8:]}", "inbound_stock_visible": True}
    if task["target_system"] == "CRM":
        return {**base, "crm_case_id": f"CRM-{task['task_id'][-8:]}", "customer_confirmation_status": "contacted"}
    return base


def test_fulfillment_case_api_rejects_confirm_when_preflight_fingerprint_changed(monkeypatch):
    order = OrderRecord(
        order_id="SO-API-STALE-PREFLIGHT-001",
        platform="京东",
        order_time=datetime(2026, 8, 22, 10, 0, 0),
        order_status="paid_waiting_fulfillment",
        region="上海",
        priority="normal",
        items=[OrderItem(sku_id="SKU-X", product_name="核心商品", quantity=1, unit_price=299)],
    )
    order_service = OrderAnalysisService(FakeOrderRepository({"SO-API-STALE-PREFLIGHT-001": order}))
    inventory_service = InventoryAnalysisService(
        FakeInventoryRepository({"SKU-X": [_inventory("WH-SH", "SKU-X", 1)]}),
        order_service,
    )
    nodes = WorkflowNodes(order_service, inventory_service, FakeKnowledgeService())
    proposal = nodes._build_execution_proposal(
        order_service.analyze_order("SO-API-STALE-PREFLIGHT-001"),
        inventory_service.analyze_inventory("SO-API-STALE-PREFLIGHT-001"),
    )
    preflight = nodes._preflight_validate(proposal).model_copy(
        update={"new_fingerprint": "fp-newer-business-context"}
    )
    store = InMemoryFulfillmentCaseStore()
    case_service = FulfillmentCaseService(
        order_service=order_service,
        inventory_service=inventory_service,
        store=store,
        proposal_builder=nodes._build_execution_proposal,
    )
    monkeypatch.setattr(fulfillment_case_routes, "get_fulfillment_case_service", lambda: case_service)
    client = _client_for_fulfillment_case_router()

    response = client.post(
        "/api/v1/fulfillment-cases/confirm",
        json={
            "proposal": proposal.model_dump(mode="json"),
            "preflight_validation": preflight.model_dump(mode="json"),
            "approver_id": "ops-1",
            "notes": "不能执行旧上下文方案",
        },
    )

    assert response.status_code == 400
    assert "实时业务上下文已变化" in response.json()["detail"]
    assert store.list_cases() == []


def test_fulfillment_case_api_rejects_direct_mutation_action(monkeypatch):
    order = OrderRecord(
        order_id="SO-API-DIRECT-MUTATION-001",
        platform="京东",
        order_time=datetime(2026, 8, 22, 10, 0, 0),
        order_status="paid_waiting_fulfillment",
        region="上海",
        priority="normal",
        items=[OrderItem(sku_id="SKU-X", product_name="核心商品", quantity=1, unit_price=299)],
    )
    order_service = OrderAnalysisService(FakeOrderRepository({"SO-API-DIRECT-MUTATION-001": order}))
    inventory_service = InventoryAnalysisService(
        FakeInventoryRepository({"SKU-X": [_inventory("WH-SH", "SKU-X", 1)]}),
        order_service,
    )
    store = InMemoryFulfillmentCaseStore()
    case_service = FulfillmentCaseService(
        order_service=order_service,
        inventory_service=inventory_service,
        store=store,
    )
    monkeypatch.setattr(fulfillment_case_routes, "get_fulfillment_case_service", lambda: case_service)
    client = _client_for_fulfillment_case_router()
    proposal = {
        "proposal_id": "prop-api-direct-mutation",
        "order_id": "SO-API-DIRECT-MUTATION-001",
        "title": "非法直接修改",
        "summary": "API 入口不能接受直接业务突变动作。",
        "status": "pending_approval",
        "actions": [
            {
                "action_id": "a1",
                "action_type": "direct_inventory_mutation",
                "sku_id": "SKU-X",
                "quantity": 1,
                "reason": "直接扣库存",
            }
        ],
        "data_fingerprint": "fp-api-direct-mutation",
    }

    response = client.post(
        "/api/v1/fulfillment-cases/confirm",
        json={
            "proposal": proposal,
            "preflight_validation": {
                "status": "valid",
                "checked_at": datetime(2026, 8, 22, 11, 0, 0).isoformat(),
                "checks": [],
                "old_fingerprint": proposal["data_fingerprint"],
                "new_fingerprint": proposal["data_fingerprint"],
                "message": "ok",
            },
            "approver_id": "ops-1",
            "notes": "不能执行直接突变",
        },
    )

    assert response.status_code == 400
    assert "Policy/Schema Validator 未通过" in response.json()["detail"]
    assert store.list_cases() == []


def test_fulfillment_case_api_requires_server_side_write_permission(monkeypatch):
    order = OrderRecord(
        order_id="SO-API-PERMISSION-001",
        platform="京东",
        order_time=datetime(2026, 8, 22, 10, 0, 0),
        order_status="paid_waiting_fulfillment",
        region="上海",
        priority="normal",
        items=[OrderItem(sku_id="SKU-X", product_name="核心商品", quantity=1, unit_price=299)],
    )
    order_service = OrderAnalysisService(FakeOrderRepository({"SO-API-PERMISSION-001": order}))
    inventory_service = InventoryAnalysisService(
        FakeInventoryRepository({"SKU-X": [_inventory("WH-SH", "SKU-X", 1)]}),
        order_service,
    )
    nodes = WorkflowNodes(order_service, inventory_service, FakeKnowledgeService())
    proposal = nodes._build_execution_proposal(
        order_service.analyze_order("SO-API-PERMISSION-001"),
        inventory_service.analyze_inventory("SO-API-PERMISSION-001"),
    )
    preflight = nodes._preflight_validate(proposal)
    store = InMemoryFulfillmentCaseStore()
    case_service = FulfillmentCaseService(
        order_service=order_service,
        inventory_service=inventory_service,
        store=store,
        proposal_builder=nodes._build_execution_proposal,
    )
    monkeypatch.setattr(fulfillment_case_routes, "get_fulfillment_case_service", lambda: case_service)
    client = _client_for_fulfillment_case_router()

    response = client.post(
        "/api/v1/fulfillment-cases/confirm",
        json={
            "proposal": proposal.model_dump(mode="json"),
            "preflight_validation": preflight.model_dump(mode="json"),
            "approver_id": "ops-1",
            "notes": "缺少服务端授权上下文",
        },
    )

    assert response.status_code == 403
    assert "wms:write" in response.json()["detail"]
    assert store.list_cases() == []


def test_fulfillment_case_api_confirm_webhook_verify_and_excellent_case(monkeypatch, tmp_path):
    """HTTP API proves the case lifecycle can resume without keeping Agent running."""
    order = OrderRecord(
        order_id="SO-API-001",
        platform="京东",
        order_time=datetime(2026, 8, 22, 10, 0, 0),
        order_status="paid_waiting_fulfillment",
        region="上海",
        priority="vip",
        inventory_reserved=False,
        package_created=False,
        waybill_created=False,
        outbound_completed=False,
        active_fulfillment_tasks=[],
        items=[OrderItem(sku_id="SKU-X", product_name="核心商品", quantity=5, unit_price=299)],
    )
    order_service = OrderAnalysisService(FakeOrderRepository({"SO-API-001": order}))
    inventory_service = InventoryAnalysisService(
        FakeInventoryRepository(
            {
                "SKU-X": [
                    _inventory("WH-SH", "SKU-X", 3),
                    _inventory("WH-HZ", "SKU-X", 2),
                ]
            }
        ),
        order_service,
    )
    nodes = WorkflowNodes(order_service, inventory_service, FakeKnowledgeService())
    proposal = nodes._build_execution_proposal(
        order_service.analyze_order("SO-API-001"),
        inventory_service.analyze_inventory("SO-API-001"),
    )
    preflight = nodes._preflight_validate(proposal)
    case_service = FulfillmentCaseService(
        order_service=order_service,
        inventory_service=inventory_service,
        store=InMemoryFulfillmentCaseStore(),
        proposal_builder=nodes._build_execution_proposal,
    )
    rebuild_calls = []

    monkeypatch.setattr(fulfillment_case_routes, "get_fulfillment_case_service", lambda: case_service)
    monkeypatch.setattr(fulfillment_case_routes, "get_settings", lambda: SimpleNamespace(knowledge_dir=str(tmp_path)))
    monkeypatch.setattr(
        fulfillment_case_routes,
        "get_knowledge_retrieval_service",
        lambda: SimpleNamespace(rebuild_index=lambda: rebuild_calls.append("rebuild") or "ok"),
    )
    monkeypatch.setattr(fulfillment_case_routes, "_hitl_tool_runtime_context", _authorized_hitl_context)
    client = _client_for_fulfillment_case_router()

    confirm_response = client.post(
        "/api/v1/fulfillment-cases/confirm",
        json={
            "proposal": proposal.model_dump(mode="json"),
            "preflight_validation": preflight.model_dump(mode="json"),
            "approver_id": "ops-1",
            "notes": "批准执行",
            "human_changes": ["API 审批时补充客服同步动作"],
            "checkpoint": {"thread_id": "thread-api-001", "node": "human_approval"},
        },
    )
    assert confirm_response.status_code == 200
    case_payload = confirm_response.json()["data"]["case"]
    case_id = case_payload["case_id"]
    assert case_payload["case_status"] == "WAITING_EXTERNAL_TASK"
    assert case_payload["checkpoint"]["thread_id"] == "thread-api-001"
    assert case_payload["current_state"]["human_changes"] == ["API 审批时补充客服同步动作"]
    assert case_payload["current_state"]["hitl_decision"] == "MODIFIED_APPROVED"
    assert all(task["status"] == "RUNNING" for task in case_payload["tasks"])
    status_response = client.get(f"/api/v1/fulfillment-cases/tasks/{case_payload['tasks'][0]['task_id']}/status")
    assert status_response.status_code == 200
    assert status_response.json()["data"]["task"]["status"] == "RUNNING"

    list_response = client.get("/api/v1/fulfillment-cases", params={"order_id": "SO-API-001"})
    assert list_response.status_code == 200
    assert list_response.json()["data"]["cases"][0]["case_id"] == case_id

    task_update_response = None
    for index, task in enumerate(case_payload["tasks"]):
        if index == 0:
            task_update_response = client.post(
                f"/api/v1/fulfillment-cases/tasks/{task['task_id']}/status-sync",
                json={
                    "status": "COMPLETED",
                    "poll_id": f"poll-{task['task_id']}",
                    "case_version": case_payload["case_version"],
                    "plan_version": task["plan_version"],
                    "idempotency_key": task["idempotency_key"],
                    "result": _business_result_for_task_payload(task),
                },
            )
        else:
            task_update_response = client.post(
                f"/api/v1/fulfillment-cases/tasks/{task['task_id']}/webhook",
                json={
                    "status": "COMPLETED",
                    "event_id": f"evt-{task['task_id']}",
                    "case_version": case_payload["case_version"],
                    "plan_version": task["plan_version"],
                    "idempotency_key": task["idempotency_key"],
                    "result": _business_result_for_task_payload(task),
                },
            )
        assert task_update_response.status_code == 200
    assert task_update_response is not None
    assert task_update_response.json()["data"]["case"]["case_status"] == "VERIFYING"

    order.inventory_reserved = True
    order.package_created = True
    order.waybill_created = True
    order.outbound_completed = True
    verify_response = client.post(f"/api/v1/fulfillment-cases/{case_id}/verify")
    assert verify_response.status_code == 200
    verification = verify_response.json()["data"]["verification"]
    assert verification["status"] == "COMPLETED"
    assert any(check["name"] == "direct_fulfillment_materialized" for check in verification["checks"])

    excellent_response = client.post(
        f"/api/v1/fulfillment-cases/{case_id}/excellent-case",
        json={"operator_id": "ops-1", "notes": "API 闭环验证成功"},
    )
    assert excellent_response.status_code == 200
    excellent_payload = excellent_response.json()["data"]
    assert excellent_payload["ingestion_job"]["status"] == "PENDING"
    assert excellent_payload["rebuild_scheduled"] is True
    assert rebuild_calls == ["rebuild"]
    final_case = case_service.get_case(case_id)
    assert final_case is not None
    ingestion = final_case.current_state["case_ingestion"]
    assert ingestion["status"] == "SUCCESS"
    assert "excellent_cases" in ingestion["knowledge_path"]

    status_response = client.get(f"/api/v1/fulfillment-cases/{case_id}/excellent-case")
    assert status_response.status_code == 200
    assert status_response.json()["data"]["ingestion_job"]["status"] == "SUCCESS"

    duplicate_response = client.post(
        f"/api/v1/fulfillment-cases/{case_id}/excellent-case",
        json={"operator_id": "ops-1", "notes": "重复点击不应重复调度"},
    )
    assert duplicate_response.status_code == 200
    duplicate_payload = duplicate_response.json()["data"]
    assert duplicate_payload["ingestion_job"]["status"] == "SUCCESS"
    assert duplicate_payload["rebuild_scheduled"] is False
    assert rebuild_calls == ["rebuild"]


def test_fulfillment_case_api_records_hitl_reject_without_dispatch(monkeypatch):
    order = OrderRecord(
        order_id="SO-API-REJECT-001",
        platform="京东",
        order_time=datetime(2026, 8, 22, 10, 0, 0),
        order_status="paid_waiting_fulfillment",
        region="上海",
        priority="vip",
        items=[OrderItem(sku_id="SKU-X", product_name="核心商品", quantity=1, unit_price=299)],
    )
    order_service = OrderAnalysisService(FakeOrderRepository({"SO-API-REJECT-001": order}))
    inventory_service = InventoryAnalysisService(
        FakeInventoryRepository({"SKU-X": [_inventory("WH-SH", "SKU-X", 1)]}),
        order_service,
    )
    nodes = WorkflowNodes(order_service, inventory_service, FakeKnowledgeService())
    proposal = nodes._build_execution_proposal(
        order_service.analyze_order("SO-API-REJECT-001"),
        inventory_service.analyze_inventory("SO-API-REJECT-001"),
    )
    case_service = FulfillmentCaseService(
        order_service=order_service,
        inventory_service=inventory_service,
        store=InMemoryFulfillmentCaseStore(),
        proposal_builder=nodes._build_execution_proposal,
    )
    monkeypatch.setattr(fulfillment_case_routes, "get_fulfillment_case_service", lambda: case_service)
    monkeypatch.setattr(fulfillment_case_routes, "_hitl_tool_runtime_context", _authorized_hitl_context)
    client = _client_for_fulfillment_case_router()

    response = client.post(
        "/api/v1/fulfillment-cases/review",
        json={
            "proposal": proposal.model_dump(mode="json"),
            "decision": "rejected",
            "approver_id": "ops-lead",
            "notes": "客户不接受当前方案，重新规划。",
            "checkpoint": {"thread_id": "thread-api-reject"},
        },
    )

    assert response.status_code == 200
    case_payload = response.json()["data"]["case"]
    assert case_payload["case_status"] == "REPLAN_REQUIRED"
    assert case_payload["tasks"] == []
    assert case_payload["checkpoint"]["hitl"] == "REJECTED"
    assert case_payload["current_state"]["dispatch_status"] == "not_dispatched"
    assert case_payload["current_state"]["pending_replan_proposal"]["plan_version"] == 2


def test_fulfillment_case_api_confirms_replan_on_existing_case(monkeypatch):
    """HTTP API can approve a regenerated plan without creating a new case."""
    order = OrderRecord(
        order_id="SO-API-REPLAN-001",
        platform="京东",
        order_time=datetime(2026, 8, 22, 10, 0, 0),
        order_status="paid_waiting_fulfillment",
        region="上海",
        priority="vip",
        items=[OrderItem(sku_id="SKU-X", product_name="核心商品", quantity=1, unit_price=299)],
    )
    order_service = OrderAnalysisService(FakeOrderRepository({"SO-API-REPLAN-001": order}))
    inventory_service = InventoryAnalysisService(
        FakeInventoryRepository({"SKU-X": [_inventory("WH-SH", "SKU-X", 1)]}),
        order_service,
    )
    nodes = WorkflowNodes(order_service, inventory_service, FakeKnowledgeService())
    proposal = nodes._build_execution_proposal(
        order_service.analyze_order("SO-API-REPLAN-001"),
        inventory_service.analyze_inventory("SO-API-REPLAN-001"),
    )
    preflight = nodes._preflight_validate(proposal)
    case_service = FulfillmentCaseService(
        order_service=order_service,
        inventory_service=inventory_service,
        store=InMemoryFulfillmentCaseStore(),
        proposal_builder=nodes._build_execution_proposal,
    )
    monkeypatch.setattr(fulfillment_case_routes, "get_fulfillment_case_service", lambda: case_service)
    monkeypatch.setattr(fulfillment_case_routes, "_hitl_tool_runtime_context", _authorized_hitl_context)
    client = _client_for_fulfillment_case_router()

    confirm_response = client.post(
        "/api/v1/fulfillment-cases/confirm",
        json={
            "proposal": proposal.model_dump(mode="json"),
            "preflight_validation": preflight.model_dump(mode="json"),
            "approver_id": "ops-1",
            "notes": "首次批准",
        },
    )
    assert confirm_response.status_code == 200
    case_payload = confirm_response.json()["data"]["case"]
    old_task = case_payload["tasks"][0]
    failed_response = client.post(
        f"/api/v1/fulfillment-cases/tasks/{old_task['task_id']}/webhook",
        json={
            "status": "FAILED",
            "event_id": "evt-api-replan-failed",
            "case_version": case_payload["case_version"],
            "plan_version": old_task["plan_version"],
            "idempotency_key": old_task["idempotency_key"],
            "result": {"reason": "外部任务失败"},
        },
    )
    assert failed_response.status_code == 200

    verify_response = client.post(f"/api/v1/fulfillment-cases/{case_payload['case_id']}/verify")
    assert verify_response.status_code == 200
    verification = verify_response.json()["data"]["verification"]
    assert verification["status"] == "REPLAN_REQUIRED"
    replacement = verification["replacement_proposal"]
    assert replacement["plan_version"] == 2

    replan_response = client.post(
        f"/api/v1/fulfillment-cases/{case_payload['case_id']}/replan/confirm",
        json={
            "proposal": replacement,
            "preflight_validation": {
                "status": "valid",
                "checked_at": datetime(2026, 8, 22, 11, 0, 0).isoformat(),
                "checks": [{"name": "replan_schema", "status": "pass"}],
                "old_fingerprint": replacement["data_fingerprint"],
                "new_fingerprint": replacement["data_fingerprint"],
                "message": "重新规划方案校验通过。",
            },
            "approver_id": "ops-lead",
            "notes": "确认新版方案",
            "human_changes": ["继续保留时效优先"],
            "checkpoint": {"thread_id": "thread-api-replan"},
        },
    )

    assert replan_response.status_code == 200
    replan_case = replan_response.json()["data"]["case"]
    assert replan_case["case_id"] == case_payload["case_id"]
    assert replan_case["case_status"] == "WAITING_EXTERNAL_TASK"
    assert replan_case["case_version"] == 3
    assert replan_case["plan_version"] == 2
    assert replan_case["current_state"]["hitl_decision"] == "REPLAN_MODIFIED_APPROVED"
    assert replan_case["current_state"]["attempted_plans"][0]["plan_version"] == 1
    assert replan_case["current_state"]["previous_tasks"][0]["task_id"] == old_task["task_id"]
    assert all(task["plan_version"] == 2 for task in replan_case["tasks"])
    assert {task["task_id"] for task in replan_case["tasks"]}.isdisjoint({old_task["task_id"]})


def test_fulfillment_case_api_manual_processing_and_completion(monkeypatch):
    """HTTP API can move MANUAL handoff through processing and back to verification."""
    order = OrderRecord(
        order_id="SO-API-MANUAL-001",
        platform="京东",
        order_time=datetime(2026, 8, 22, 10, 0, 0),
        order_status="paid_waiting_fulfillment",
        region="上海",
        priority="vip",
        items=[OrderItem(sku_id="SKU-X", product_name="核心商品", quantity=1, unit_price=299)],
    )
    order_service = OrderAnalysisService(FakeOrderRepository({"SO-API-MANUAL-001": order}))
    inventory_service = InventoryAnalysisService(
        FakeInventoryRepository({"SKU-X": [_inventory("WH-SH", "SKU-X", 0)]}),
        order_service,
    )
    proposal = WorkflowNodes(order_service, inventory_service, FakeKnowledgeService())._build_execution_proposal(
        order_service.analyze_order("SO-API-MANUAL-001"),
        inventory_service.analyze_inventory("SO-API-MANUAL-001"),
    )
    case_service = FulfillmentCaseService(
        order_service=order_service,
        inventory_service=inventory_service,
        store=InMemoryFulfillmentCaseStore(),
    )
    case = case_service.confirm(
        proposal=proposal,
        preflight_validation=WorkflowNodes(order_service, inventory_service, FakeKnowledgeService())._preflight_validate(proposal),
        approver_id="ops-1",
        notes="首次批准",
    )
    case_service.store.save_case(
        case.model_copy(
            update={
                "case_status": "MANUAL",
                "replan_count": 4,
                "current_state": {
                    **case.current_state,
                    "phase": "MANUAL_WAITING",
                    "manual_handoff_package": {"case_id": case.case_id, "order_id": case.order_id},
                },
            }
        )
    )
    monkeypatch.setattr(fulfillment_case_routes, "get_fulfillment_case_service", lambda: case_service)
    client = _client_for_fulfillment_case_router()

    start_response = client.post(
        f"/api/v1/fulfillment-cases/{case.case_id}/manual/start",
        json={"operator_id": "manual-api-1", "notes": "人工开始处理"},
    )
    assert start_response.status_code == 200
    assert start_response.json()["data"]["case"]["current_state"]["phase"] == "MANUAL_PROCESSING"

    complete_response = client.post(
        f"/api/v1/fulfillment-cases/{case.case_id}/manual/complete",
        json={
            "operator_id": "manual-api-1",
            "notes": "客户接受延期",
            "business_result": {
                "crm_case_id": "CRM-API-MANUAL-001",
                "customer_confirmation_status": "accepted",
            },
        },
    )
    assert complete_response.status_code == 200
    resumed_case = complete_response.json()["data"]["case"]
    assert resumed_case["case_status"] == "VERIFYING"
    assert resumed_case["current_state"]["phase"] == "VERIFYING"
    assert all(task["status"] == "COMPLETED" for task in resumed_case["tasks"])

    verify_response = client.post(f"/api/v1/fulfillment-cases/{case.case_id}/verify")
    assert verify_response.status_code == 200
    assert verify_response.json()["data"]["verification"]["status"] == "COMPLETED"
