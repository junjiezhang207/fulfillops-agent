"""工具结构化输出单元测试 — 验证所有工具返回合法 JSON 格式。

大厂标准：工具输出格式变更是 Agent 质量退化的常见原因，
每次发布前必须验证 JSON schema 不变。

全部无 LLM 依赖，使用 mock service，CI 毫秒级完成。
"""

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import StructuredTool

from app.agents.tools.contracts import ToolRuntimeContext, tool_runtime_context
from app.agents.tools.factory import (
    make_fulfillment_plan_tool,
    make_inventory_tool,
    make_knowledge_tool,
    make_order_tool,
    make_source_context_tool,
    make_substitute_tool,
    make_warehouse_tool,
)
from app.agents.tools.registry import ToolServiceBundle, get_tool_registry
from app.agents.tools.telemetry import get_tool_telemetry
from app.agents.tools.wrapper import wrap_tool_with_resilience


# ── Mock 数据构造 ────────────────────────────────────────────────────────────

def _mock_order_result():
    m = MagicMock()
    m.summary = "订单 SO123 含 2 个 SKU，总计 50 件"
    m.item_count = 2
    m.total_quantity = 50
    m.platform = "天猫"
    m.region = "华东"
    m.customer_level = "vip"
    m.priority = "high"
    return m


def _mock_inventory_result():
    m = MagicMock()
    m.summary = "库存充足，可全量发货"
    m.fulfillment_ready = True
    m.insufficient_skus = []
    m.item_count = 2
    return m


def _mock_knowledge_result():
    summary = MagicMock()
    summary.key_rules = ["缺货时优先替换", "VIP 客户 4h 内处理"]
    summary.suggested_actions = ["联系客服确认替换方案"]
    result = MagicMock()
    result.hits = [MagicMock(), MagicMock()]
    result.answer_summary = summary
    result.matched_categories = ["priority_orders"]
    return result


def _mock_warehouse_result():
    wh1 = MagicMock()
    wh1.warehouse_name = "华东仓"
    wh1.available_quantity = 100
    wh1.reserved_quantity = 10
    result = MagicMock()
    result.summary = "华东仓 100 件可用"
    result.warehouse_list = [wh1]
    return result


def _mock_substitute_result():
    sub = MagicMock()
    sub.sku_id = "SKU-ALT-001"
    sub.compatibility = "完全兼容"
    result = MagicMock()
    result.summary = "找到 1 个替代品"
    result.substitutes = [sub]
    return result


def _mock_fulfillment_plan():
    action = MagicMock()
    action.product_name = "iPhone 保护壳"
    action.quantity = 50
    action.action_type = "direct_ship"
    action.warehouse_name = "华东仓"
    action.substitute_product = None
    action.estimated_days = 3
    action.note = ""
    plan = MagicMock()
    plan.summary = "快速路径发货"
    plan.actions = [action]
    return plan


def _mock_source_context(source_system: str):
    return {
        "order_id": "SO123",
        "intent": "fulfillment_action",
        "source_system": source_system,
        "read_only": True,
        "field_groups_loaded": ["core", "inventory", "logistics"],
        "context_completeness": {"status": "complete"},
        "detail_paths": ["sku_warehouse_inventory"],
        "details": {"sku_warehouse_inventory": [{"sku_id": "SKU-001", "available_stock": 5}]},
    }


# ── JSON Schema 验证辅助 ─────────────────────────────────────────────────────

def _parse_and_validate(raw: str) -> dict:
    """解析工具输出 JSON，验证基本 schema。"""
    data = json.loads(raw)
    assert "status" in data, "缺少 status 字段"
    assert "summary" in data, "缺少 summary 字段"
    assert "data" in data, "缺少 data 字段"
    assert data["schema_version"] == "1.0"
    assert data["status"] == "ok", f"status 应为 ok，实际为 {data['status']}"
    return data


# ── 各工具输出格式测试 ────────────────────────────────────────────────────────

class TestOrderToolOutput:
    def test_returns_valid_json(self):
        svc = MagicMock()
        svc.analyze_order.return_value = _mock_order_result()
        tool = make_order_tool(svc)
        raw = tool.invoke({"order_id": "SO123"})
        data = _parse_and_validate(raw)
        assert data["data"]["order_id"] == "SO123"
        assert isinstance(data["data"]["item_count"], int)
        assert isinstance(data["data"]["total_quantity"], int)

    def test_error_raises_tool_exception(self):
        svc = MagicMock()
        svc.analyze_order.side_effect = ValueError("订单不存在")
        tool = make_order_tool(svc)
        result = tool.invoke({"order_id": "NOTFOUND"})
        data = json.loads(result)
        assert data["status"] == "error"
        assert data["error"]["code"] == "order_query_failed"

    def test_validation_error_returns_structured_json(self):
        svc = MagicMock()
        tool = make_order_tool(svc)
        result = tool.invoke({})
        data = json.loads(result)
        assert data["status"] == "error"
        assert data["error"]["code"] == "validation_error"


class TestInventoryToolOutput:
    def test_returns_valid_json(self):
        svc = MagicMock()
        svc.analyze_inventory.return_value = _mock_inventory_result()
        tool = make_inventory_tool(svc)
        raw = tool.invoke({"order_id": "SO123"})
        data = _parse_and_validate(raw)
        assert "fulfillment_ready" in data["data"]
        assert "insufficient_skus" in data["data"]
        assert isinstance(data["data"]["insufficient_skus"], list)


class TestKnowledgeToolOutput:
    def test_returns_valid_json(self):
        svc = MagicMock()
        svc.retrieve.return_value = _mock_knowledge_result()
        tool = make_knowledge_tool(svc)
        raw = tool.invoke({"order_id": "SO123", "question": "缺货怎么处理"})
        data = _parse_and_validate(raw)
        assert "key_rules" in data["data"]
        assert "suggested_actions" in data["data"]
        assert isinstance(data["data"]["hit_count"], int)

    def test_order_id_is_optional(self):
        svc = MagicMock()
        svc.retrieve.return_value = _mock_knowledge_result()
        tool = make_knowledge_tool(svc)

        raw = tool.invoke({"question": "stockout SOP"})

        data = _parse_and_validate(raw)
        assert data["data"]["hit_count"] == 2
        svc.retrieve.assert_called_once()
        assert svc.retrieve.call_args.kwargs["order_id"] is None


class TestWarehouseToolOutput:
    def test_returns_valid_json(self):
        svc = MagicMock()
        svc.search_sku_inventory.return_value = _mock_warehouse_result()
        tool = make_warehouse_tool(svc)
        raw = tool.invoke({"sku_id": "SKU-001"})
        data = _parse_and_validate(raw)
        assert "warehouses" in data["data"]
        assert "total_available" in data["data"]
        assert data["data"]["sku_id"] == "SKU-001"


class TestSubstituteToolOutput:
    def test_returns_valid_json(self):
        svc = MagicMock()
        svc.search_substitutes.return_value = _mock_substitute_result()
        tool = make_substitute_tool(svc)
        raw = tool.invoke({"sku_id": "SKU-001"})
        data = _parse_and_validate(raw)
        assert "original_sku" in data["data"]
        assert "has_substitutes" in data["data"]
        assert "substitutes" in data["data"]


class TestFulfillmentPlanToolOutput:
    def test_returns_valid_json(self):
        svc = MagicMock()
        svc.generate_plan.return_value = _mock_fulfillment_plan()
        tool = make_fulfillment_plan_tool(svc)
        raw = tool.invoke({"order_id": "SO123"})
        data = _parse_and_validate(raw)
        assert "actions" in data["data"]
        assert "action_count" in data["data"]
        assert isinstance(data["data"]["actions"], list)
        if data["data"]["actions"]:
            action = data["data"]["actions"][0]
            assert "product" in action
            assert "action_type" in action


class TestProgressiveReadOnlyContextTools:
    def test_source_context_tool_returns_read_only_source_payload(self):
        svc = MagicMock()
        svc.get_source_detail.return_value = _mock_source_context("WMS")
        tool = make_source_context_tool(
            svc,
            name="get_inventory_warehouse_detail",
            source_system="WMS",
            description="test",
        )

        raw = tool.invoke({"order_id": "SO123", "question": "查库存"})
        data = _parse_and_validate(raw)

        assert data["data"]["source_system"] == "WMS"
        assert data["data"]["read_only"] is True
        assert "details" in data["data"]
        svc.get_source_detail.assert_called_once()

    def test_tool_registry_exposes_six_progressive_read_only_tools(self):
        registry = get_tool_registry()
        names = {
            "get_order_detail",
            "get_inventory_warehouse_detail",
            "get_shipping_detail",
            "get_supply_chain_detail",
            "get_product_constraints",
            "get_customer_case_context",
        }

        catalog_names = {item["name"] for item in registry.catalog(use_case="agent", include_metrics=False)}

        assert names <= catalog_names
        for name in names:
            manifest = registry.get(name).manifest
            assert manifest.side_effects is False
            assert manifest.data_freshness == "realtime"

    def test_registered_read_only_tool_uses_gateway_permissions(self):
        svc = MagicMock()
        svc.get_source_detail.return_value = _mock_source_context("WMS")
        tools = get_tool_registry().build_tools(
            services=ToolServiceBundle(context_service=svc),
            use_case="agent",
            include_names=("get_inventory_warehouse_detail",),
        )
        wrapped = wrap_tool_with_resilience(tools[0], enable_cache=False, enable_circuit_breaker=False)
        context = ToolRuntimeContext(
            tenant_id="tenant-a",
            user_id="u1",
            roles=["agent_user"],
            permissions=["oms:read"],
        )

        with tool_runtime_context(context):
            result = wrapped.invoke({"order_id": "SO123"})

        data = json.loads(result)
        assert data["status"] == "error"
        assert data["error"]["code"] == "permission_denied"


class TestEnterpriseToolControls:
    def test_wrapped_tool_denies_missing_permission(self):
        def _tool(order_id: str) -> str:
            return json.dumps({"status": "ok", "data": {"order_id": order_id}, "summary": "ok"})

        base = StructuredTool.from_function(
            func=_tool,
            name="analyze_order",
            description="test",
        )
        wrapped = wrap_tool_with_resilience(base, enable_cache=False, enable_circuit_breaker=False)

        context = ToolRuntimeContext(
            tenant_id="tenant-a",
            user_id="u1",
            roles=["agent_user"],
            permissions=[],
        )
        with tool_runtime_context(context):
            result = wrapped.invoke({"order_id": "SO123"})

        data = json.loads(result)
        assert data["status"] == "error"
        assert data["error"]["code"] == "permission_denied"

    def test_tool_telemetry_records_success(self):
        telemetry = get_tool_telemetry()
        telemetry.reset()

        def _tool(value: str) -> str:
            return "ok"

        wrapped = wrap_tool_with_resilience(
            StructuredTool.from_function(func=_tool, name="unknown_test_tool", description="test"),
            enable_cache=False,
            enable_circuit_breaker=False,
        )
        assert wrapped.invoke({"value": "x"}) == "ok"

        metric = telemetry.snapshot["unknown_test_tool"]
        assert metric["calls"] == 1
        assert metric["success"] == 1
