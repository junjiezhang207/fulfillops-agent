"""工具结构化输出单元测试 — 验证所有工具返回合法 JSON 格式。

大厂标准：工具输出格式变更是 Agent 质量退化的常见原因，
每次发布前必须验证 JSON schema 不变。

全部无 LLM 依赖，使用 mock service，CI 毫秒级完成。
"""

import json
from unittest.mock import MagicMock

import pytest

from app.agents.tools.factory import (
    make_fulfillment_plan_tool,
    make_inventory_tool,
    make_knowledge_tool,
    make_order_tool,
    make_substitute_tool,
    make_warehouse_tool,
)


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


# ── JSON Schema 验证辅助 ─────────────────────────────────────────────────────

def _parse_and_validate(raw: str) -> dict:
    """解析工具输出 JSON，验证基本 schema。"""
    data = json.loads(raw)
    assert "status" in data, "缺少 status 字段"
    assert "summary" in data, "缺少 summary 字段"
    assert "data" in data, "缺少 data 字段"
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
        # handle_tool_error=True 时返回错误字符串而非抛出
        result = tool.invoke({"order_id": "NOTFOUND"})
        assert isinstance(result, str)


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
