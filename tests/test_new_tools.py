"""新工具服务的单元测试：仓库、替代品、履约方案。"""

import pytest

from app.domain.fulfillment.plan_service import FulfillmentPlanService
from app.domain.inventory.analysis import InventoryAnalysisService
from app.domain.orders.analysis import OrderAnalysisService
from app.domain.fulfillment.substitute_sku import SubstituteSkuService
from app.domain.inventory.warehouse_service import WarehouseService
from app.repositories.in_memory_inventory_repository import (
    InMemoryInventoryRepository,
)
from app.repositories.in_memory_order_repository import InMemoryOrderRepository


@pytest.fixture
def warehouse_service():
    return WarehouseService()


@pytest.fixture
def substitute_service():
    return SubstituteSkuService()


@pytest.fixture
def fulfillment_service():
    order_repo = InMemoryOrderRepository()
    order_service = OrderAnalysisService(order_repo)
    inventory_repo = InMemoryInventoryRepository()
    inventory_service = InventoryAnalysisService(
        inventory_repository=inventory_repo,
        order_analysis_service=order_service,
    )
    warehouse_service = WarehouseService()
    substitute_service = SubstituteSkuService()

    return FulfillmentPlanService(
        inventory_service=inventory_service,
        warehouse_service=warehouse_service,
        substitute_service=substitute_service,
    )


class TestWarehouseService:
    """仓库库存查询服务测试。"""

    def test_search_sku_with_inventory(self, warehouse_service):
        """测试查询有库存的 SKU。"""
        result = warehouse_service.search_sku_inventory("SKU-IPHONE-CASE-001")

        assert result.sku_id == "SKU-IPHONE-CASE-001"
        assert result.total_available > 0
        assert len(result.warehouse_list) == 3  # 三个仓库
        assert result.summary

    def test_search_sku_warehouse_distribution(self, warehouse_service):
        """测试库存分布。"""
        result = warehouse_service.search_sku_inventory("SKU-CHARGER-020W-002")

        # 验证各仓库的库存都被正确查询
        warehouse_names = [wh.warehouse_name for wh in result.warehouse_list]
        assert "上海金桥仓" in warehouse_names
        assert "广州南沙仓" in warehouse_names
        assert "北京大兴仓" in warehouse_names

        # 总库存应该等于各仓库可用库存之和
        total = sum(wh.available_quantity for wh in result.warehouse_list)
        assert result.total_available == total

    def test_warehouse_sorted_by_availability(self, warehouse_service):
        """测试仓库按库存降序排列。"""
        result = warehouse_service.search_sku_inventory("SKU-CABLE-TYPEC-003")

        for i in range(len(result.warehouse_list) - 1):
            assert (
                result.warehouse_list[i].available_quantity
                >= result.warehouse_list[i + 1].available_quantity
            )


class TestSubstituteSkuService:
    """替代 SKU 查询服务测试。"""

    def test_find_substitutes(self, substitute_service):
        """测试查询替代方案。"""
        result = substitute_service.search_substitutes("SKU-IPHONE-CASE-001")

        assert result.original_sku_id == "SKU-IPHONE-CASE-001"
        assert len(result.substitutes) > 0
        assert result.summary

    def test_substitutes_sorted_by_compatibility(self, substitute_service):
        """测试替代方案按兼容度降序排列。"""
        result = substitute_service.search_substitutes("SKU-CHARGER-020W-002")

        for i in range(len(result.substitutes) - 1):
            assert (
                result.substitutes[i].compatibility_score
                >= result.substitutes[i + 1].compatibility_score
            )

    def test_no_substitutes_for_unknown_sku(self, substitute_service):
        """测试未知 SKU 无替代方案。"""
        result = substitute_service.search_substitutes("SKU-UNKNOWN-999")

        assert len(result.substitutes) == 0
        assert "暂无推荐" in result.summary

    def test_substitute_sku_has_inventory(self, substitute_service):
        """测试替代方案本身有库存。"""
        result = substitute_service.search_substitutes("SKU-IPHONE-CASE-001")

        for sub in result.substitutes:
            assert sub.available_quantity > 0


class TestFulfillmentPlanService:
    """履约方案生成服务测试。"""

    def test_generate_plan_fast_track(self, fulfillment_service):
        """测试库存充足时生成 fast_track 方案。"""
        result = fulfillment_service.generate_plan("SO202502140001")

        assert result.order_id == "SO202502140001"
        assert result.plan_strategy == "fast_track"
        assert result.estimated_completion_days == 3
        assert len(result.actions) > 0

    def test_plan_actions_detail(self, fulfillment_service):
        """测试方案中的动作详情。"""
        result = fulfillment_service.generate_plan("SO202502140001")

        for action in result.actions:
            assert action.sku_id
            assert action.product_name
            assert action.quantity
            assert action.action_type in ["ship_from_warehouse", "substitute", "backorder"]

    def test_plan_warehouse_assignment(self, fulfillment_service):
        """测试方案中的仓库分配。"""
        result = fulfillment_service.generate_plan("SO202502140001")

        # fast_track 方案中的所有动作都应该有仓库分配
        for action in result.actions:
            if action.action_type == "ship_from_warehouse":
                assert action.warehouse_id is not None
                assert action.warehouse_name is not None

    def test_plan_cost_impact(self, fulfillment_service):
        """测试方案中的成本影响计算。"""
        result = fulfillment_service.generate_plan("SO202502140001")

        # fast_track 无替代，成本影响应该为 0
        assert result.total_cost_impact == 0

    def test_plan_summary_generation(self, fulfillment_service):
        """测试方案摘要生成。"""
        result = fulfillment_service.generate_plan("SO202502140002")

        assert result.summary
        assert "履约方案" in result.summary or "fast_track" in result.summary or "mixed" in result.summary


class TestIntegration:
    """集成测试：多个服务协同。"""

    def test_warehouse_and_substitute_integration(
        self, warehouse_service, substitute_service
    ):
        """测试仓库和替代方案的协同。"""
        # 查询某个 SKU 的库存分布
        warehouse_result = warehouse_service.search_sku_inventory(
            "SKU-PHONE-CASE-BASIC-002"
        )
        # 查询其原 SKU 的替代方案
        substitute_result = substitute_service.search_substitutes("SKU-IPHONE-CASE-001")

        # 确保替代方案中包含了我们上面查询的 SKU
        substitute_skus = {sub.sku_id for sub in substitute_result.substitutes}
        assert "SKU-PHONE-CASE-BASIC-002" in substitute_skus

    def test_fulfillment_uses_warehouse_and_substitute(self, fulfillment_service):
        """测试履约方案使用了仓库和替代方案的信息。"""
        plan = fulfillment_service.generate_plan("SO202502140001")

        # 验证方案中的动作使用了仓库信息
        warehouse_actions = [a for a in plan.actions if a.warehouse_id]
        assert len(warehouse_actions) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
