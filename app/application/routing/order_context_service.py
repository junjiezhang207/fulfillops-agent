"""Application service for OrderContext JSON and progressive details."""

from __future__ import annotations

from typing import Any

from app.domain.fulfillment.context_builder import OrderContextBuilder, OrderContextEnvelope
from app.domain.inventory.analysis import InventoryAnalysisService
from app.domain.orders.analysis import OrderAnalysisService


class OrderContextService:
    """Program-owned context loading for fulfillment Agents."""

    SOURCE_DETAIL_PATHS = {
        "OMS": ["order_main", "order_items", "fulfillment_state"],
        "WMS": ["candidate_warehouses", "sku_warehouse_inventory"],
        "TMS": ["logistics_options"],
        "ERP": ["replenishment_options"],
        "PIM": ["product_restrictions"],
        "CRM": ["customer_risk"],
    }

    def __init__(
        self,
        *,
        order_service: OrderAnalysisService,
        inventory_service: InventoryAnalysisService,
        context_builder: OrderContextBuilder | None = None,
    ) -> None:
        self.order_service = order_service
        self.inventory_service = inventory_service
        self.context_builder = context_builder or OrderContextBuilder()

    def build_context(
        self,
        *,
        order_id: str,
        intent: str = "fulfillment_action",
        question: str | None = None,
    ) -> OrderContextEnvelope:
        order_result = self.order_service.analyze_order(order_id)
        inventory_result = self.inventory_service.analyze_inventory(order_id)
        return self.context_builder.build_from_results(
            order_result,
            inventory_result,
            intent=intent,
            question=question,
        )

    def get_context_detail(
        self,
        *,
        order_id: str,
        detail_path: str,
        intent: str = "fulfillment_action",
        question: str | None = None,
    ) -> dict[str, Any]:
        envelope = self.build_context(order_id=order_id, intent=intent, question=question)
        detail = self.context_builder.get_context_detail(envelope.full_context, detail_path)
        return {
            "order_id": order_id,
            "intent": intent,
            "field_groups_loaded": envelope.field_groups_loaded,
            "context_completeness": envelope.completeness,
            "detail": detail,
        }

    def get_source_detail(
        self,
        *,
        order_id: str,
        source_system: str,
        intent: str = "fulfillment_action",
        question: str | None = None,
    ) -> dict[str, Any]:
        """按源系统返回只读上下文明细。

        这是 6 个渐进式只读 Tool 的后端共享实现，保证它们都从最新
        OrderContext 构建，不读取短期记忆中的旧业务快照。
        """
        source = source_system.strip().upper()
        detail_paths = self.SOURCE_DETAIL_PATHS.get(source)
        if not detail_paths:
            raise ValueError(f"不支持的源系统：{source_system}")
        envelope = self.build_context(order_id=order_id, intent=intent, question=question)
        details = {
            path: self.context_builder.get_context_detail(envelope.full_context, path)["value"]
            for path in detail_paths
        }
        return {
            "order_id": order_id,
            "intent": intent,
            "source_system": source,
            "read_only": True,
            "field_groups_loaded": envelope.field_groups_loaded,
            "context_completeness": envelope.completeness,
            "detail_paths": detail_paths,
            "details": details,
        }
