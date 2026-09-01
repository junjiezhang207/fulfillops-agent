"""Business context adapters for OMS/WMS/TMS/ERP/PIM/CRM facts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable


LogisticsQuoteFn = Callable[[str, str, int], dict[str, Any]]
EtaLabelFn = Callable[[int], str]


@dataclass(frozen=True)
class AdapterGovernancePolicy:
    """Shared adapter controls recorded with every source-system load."""

    timeout_ms: int = 3000
    max_retries: int = 2
    pagination_strategy: str = "load_all_pages_before_merge"
    field_mapping_version: str = "fulfillops.order_context.v1"


class BusinessContextAdapterBundle:
    """Normalize facts from business domains before Context Builder merges them."""

    carriers = ("顺丰标快", "京东快运", "中通标准")

    def __init__(
        self,
        *,
        logistics_quote: LogisticsQuoteFn,
        eta_label: EtaLabelFn,
        governance: AdapterGovernancePolicy | None = None,
    ) -> None:
        self._logistics_quote = logistics_quote
        self._eta_label = eta_label
        self.governance = governance or AdapterGovernancePolicy()

    def load_oms(self, order_result: Any) -> dict[str, Any]:
        current_warehouse = getattr(order_result, "current_warehouse_id", None)
        created_at = getattr(order_result, "created_at", None)
        promise_delivery_time = getattr(order_result, "promise_delivery_time", None)
        return {
            "order_main": {
                "order_id": order_result.order_id,
                "order_status": order_result.order_status,
                "created_at": created_at.isoformat() if created_at else None,
                "promise_delivery_time": promise_delivery_time.isoformat() if promise_delivery_time else None,
                "current_warehouse_id": current_warehouse,
                "shipping_region": getattr(order_result, "shipping_region", None) or order_result.region,
                "fulfillment_type": getattr(order_result, "fulfillment_type", "普通"),
            },
            "order_items": [
                {
                    "sku_id": item.sku_id,
                    "quantity": item.quantity,
                    "allocated_quantity": item.allocated_quantity,
                    "shipped_quantity": item.shipped_quantity,
                    "sku_status": item.sku_status,
                    "split_allowed": item.split_allowed,
                    "special_storage": item.special_storage,
                    "sku_type": item.sku_type or "standard",
                }
                for item in order_result.items
            ],
            "fulfillment_state": getattr(order_result, "fulfillment_status_flags", {}) or {
                "already_split": False,
                "inventory_reserved": False,
                "package_created": False,
                "waybill_created": False,
                "outbound_completed": False,
                "active_fulfillment_tasks": [],
            },
            "adapter_metadata": {
                **self._adapter_metadata(
                    system="OMS",
                    source="OrderAnalysisService",
                    normalized_fields=("order_main", "order_items", "fulfillment_state"),
                    id_mappings=(("order_result.order_id", "order_main.order_id"),),
                    pages_loaded=1,
                ),
            },
        }

    def load_wms(self, order_result: Any, inventory_result: Any) -> dict[str, Any]:
        candidate_ids: list[str] = []
        warehouse_by_id: dict[str, dict[str, Any]] = {}
        inventory_by_pair: dict[tuple[str, str], Any] = {}

        for check in inventory_result.sku_checks:
            for record in check.warehouse_records:
                if record.warehouse_id not in warehouse_by_id:
                    candidate_ids.append(record.warehouse_id)
                    warehouse_by_id[record.warehouse_id] = {
                        "warehouse_id": record.warehouse_id,
                        "warehouse_name": record.warehouse_name,
                        "warehouse_status": record.warehouse_status,
                        "service_region": record.service_region or record.region,
                        "supported_sku_types": record.supported_sku_types or ["standard"],
                        "cutoff_time": record.cutoff_time or "18:00",
                        "capacity_status": record.capacity_status,
                    }
                inventory_by_pair[(check.sku_id, record.warehouse_id)] = record

        current_warehouse = getattr(order_result, "current_warehouse_id", None)
        if current_warehouse and current_warehouse not in warehouse_by_id:
            candidate_ids.insert(0, current_warehouse)
            warehouse_by_id[current_warehouse] = {
                "warehouse_id": current_warehouse,
                "warehouse_name": current_warehouse,
                "warehouse_status": "unknown",
                "service_region": getattr(order_result, "shipping_region", None) or order_result.region,
                "supported_sku_types": ["standard"],
                "cutoff_time": "18:00",
                "capacity_status": "unknown",
            }

        rows: list[dict[str, Any]] = []
        for item in order_result.items:
            for warehouse_id in candidate_ids:
                record = inventory_by_pair.get((item.sku_id, warehouse_id))
                rows.append({
                    "sku_id": item.sku_id,
                    "warehouse_id": warehouse_id,
                    "on_hand_stock": getattr(record, "on_hand_stock", 0) if record else 0,
                    "available_stock": record.available_stock if record else 0,
                    "reserved_stock": getattr(record, "reserved_stock", 0) if record else 0,
                    "locked_stock": record.locked_stock if record else 0,
                    "inbound_stock": getattr(record, "inbound_stock", 0) if record else 0,
                    "expected_inbound_time": (
                        record.expected_inbound_time.isoformat()
                        if record and getattr(record, "expected_inbound_time", None)
                        else None
                    ),
                    "inventory_version": getattr(record, "inventory_version", None) if record else None,
                    "updated_at": record.updated_at.isoformat() if record else None,
                    "immediate_fulfillment_field": "available_stock",
                })
        return {
            "candidate_ids": candidate_ids,
            "warehouse_by_id": warehouse_by_id,
            "candidate_warehouses": [warehouse_by_id[warehouse_id] for warehouse_id in candidate_ids],
            "sku_warehouse_inventory": rows,
            "adapter_metadata": {
                **self._adapter_metadata(
                    system="WMS",
                    source="InventoryAnalysisService",
                    normalized_fields=("candidate_warehouses", "sku_warehouse_inventory"),
                    id_mappings=(
                        ("sku_check.sku_id", "sku_warehouse_inventory.sku_id"),
                        ("warehouse_record.warehouse_id", "candidate_warehouses.warehouse_id"),
                    ),
                    pages_loaded=max(1, len(inventory_result.sku_checks)),
                ),
            },
        }

    def load_tms(self, order_result: Any, wms_context: dict[str, Any]) -> dict[str, Any]:
        options: list[dict[str, Any]] = []
        for warehouse_id in wms_context["candidate_ids"]:
            warehouse = wms_context["warehouse_by_id"][warehouse_id]
            for carrier in self.carriers:
                quote = self._logistics_quote(order_result.region, carrier, order_result.total_quantity)
                serviceable = (
                    warehouse["warehouse_status"] == "normal"
                    and warehouse["capacity_status"] in {"normal", "available", "ok"}
                )
                options.append({
                    "warehouse_id": warehouse_id,
                    "carrier": carrier,
                    "channel": carrier,
                    "serviceable": serviceable,
                    "estimated_delivery_time": self._eta_label(int(quote["eta_hours"])),
                    "eta_hours": int(quote["eta_hours"]),
                    "shipping_cost": quote["price"],
                    "cutoff_time": warehouse["cutoff_time"],
                    "channel_status": "available" if serviceable else "blocked",
                })
        return {
            "logistics_options": options,
            "adapter_metadata": {
                **self._adapter_metadata(
                    system="TMS",
                    source="deterministic_logistics_quote_adapter",
                    normalized_fields=("logistics_options",),
                    id_mappings=(("candidate_warehouses.warehouse_id", "logistics_options.warehouse_id"),),
                    pages_loaded=max(1, len(wms_context.get("candidate_ids", []))),
                ),
            },
        }

    def load_erp(self, wms_context: dict[str, Any]) -> dict[str, Any]:
        by_sku: dict[str, dict[str, Any]] = {}
        for row in wms_context["sku_warehouse_inventory"]:
            current = by_sku.setdefault(
                row["sku_id"],
                {"sku_id": row["sku_id"], "total_inbound_stock": 0, "earliest_expected_inbound_time": None},
            )
            current["total_inbound_stock"] += int(row.get("inbound_stock") or 0)
            inbound_time = row.get("expected_inbound_time")
            if inbound_time and (
                current["earliest_expected_inbound_time"] is None
                or inbound_time < current["earliest_expected_inbound_time"]
            ):
                current["earliest_expected_inbound_time"] = inbound_time
        return {
            "replenishment_options": list(by_sku.values()),
            "adapter_metadata": {
                **self._adapter_metadata(
                    system="ERP",
                    source="inventory_inbound_projection",
                    normalized_fields=("replenishment_options",),
                    id_mappings=(("sku_warehouse_inventory.sku_id", "replenishment_options.sku_id"),),
                    pages_loaded=1,
                ),
            },
        }

    def load_pim(self, order_result: Any) -> dict[str, Any]:
        return {
            "product_restrictions": [
                {
                    "sku_id": item.sku_id,
                    "sku_type": item.sku_type or "standard",
                    "split_allowed": item.split_allowed,
                    "special_storage": item.special_storage,
                    "requires_special_warehouse": bool(item.special_storage),
                }
                for item in order_result.items
            ],
            "adapter_metadata": {
                **self._adapter_metadata(
                    system="PIM",
                    source="order_item_product_attributes",
                    normalized_fields=("product_restrictions",),
                    id_mappings=(("order_items.sku_id", "product_restrictions.sku_id"),),
                    pages_loaded=max(1, len(order_result.items)),
                ),
            },
        }

    def load_crm(self, order_result: Any, question: str | None) -> dict[str, Any]:
        priority = (order_result.priority or "").lower()
        return {
            "customer_risk": {
                "priority": order_result.priority,
                "is_vip_or_urgent": any(marker in priority for marker in ("vip", "urgent", "high")),
                "question_mentions_customer": any(
                    marker in (question or "")
                    for marker in ("客户", "客服", "客诉", "投诉", "赔付")
                ),
                "complaint_open": False,
                "crm_task_required_before_customer_promise": any(marker in priority for marker in ("vip", "urgent")),
            },
            "adapter_metadata": {
                **self._adapter_metadata(
                    system="CRM",
                    source="customer_priority_projection",
                    normalized_fields=("customer_risk",),
                    id_mappings=(("order_main.order_id", "customer_risk.order_id_context"),),
                    pages_loaded=1,
                ),
            },
        }

    def _adapter_metadata(
        self,
        *,
        system: str,
        source: str,
        normalized_fields: tuple[str, ...],
        id_mappings: tuple[tuple[str, str], ...],
        pages_loaded: int,
    ) -> dict[str, Any]:
        return {
            "system": system,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "status": "success",
            "field_mapping_version": self.governance.field_mapping_version,
            "normalized_fields": list(normalized_fields),
            "id_mapping": [
                {"from": source_field, "to": target_field}
                for source_field, target_field in id_mappings
            ],
            "pagination": {
                "strategy": self.governance.pagination_strategy,
                "pages_loaded": pages_loaded,
                "complete": True,
            },
            "timeout": {"timeout_ms": self.governance.timeout_ms},
            "retry": {
                "max_retries": self.governance.max_retries,
                "attempts": 1,
                "exhausted": False,
            },
        }
