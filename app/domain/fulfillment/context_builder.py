"""OrderContext Builder for fulfillment decisions.

The builder is deterministic application code: it standardizes facts from the
business services before the Agent sees them.  The Agent receives fixed core
fields plus intent-related field groups, while large detail sections stay
available by path through ``get_context_detail``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Callable

from app.domain.fulfillment.business_context_adapters import BusinessContextAdapterBundle


LogisticsQuoteFn = Callable[[str, str, int], dict[str, Any]]
EtaLabelFn = Callable[[int], str]


_CONTEXT_ADAPTER_EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="order-context-adapter",
)


FIELD_GROUPS: dict[str, list[str]] = {
    "core": ["order_main", "order_items"],
    "warehouse": ["candidate_warehouses"],
    "inventory": ["sku_warehouse_inventory"],
    "logistics": ["logistics_options"],
    "fulfillment_state": ["fulfillment_state"],
    "replenishment": ["replenishment_options"],
    "product_restrictions": ["product_restrictions"],
    "customer_risk": ["customer_risk"],
}

INTENT_FIELD_GROUPS: dict[str, list[str]] = {
    "fulfillment_action": [
        "core",
        "warehouse",
        "inventory",
        "logistics",
        "fulfillment_state",
        "product_restrictions",
        "replenishment",
    ],
    "stockout_resolution": [
        "core",
        "inventory",
        "warehouse",
        "logistics",
        "fulfillment_state",
        "replenishment",
        "customer_risk",
    ],
    "logistics_exception": ["core", "warehouse", "logistics", "fulfillment_state"],
    "case_review": ["core", "inventory", "fulfillment_state", "customer_risk"],
}


@dataclass(frozen=True)
class OrderContextEnvelope:
    """Full normalized context plus the selected slice passed to the Agent."""

    order_id: str
    intent: str
    full_context: dict[str, Any]
    selected_context: dict[str, Any]
    field_groups_loaded: list[str]
    completeness: dict[str, Any]


class OrderContextBuilder:
    """Build and slice OrderContext JSON from deterministic business facts."""

    carriers = ("顺丰标快", "京东快运", "中通标准")

    def __init__(
        self,
        *,
        logistics_quote: LogisticsQuoteFn | None = None,
        eta_label: EtaLabelFn | None = None,
        adapters: BusinessContextAdapterBundle | None = None,
    ) -> None:
        self._logistics_quote = logistics_quote or self._default_logistics_quote
        self._eta_label = eta_label or self._default_eta_label
        self.adapters = adapters or BusinessContextAdapterBundle(
            logistics_quote=self._logistics_quote,
            eta_label=self._eta_label,
        )

    def build_from_results(
        self,
        order_result: Any,
        inventory_result: Any,
        *,
        intent: str = "fulfillment_action",
        question: str | None = None,
    ) -> OrderContextEnvelope:
        """Build a complete context, then select the field groups for intent."""

        full_context = self._build_full_context(order_result, inventory_result, question=question)
        field_groups = self.resolve_field_groups(intent=intent, question=question)
        selected_context = self.select_field_groups(full_context, field_groups)
        completeness = self._validate_completeness(selected_context)
        selected_context["field_groups_loaded"] = field_groups
        selected_context["context_completeness"] = completeness
        return OrderContextEnvelope(
            order_id=order_result.order_id,
            intent=intent,
            full_context=full_context,
            selected_context=selected_context,
            field_groups_loaded=field_groups,
            completeness=completeness,
        )

    def resolve_field_groups(self, *, intent: str, question: str | None = None) -> list[str]:
        groups = list(INTENT_FIELD_GROUPS.get(intent, INTENT_FIELD_GROUPS["fulfillment_action"]))
        text = (question or "").lower()
        if any(marker in text for marker in ("物流", "carrier", "channel", "快递", "时效")):
            groups.append("logistics")
        if any(marker in text for marker in ("调拨", "补货", "在途", "缺货", "stockout")):
            groups.extend(["inventory", "replenishment"])
        if any(marker in text for marker in ("客诉", "客户", "客服", "赔付", "投诉")):
            groups.append("customer_risk")

        ordered: list[str] = []
        for group in ["core", *groups]:
            if group in FIELD_GROUPS and group not in ordered:
                ordered.append(group)
        return ordered

    def select_field_groups(self, full_context: dict[str, Any], field_groups: list[str]) -> dict[str, Any]:
        selected: dict[str, Any] = {
            "loaded_at": full_context["loaded_at"],
            "adapter_trace": full_context["adapter_trace"],
            "adapter_errors": full_context.get("adapter_errors", []),
            "source_systems": full_context["source_systems"],
            "data_loading_policy": full_context["data_loading_policy"],
            "detail_paths": full_context["detail_paths"],
        }
        for field in full_context["data_loading_policy"]["fixed_required_by_program"]:
            if field in full_context:
                selected[field] = full_context[field]
        for group in field_groups:
            for field in FIELD_GROUPS.get(group, []):
                if field in full_context:
                    selected[field] = full_context[field]
        return selected

    @staticmethod
    def get_context_detail(full_context: dict[str, Any], path: str) -> dict[str, Any]:
        """Return one expandable detail group or dotted path from full context."""

        normalized = path.strip().strip("/")
        if not normalized:
            return {"path": path, "found": True, "value": full_context}
        if normalized in FIELD_GROUPS:
            return {
                "path": normalized,
                "found": True,
                "value": {
                    field: full_context.get(field)
                    for field in FIELD_GROUPS[normalized]
                    if field in full_context
                },
            }
        current: Any = full_context
        for part in normalized.replace("/", ".").split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return {"path": path, "found": False, "value": None}
        return {"path": path, "found": True, "value": current}

    def _build_full_context(self, order_result: Any, inventory_result: Any, *, question: str | None) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        adapter_errors: list[dict[str, Any]] = []
        oms_context = self._load_adapter_context(
            "OMS",
            lambda: self.adapters.load_oms(order_result),
            self._failed_source_context(
                "OMS",
                "OrderAnalysisService",
                ("order_main", "order_items", "fulfillment_state"),
            ),
            adapter_errors,
        )
        wms_context = self._load_adapter_context(
            "WMS",
            lambda: self.adapters.load_wms(order_result, inventory_result),
            self._failed_source_context(
                "WMS",
                "InventoryAnalysisService",
                ("candidate_ids", "warehouse_by_id", "candidate_warehouses", "sku_warehouse_inventory"),
            ),
            adapter_errors,
        )
        tms_context = self._load_adapter_context(
            "TMS",
            lambda: self.adapters.load_tms(order_result, wms_context),
            self._failed_source_context("TMS", "deterministic_logistics_quote_adapter", ("logistics_options",)),
            adapter_errors,
        )
        erp_context = self._load_adapter_context(
            "ERP",
            lambda: self.adapters.load_erp(wms_context),
            self._failed_source_context("ERP", "inventory_inbound_projection", ("replenishment_options",)),
            adapter_errors,
        )
        pim_context = self._load_adapter_context(
            "PIM",
            lambda: self.adapters.load_pim(order_result),
            self._failed_source_context("PIM", "order_item_product_attributes", ("product_restrictions",)),
            adapter_errors,
        )
        crm_context = self._load_adapter_context(
            "CRM",
            lambda: self.adapters.load_crm(order_result, question),
            self._failed_source_context("CRM", "customer_priority_projection", ("customer_risk",)),
            adapter_errors,
        )

        return {
            "loaded_at": now,
            "adapter_errors": adapter_errors,
            "adapter_trace": {
                "OMS": oms_context["adapter_metadata"],
                "WMS": wms_context["adapter_metadata"],
                "TMS": tms_context["adapter_metadata"],
                "ERP": erp_context["adapter_metadata"],
                "PIM": pim_context["adapter_metadata"],
                "CRM": crm_context["adapter_metadata"],
            },
            "source_systems": {
                "OMS": ["order_main", "order_items", "fulfillment_state"],
                "WMS": ["candidate_warehouses", "sku_warehouse_inventory"],
                "TMS": ["logistics_options"],
                "ERP": ["replenishment_options"],
                "PIM": ["product_restrictions"],
                "CRM": ["customer_risk"],
            },
            "data_loading_policy": {
                "fixed_required_by_program": [
                    "order_main",
                    "order_items",
                    "candidate_warehouses",
                    "sku_warehouse_inventory",
                    "logistics_options",
                    "fulfillment_state",
                ],
                "field_group_strategy": "core_plus_intent_groups",
                "agent_may_expand_only_for_large_details": True,
                "immediate_inventory_judgement_field": "available_stock",
                "vector_store_backend": "pgvector",
                "pgvector_status": "default_postgresql_vector_backend",
            },
            "detail_paths": {
                "warehouse": "candidate_warehouses",
                "inventory": "sku_warehouse_inventory",
                "logistics": "logistics_options",
                "replenishment": "replenishment_options",
                "product_restrictions": "product_restrictions",
                "customer_risk": "customer_risk",
            },
            "order_main": oms_context.get("order_main", {}),
            "order_items": oms_context.get("order_items", []),
            "candidate_warehouses": wms_context.get("candidate_warehouses", []),
            "sku_warehouse_inventory": wms_context.get("sku_warehouse_inventory", []),
            "logistics_options": tms_context.get("logistics_options", []),
            "fulfillment_state": oms_context.get("fulfillment_state", {}),
            "replenishment_options": erp_context.get("replenishment_options", []),
            "product_restrictions": pim_context.get("product_restrictions", []),
            "customer_risk": crm_context.get("customer_risk", {}),
        }

    def _load_adapter_context(
        self,
        system: str,
        loader: Callable[[], dict[str, Any]],
        fallback: dict[str, Any],
        adapter_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Load one source-system adapter and preserve a DATA_CONFLICT-able failure."""

        max_retries = max(0, int(getattr(self.adapters.governance, "max_retries", 0)))
        last_error: Exception | None = None
        started = monotonic()
        for attempt in range(1, max_retries + 2):
            try:
                context = self._invoke_adapter_loader(system, loader)
                metadata = dict(context.get("adapter_metadata") or {})
                metadata.setdefault("system", system)
                metadata.setdefault("status", "success")
                retry = dict(metadata.get("retry") or {})
                retry["attempts"] = attempt
                retry.setdefault("max_retries", max_retries)
                retry["exhausted"] = False
                metadata["retry"] = retry
                metadata["latency_ms"] = round((monotonic() - started) * 1000, 2)
                context["adapter_metadata"] = metadata
                return context
            except Exception as exc:  # noqa: BLE001 - adapter failures must become traceable business conflicts.
                last_error = exc

        failed = dict(fallback)
        metadata = dict(failed.get("adapter_metadata") or {})
        error_message = str(last_error) if last_error else "unknown adapter failure"
        error_code = "ADAPTER_TIMEOUT" if isinstance(last_error, TimeoutError) else "ADAPTER_LOAD_FAILED"
        metadata.update({
            "system": system,
            "status": "failed",
            "error_code": error_code,
            "error_message": error_message,
            "latency_ms": round((monotonic() - started) * 1000, 2),
            "retry": {
                "max_retries": max_retries,
                "attempts": max_retries + 1,
                "exhausted": True,
            },
            "timeout": {"timeout_ms": getattr(self.adapters.governance, "timeout_ms", 3000)},
            "pagination": {
                "strategy": getattr(self.adapters.governance, "pagination_strategy", "load_all_pages_before_merge"),
                "pages_loaded": 0,
                "complete": False,
            },
        })
        failed["adapter_metadata"] = metadata
        adapter_errors.append({
            "source_system": system,
            "error_code": error_code,
            "error_message": error_message,
            "retry_exhausted": True,
        })
        return failed

    def _invoke_adapter_loader(self, system: str, loader: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        timeout_ms = max(1, int(getattr(self.adapters.governance, "timeout_ms", 3000)))
        future = _CONTEXT_ADAPTER_EXECUTOR.submit(loader)
        try:
            return future.result(timeout=timeout_ms / 1000)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"{system} adapter load exceeded {timeout_ms}ms") from exc

    def _failed_source_context(self, system: str, source: str, fields: tuple[str, ...]) -> dict[str, Any]:
        context: dict[str, Any] = {field: {} if field in {"order_main", "fulfillment_state", "warehouse_by_id", "customer_risk"} else [] for field in fields}
        context["adapter_metadata"] = {
            "system": system,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "status": "failed",
            "field_mapping_version": getattr(self.adapters.governance, "field_mapping_version", "fulfillops.order_context.v1"),
            "normalized_fields": list(fields),
            "id_mapping": [],
        }
        return context

    @staticmethod
    def _order_main(order_result: Any, current_warehouse: str | None) -> dict[str, Any]:
        created_at = getattr(order_result, "created_at", None)
        promise_delivery_time = getattr(order_result, "promise_delivery_time", None)
        return {
            "order_id": order_result.order_id,
            "order_status": order_result.order_status,
            "created_at": created_at.isoformat() if created_at else None,
            "promise_delivery_time": promise_delivery_time.isoformat() if promise_delivery_time else None,
            "current_warehouse_id": current_warehouse,
            "shipping_region": getattr(order_result, "shipping_region", None) or order_result.region,
            "fulfillment_type": getattr(order_result, "fulfillment_type", "普通"),
        }

    @staticmethod
    def _order_items(order_result: Any) -> list[dict[str, Any]]:
        return [
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
        ]

    @staticmethod
    def _sku_warehouse_inventory(
        order_result: Any,
        candidate_ids: list[str],
        inventory_by_pair: dict[tuple[str, str], Any],
    ) -> list[dict[str, Any]]:
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
        return rows

    def _logistics_options(
        self,
        order_result: Any,
        candidate_ids: list[str],
        warehouse_by_id: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        options: list[dict[str, Any]] = []
        for warehouse_id in candidate_ids:
            warehouse = warehouse_by_id[warehouse_id]
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
        return options

    @staticmethod
    def _replenishment_options(sku_inventory_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_sku: dict[str, dict[str, Any]] = {}
        for row in sku_inventory_rows:
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
        return list(by_sku.values())

    @staticmethod
    def _product_restrictions(order_result: Any) -> list[dict[str, Any]]:
        return [
            {
                "sku_id": item.sku_id,
                "sku_type": item.sku_type or "standard",
                "split_allowed": item.split_allowed,
                "special_storage": item.special_storage,
                "requires_special_warehouse": bool(item.special_storage),
            }
            for item in order_result.items
        ]

    @staticmethod
    def _customer_risk(order_result: Any, question: str | None) -> dict[str, Any]:
        priority = (order_result.priority or "").lower()
        return {
            "priority": order_result.priority,
            "is_vip_or_urgent": any(marker in priority for marker in ("vip", "urgent", "high")),
            "question_mentions_customer": any(
                marker in (question or "")
                for marker in ("客户", "客服", "客诉", "投诉", "赔付")
            ),
            "complaint_open": False,
            "crm_task_required_before_customer_promise": any(marker in priority for marker in ("vip", "urgent")),
        }

    @staticmethod
    def _validate_completeness(selected_context: dict[str, Any]) -> dict[str, Any]:
        required = selected_context.get("data_loading_policy", {}).get("fixed_required_by_program", [])
        missing = [
            field
            for field in required
            if field not in selected_context or selected_context.get(field) in (None, [], {})
        ]
        adapter_failures = [
            {
                "source_system": system,
                "error_code": metadata.get("error_code", "ADAPTER_LOAD_FAILED"),
                "error_message": metadata.get("error_message", ""),
            }
            for system, metadata in selected_context.get("adapter_trace", {}).items()
            if metadata.get("status") not in (None, "success")
        ]
        status = "complete" if not missing and not adapter_failures else "DATA_CONFLICT"
        return {
            "status": status,
            "missing_required_fields": missing,
            "adapter_failures": adapter_failures,
            "loaded_required_fields": [field for field in required if field not in missing],
            "large_details_expandable_by_path": list(selected_context.get("detail_paths", {}).values()),
            "recovery_action": (
                "continue"
                if status == "complete"
                else "reload_business_sources_or_request_human_verification"
            ),
        }

    @staticmethod
    def _default_logistics_quote(region: str, carrier: str, quantity: int) -> dict[str, Any]:
        base_price = 10.0 + max(quantity - 1, 0) * 2.0
        eta = 48
        if "顺丰" in carrier:
            base_price += 8.0
            eta = 24
        elif "京东" in carrier:
            base_price += 5.0
            eta = 36
        if region and any(marker in region for marker in ("新疆", "西藏", "内蒙古")):
            eta += 48
            base_price += 12.0
        return {"price": round(base_price, 2), "eta_hours": eta}

    @staticmethod
    def _default_eta_label(hours: int) -> str:
        if hours <= 24:
            return "约 1 天"
        if hours <= 48:
            return "约 2 天"
        return f"约 {max(1, round(hours / 24))} 天"
