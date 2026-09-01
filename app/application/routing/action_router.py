"""Deterministic router from Agent proposal actions to business systems."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from app.schemas.fulfillment_case import RoutedExternalTask
from app.schemas.workflow import ExecutionProposal, ProposalAction


COLLABORATIVE_WRITE_TOOL_BY_SYSTEM: dict[str, str] = {
    "WMS": "create_warehouse_request",
    "TMS": "create_logistics_request",
    "ERP": "create_supply_chain_request",
    "CRM": "create_customer_service_task",
}

COLLABORATIVE_TASK_PAYLOAD_SCHEMA_VERSION = "fulfillops.collaborative_task.v1"


ACTION_ROUTE_MAP: dict[str, tuple[str, str, str]] = {
    "ship_from_warehouse": ("WMS", "create_warehouse_request", "FULFILL_FROM_WAREHOUSE"),
    "switch_warehouse": ("WMS", "create_warehouse_request", "SWITCH_FULFILLMENT_WAREHOUSE"),
    "split_order": ("WMS", "create_warehouse_request", "SPLIT_ORDER_FULFILLMENT"),
    "merge_order": ("WMS", "create_warehouse_request", "MERGE_ORDER_FULFILLMENT"),
    "inventory_transfer": ("WMS", "create_warehouse_request", "INVENTORY_TRANSFER"),
    "change_carrier": ("TMS", "create_logistics_request", "CHANGE_CARRIER"),
    "logistics_exception": ("TMS", "create_logistics_request", "LOGISTICS_EXCEPTION"),
    "replenishment": ("ERP", "create_supply_chain_request", "REPLENISHMENT_REQUEST"),
    "stockout_resolution": ("CRM", "create_customer_service_task", "CUSTOMER_STOCKOUT_CONFIRMATION"),
    "customer_complaint": ("CRM", "create_customer_service_task", "CUSTOMER_COMPLAINT_FOLLOWUP"),
}


class ActionRouter:
    """Create external task requests without mutating orders or inventory."""

    def route(
        self,
        *,
        case_id: str,
        proposal: ExecutionProposal,
        case_version: int = 1,
    ) -> list[RoutedExternalTask]:
        tasks: list[RoutedExternalTask] = []
        for index, action in enumerate(proposal.actions, start=1):
            target_system, domain_service, external_task_type = self._route_action(action)
            collaborative_tool_name = COLLABORATIVE_WRITE_TOOL_BY_SYSTEM[target_system]
            task_id = self._task_id(case_id, proposal.plan_version, action.action_id, target_system, index)
            now = datetime.now(timezone.utc)
            tasks.append(
                RoutedExternalTask(
                    task_id=task_id,
                    case_id=case_id,
                    proposal_id=proposal.proposal_id,
                    action_id=action.action_id,
                    action_type=action.action_type,
                    target_system=target_system,
                    domain_service=domain_service,
                    collaborative_tool_name=collaborative_tool_name,
                    external_task_type=external_task_type,
                    payload=self._payload(case_id, case_version, proposal, action, collaborative_tool_name),
                    status="CREATED",
                    case_version=case_version,
                    plan_version=proposal.plan_version,
                    dependency_ids=list(action.depends_on),
                    idempotency_key=self._idempotency_key(case_id, proposal, action),
                    created_at=now,
                    updated_at=now,
                )
            )
        return tasks

    @staticmethod
    def _route_action(action: ProposalAction) -> tuple[str, str, str]:
        route = ACTION_ROUTE_MAP.get(action.action_type)
        if route is None:
            raise ValueError(f"未知或未授权的协同动作类型：{action.action_type}")
        return route

    @staticmethod
    def _task_id(case_id: str, plan_version: int, action_id: str, target_system: str, index: int) -> str:
        digest = hashlib.sha1(
            f"{case_id}:{plan_version}:{action_id}:{target_system}:{index}".encode("utf-8")
        ).hexdigest()[:10]
        return f"task-{target_system.lower()}-{digest}"

    @staticmethod
    def _idempotency_key(case_id: str, proposal: ExecutionProposal, action: ProposalAction) -> str:
        context_version = ActionRouter._context_version(proposal)
        raw = f"{case_id}:{proposal.plan_version}:{context_version}:{action.action_id}:{action.action_type}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _context_version(proposal: ExecutionProposal) -> str:
        if proposal.context_version:
            return proposal.context_version
        return f"ctx-{proposal.data_fingerprint[:12]}"

    @staticmethod
    def _payload(
        case_id: str,
        case_version: int,
        proposal: ExecutionProposal,
        action: ProposalAction,
        collaborative_tool_name: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": COLLABORATIVE_TASK_PAYLOAD_SCHEMA_VERSION,
            "case_id": case_id,
            "case_version": case_version,
            "order_id": proposal.order_id,
            "proposal_id": proposal.proposal_id,
            "action_id": action.action_id,
            "action_type": action.action_type,
            "collaborative_tool_name": collaborative_tool_name,
            "sku_id": action.sku_id,
            "quantity": action.quantity,
            "from_warehouse": action.from_warehouse,
            "to_warehouse": action.to_warehouse,
            "carrier": action.carrier,
            "reason": action.reason,
            "sop_evidence": proposal.rule_citations,
            "data_fingerprint": proposal.data_fingerprint,
            "plan_version": proposal.plan_version,
            "context_version": ActionRouter._context_version(proposal),
            "goal_type": proposal.goal_type,
            "success_criteria": proposal.success_criteria,
            "reversible": action.reversible,
        }
