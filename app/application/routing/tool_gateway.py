"""Tool Gateway for submitting routed tasks to business-system adapters.

The Agent never mutates OMS/WMS/TMS/ERP/CRM records directly.  It produces
standard actions; ActionRouter maps them to target systems; this gateway calls
the corresponding adapter and records the external task receipt.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from app.agents.tools.contracts import authorize_tool_call
from app.agents.tools.guardrails import get_tool_sanitizer
from app.application.routing.action_router import (
    COLLABORATIVE_TASK_PAYLOAD_SCHEMA_VERSION,
    COLLABORATIVE_WRITE_TOOL_BY_SYSTEM,
)
from app.observability.business_trace import add_trace_step
from app.schemas.fulfillment_case import RoutedExternalTask


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_TOOL_GATEWAY_EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="collaborative-tool-gateway",
)


@dataclass(frozen=True)
class ExternalTaskReceipt:
    """Receipt returned by an original business system task API."""

    external_ref: str
    status: str
    submitted_at: datetime
    message: str


class BusinessSystemAdapter(Protocol):
    system_name: str

    def create_task(self, task: RoutedExternalTask) -> ExternalTaskReceipt: ...


class ToolGatewayCircuitOpen(RuntimeError):
    """Raised when a target system is temporarily circuit-broken."""


@dataclass
class GatewayCircuitBreaker:
    """Small per-system circuit breaker for collaborative write adapters."""

    failure_threshold: int = 3
    recovery_seconds: float = 30.0
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"

    def is_open(self) -> bool:
        if self.state != "open":
            return False
        if time.monotonic() - self.last_failure_time >= self.recovery_seconds:
            self.state = "half_open"
            return False
        return True

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.state = "closed"

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()
        if self.consecutive_failures >= self.failure_threshold:
            self.state = "open"


class DeterministicBusinessSystemAdapter:
    """Local adapter used until real WMS/TMS/ERP/CRM APIs are wired in."""

    def __init__(self, system_name: str) -> None:
        self.system_name = system_name

    def create_task(self, task: RoutedExternalTask) -> ExternalTaskReceipt:
        digest = hashlib.sha1(
            f"{self.system_name}:{task.case_id}:{task.task_id}:{task.external_task_type}".encode("utf-8")
        ).hexdigest()[:12]
        return ExternalTaskReceipt(
            external_ref=f"{self.system_name.lower()}-{digest}",
            status="RUNNING",
            submitted_at=_utc_now(),
            message=(
                f"{self.system_name} 已通过 {task.collaborative_tool_name or task.domain_service} "
                f"接收 {task.external_task_type}，后续由原系统队列分配处理人。"
            ),
        )


class ToolGateway:
    """Submit routed tasks to target adapters and attach receipts."""

    def __init__(
        self,
        adapters: dict[str, BusinessSystemAdapter] | None = None,
        *,
        max_retries: int = 2,
        timeout_seconds: float = 10.0,
        failure_threshold: int = 3,
        recovery_seconds: float = 30.0,
    ) -> None:
        self.adapters = adapters or {
            system: DeterministicBusinessSystemAdapter(system)
            for system in ("WMS", "TMS", "ERP", "CRM")
        }
        self.max_retries = max(0, int(max_retries))
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self._sanitizer = get_tool_sanitizer()
        self._breakers = {
            system: GatewayCircuitBreaker(
                failure_threshold=max(1, int(failure_threshold)),
                recovery_seconds=max(0.1, float(recovery_seconds)),
            )
            for system in self.adapters
        }
        self._submitted_by_idempotency: dict[str, RoutedExternalTask] = {}

    def submit(self, task: RoutedExternalTask) -> RoutedExternalTask:
        started = time.monotonic()
        self._validate_collaborative_write_task(task)
        existing = self._submitted_by_idempotency.get(task.idempotency_key)
        if existing is not None:
            if (
                existing.task_id != task.task_id
                or existing.case_id != task.case_id
                or existing.plan_version != task.plan_version
            ):
                raise ValueError("协同写任务 idempotency_key 已被不同任务使用，拒绝复用。")
            self._record_submit_trace(existing, started, "idempotent_replay")
            return existing
        adapter = self.adapters.get(task.target_system)
        if adapter is None:
            raise ValueError(f"未配置 {task.target_system} 适配器，拒绝静默吞掉外部任务。")
        breaker = self._breakers.setdefault(task.target_system, GatewayCircuitBreaker())
        if breaker.is_open():
            exc = ToolGatewayCircuitOpen(
                f"{task.target_system} 协同写通道熔断中，暂不提交 {task.task_id}。"
            )
            self._record_submit_trace(task, started, "circuit_open", error=exc)
            raise exc

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                receipt = self._submit_with_timeout(adapter, task)
                self._validate_receipt(receipt)
                breaker.record_success()
                updated = self._task_with_receipt(task, adapter, receipt, retry_count=attempt)
                self._submitted_by_idempotency[task.idempotency_key] = updated
                self._record_submit_trace(updated, started, "success", retry_count=attempt)
                return updated
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    breaker.record_failure()
                    self._record_submit_trace(
                        task,
                        started,
                        "error",
                        retry_count=attempt,
                        error=exc,
                    )
                    raise RuntimeError(
                        f"{task.target_system} 协同写任务提交失败：{exc}"
                    ) from exc

        raise RuntimeError(f"{task.target_system} 协同写任务提交失败：{last_exc}")

    def _submit_with_timeout(
        self,
        adapter: BusinessSystemAdapter,
        task: RoutedExternalTask,
    ) -> ExternalTaskReceipt:
        future = _TOOL_GATEWAY_EXECUTOR.submit(adapter.create_task, task)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"{task.collaborative_tool_name} 超过 {self.timeout_seconds:.1f}s") from exc

    def _task_with_receipt(
        self,
        task: RoutedExternalTask,
        adapter: BusinessSystemAdapter,
        receipt: ExternalTaskReceipt,
        *,
        retry_count: int,
    ) -> RoutedExternalTask:
        external_ref = self._sanitizer.sanitize(receipt.external_ref, source=task.collaborative_tool_name)
        message = self._sanitizer.sanitize(receipt.message, source=task.collaborative_tool_name)
        return task.model_copy(
            update={
                "status": receipt.status,
                "updated_at": receipt.submitted_at,
                "result": {
                    **task.result,
                    "external_ref": external_ref,
                    "submitted_at": receipt.submitted_at.isoformat(),
                    "submission_message": message,
                    "adapter": adapter.system_name,
                    "domain_service": task.domain_service,
                    "collaborative_tool_name": task.collaborative_tool_name,
                    "tool_gateway": {
                        "status": "submitted",
                        "retry_count": retry_count,
                        "timeout_seconds": self.timeout_seconds,
                        "permission_checked": True,
                        "output_sanitized": external_ref != receipt.external_ref or message != receipt.message,
                    },
                },
            }
        )

    @staticmethod
    def _validate_collaborative_write_task(task: RoutedExternalTask) -> None:
        expected_tool = COLLABORATIVE_WRITE_TOOL_BY_SYSTEM.get(task.target_system)
        if not expected_tool or task.collaborative_tool_name != expected_tool:
            raise ValueError(
                f"{task.target_system} 协同写工具不匹配："
                f"expected={expected_tool}, actual={task.collaborative_tool_name}"
            )
        if task.payload.get("hitl") != "APPROVED":
            raise ValueError("协同写任务缺少 HITL=APPROVED，拒绝创建外部任务。")
        if task.plan_version < 1 or task.case_version < 1:
            raise ValueError("协同写任务缺少有效 case_version / plan_version。")
        if not task.idempotency_key:
            raise ValueError("协同写任务缺少 idempotency_key。")
        ToolGateway._validate_payload_schema(task)
        authorize_tool_call(task.collaborative_tool_name)

    @staticmethod
    def _validate_payload_schema(task: RoutedExternalTask) -> None:
        payload = task.payload or {}
        schema_version = payload.get("schema_version")
        if schema_version != COLLABORATIVE_TASK_PAYLOAD_SCHEMA_VERSION:
            raise ValueError(
                "协同写任务 payload schema_version 无效："
                f"expected={COLLABORATIVE_TASK_PAYLOAD_SCHEMA_VERSION}, actual={schema_version}"
            )
        required_common = {
            "case_id": task.case_id,
            "case_version": task.case_version,
            "order_id": None,
            "proposal_id": task.proposal_id,
            "action_id": task.action_id,
            "action_type": task.action_type,
            "collaborative_tool_name": task.collaborative_tool_name,
            "plan_version": task.plan_version,
            "context_version": None,
            "data_fingerprint": None,
            "success_criteria": None,
        }
        missing_or_mismatch: list[str] = []
        for field, expected in required_common.items():
            value = payload.get(field)
            if value in (None, ""):
                missing_or_mismatch.append(field)
                continue
            if expected is not None and value != expected:
                missing_or_mismatch.append(field)
        if not isinstance(payload.get("success_criteria"), dict):
            missing_or_mismatch.append("success_criteria")
        if missing_or_mismatch:
            raise ValueError(
                "协同写任务 payload schema 校验失败，字段缺失或不一致："
                + ", ".join(sorted(set(missing_or_mismatch)))
            )

        action_errors = ToolGateway._validate_action_payload(task.action_type, payload)
        if action_errors:
            raise ValueError(
                "协同写任务 payload 动作参数校验失败："
                + ", ".join(action_errors)
            )

    @staticmethod
    def _validate_action_payload(action_type: str, payload: dict) -> list[str]:
        errors: list[str] = []

        def require_text(field: str) -> None:
            if not str(payload.get(field) or "").strip():
                errors.append(field)

        def require_quantity() -> None:
            quantity = payload.get("quantity")
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                errors.append("quantity")

        if action_type in {"ship_from_warehouse", "split_order"}:
            require_text("sku_id")
            require_quantity()
            require_text("from_warehouse")
            require_text("carrier")
        elif action_type == "merge_order":
            require_quantity()
            require_text("from_warehouse")
            require_text("carrier")
        elif action_type in {"switch_warehouse", "inventory_transfer"}:
            require_text("sku_id")
            require_quantity()
            require_text("to_warehouse")
        elif action_type in {"change_carrier", "logistics_exception"}:
            require_text("carrier")
            require_text("reason")
        elif action_type == "replenishment":
            require_text("sku_id")
            require_quantity()
        elif action_type in {"stockout_resolution", "customer_complaint"}:
            require_text("reason")
        else:
            errors.append("action_type")
        return errors

    @staticmethod
    def _validate_receipt(receipt: ExternalTaskReceipt) -> None:
        if not receipt.external_ref:
            raise ValueError("协同写任务回执缺少 external_ref。")
        if receipt.status not in {"CREATED", "READY", "RUNNING", "COMPLETED", "FAILED", "REJECTED"}:
            raise ValueError(f"协同写任务回执状态非法：{receipt.status}")

    def _record_submit_trace(
        self,
        task: RoutedExternalTask,
        started: float,
        status: str,
        *,
        retry_count: int = 0,
        error: Exception | None = None,
    ) -> None:
        add_trace_step(
            step_type="tool_gateway",
            name=task.collaborative_tool_name,
            status=status,
            duration_ms=(time.monotonic() - started) * 1000,
            summary=(
                f"{task.target_system} 协同写任务幂等复用已提交回执"
                if status == "idempotent_replay"
                else f"{task.target_system} 协同写任务已提交"
                if error is None
                else f"{task.target_system} 协同写任务提交失败"
            ),
            error_code=error.__class__.__name__ if error else None,
            error_message=str(error) if error else None,
            input_summary={
                "case_id": task.case_id,
                "task_id": task.task_id,
                "target_system": task.target_system,
                "action_type": task.action_type,
            },
            metadata={
                "retry_count": retry_count,
                "timeout_seconds": self.timeout_seconds,
                "case_version": task.case_version,
                "plan_version": task.plan_version,
                "idempotency_key_present": bool(task.idempotency_key),
            },
        )
