"""Schemas for fulfillment action routing and external task lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.workflow import ExecutionProposal, PreflightValidation


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RoutedExternalTask(BaseModel):
    """Task created in the target business system queue."""

    task_id: str = Field(..., description="Local idempotent task id.")
    case_id: str = Field(..., description="Fulfillment case id.")
    proposal_id: str = Field(..., description="Source proposal id.")
    action_id: str = Field(..., description="Source action id.")
    action_type: str = Field(..., description="Standard action type from the planner.")
    target_system: str = Field(..., description="WMS / TMS / ERP / CRM.")
    domain_service: str = Field(..., description="Domain service/API adapter to call.")
    collaborative_tool_name: str = Field(
        default="",
        description=(
            "协同写工具名称；只创建请求/任务，不直接修改订单、库存、物流或客户记录。"
        ),
    )
    external_task_type: str = Field(..., description="Task type in target system.")
    payload: dict[str, Any] = Field(default_factory=dict, description="Parameters passed to original API.")
    status: str = Field(
        default="CREATED",
        description="CREATED / READY / BLOCKED / RUNNING / COMPLETED / FAILED / REJECTED",
    )
    case_version: int = Field(default=1, ge=1, description="创建任务时的 Case 版本。")
    plan_version: int = Field(default=1, ge=1, description="创建任务时的 Plan 版本。")
    dependency_ids: list[str] = Field(default_factory=list, description="前置外部任务 ID。")
    idempotency_key: str = Field(default="", description="外部任务创建幂等键。")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    result: dict[str, Any] = Field(default_factory=dict)


class FulfillmentCase(BaseModel):
    """A case paused while external warehouse/logistics/ERP/CRM tasks run."""

    case_id: str
    order_id: str
    proposal_id: str
    case_status: str = Field(
        default="WAITING_EXTERNAL_TASK",
        description=(
            "NEW / ANALYZING / REVIEWING / DISPATCHING / WAITING_EXTERNAL_TASK / "
            "VERIFYING / COMPLETED / CLOSED / FAILED / REJECTED / REPLAN_REQUIRED / MANUAL"
        ),
    )
    case_version: int = Field(default=1, ge=1)
    plan: ExecutionProposal
    plan_version: int = Field(default=1, ge=1)
    context_version: str = Field(default="")
    action_dag: dict[str, Any] = Field(default_factory=dict)
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    replan_count: int = Field(default=0, ge=0)
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    current_state: dict[str, Any] = Field(default_factory=dict)
    tasks: list[RoutedExternalTask] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    verification: dict[str, Any] = Field(default_factory=dict)


class FulfillmentCaseConfirmRequest(BaseModel):
    """Approve a validated proposal and create external tasks."""

    proposal: ExecutionProposal
    preflight_validation: PreflightValidation
    approver_id: str = Field(default="operator")
    notes: str = Field(default="")
    human_changes: list[str] = Field(
        default_factory=list,
        description="HITL Modify 分支产生的人工修改记录；为空表示直接 approve。",
    )
    checkpoint: dict[str, Any] = Field(default_factory=dict)


class FulfillmentCaseReplanConfirmRequest(FulfillmentCaseConfirmRequest):
    """Approve a regenerated plan for an existing case after Replan."""


class FulfillmentCaseReviewRequest(BaseModel):
    """Record a HITL review decision before any external task is dispatched."""

    proposal: ExecutionProposal
    decision: Literal["rejected", "ask_followup"] = Field(
        default="rejected",
        description="HITL review decision: rejected returns to Planner, ask_followup keeps the case in review.",
    )
    approver_id: str = Field(default="operator")
    notes: str = Field(default="")
    checkpoint: dict[str, Any] = Field(default_factory=dict)


class ExternalTaskWebhookRequest(BaseModel):
    """Callback from WMS/TMS/ERP/CRM task queue."""

    status: str = Field(..., description="RUNNING / COMPLETED / FAILED / REJECTED")
    event_id: str | None = Field(default=None, description="外部事件 ID，用于幂等和审计。")
    case_version: int | None = Field(default=None, ge=1, description="回调携带的 Case 版本。")
    plan_version: int | None = Field(default=None, ge=1, description="回调携带的 Plan 版本。")
    idempotency_key: str | None = Field(default=None, description="回调幂等键。")
    external_ref: str | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    operator: str | None = None
    message: str | None = None


class ExternalTaskStatusPollRequest(ExternalTaskWebhookRequest):
    """Status API snapshot polled from WMS/TMS/ERP/CRM.

    It intentionally mirrors webhook fields so polling and callbacks share the
    same idempotency, case_version and plan_version validation rules.
    """

    poll_id: str | None = Field(default=None, description="本次状态轮询 ID，用于幂等和审计。")


class FulfillmentCaseVerifyResult(BaseModel):
    """Result after reloading OMS/WMS/TMS/ERP/PIM/CRM facts and validating actual state."""

    case_id: str
    order_id: str
    status: str
    message: str
    checks: list[dict[str, Any]] = Field(default_factory=list)
    replan_required: bool = False
    replacement_proposal: ExecutionProposal | None = None


class ExcellentCaseRequest(BaseModel):
    """Persist a successful case as future retrieval material."""

    operator_id: str = Field(default="operator")
    notes: str = Field(default="")


class ManualProcessingRequest(BaseModel):
    """Mark a MANUAL case as being handled by a human operator."""

    operator_id: str = Field(default="operator")
    notes: str = Field(default="")


class ManualCompletionRequest(BaseModel):
    """Record manual business completion and resume verification."""

    operator_id: str = Field(default="operator")
    notes: str = Field(default="")
    business_result: dict[str, Any] = Field(
        default_factory=dict,
        description="人工在 WMS/TMS/ERP/CRM 完成真实操作后的业务证明。",
    )


class ExcellentCaseRecord(BaseModel):
    """Desensitized summary saved to PostgreSQL and indexed by PGVector later."""

    excellent_case_id: str
    case_id: str
    order_id_masked: str
    scenario: str
    key_conditions: list[str] = Field(default_factory=list)
    final_plan: list[dict[str, Any]] = Field(default_factory=list)
    human_changes: list[str] = Field(default_factory=list)
    final_result: dict[str, Any] = Field(default_factory=dict)
    sop_evidence: list[str] = Field(default_factory=list)
    execution_result: dict[str, Any] = Field(default_factory=dict)
    vector_backend: str = Field(default="pgvector")
    extraction_use_case: str = Field(default="case_extraction")
    extraction_mode: str = Field(default="deterministic_local")
    extraction_warnings: list[str] = Field(default_factory=list)
    prompt_version: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class CaseIngestionJob(BaseModel):
    """Background job state for excellent-case vector ingestion."""

    job_id: str
    case_id: str
    status: str = Field(default="PENDING", description="PENDING / PROCESSING / SUCCESS / FAILED")
    retry_count: int = Field(default=0, ge=0)
    operator_id: str = Field(default="operator")
    notes: str = Field(default="")
    excellent_case_id: str | None = None
    knowledge_path: str | None = None
    error: str | None = None
    dead_letter: bool = False
    manual_queue: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
