"""Fulfillment case APIs after HITL approval."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from app.agents.tools.contracts import ToolRuntimeContext, get_tool_runtime_context, tool_runtime_context
from app.application.routing.excellent_case_knowledge import ExcellentCaseKnowledgeWriter
from app.application.routing.fulfillment_case_service import (
    FulfillmentCaseService,
    PostgreSQLFulfillmentCaseStore,
)
from app.core.config import get_settings
from app.core.service_registry import (
    get_enterprise_data_repository,
    get_inventory_analysis_service,
    get_knowledge_retrieval_service,
    get_order_analysis_service,
)
from app.schemas.common import ApiResponse
from app.schemas.fulfillment_case import (
    ExcellentCaseRequest,
    FulfillmentCaseReplanConfirmRequest,
    FulfillmentCaseReviewRequest,
    ExternalTaskStatusPollRequest,
    ExternalTaskWebhookRequest,
    FulfillmentCaseConfirmRequest,
    ManualCompletionRequest,
    ManualProcessingRequest,
)
from app.workflows.fulfillment.nodes import WorkflowNodes

router = APIRouter(prefix="/fulfillment-cases")


def _case_dispatch_message(case_status: str) -> str:
    if case_status == "WAITING_EXTERNAL_TASK":
        return "已创建外部系统任务，案件进入 WAITING_EXTERNAL_TASK。"
    if case_status == "REPLAN_REQUIRED":
        return "外部系统任务创建失败，案件已进入 REPLAN_REQUIRED。"
    if case_status == "MANUAL":
        return "外部系统任务创建失败且重规划次数已耗尽，案件已进入 MANUAL。"
    return f"外部系统任务处理完成，案件状态：{case_status}"


@lru_cache
def get_fulfillment_case_service() -> FulfillmentCaseService:
    order_service = get_order_analysis_service()
    inventory_service = get_inventory_analysis_service()
    knowledge_service = get_knowledge_retrieval_service()
    nodes = WorkflowNodes(
        order_service=order_service,
        inventory_service=inventory_service,
        knowledge_service=knowledge_service,
    )
    return FulfillmentCaseService(
        order_service=order_service,
        inventory_service=inventory_service,
        store=PostgreSQLFulfillmentCaseStore(get_enterprise_data_repository().engine),
        proposal_builder=nodes._build_execution_proposal,
    )


@router.post("/confirm", response_model=ApiResponse)
def confirm_fulfillment_case(request: FulfillmentCaseConfirmRequest):
    """Create WMS/TMS/ERP/CRM tasks after a valid preflight check."""

    try:
        with _hitl_tool_runtime_context():
            case = get_fulfillment_case_service().confirm(
                proposal=request.proposal,
                preflight_validation=request.preflight_validation,
                approver_id=request.approver_id,
                notes=request.notes,
                human_changes=request.human_changes,
                checkpoint=request.checkpoint,
            )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建履约外部任务失败：{exc}",
        ) from exc
    return ApiResponse(
        success=True,
        message=_case_dispatch_message(case.case_status),
        data={"case": case.model_dump(mode="json")},
    )


@router.post("/review", response_model=ApiResponse)
def review_fulfillment_case(request: FulfillmentCaseReviewRequest):
    """Record a HITL reject/follow-up decision before external task dispatch."""

    try:
        case = get_fulfillment_case_service().review_proposal(
            proposal=request.proposal,
            decision=request.decision,
            approver_id=request.approver_id,
            notes=request.notes,
            checkpoint=request.checkpoint,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"记录履约方案审核结果失败：{exc}",
        ) from exc
    return ApiResponse(
        success=True,
        message=f"审核结果已记录，案件状态：{case.case_status}",
        data={"case": case.model_dump(mode="json")},
    )


@router.post("/{case_id}/replan/confirm", response_model=ApiResponse)
def confirm_replan_fulfillment_case(case_id: str, request: FulfillmentCaseReplanConfirmRequest):
    """Approve a regenerated plan and dispatch its new external tasks."""

    try:
        with _hitl_tool_runtime_context():
            case = get_fulfillment_case_service().confirm_replan(
                case_id=case_id,
                proposal=request.proposal,
                preflight_validation=request.preflight_validation,
                approver_id=request.approver_id,
                notes=request.notes,
                human_changes=request.human_changes,
                checkpoint=request.checkpoint,
            )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"确认重新规划方案失败：{exc}",
        ) from exc
    return ApiResponse(
        success=True,
        message=f"重新规划方案已确认，案件状态：{case.case_status}",
        data={"case": case.model_dump(mode="json")},
    )


def _hitl_tool_runtime_context():
    context = get_tool_runtime_context()
    if context.is_system:
        context = ToolRuntimeContext(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            roles=["unauthenticated"],
            permissions=[],
            request_id=context.request_id,
        )
    return tool_runtime_context(context)


@router.get("", response_model=ApiResponse)
def list_fulfillment_cases(order_id: str | None = Query(None), limit: int = Query(50)):
    cases = get_fulfillment_case_service().list_cases(order_id=order_id, limit=limit)
    return ApiResponse(
        success=True,
        message=f"Fulfillment cases: {len(cases)}",
        data={"cases": [case.model_dump(mode="json") for case in cases]},
    )


@router.get("/{case_id}", response_model=ApiResponse)
def get_fulfillment_case(case_id: str):
    case = get_fulfillment_case_service().get_case(case_id)
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="案件不存在")
    return ApiResponse(success=True, message="Fulfillment case detail.", data={"case": case.model_dump(mode="json")})


@router.get("/tasks/{task_id}/status", response_model=ApiResponse)
def get_fulfillment_task_status(task_id: str):
    """Status API for polling an external task mirrored in the local case store."""

    task = get_fulfillment_case_service().get_task(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="外部任务不存在")
    return ApiResponse(
        success=True,
        message=f"外部任务状态：{task.status}",
        data={"task": task.model_dump(mode="json")},
    )


@router.post("/tasks/{task_id}/status-sync", response_model=ApiResponse)
def sync_fulfillment_task_status(task_id: str, request: ExternalTaskStatusPollRequest):
    """Apply a status snapshot polled from WMS/TMS/ERP/CRM Status API."""

    try:
        case = get_fulfillment_case_service().sync_task_from_status_api(
            task_id=task_id,
            status_update=request,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ApiResponse(
        success=True,
        message=f"状态 API 同步完成，案件状态：{case.case_status}",
        data={"case": case.model_dump(mode="json")},
    )


@router.post("/tasks/{task_id}/webhook", response_model=ApiResponse)
def fulfillment_task_webhook(task_id: str, request: ExternalTaskWebhookRequest):
    """Receive task updates from WMS/TMS/ERP/CRM."""

    try:
        case = get_fulfillment_case_service().update_task_from_webhook(task_id=task_id, webhook=request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ApiResponse(
        success=True,
        message=f"任务状态已更新，案件状态：{case.case_status}",
        data={"case": case.model_dump(mode="json")},
    )


@router.post("/{case_id}/verify", response_model=ApiResponse)
def verify_fulfillment_case(case_id: str):
    """Reload current OMS/WMS facts and verify whether the real target state is reached."""

    try:
        result = get_fulfillment_case_service().verify(case_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ApiResponse(
        success=result.status == "COMPLETED",
        message=result.message,
        data={"verification": result.model_dump(mode="json")},
    )


@router.post("/{case_id}/manual/start", response_model=ApiResponse)
def start_manual_fulfillment_case(case_id: str, request: ManualProcessingRequest):
    """Mark a manual handoff case as being processed by a human operator."""

    try:
        case = get_fulfillment_case_service().start_manual_processing(
            case_id=case_id,
            operator_id=request.operator_id,
            notes=request.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return ApiResponse(
        success=True,
        message="人工接管已开始处理。",
        data={"case": case.model_dump(mode="json")},
    )


@router.post("/{case_id}/manual/complete", response_model=ApiResponse)
def complete_manual_fulfillment_case(case_id: str, request: ManualCompletionRequest):
    """Record manual completion and resume the case at VERIFYING."""

    try:
        case = get_fulfillment_case_service().complete_manual_resolution(
            case_id=case_id,
            operator_id=request.operator_id,
            notes=request.notes,
            business_result=request.business_result,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return ApiResponse(
        success=True,
        message="人工处理完成，案件已恢复到 VERIFYING。",
        data={"case": case.model_dump(mode="json")},
    )


@router.post("/{case_id}/excellent-case", response_model=ApiResponse)
def persist_excellent_case(case_id: str, request: ExcellentCaseRequest, background_tasks: BackgroundTasks):
    """Save a completed case as future PGVector-indexed excellent-case material."""

    try:
        service = get_fulfillment_case_service()
        case_before = service.get_case(case_id)
        existing = (case_before.current_state.get("case_ingestion") if case_before else None) or {}
        job = service.start_case_ingestion(
            case_id=case_id,
            operator_id=request.operator_id,
            notes=request.notes,
        )
        rebuild_scheduled = _should_schedule_case_ingestion(existing, job)
        if rebuild_scheduled:
            background_tasks.add_task(
                _run_case_ingestion,
                case_id,
                request.operator_id,
                request.notes,
            )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return ApiResponse(
        success=True,
        message=(
            "优秀案例沉淀任务已提交，后台索引任务会写入 PGVector。"
            if rebuild_scheduled
            else "优秀案例沉淀任务已存在或已进入待处理队列，本次提交按 case_id 幂等返回。"
        ),
        data={
            "ingestion_job": job.model_dump(mode="json"),
            "rebuild_scheduled": rebuild_scheduled,
        },
    )


@router.get("/{case_id}/excellent-case", response_model=ApiResponse)
def get_excellent_case_ingestion(case_id: str):
    """Return the current excellent-case ingestion job for a case."""

    case = get_fulfillment_case_service().get_case(case_id)
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="案件不存在")
    job = case.current_state.get("case_ingestion")
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="优秀案例沉淀任务不存在")
    return ApiResponse(
        success=True,
        message=f"优秀案例沉淀任务状态：{job.get('status')}",
        data={"ingestion_job": job},
    )


def _should_schedule_case_ingestion(existing: dict, job) -> bool:
    if job.status != "PENDING" or job.dead_letter:
        return False
    existing_status = existing.get("status")
    if existing_status in {"PENDING", "PROCESSING", "SUCCESS"}:
        return False
    if existing.get("dead_letter"):
        return False
    return True


def _run_case_ingestion(case_id: str, operator_id: str, notes: str) -> None:
    service = get_fulfillment_case_service()
    service.mark_case_ingestion_processing(case_id)
    try:
        record = service.persist_excellent_case(
            case_id=case_id,
            operator_id=operator_id,
            notes=notes,
        )
        knowledge_path = ExcellentCaseKnowledgeWriter(get_settings().knowledge_dir).write(record)
        get_knowledge_retrieval_service().rebuild_index()
        service.mark_case_ingestion_success(
            case_id=case_id,
            excellent_case_id=record.excellent_case_id,
            knowledge_path=knowledge_path,
        )
    except Exception as exc:
        service.mark_case_ingestion_failed(case_id=case_id, error=str(exc))
