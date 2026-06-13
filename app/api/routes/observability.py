"""业务可观测 Trace Center API。

这些接口给前端 Trace Center 使用，底层查询 MySQL trace store。
这样不接外部监控平台时，也能在数据库里保留订单决策链路、步骤和审计事件。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.observability.trace_store import get_trace_store
from app.schemas.common import ApiResponse


router = APIRouter(prefix="/observability")


@router.get("/traces", response_model=ApiResponse)
def list_traces(
    order_id: str | None = Query(None),
    status: str | None = Query(None),
    tool_name: str | None = Query(None),
    route: str | None = Query(None),
    step_type: str | None = Query(None),
    model_id: str | None = Query(None),
    prompt_id: str | None = Query(None),
    min_duration_ms: float | None = Query(None, ge=0),
    started_from: str | None = Query(None),
    started_to: str | None = Query(None),
    error: str | None = Query(None),
    include_internal: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
):
    traces = get_trace_store().list_traces(
        order_id=order_id,
        status=status,
        tool_name=tool_name,
        route=route,
        step_type=step_type,
        model_id=model_id,
        prompt_id=prompt_id,
        min_duration_ms=min_duration_ms,
        started_from=started_from,
        started_to=started_to,
        error=error,
        limit=limit,
        include_internal=include_internal,
    )
    return ApiResponse(success=True, message="Trace list", data={"traces": traces})


@router.get("/traces/{trace_id}", response_model=ApiResponse)
def get_trace(trace_id: str):
    trace = get_trace_store().get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    trace["audit_events"] = get_trace_store().list_audit_events(trace_id=trace_id, limit=100)
    return ApiResponse(success=True, message="Trace detail", data=trace)


@router.get("/audit-events", response_model=ApiResponse)
def list_audit_events(
    trace_id: str | None = Query(None),
    order_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    events = get_trace_store().list_audit_events(
        trace_id=trace_id,
        order_id=order_id,
        limit=limit,
    )
    return ApiResponse(success=True, message="Audit events", data={"events": events})
