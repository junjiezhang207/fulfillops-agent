"""固定业务工作流 API。

本模块暴露 ``WorkflowService`` 的 HTTP 入口。与 ReAct Agent 不同，Workflow 是一条
固定履约链路：订单分析 → 库存判断 → 知识检索/人工审批 → 最终答案。

端点：
    POST /api/v1/workflow/run          — 同步执行（含幂等性缓存）
    POST /api/v1/workflow/run/stream   — SSE 流式执行（每个节点完成即推送）
    POST /api/v1/workflow/run/timeout  — 带超时的异步执行（超过 N 秒返回 408）

Workflow 适合确定性强、可审计、业务节点固定的 SOP 链路；
Agent 适合开放追问和跨工具推理。API 层在这里额外做限流和超时保护。
"""

import json
import uuid

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.core.rate_limiter import check_workflow_rate_limit
from app.core.service_registry import (
    get_inventory_analysis_service,
    get_knowledge_retrieval_service,
    get_order_analysis_service,
)
from app.schemas.common import ApiResponse
from app.schemas.workflow import WorkflowRunRequest
from app.domain.orders.analysis import OrderNotFoundError
from app.application.workflow.workflow_service import WorkflowService, WorkflowTimeoutError

router = APIRouter(prefix="/workflow")

# ── 模块级依赖装配 ─────────────────────────────────────────────────────────────
# 这里在模块加载时装配 service，适合本项目这种单体学习项目。
# 如果变成大型服务，可以改成 FastAPI Depends 或容器化依赖注入。
_order_analysis_service = get_order_analysis_service()
_inventory_analysis_service = get_inventory_analysis_service()
_knowledge_retrieval_service = get_knowledge_retrieval_service()
_workflow_service = WorkflowService(
    order_service=_order_analysis_service,
    inventory_service=_inventory_analysis_service,
    knowledge_service=_knowledge_retrieval_service,
)


# ── 端点 1：同步执行（含幂等性）────────────────────────────────────────────────

@router.post("/run", response_model=ApiResponse)
def run_workflow(request: WorkflowRunRequest, http_request: Request) -> ApiResponse:
    """同步执行工作流，相同请求 5 分钟内直接返回缓存结果。

    错误处理：
      - ValueError        → 400（dispatch 参数缺失）
      - OrderNotFoundError → 404（兜底，通常已在节点内捕获）
    """
    # 限流放在真正执行业务前面，防止超额请求消耗数据库、RAG 或模型资源。
    check_workflow_rate_limit(http_request, request.order_id)
    try:
        result = _workflow_service.run(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except OrderNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return ApiResponse(
        success=True,
        message="工作流执行完成。",
        data=result.model_dump(mode="json"),
    )


@router.get("/approvals/pending", response_model=ApiResponse)
def list_pending_approvals(risk_level: str | None = Query(None)) -> ApiResponse:
    """查询 MySQL 中待人工审核的 HITL 工单。"""
    rows = _workflow_service.list_pending_approvals(risk_level)
    return ApiResponse(
        success=True,
        message=f"共 {len(rows)} 条待审核任务。",
        data={"items": [row.model_dump(mode="json") for row in rows]},
    )


@router.get("/approvals/history/{order_id}", response_model=ApiResponse)
def get_approval_history(order_id: str) -> ApiResponse:
    """查询某个订单的人工审核历史。"""
    rows = _workflow_service.get_approval_history(order_id)
    return ApiResponse(
        success=True,
        message=f"共 {len(rows)} 条审核历史。",
        data={"items": [row.model_dump(mode="json") for row in rows]},
    )


@router.get("/approvals/stats", response_model=ApiResponse)
def get_hitl_stats() -> ApiResponse:
    """查询 HITL 审核统计。"""
    return ApiResponse(success=True, message="HITL 统计完成。", data=_workflow_service.get_hitl_stats())


# ── 端点 2：SSE 流式执行（每个节点完成即推送）──────────────────────────────────

@router.post("/run/stream")
async def run_workflow_stream(
    request: WorkflowRunRequest,
    http_request: Request,
    timeout: float = Query(default=60.0, ge=5.0, le=300.0, description="超时秒数"),
) -> StreamingResponse:
    """SSE 流式执行工作流，每完成一个节点立即推送事件。

    事件格式（text/event-stream）：
      data: {"type": "node_complete", "node": "order_analysis", "data": {...}}
      data: {"type": "node_complete", "node": "inventory_analysis", "data": {...}}
      data: {"type": "completed",     "final_state": {...}}
      data: {"type": "interrupted",   "interrupt": {...}}
      data: {"type": "error",         "message": "..."}

    前端用法：
      const es = new EventSource('/api/v1/workflow/run/stream');
      es.onmessage = (e) => console.log(JSON.parse(e.data));
    """
    check_workflow_rate_limit(http_request, request.order_id)
    thread_id = f"sse-{request.order_id}-{uuid.uuid4().hex[:8]}"

    async def _event_generator():
        # SSE 的关键是“边执行边 yield”。每个 yield 都会被前端立即收到，
        # 适合展示节点进度，而不是等整个 Workflow 完成后才一次性返回。
        try:
            async for event in _workflow_service.run_stream_async(
                request, thread_id, timeout=timeout
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            error_event = {"type": "error", "message": str(exc)}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲，确保实时推送
        },
    )


# ── 端点 3：带超时的异步执行（超过 N 秒返回 408）────────────────────────────────

@router.post("/run/timeout", response_model=ApiResponse)
async def run_workflow_with_timeout(
    request: WorkflowRunRequest,
    http_request: Request,
    timeout: float = Query(default=30.0, ge=5.0, le=120.0, description="超时秒数"),
) -> ApiResponse:
    """带超时保护的异步执行，超过 timeout 秒返回 408 Request Timeout。

    适用场景：对响应时间有 SLA 要求的场景（如实时交互界面）。
    """
    # timeout 版本主要服务有 SLA 的交互场景：用户愿意要一个明确超时，
    # 而不是一直等待一个不可控的长请求。
    check_workflow_rate_limit(http_request, request.order_id)
    try:
        result = await _workflow_service.run_with_timeout(request, timeout=timeout)
    except WorkflowTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return ApiResponse(
        success=True,
        message="工作流执行完成。",
        data=result.model_dump(mode="json"),
    )
