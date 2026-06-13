"""创建业务 Trace 上下文的 FastAPI 中间件。

中间件故意保持很薄：读取请求头里的身份信息，启动 ContextVar trace，让路由正常执行，
最后把完整 trace 写入 MySQL，并在响应头回写 ``X-Trace-Id``。具体的业务步骤由
Hybrid、Workflow、Tool、RAG 等服务自己追加；中间件只负责请求边界。
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import set_trace_id
from app.observability.business_trace import (
    finish_trace,
    get_current_trace,
    reset_trace,
    start_trace,
    update_current_trace,
)


SKIP_TRACE_PATH_PREFIXES = (
    "/api/v1/observability",
    "/api/v1/health",
    "/api/v1/metrics",
)


def should_skip_business_trace(path: str) -> bool:
    """Return True for platform observability endpoints that should not self-trace."""

    return any(path.startswith(prefix) for prefix in SKIP_TRACE_PATH_PREFIXES)


class BusinessTraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if should_skip_business_trace(request.url.path):
            return await call_next(request)

        incoming_trace_id = request.headers.get("X-Trace-Id")
        trace_id = set_trace_id(incoming_trace_id)
        token = start_trace(
            trace_id=trace_id,
            request_id=request.headers.get("X-Request-Id") or trace_id,
            session_id=request.headers.get("X-Session-Id") or request.query_params.get("thread_id"),
            tenant_id=request.headers.get("X-Tenant-Id") or "default",
            user_id=request.headers.get("X-User-Id"),
            order_id=request.query_params.get("order_id"),
            route=request.url.path,
            metadata={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None,
            },
        )
        try:
            response = await call_next(request)
            current = get_current_trace()
            status = "success" if response.status_code < 400 else "error"
            if response.status_code < 400 and current and current.status not in {"running", "success"}:
                status = current.status
            update_current_trace(status=status)
            finish_trace(status=status, error_code=str(response.status_code) if status == "error" else None)
            response.headers["X-Trace-Id"] = trace_id
            return response
        except Exception as exc:
            finish_trace(
                status="error",
                error_code=exc.__class__.__name__,
                error_message=str(exc),
            )
            raise
        finally:
            reset_trace(token)
