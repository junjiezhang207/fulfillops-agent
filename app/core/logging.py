"""结构化日志 — JSON 格式 + 全链路 trace_id 透传。

升级内容：
  旧版：basicConfig 输出纯文本，没有 trace_id，无法关联同一请求的多条日志
  新版：
    - JSON 格式，每行一个 JSON 对象，方便 ELK/Loki/Datadog 采集
    - ContextVar 持有当前请求的 trace_id，自动注入每条日志
    - FastAPI 中间件从 X-Trace-Id 请求头读取或生成 trace_id，响应头回写
    - trace_id 可在任何模块通过 get_trace_id() 读取，无需传参

使用方式：
    from app.core.logging import get_logger, get_trace_id, set_trace_id

    logger = get_logger(__name__)
    logger.info("订单查询完成", extra={"order_id": "SO123", "hits": 5})
    # → {"timestamp": "...", "level": "INFO", "logger": "...", "message": "...",
    #    "trace_id": "abc123", "order_id": "SO123", "hits": 5}
"""

from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ── trace_id 上下文变量（每个协程/线程独立）─────────────────────────────────
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """获取当前请求的 trace_id（空字符串表示不在请求上下文中）。"""
    return _trace_id_var.get()


def set_trace_id(trace_id: str | None = None) -> str:
    """设置当前请求的 trace_id，返回实际使用的值。"""
    tid = trace_id or uuid.uuid4().hex[:16]
    _trace_id_var.set(tid)
    return tid


# ── JSON Formatter ─────────────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """将日志记录格式化为单行 JSON，包含 trace_id 和 extra 字段。"""

    # 标准 LogRecord 自带的字段，不重复输出到 extra
    _SKIP_FIELDS = frozenset({
        "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "name", "message", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "trace_id": get_trace_id(),
        }
        # 附加 extra 字段（用户通过 logger.info(..., extra={...}) 传入）
        for key, val in record.__dict__.items():
            if key not in self._SKIP_FIELDS and not key.startswith("_"):
                data[key] = val

        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)

        return json.dumps(data, ensure_ascii=False, default=str)


# ── FastAPI 中间件 ─────────────────────────────────────────────────────────────

class TraceIdMiddleware(BaseHTTPMiddleware):
    """从请求头读取或生成 trace_id，注入 ContextVar，响应头回写。

    客户端可以传入 X-Trace-Id 头来保持自己生成的 ID；
    不传则自动生成一个 16 位十六进制 ID。
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = request.headers.get("X-Trace-Id")
        trace_id = set_trace_id(incoming)
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response


# ── 初始化入口 ────────────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO") -> None:
    """初始化项目日志（JSON 格式，替换旧版 basicConfig 纯文本）。

    在 app/main.py 的 create_app() 里调用，确保应用启动时完成配置。
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)

    # 抑制第三方库的过度日志
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取已配置的 logger（推荐在每个模块顶部调用）。

    使用方式：
        logger = get_logger(__name__)
        logger.info("查询完成", extra={"order_id": "SO123"})
    """
    return logging.getLogger(name)
