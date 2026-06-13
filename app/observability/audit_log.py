"""人工与 AI 履约决策的结构化审计日志。

Trace step 回答“这次请求经历了什么步骤”；Audit event 回答“谁对哪个订单做了什么
决策”。企业系统里两者必须分开，因为审计日志通常有更严格的留存周期、访问权限
和合规审查要求。
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from app.observability.business_trace import (
    get_current_trace,
    now_iso,
    sanitize_payload,
    summarize_payload,
)
from app.observability.trace_store import get_trace_store


@dataclass
class AuditEvent:
    id: str
    event_type: str
    action: str
    actor_type: str = "user"
    status: str = "success"
    created_at: str = field(default_factory=now_iso)
    trace_id: str | None = None
    request_id: str | None = None
    tenant_id: str = "default"
    user_id: str | None = None
    order_id: str | None = None
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def record_audit_event(
    *,
    event_type: str,
    action: str,
    actor_type: str = "user",
    status: str = "success",
    order_id: str | None = None,
    user_id: str | None = None,
    summary: str = "",
    metadata: dict[str, Any] | None = None,
) -> AuditEvent:
    """持久化一个脱敏审计事件，并自动补齐当前 trace 的上下文信息。"""

    trace = get_current_trace()
    event = AuditEvent(
        id=f"audit-{uuid.uuid4().hex[:16]}",
        event_type=event_type,
        action=action,
        actor_type=actor_type,
        status=status,
        trace_id=trace.trace_id if trace else None,
        request_id=trace.request_id if trace else None,
        tenant_id=trace.tenant_id if trace else "default",
        user_id=user_id or (trace.user_id if trace else None),
        order_id=order_id or (trace.order_id if trace else None),
        summary=summarize_payload(summary) if summary else "",
        metadata=sanitize_payload(metadata or {}),
    )
    # 生产模式下审计日志是合规链路的一部分，写入失败必须暴露给调用方。
    # 如果这里静默吞掉异常，人工审核、Trace Center 和事后追责都会出现断点。
    get_trace_store().save_audit_event(asdict(event))
    return event
