"""业务决策链路 Trace 上下文。

这个模块是自研可观测系统的主干，但它不是日志或 Prometheus 的替代品：

* 日志回答“代码在某个时间点做了什么”。
* Prometheus 回答“整体成功率、耗时、错误率是否健康”。
* BusinessTrace 回答“一次履约决策从用户输入到最终输出经历了哪些业务步骤”。

这里最关键的设计是 ``ContextVar``。FastAPI 会在同一个进程里并发处理多个请求，
如果用普通全局变量，A 用户和 B 用户的 trace 很容易串在一起。ContextVar 会跟随
当前 async task，因此 Tool、RAG、Workflow 可以在不层层传递 ``trace_id`` 的情况下
把步骤追加到当前请求的 trace 中。
"""

from __future__ import annotations

import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

from app.observability.trace_store import get_trace_store


MAX_TEXT_LENGTH = 600
FULFILLOPS_TRACE_SCHEMA_VERSION = "fulfillops.trace.v1"
FULFILLOPS_TRACE_MODULES = (
    "memory",
    "context",
    "rag",
    "model_gateway",
    "planner",
    "tool_gateway",
    "verification",
)
SENSITIVE_KEYS = {
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "phone",
    "email",
    "address",
    "customer_name",
    "receiver",
    "recipient",
}
SENSITIVE_PATTERN = re.compile(
    r"(?i)(bearer\s+[a-z0-9._-]+|sk-[a-z0-9_-]+|1[3-9]\d{9}|[\w.+-]+@[\w.-]+\.\w+)"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def _truncate(text: str, limit: int = MAX_TEXT_LENGTH) -> str:
    return text if len(text) <= limit else f"{text[:limit]}...<truncated>"


def sanitize_payload(value: Any, *, max_length: int = MAX_TEXT_LENGTH) -> Any:
    """生成可以写入 trace/audit 表的安全摘要。

    Trace 数据库用于排障和审计，不用于回放用户的完整隐私对话。
    所以这里会递归处理 dict/list：敏感字段直接打码，常见手机号、邮箱、token
    用正则脱敏，长文本再做截断，避免一次异常请求把本地数据库撑爆。
    """

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate(SENSITIVE_PATTERN.sub("<redacted>", value), max_length)
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in SENSITIVE_KEYS:
                safe[key_text] = "<redacted>"
            else:
                safe[key_text] = sanitize_payload(item, max_length=max_length)
        return safe
    if isinstance(value, (list, tuple, set)):
        return [sanitize_payload(item, max_length=max_length) for item in list(value)[:20]]
    if hasattr(value, "model_dump"):
        return sanitize_payload(value.model_dump(mode="json"), max_length=max_length)
    if hasattr(value, "dict"):
        return sanitize_payload(value.dict(), max_length=max_length)
    return _truncate(str(value), max_length)


def summarize_payload(value: Any, *, max_length: int = MAX_TEXT_LENGTH) -> str:
    safe = sanitize_payload(value, max_length=max_length)
    if isinstance(safe, str):
        return safe
    return _truncate(str(safe), max_length)


@dataclass
class TraceStep:
    id: str
    type: str
    name: str
    parent_id: str | None = None
    status: str = "success"
    started_at: str = field(default_factory=now_iso)
    ended_at: str | None = None
    duration_ms: float | None = None
    summary: str = ""
    error_code: str | None = None
    error_message: str | None = None
    input_summary: str | None = None
    output_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TraceContext:
    trace_id: str
    request_id: str
    started_at: str = field(default_factory=now_iso)
    ended_at: str | None = None
    duration_ms: float | None = None
    status: str = "running"
    route: str | None = None
    session_id: str | None = None
    tenant_id: str = "default"
    user_id: str | None = None
    order_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[TraceStep] = field(default_factory=list)
    _started_monotonic: float = field(default_factory=time.monotonic, repr=False)


_current_trace: ContextVar[TraceContext | None] = ContextVar(
    "business_trace_context",
    default=None,
)

# 轻量级进程内指标快照。业务代码只依赖 observability 包，不反向 import FastAPI
# metrics 路由，避免形成“底层业务模块依赖 API 层”的倒置依赖。
_metrics: dict[str, Any] = {
    "business_request_total": {},
    "tool_call_total": {},
    "rag_retrieval_total": {},
    "rag_empty_result_total": 0,
    "workflow_interrupted_total": {},
    "hitl_trigger_total": {},
    "guardrail_block_total": {},
    "prompt_injection_detected_total": {},
}


def _inc_bucket(name: str, labels: tuple[Any, ...] = (), amount: int = 1) -> None:
    bucket = _metrics.setdefault(name, {})
    bucket[labels] = bucket.get(labels, 0) + amount


def get_observability_metrics_snapshot() -> dict[str, Any]:
    return {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in _metrics.items()
    }


def record_prompt_injection_detected(source: str) -> None:
    """Record a prompt-injection detection for governance metrics."""

    _inc_bucket("prompt_injection_detected_total", (source or "unknown",))


def start_trace(
    *,
    trace_id: str | None = None,
    request_id: str | None = None,
    session_id: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
    order_id: str | None = None,
    route: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Token:
    trace_id = trace_id or uuid.uuid4().hex[:16]
    context = TraceContext(
        trace_id=trace_id,
        request_id=request_id or trace_id,
        session_id=session_id,
        tenant_id=tenant_id or "default",
        user_id=user_id,
        order_id=order_id,
        route=route,
        metadata=sanitize_payload(metadata or {}),
    )
    return _current_trace.set(context)


def reset_trace(token: Token) -> None:
    _current_trace.reset(token)


def get_current_trace() -> TraceContext | None:
    return _current_trace.get()


def update_current_trace(**updates: Any) -> None:
    trace = get_current_trace()
    if trace is None:
        return
    for key, value in updates.items():
        if not hasattr(trace, key) or value is None:
            continue
        if key == "metadata" and isinstance(value, dict):
            trace.metadata.update(sanitize_payload(value))
        else:
            setattr(trace, key, sanitize_payload(value))


def add_trace_step(
    *,
    step_type: str,
    name: str,
    parent_id: str | None = None,
    status: str = "success",
    duration_ms: float | None = None,
    summary: str = "",
    error_code: str | None = None,
    error_message: str | None = None,
    input_summary: Any = None,
    output_summary: Any = None,
    metadata: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> TraceStep | None:
    """向当前 trace 追加一个步骤，并同步更新聚合指标快照。"""

    trace = get_current_trace()
    if trace is None:
        return None
    step = TraceStep(
        id=_new_id("step"),
        type=step_type,
        name=name,
        parent_id=parent_id,
        status=status,
        started_at=started_at or now_iso(),
        ended_at=ended_at or now_iso(),
        duration_ms=round(duration_ms, 3) if duration_ms is not None else None,
        summary=summarize_payload(summary) if summary else "",
        error_code=sanitize_payload(error_code),
        error_message=summarize_payload(error_message) if error_message else None,
        input_summary=summarize_payload(input_summary) if input_summary is not None else None,
        output_summary=summarize_payload(output_summary) if output_summary is not None else None,
        metadata=sanitize_payload(metadata or {}),
        evidence=sanitize_payload(evidence or []),
    )
    trace.steps.append(step)
    if step_type == "tool":
        _inc_bucket("tool_call_total", (name, status))
    elif step_type == "rag":
        _inc_bucket("rag_retrieval_total", (status,))
        if metadata and metadata.get("empty_result"):
            _metrics["rag_empty_result_total"] = int(_metrics.get("rag_empty_result_total", 0)) + 1
    elif step_type == "workflow" and status in {"interrupted", "pending_human"}:
        _inc_bucket("workflow_interrupted_total", (name,))
    elif step_type == "hitl":
        _inc_bucket("hitl_trigger_total", (status,))
    elif step_type == "guardrail":
        _inc_bucket("guardrail_block_total", (name,))
    return step


@contextmanager
def trace_step(
    *,
    step_type: str,
    name: str,
    summary: str = "",
    input_summary: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    """测量同步代码块耗时，并在成功/失败时自动写入一个 trace step。"""

    started = time.monotonic()
    try:
        yield
    except Exception as exc:
        add_trace_step(
            step_type=step_type,
            name=name,
            status="error",
            duration_ms=(time.monotonic() - started) * 1000,
            summary=summary,
            error_code=exc.__class__.__name__,
            error_message=str(exc),
            input_summary=input_summary,
            metadata=metadata,
        )
        raise
    else:
        add_trace_step(
            step_type=step_type,
            name=name,
            status="success",
            duration_ms=(time.monotonic() - started) * 1000,
            summary=summary,
            input_summary=input_summary,
            metadata=metadata,
        )


def snapshot_current_trace(*, include_running_duration: bool = True) -> dict[str, Any] | None:
    trace = get_current_trace()
    if trace is None:
        return None
    data = asdict(trace)
    data.pop("_started_monotonic", None)
    if include_running_duration and data.get("duration_ms") is None:
        data["duration_ms"] = round((time.monotonic() - trace._started_monotonic) * 1000, 3)
    return sanitize_payload(data, max_length=MAX_TEXT_LENGTH)


def _trace_dict_from_context(trace: TraceContext | dict[str, Any] | None) -> dict[str, Any] | None:
    if trace is None:
        return snapshot_current_trace()
    if isinstance(trace, TraceContext):
        data = asdict(trace)
        data.pop("_started_monotonic", None)
        if data.get("duration_ms") is None:
            data["duration_ms"] = round((time.monotonic() - trace._started_monotonic) * 1000, 3)
        return sanitize_payload(data, max_length=MAX_TEXT_LENGTH)
    return sanitize_payload(trace, max_length=MAX_TEXT_LENGTH)


def _lower_text(*parts: Any) -> str:
    return " ".join(str(part or "") for part in parts).lower()


def _step_module(step: dict[str, Any]) -> str:
    step_type = str(step.get("type") or "").lower()
    name = str(step.get("name") or "").lower()
    metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
    haystack = _lower_text(step_type, name, metadata)
    if step_type in {"memory", "short_term_memory", "long_term_memory"}:
        return "memory"
    if step_type in {"rag", "retrieval"}:
        return "rag"
    if step_type in {"llm", "model", "model_gateway"}:
        return "model_gateway"
    if step_type in {"planner", "workflow", "workflow_node"}:
        return "planner"
    if step_type in {"tool", "tool_gateway", "mcp"}:
        return "tool_gateway"
    if step_type in {"verification", "verify"}:
        return "verification"
    if step_type in {"context", "business_context", "router"} or "context" in haystack:
        return "context"
    if "rag" in haystack or "sop_collection" in haystack:
        return "rag"
    if "model_gateway" in haystack or "use_case" in metadata:
        return "model_gateway"
    if "plan" in haystack or "replan" in haystack:
        return "planner"
    if "tool_gateway" in haystack:
        return "tool_gateway"
    if "verify" in haystack or "verifying" in haystack:
        return "verification"
    return "context"


def _compact_module_step(step: dict[str, Any], module: str) -> dict[str, Any]:
    return {
        "step_id": step.get("id"),
        "module": module,
        "type": step.get("type"),
        "name": step.get("name"),
        "parent_id": step.get("parent_id"),
        "status": step.get("status"),
        "started_at": step.get("started_at"),
        "ended_at": step.get("ended_at"),
        "duration_ms": step.get("duration_ms"),
        "summary": step.get("summary") or "",
        "input_summary": step.get("input_summary"),
        "output_summary": step.get("output_summary"),
        "metadata": step.get("metadata") or {},
        "evidence": step.get("evidence") or [],
    }


def _lifecycle_event_type(step: dict[str, Any]) -> str | None:
    metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
    explicit_event = metadata.get("event_type")
    if explicit_event:
        return str(explicit_event).upper()

    haystack = _lower_text(step.get("type"), step.get("name"), step.get("status"), step.get("summary"), metadata)
    if "hitl" in haystack and "approv" in haystack:
        return "HITL_APPROVED"
    if any(word in haystack for word in ("checkpoint", "checkpointer")):
        return "CHECKPOINT_SAVED"
    if any(word in haystack for word in ("waiting", "pending_human", "interrupted")):
        return "ENTER_WAITING"
    if "webhook" in haystack:
        return "WEBHOOK_RECEIVED"
    if "resume" in haystack or "resumed" in haystack:
        return "WORKFLOW_RESUMED"
    if "case_status" in haystack or "status_changed" in haystack:
        return "CASE_STATUS_CHANGED"
    if "plan_version" in haystack or "version_changed" in haystack:
        return "PLAN_VERSION_CHANGED"
    if "manual" in haystack or "handoff" in haystack:
        return "MANUAL_HANDOFF"
    return None


def _compact_event(step: dict[str, Any], event_type: str) -> dict[str, Any]:
    return {
        "event_id": step.get("id"),
        "event_type": event_type,
        "name": step.get("name"),
        "status": step.get("status"),
        "timestamp": step.get("ended_at") or step.get("started_at"),
        "summary": step.get("summary") or "",
        "metadata": step.get("metadata") or {},
    }


def _number_from_metadata(metadata: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def build_fulfillops_trace_document(trace: TraceContext | dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Build the FulfillOps governance trace document.

    The legacy trace snapshot is optimized for storage. This view is optimized for
    audit/replay and mirrors the flow document's top-level contract:
    trace_meta, execution_context, summary, modules, events, errors.
    """

    data = _trace_dict_from_context(trace)
    if data is None:
        return None

    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    modules: dict[str, list[dict[str, Any]]] = {module: [] for module in FULFILLOPS_TRACE_MODULES}
    events: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    model_call_count = 0
    tool_call_count = 0
    total_tokens = 0.0
    total_cost = 0.0
    retry_count = 0
    fallback_count = 0

    for raw_step in data.get("steps") or []:
        if not isinstance(raw_step, dict):
            continue
        step = sanitize_payload(raw_step, max_length=MAX_TEXT_LENGTH)
        module = _step_module(step)
        modules[module].append(_compact_module_step(step, module))

        step_metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        if module == "model_gateway" and step_metadata.get("actual_model_call") is not False:
            model_call_count += 1
        if module == "tool_gateway":
            tool_call_count += 1
        step_total_tokens = _number_from_metadata(step_metadata, ("total_tokens", "tokens"))
        if step_total_tokens:
            total_tokens += step_total_tokens
        else:
            total_tokens += _number_from_metadata(step_metadata, ("prompt_tokens",))
            total_tokens += _number_from_metadata(step_metadata, ("completion_tokens",))
        total_cost += _number_from_metadata(step_metadata, ("total_cost_usd", "cost_usd", "estimated_cost_usd"))
        retry_count += int(_number_from_metadata(step_metadata, ("retry_count", "retries")))
        if step_metadata.get("fallback") or step_metadata.get("fallback_used") or "fallback" in _lower_text(step.get("name")):
            fallback_count += 1

        event_type = _lifecycle_event_type(step)
        if event_type:
            events.append(_compact_event(step, event_type))

        status = str(step.get("status") or "").lower()
        if status in {"error", "failed", "failure", "blocked"} or step.get("error_code") or step.get("error_message"):
            errors.append(
                {
                    "step_id": step.get("id"),
                    "module": module,
                    "name": step.get("name"),
                    "status": step.get("status"),
                    "error_code": step.get("error_code"),
                    "error_message": step.get("error_message"),
                    "timestamp": step.get("ended_at") or step.get("started_at"),
                }
            )

    if data.get("error_code") or data.get("error_message"):
        errors.insert(
            0,
            {
                "step_id": None,
                "module": "trace",
                "name": "trace",
                "status": data.get("status"),
                "error_code": data.get("error_code"),
                "error_message": data.get("error_message"),
                "timestamp": data.get("ended_at") or data.get("started_at"),
            },
        )

    return sanitize_payload(
        {
            "trace_meta": {
                "schema_version": FULFILLOPS_TRACE_SCHEMA_VERSION,
                "trace_id": data.get("trace_id"),
                "request_id": data.get("request_id"),
                "run_id": metadata.get("run_id"),
                "case_id": metadata.get("case_id"),
                "order_id": data.get("order_id") or metadata.get("order_id"),
                "tenant_id": data.get("tenant_id"),
                "route": data.get("route"),
                "status": data.get("status"),
                "started_at": data.get("started_at"),
                "ended_at": data.get("ended_at"),
                "workflow_version": metadata.get("workflow_version"),
                "context_version": metadata.get("context_version"),
                "plan_version": metadata.get("plan_version"),
            },
            "execution_context": {
                "session_id": data.get("session_id"),
                "user_id": data.get("user_id"),
                "channel": metadata.get("channel"),
                "environment": metadata.get("environment"),
                "agent_version": metadata.get("agent_version"),
                "model_gateway_profile": metadata.get("model_gateway_profile"),
                "source_systems": metadata.get("source_systems"),
            },
            "summary": {
                "status": data.get("status"),
                "duration_ms": data.get("duration_ms"),
                "step_count": sum(len(steps) for steps in modules.values()),
                "model_call_count": model_call_count,
                "tool_call_count": tool_call_count,
                "retry_count": retry_count,
                "fallback_count": fallback_count,
                "total_tokens": int(total_tokens),
                "total_cost_usd": round(total_cost, 6),
                "error_count": len(errors),
            },
            "modules": modules,
            "events": events,
            "errors": errors,
        },
        max_length=MAX_TEXT_LENGTH,
    )


def snapshot_fulfillops_trace() -> dict[str, Any] | None:
    return build_fulfillops_trace_document()


def finish_trace(
    *,
    status: str = "success",
    error_code: str | None = None,
    error_message: str | None = None,
    persist: bool = True,
) -> dict[str, Any] | None:
    trace = get_current_trace()
    if trace is None:
        return None
    trace.status = status
    trace.error_code = error_code
    trace.error_message = summarize_payload(error_message) if error_message else None
    trace.ended_at = now_iso()
    trace.duration_ms = round((time.monotonic() - trace._started_monotonic) * 1000, 3)
    _inc_bucket("business_request_total", (trace.route or "unknown", status))
    data = snapshot_current_trace(include_running_duration=False)
    if persist and data is not None:
        try:
            get_trace_store().save_trace(data)
        except Exception:
            # 可观测写入必须是 best-effort：业务请求不能因为本地 trace 数据库临时不可写
            # 就变成失败。生产环境可以把这里替换为可靠队列或数据库重试。
            pass
    return data
