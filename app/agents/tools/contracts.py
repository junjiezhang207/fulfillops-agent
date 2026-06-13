"""Agent 工具的企业级契约和治理元数据。

这个模块负责稳定 LLM 看到的工具接口，同时补齐生产系统通常需要的能力：
显式参数 schema、schema 版本、结构化错误、运行时调用者上下文和权限校验。
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any, Iterator

from pydantic import BaseModel, Field


TOOL_SCHEMA_VERSION = "1.0"


class ToolStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


class ToolRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolCachePolicy(str, Enum):
    NONE = "none"
    TTL = "ttl"


class ToolError(BaseModel):
    code: str = Field(..., description="稳定、机器可读的错误码。")
    message: str = Field(..., description="面向用户或模型的安全错误信息。")
    retryable: bool = Field(default=False, description="重试是否可能解决问题。")
    details: dict[str, Any] = Field(default_factory=dict)


class ToolEnvelope(BaseModel):
    schema_version: str = Field(default=TOOL_SCHEMA_VERSION)
    status: ToolStatus
    data: dict[str, Any] = Field(default_factory=dict)
    summary: str = Field(default="")
    error: ToolError | None = None


class ToolRuntimeContext(BaseModel):
    tenant_id: str = Field(default="default")
    user_id: str = Field(default="system")
    roles: list[str] = Field(default_factory=lambda: ["system"])
    permissions: list[str] = Field(default_factory=lambda: ["*"])
    request_id: str = Field(default="")

    def has_permission(self, permission: str) -> bool:
        return "*" in self.permissions or permission in self.permissions

    @property
    def is_system(self) -> bool:
        return "system" in self.roles or "*" in self.permissions


class ToolManifest(BaseModel):
    name: str
    description: str
    owner: str = "fulfillment-platform"
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW
    cache_policy: ToolCachePolicy = ToolCachePolicy.NONE
    ttl_seconds: int | None = None
    side_effects: bool = False
    required_permissions: list[str] = Field(default_factory=list)
    data_freshness: str = "realtime"


class AnalyzeOrderArgs(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=64, description="订单 ID，例如 SO202502140001")


class CheckInventoryArgs(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=64, description="订单 ID，例如 SO202502140001")


class RetrieveKnowledgeArgs(BaseModel):
    order_id: str | None = Field(
        default=None,
        max_length=64,
        description="订单 ID，可选；不传时只检索知识库文档。",
    )
    question: str = Field(..., min_length=1, max_length=500, description="需要检索的业务问题")
    categories: str = Field(default="", max_length=300, description="可选，逗号分隔的知识类别")


class SearchWarehouseInventoryArgs(BaseModel):
    sku_id: str = Field(..., min_length=1, max_length=128, description="SKU 编码，例如 SKU-IPHONE-CASE-001")


class FindSubstituteSkuArgs(BaseModel):
    sku_id: str = Field(..., min_length=1, max_length=128, description="原始 SKU 编码")


class GenerateFulfillmentPlanArgs(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=64, description="订单 ID，例如 SO202502140001")


DEFAULT_AGENT_TOOL_PERMISSIONS = [
    "orders:read",
    "inventory:read",
    "knowledge:read",
    "warehouse:read",
    "catalog:read",
    "fulfillment:plan",
]


TOOL_MANIFESTS: dict[str, ToolManifest] = {
    "analyze_order": ToolManifest(
        name="analyze_order",
        description="查询订单结构化详情。",
        risk_level=ToolRiskLevel.MEDIUM,
        required_permissions=["orders:read"],
        data_freshness="realtime",
    ),
    "check_inventory": ToolManifest(
        name="check_inventory",
        description="检查订单库存和缺货情况。",
        risk_level=ToolRiskLevel.MEDIUM,
        required_permissions=["inventory:read"],
        data_freshness="realtime",
    ),
    "retrieve_knowledge": ToolManifest(
        name="retrieve_knowledge",
        description="检索履约规则、SOP 和业务政策。",
        cache_policy=ToolCachePolicy.TTL,
        ttl_seconds=600,
        required_permissions=["knowledge:read"],
        data_freshness="ttl_10m",
    ),
    "search_warehouse_inventory": ToolManifest(
        name="search_warehouse_inventory",
        description="查询 SKU 在各仓库的库存分布。",
        risk_level=ToolRiskLevel.MEDIUM,
        required_permissions=["warehouse:read"],
        data_freshness="realtime",
    ),
    "find_substitute_sku": ToolManifest(
        name="find_substitute_sku",
        description="查询缺货 SKU 的替代品。",
        cache_policy=ToolCachePolicy.TTL,
        ttl_seconds=300,
        required_permissions=["catalog:read"],
        data_freshness="ttl_5m",
    ),
    "generate_fulfillment_plan": ToolManifest(
        name="generate_fulfillment_plan",
        description="生成履约方案建议。",
        risk_level=ToolRiskLevel.MEDIUM,
        required_permissions=["fulfillment:plan"],
        data_freshness="realtime",
    ),
}


_TOOL_CONTEXT: ContextVar[ToolRuntimeContext] = ContextVar(
    "tool_runtime_context",
    default=ToolRuntimeContext(),
)


def get_tool_runtime_context() -> ToolRuntimeContext:
    return _TOOL_CONTEXT.get()


@contextmanager
def tool_runtime_context(context: ToolRuntimeContext) -> Iterator[None]:
    token = _TOOL_CONTEXT.set(context)
    try:
        yield
    finally:
        _TOOL_CONTEXT.reset(token)


def make_agent_tool_context(
    session_id: str,
    tenant_id: str = "default",
    user_id: str | None = None,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
    request_id: str = "",
) -> ToolRuntimeContext:
    return ToolRuntimeContext(
        tenant_id=tenant_id or "default",
        user_id=user_id or session_id or "anonymous",
        roles=roles if roles is not None else ["agent_user"],
        permissions=permissions if permissions is not None else list(DEFAULT_AGENT_TOOL_PERMISSIONS),
        request_id=request_id,
    )


def authorize_tool_call(tool_name: str, context: ToolRuntimeContext | None = None) -> None:
    manifest = TOOL_MANIFESTS.get(tool_name)
    if manifest is None:
        return
    context = context or get_tool_runtime_context()
    if context.is_system:
        return
    missing = [
        permission
        for permission in manifest.required_permissions
        if not context.has_permission(permission)
    ]
    if missing:
        raise PermissionError(
            f"工具 {tool_name} 权限不足，缺少权限：{', '.join(missing)}"
        )


def ok_envelope(data: dict[str, Any], summary: str) -> str:
    envelope = ToolEnvelope(status=ToolStatus.OK, data=data, summary=summary)
    return envelope.model_dump_json(exclude_none=True)


def error_envelope(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> str:
    envelope = ToolEnvelope(
        status=ToolStatus.ERROR,
        summary=message,
        error=ToolError(
            code=code,
            message=message,
            retryable=retryable,
            details=details or {},
        ),
    )
    return envelope.model_dump_json(exclude_none=True)


def is_tool_success(raw: object) -> bool:
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return True
    return parsed.get("status") == ToolStatus.OK.value


def parse_tool_error(raw: object) -> ToolError | None:
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return None
    if parsed.get("status") != ToolStatus.ERROR.value:
        return None
    error = parsed.get("error") or {}
    try:
        return ToolError(**error)
    except Exception:
        return ToolError(code="tool_error", message=str(parsed.get("summary") or "工具调用失败"))
