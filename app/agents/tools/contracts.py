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


class GetContextDetailArgs(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=64, description="订单 ID，例如 SO202502140001")
    detail_path: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="需要展开的上下文字段组或路径，例如 inventory / logistics / product_restrictions。",
    )
    intent: str = Field(default="fulfillment_action", max_length=64, description="当前规划意图。")
    question: str = Field(default="", max_length=500, description="运营问题，可用于字段组判断。")


class SourceContextDetailArgs(BaseModel):
    order_id: str = Field(..., min_length=1, max_length=64, description="订单 ID，例如 SO202502140001")
    intent: str = Field(default="fulfillment_action", max_length=64, description="当前规划意图。")
    question: str = Field(default="", max_length=500, description="运营问题，可用于字段组判断。")


DEFAULT_AGENT_TOOL_PERMISSIONS = [
    "orders:read",
    "inventory:read",
    "knowledge:read",
    "warehouse:read",
    "catalog:read",
    "fulfillment:plan",
    "context:read",
    "oms:read",
    "wms:read",
    "tms:read",
    "erp:read",
    "pim:read",
    "crm:read",
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
    "get_context_detail": ToolManifest(
        name="get_context_detail",
        description="按路径展开 OrderContext 大体量明细，不重新加载整包 JSON 给模型。",
        risk_level=ToolRiskLevel.MEDIUM,
        required_permissions=["context:read"],
        data_freshness="realtime",
    ),
    "get_order_detail": ToolManifest(
        name="get_order_detail",
        description="只读获取 OMS 订单、SKU 和已发生履约状态。",
        risk_level=ToolRiskLevel.MEDIUM,
        required_permissions=["oms:read"],
        data_freshness="realtime",
    ),
    "get_inventory_warehouse_detail": ToolManifest(
        name="get_inventory_warehouse_detail",
        description="只读获取 WMS 候选仓、仓库能力和 SKU×仓库存。",
        risk_level=ToolRiskLevel.MEDIUM,
        required_permissions=["wms:read"],
        data_freshness="realtime",
    ),
    "get_shipping_detail": ToolManifest(
        name="get_shipping_detail",
        description="只读获取 TMS 物流渠道、可配送性、价格和时效。",
        risk_level=ToolRiskLevel.MEDIUM,
        required_permissions=["tms:read"],
        data_freshness="realtime",
    ),
    "get_supply_chain_detail": ToolManifest(
        name="get_supply_chain_detail",
        description="只读获取 ERP 补货、采购和在途库存信息。",
        risk_level=ToolRiskLevel.MEDIUM,
        required_permissions=["erp:read"],
        data_freshness="realtime",
    ),
    "get_product_constraints": ToolManifest(
        name="get_product_constraints",
        description="只读获取 PIM 商品履约限制，如冷链、危险品、是否允许拆单。",
        risk_level=ToolRiskLevel.MEDIUM,
        required_permissions=["pim:read"],
        data_freshness="realtime",
    ),
    "get_customer_case_context": ToolManifest(
        name="get_customer_case_context",
        description="只读获取 CRM 客诉、客服承诺和客户风险上下文。",
        risk_level=ToolRiskLevel.MEDIUM,
        required_permissions=["crm:read"],
        data_freshness="realtime",
    ),
    "create_warehouse_request": ToolManifest(
        name="create_warehouse_request",
        description="创建 WMS 仓储协同申请；只创建任务/申请，不直接改订单或库存。",
        risk_level=ToolRiskLevel.HIGH,
        side_effects=True,
        required_permissions=["wms:write"],
        data_freshness="write_request",
    ),
    "create_logistics_request": ToolManifest(
        name="create_logistics_request",
        description="创建 TMS 物流协同任务；只创建任务/申请，不直接改物流核心数据。",
        risk_level=ToolRiskLevel.HIGH,
        side_effects=True,
        required_permissions=["tms:write"],
        data_freshness="write_request",
    ),
    "create_supply_chain_request": ToolManifest(
        name="create_supply_chain_request",
        description="创建 ERP 供应链协同申请；只创建任务/申请，不直接改采购或在途库存。",
        risk_level=ToolRiskLevel.HIGH,
        side_effects=True,
        required_permissions=["erp:write"],
        data_freshness="write_request",
    ),
    "create_customer_service_task": ToolManifest(
        name="create_customer_service_task",
        description="创建 CRM/客服协同任务；只创建任务/申请，不直接改客户核心记录。",
        risk_level=ToolRiskLevel.HIGH,
        side_effects=True,
        required_permissions=["crm:write"],
        data_freshness="write_request",
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
