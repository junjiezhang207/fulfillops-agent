"""Agent 工具的统一注册中心。

具体 LangChain 工具如何创建仍然由 factory 模块负责；注册中心负责企业级元数据：
工具在哪些链路可用、是否允许缓存、属于哪个专家分组，以及 UI/管理接口如何查看。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from langchain_core.tools import BaseTool

from app.agents.tools.contracts import TOOL_MANIFESTS, ToolCachePolicy, ToolManifest
from app.agents.tools.factory import (
    make_context_detail_tool,
    make_fulfillment_plan_tool,
    make_inventory_tool,
    make_knowledge_tool,
    make_order_tool,
    make_source_context_tool,
    make_substitute_tool,
    make_warehouse_tool,
)
from app.agents.tools.telemetry import get_tool_telemetry


@dataclass(frozen=True)
class ToolServiceBundle:
    """构建业务工具所需的服务集合。

    用一个小 dataclass 显式表达工具依赖，避免所有工具都被迫依赖全部 service。
    """

    order_service: object | None = None
    inventory_service: object | None = None
    knowledge_service: object | None = None
    warehouse_service: object | None = None
    substitute_service: object | None = None
    fulfillment_service: object | None = None
    context_service: object | None = None


ToolBuilder = Callable[[ToolServiceBundle], BaseTool]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    builder: ToolBuilder
    manifest: ToolManifest
    use_cases: tuple[str, ...]
    groups: tuple[str, ...] = ()

    @property
    def cache_enabled(self) -> bool:
        return self.manifest.cache_policy == ToolCachePolicy.TTL

    def supports(self, use_case: str, group: str | None = None) -> bool:
        if use_case not in self.use_cases:
            return False
        return group is None or group in self.groups

    def metadata(self) -> dict[str, Any]:
        return {
            **self.manifest.model_dump(mode="json"),
            "use_cases": list(self.use_cases),
            "groups": list(self.groups),
            "cache_enabled": self.cache_enabled,
        }


def _require(value: object | None, field_name: str, tool_name: str) -> object:
    if value is None:
        raise ValueError(f"构建工具 {tool_name} 需要服务 {field_name}")
    return value


class ToolRegistry:
    """可用工具及其治理元数据的注册表。"""

    def __init__(self, definitions: list[ToolDefinition] | None = None) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        for definition in definitions or []:
            self.register(definition)

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"工具重复注册：{definition.name}")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._definitions.get(name)

    def definitions(
        self,
        use_case: str | None = None,
        group: str | None = None,
    ) -> list[ToolDefinition]:
        items = list(self._definitions.values())
        if use_case:
            items = [item for item in items if item.supports(use_case, group)]
        return items

    def build_tools(
        self,
        services: ToolServiceBundle,
        use_case: str,
        group: str | None = None,
        include_names: list[str] | tuple[str, ...] | None = None,
        exclude_names: list[str] | tuple[str, ...] | None = None,
    ) -> list[BaseTool]:
        include = set(include_names or [])
        exclude = set(exclude_names or [])
        definitions = self.definitions(use_case=use_case, group=group)
        if include:
            definitions = [item for item in definitions if item.name in include]
        if exclude:
            definitions = [item for item in definitions if item.name not in exclude]
        return [definition.builder(services) for definition in definitions]

    def catalog(
        self,
        use_case: str | None = None,
        group: str | None = None,
        include_metrics: bool = True,
    ) -> list[dict[str, Any]]:
        metrics: Mapping[str, dict[str, Any]] = (
            get_tool_telemetry().snapshot if include_metrics else {}
        )
        rows = []
        for definition in self.definitions(use_case=use_case, group=group):
            row = definition.metadata()
            row["metrics"] = metrics.get(definition.name, {})
            rows.append(row)
        return rows


def _default_definitions() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="analyze_order",
            builder=lambda services: make_order_tool(_require(services.order_service, "order_service", "analyze_order")),
            manifest=TOOL_MANIFESTS["analyze_order"],
            use_cases=("agent", "plan_execute", "multi_agent"),
            groups=("multi_agent.risk_agent",),
        ),
        ToolDefinition(
            name="check_inventory",
            builder=lambda services: make_inventory_tool(_require(services.inventory_service, "inventory_service", "check_inventory")),
            manifest=TOOL_MANIFESTS["check_inventory"],
            use_cases=("agent", "plan_execute", "multi_agent", "mcp"),
            groups=("multi_agent.inventory_agent",),
        ),
        ToolDefinition(
            name="retrieve_knowledge",
            builder=lambda services: make_knowledge_tool(_require(services.knowledge_service, "knowledge_service", "retrieve_knowledge")),
            manifest=TOOL_MANIFESTS["retrieve_knowledge"],
            use_cases=("agent", "plan_execute", "multi_agent", "mcp"),
            groups=("multi_agent.risk_agent",),
        ),
        ToolDefinition(
            name="search_warehouse_inventory",
            builder=lambda services: make_warehouse_tool(_require(services.warehouse_service, "warehouse_service", "search_warehouse_inventory")),
            manifest=TOOL_MANIFESTS["search_warehouse_inventory"],
            use_cases=("agent", "plan_execute", "multi_agent", "mcp"),
            groups=("multi_agent.inventory_agent",),
        ),
        ToolDefinition(
            name="find_substitute_sku",
            builder=lambda services: make_substitute_tool(_require(services.substitute_service, "substitute_service", "find_substitute_sku")),
            manifest=TOOL_MANIFESTS["find_substitute_sku"],
            use_cases=("agent", "plan_execute", "multi_agent", "mcp"),
            groups=("multi_agent.fulfillment_agent",),
        ),
        ToolDefinition(
            name="generate_fulfillment_plan",
            builder=lambda services: make_fulfillment_plan_tool(_require(services.fulfillment_service, "fulfillment_service", "generate_fulfillment_plan")),
            manifest=TOOL_MANIFESTS["generate_fulfillment_plan"],
            use_cases=("agent", "plan_execute", "multi_agent", "mcp"),
            groups=("multi_agent.fulfillment_agent",),
        ),
        ToolDefinition(
            name="get_context_detail",
            builder=lambda services: make_context_detail_tool(_require(services.context_service, "context_service", "get_context_detail")),
            manifest=TOOL_MANIFESTS["get_context_detail"],
            use_cases=("agent", "plan_execute", "multi_agent", "mcp"),
            groups=("multi_agent.fulfillment_agent",),
        ),
        ToolDefinition(
            name="get_order_detail",
            builder=lambda services: make_source_context_tool(
                _require(services.context_service, "context_service", "get_order_detail"),
                name="get_order_detail",
                source_system="OMS",
                description="只读获取 OMS 订单主信息、SKU 明细和已发生履约状态。",
            ),
            manifest=TOOL_MANIFESTS["get_order_detail"],
            use_cases=("agent", "plan_execute", "multi_agent", "mcp"),
            groups=("multi_agent.risk_agent", "multi_agent.fulfillment_agent"),
        ),
        ToolDefinition(
            name="get_inventory_warehouse_detail",
            builder=lambda services: make_source_context_tool(
                _require(services.context_service, "context_service", "get_inventory_warehouse_detail"),
                name="get_inventory_warehouse_detail",
                source_system="WMS",
                description="只读获取 WMS 候选仓、仓库能力和 SKU×仓库存，立即履约判断必须使用 available_stock。",
            ),
            manifest=TOOL_MANIFESTS["get_inventory_warehouse_detail"],
            use_cases=("agent", "plan_execute", "multi_agent", "mcp"),
            groups=("multi_agent.inventory_agent", "multi_agent.fulfillment_agent"),
        ),
        ToolDefinition(
            name="get_shipping_detail",
            builder=lambda services: make_source_context_tool(
                _require(services.context_service, "context_service", "get_shipping_detail"),
                name="get_shipping_detail",
                source_system="TMS",
                description="只读获取 TMS 物流渠道、可配送性、运费、截单时间和预计送达时间。",
            ),
            manifest=TOOL_MANIFESTS["get_shipping_detail"],
            use_cases=("agent", "plan_execute", "multi_agent", "mcp"),
            groups=("multi_agent.fulfillment_agent",),
        ),
        ToolDefinition(
            name="get_supply_chain_detail",
            builder=lambda services: make_source_context_tool(
                _require(services.context_service, "context_service", "get_supply_chain_detail"),
                name="get_supply_chain_detail",
                source_system="ERP",
                description="只读获取 ERP 补货、采购和在途库存信息。",
            ),
            manifest=TOOL_MANIFESTS["get_supply_chain_detail"],
            use_cases=("agent", "plan_execute", "multi_agent", "mcp"),
            groups=("multi_agent.fulfillment_agent",),
        ),
        ToolDefinition(
            name="get_product_constraints",
            builder=lambda services: make_source_context_tool(
                _require(services.context_service, "context_service", "get_product_constraints"),
                name="get_product_constraints",
                source_system="PIM",
                description="只读获取 PIM 商品履约限制，如冷链、危险品、拆单限制和商品类型。",
            ),
            manifest=TOOL_MANIFESTS["get_product_constraints"],
            use_cases=("agent", "plan_execute", "multi_agent", "mcp"),
            groups=("multi_agent.fulfillment_agent",),
        ),
        ToolDefinition(
            name="get_customer_case_context",
            builder=lambda services: make_source_context_tool(
                _require(services.context_service, "context_service", "get_customer_case_context"),
                name="get_customer_case_context",
                source_system="CRM",
                description="只读获取 CRM 客诉、客服承诺和客户风险上下文。",
            ),
            manifest=TOOL_MANIFESTS["get_customer_case_context"],
            use_cases=("agent", "plan_execute", "multi_agent", "mcp"),
            groups=("multi_agent.risk_agent", "multi_agent.fulfillment_agent"),
        ),
    ]


_REGISTRY = ToolRegistry(_default_definitions())


def get_tool_registry() -> ToolRegistry:
    return _REGISTRY
