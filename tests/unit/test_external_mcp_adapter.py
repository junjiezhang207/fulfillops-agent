import json

from app.agents.tools.mcp_adapter import (
    ExternalMCPToolAdapter,
    ExternalMCPToolConfig,
    MCPWhitelistPolicy,
    register_external_mcp_tools,
)
from app.agents.tools.registry import ToolRegistry, ToolServiceBundle


def test_external_mcp_tools_register_only_when_whitelisted():
    registry = ToolRegistry([])
    configs = [
        ExternalMCPToolConfig(
            server_id="customer-service",
            tool_name="create_ticket",
            description="创建客服协同任务",
            endpoint="http://customer-service.example/mcp",
            required_parameters=("order_id",),
            registry_name="mcp_customer_service_create_ticket",
        ),
        ExternalMCPToolConfig(
            server_id="unknown",
            tool_name="dangerous_write",
            description="未授权外部工具",
            endpoint="http://unknown.example/mcp",
            registry_name="mcp_unknown_dangerous_write",
        ),
    ]

    registered = register_external_mcp_tools(
        registry,
        configs,
        policy=MCPWhitelistPolicy(
            allowed_servers=("customer-service",),
            allowed_tools=("mcp_customer_service_create_ticket",),
        ),
        invoker_factory=lambda _config: lambda arguments: {"echo": arguments},
    )

    assert registered == ["mcp_customer_service_create_ticket"]
    catalog = registry.catalog(use_case="mcp", include_metrics=False)
    assert [item["name"] for item in catalog] == ["mcp_customer_service_create_ticket"]
    assert catalog[0]["owner"] == "mcp:customer-service"


def test_external_mcp_adapter_validates_schema_and_invokes_with_envelope():
    registry = ToolRegistry([])
    config = ExternalMCPToolConfig(
        server_id="carrier",
        tool_name="quote",
        description="第三方物流报价",
        endpoint="http://carrier.example/mcp",
        required_parameters=("order_id",),
        registry_name="mcp_carrier_quote",
    )
    register_external_mcp_tools(
        registry,
        [config],
        policy=MCPWhitelistPolicy(allowed_servers=("carrier",), allowed_tools=("quote",)),
        invoker_factory=lambda _config: lambda arguments: {"price": 12.5, "order_id": arguments["order_id"]},
    )
    [tool] = registry.build_tools(ToolServiceBundle(), use_case="mcp")

    ok = json.loads(tool.invoke({"order_id": "SO-MCP-001"}))
    assert ok["status"] == "ok"
    assert ok["data"]["transport"] == "streamable-http"
    assert ok["data"]["result"]["price"] == 12.5

    invalid = json.loads(tool.invoke({"order_id": ""}))
    assert invalid["status"] == "error"
    assert invalid["error"]["code"] == "mcp_schema_validation_failed"


def test_external_mcp_adapter_validates_declared_argument_types_before_invoking():
    calls = []
    config = ExternalMCPToolConfig(
        server_id="carrier",
        tool_name="quote",
        description="第三方物流报价",
        endpoint="http://carrier.example/mcp",
        required_parameters=("order_id", "quantity"),
        parameter_schema={
            "type": "object",
            "required": ["order_id", "quantity"],
            "properties": {
                "order_id": {"type": "string"},
                "quantity": {"type": "integer"},
                "options": {"type": "object"},
            },
        },
    )
    adapter = ExternalMCPToolAdapter(
        config,
        policy=MCPWhitelistPolicy(allowed_servers=("carrier",), allowed_tools=("quote",)),
        invoker=lambda arguments: calls.append(arguments) or {"price": 12.5},
    )

    invalid = json.loads(adapter.invoke(order_id="SO-MCP-002", quantity="five", options=[]))

    assert invalid["status"] == "error"
    assert invalid["error"]["code"] == "mcp_schema_validation_failed"
    assert {error["loc"][0] for error in invalid["error"]["details"]["errors"]} == {"quantity", "options"}
    assert calls == []
