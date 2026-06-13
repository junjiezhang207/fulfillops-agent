import json

from langchain_core.tools import StructuredTool

from app.agents.tools.telemetry import get_tool_telemetry
from app.agents.tools.wrapper import wrap_tool_with_resilience
from app.mcp import server as mcp_server


def _wrapped_test_tool(name: str, response: str) -> StructuredTool:
    """构造一个经过真实 wrapper 的轻量工具。

    测试 MCP 适配层时不需要启动完整库存/RAG 服务；但我们仍然要走
    ``wrap_tool_with_resilience``，这样权限、错误 envelope 和 telemetry 的行为
    与生产 MCP Server 保持一致。
    """

    def _tool(order_id: str) -> str:
        return response

    return wrap_tool_with_resilience(
        StructuredTool.from_function(func=_tool, name=name, description="test"),
        enable_cache=False,
        enable_circuit_breaker=False,
    )


def test_mcp_tool_adapter_returns_same_error_envelope_for_permission_denied(monkeypatch):
    telemetry = get_tool_telemetry()
    telemetry.reset()
    monkeypatch.setenv("MCP_TOOL_PERMISSIONS", "inventory:read")
    monkeypatch.setattr(
        mcp_server,
        "_MCP_TOOL_MAP",
        {"analyze_order": _wrapped_test_tool("analyze_order", '{"status":"ok","data":{},"summary":"ok"}')},
    )

    raw = mcp_server._invoke_registry_tool("analyze_order", {"order_id": "SO123"})
    data = json.loads(raw)

    assert data["status"] == "error"
    assert data["error"]["code"] == "permission_denied"
    assert telemetry.snapshot["analyze_order"]["permission_denied"] == 1


def test_mcp_tool_adapter_records_success_telemetry(monkeypatch):
    telemetry = get_tool_telemetry()
    telemetry.reset()
    monkeypatch.setenv("MCP_TOOL_PERMISSIONS", "orders:read")
    monkeypatch.setattr(
        mcp_server,
        "_MCP_TOOL_MAP",
        {
            "analyze_order": _wrapped_test_tool(
                "analyze_order",
                '{"schema_version":"1.0","status":"ok","data":{"order_id":"SO123"},"summary":"ok"}',
            )
        },
    )

    raw = mcp_server._invoke_registry_tool("analyze_order", {"order_id": "SO123"})
    data = json.loads(raw)

    assert data["status"] == "ok"
    assert data["data"]["order_id"] == "SO123"
    assert telemetry.snapshot["analyze_order"]["success"] == 1


def test_mcp_tool_meta_reuses_registry_manifest():
    meta = mcp_server._tool_meta("retrieve_knowledge")

    assert meta["exposed_via"] == "mcp"
    assert meta["tool_registry"]["name"] == "retrieve_knowledge"
    assert meta["tool_registry"]["required_permissions"] == ["knowledge:read"]
