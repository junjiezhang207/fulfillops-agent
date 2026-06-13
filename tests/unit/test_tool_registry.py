from unittest.mock import MagicMock

import pytest

from app.agents.tools.registry import ToolServiceBundle, get_tool_registry


def _services() -> ToolServiceBundle:
    return ToolServiceBundle(
        order_service=MagicMock(),
        inventory_service=MagicMock(),
        knowledge_service=MagicMock(),
        warehouse_service=MagicMock(),
        substitute_service=MagicMock(),
        fulfillment_service=MagicMock(),
    )


def test_tool_registry_builds_agent_extension_tools():
    tools = get_tool_registry().build_tools(
        services=_services(),
        use_case="agent",
        include_names=(
            "search_warehouse_inventory",
            "find_substitute_sku",
            "generate_fulfillment_plan",
        ),
    )

    assert [tool.name for tool in tools] == [
        "search_warehouse_inventory",
        "find_substitute_sku",
        "generate_fulfillment_plan",
    ]


def test_tool_registry_filters_multi_agent_specialist_groups():
    registry = get_tool_registry()
    services = _services()

    inventory_tools = registry.build_tools(
        services=services,
        use_case="multi_agent",
        group="multi_agent.inventory_agent",
    )
    fulfillment_tools = registry.build_tools(
        services=services,
        use_case="multi_agent",
        group="multi_agent.fulfillment_agent",
    )
    risk_tools = registry.build_tools(
        services=services,
        use_case="multi_agent",
        group="multi_agent.risk_agent",
    )

    assert [tool.name for tool in inventory_tools] == ["check_inventory", "search_warehouse_inventory"]
    assert [tool.name for tool in fulfillment_tools] == ["find_substitute_sku", "generate_fulfillment_plan"]
    assert [tool.name for tool in risk_tools] == ["analyze_order", "retrieve_knowledge"]


def test_tool_registry_catalog_exposes_governance_metadata():
    catalog = get_tool_registry().catalog(use_case="agent", include_metrics=False)
    by_name = {item["name"]: item for item in catalog}

    assert by_name["retrieve_knowledge"]["cache_enabled"] is True
    assert by_name["retrieve_knowledge"]["required_permissions"] == ["knowledge:read"]
    assert by_name["check_inventory"]["risk_level"] == "medium"


def test_tool_registry_exposes_same_tools_for_mcp_use_case():
    catalog = get_tool_registry().catalog(use_case="mcp", include_metrics=False)
    by_name = {item["name"]: item for item in catalog}

    assert set(by_name) == {
        "analyze_order",
        "check_inventory",
        "retrieve_knowledge",
        "search_warehouse_inventory",
        "find_substitute_sku",
        "generate_fulfillment_plan",
    }
    assert by_name["retrieve_knowledge"]["required_permissions"] == ["knowledge:read"]
    assert by_name["find_substitute_sku"]["cache_enabled"] is True


def test_tool_registry_missing_service_fails_fast():
    with pytest.raises(ValueError, match="order_service"):
        get_tool_registry().build_tools(
            services=ToolServiceBundle(),
            use_case="agent",
            include_names=("analyze_order",),
        )
