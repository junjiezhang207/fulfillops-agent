"""External MCP tool adapter for ToolRegistry integration.

This module is intentionally small: it describes how external MCP tools enter
the same registry/governance path as local tools without turning the Agent into
an unrestricted remote tool caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError, create_model

from app.agents.tools.contracts import (
    ToolCachePolicy,
    ToolManifest,
    ToolRiskLevel,
    error_envelope,
    ok_envelope,
)
from app.agents.tools.registry import ToolDefinition, ToolRegistry


MCPInvoker = Callable[[dict[str, Any]], dict[str, Any]]

_JSON_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict[str, Any],
    "array": list[Any],
}


@dataclass(frozen=True)
class ExternalMCPToolConfig:
    """Config discovered from an allow-listed external MCP server."""

    server_id: str
    tool_name: str
    description: str
    endpoint: str
    transport: str = "streamable-http"
    required_parameters: tuple[str, ...] = ()
    parameter_schema: dict[str, Any] = field(default_factory=dict)
    required_permissions: tuple[str, ...] = ("mcp:external",)
    side_effects: bool = False
    registry_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def registry_tool_name(self) -> str:
        return self.registry_name or f"mcp_{self.server_id}_{self.tool_name}".replace("-", "_")


@dataclass(frozen=True)
class MCPWhitelistPolicy:
    """Server/tool whitelist used before registration and invocation."""

    allowed_servers: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()

    def allows(self, config: ExternalMCPToolConfig) -> bool:
        server_allowed = not self.allowed_servers or config.server_id in self.allowed_servers
        tool_allowed = (
            not self.allowed_tools
            or config.tool_name in self.allowed_tools
            or config.registry_tool_name in self.allowed_tools
        )
        return server_allowed and tool_allowed


class ExternalMCPToolAdapter:
    """Thin adapter around a discovered MCP tool.

    The callable invoker is injected so production can use a real Streamable HTTP
    MCP client while tests can remain deterministic.
    """

    def __init__(
        self,
        config: ExternalMCPToolConfig,
        *,
        policy: MCPWhitelistPolicy,
        invoker: MCPInvoker,
    ) -> None:
        self.config = config
        self.policy = policy
        self.invoker = invoker

    def invoke(self, **arguments: Any) -> str:
        if not self.policy.allows(self.config):
            return error_envelope(
                "mcp_tool_not_whitelisted",
                f"MCP tool {self.config.tool_name} on {self.config.server_id} is not whitelisted.",
                details={
                    "server_id": self.config.server_id,
                    "tool_name": self.config.tool_name,
                    "registry_tool_name": self.config.registry_tool_name,
                },
            )
        schema_error = _validate_mcp_arguments(self.config, arguments)
        if schema_error:
            return schema_error
        result = self.invoker(arguments)
        return ok_envelope(
            {
                "server_id": self.config.server_id,
                "tool_name": self.config.tool_name,
                "transport": self.config.transport,
                "endpoint": self.config.endpoint,
                "result": result,
            },
            f"MCP tool {self.config.tool_name} invoked through {self.config.transport}.",
        )


def _validate_mcp_arguments(config: ExternalMCPToolConfig, arguments: dict[str, Any]) -> str | None:
    required_parameters = _required_parameter_names(config)
    missing = [name for name in required_parameters if arguments.get(name) in (None, "")]
    if missing:
        return error_envelope(
            "mcp_schema_validation_failed",
            f"MCP tool arguments missing required fields: {', '.join(missing)}",
            details={"missing": missing},
        )
    type_errors = _mcp_argument_type_errors(config, arguments)
    if type_errors:
        return error_envelope(
            "mcp_schema_validation_failed",
            "MCP tool arguments do not match the declared schema.",
            details={"errors": type_errors},
        )
    if config.parameter_schema:
        args_model = _args_schema_for_mcp_tool(config)
        try:
            args_model.model_validate(arguments)
        except ValidationError as exc:
            return error_envelope(
                "mcp_schema_validation_failed",
                "MCP tool arguments do not match the declared schema.",
                details={"errors": exc.errors()},
            )
    return None


def register_external_mcp_tools(
    registry: ToolRegistry,
    configs: list[ExternalMCPToolConfig],
    *,
    policy: MCPWhitelistPolicy,
    invoker_factory: Callable[[ExternalMCPToolConfig], MCPInvoker],
) -> list[str]:
    """Register allow-listed external MCP tools into the central ToolRegistry."""

    registered: list[str] = []
    for config in configs:
        if not policy.allows(config):
            continue
        manifest = ToolManifest(
            name=config.registry_tool_name,
            description=f"{config.description}（External MCP: {config.server_id}/{config.tool_name}）",
            owner=f"mcp:{config.server_id}",
            risk_level=ToolRiskLevel.HIGH if config.side_effects else ToolRiskLevel.MEDIUM,
            cache_policy=ToolCachePolicy.NONE,
            side_effects=config.side_effects,
            required_permissions=list(config.required_permissions),
            data_freshness="realtime",
        )

        def _builder(_services, *, mcp_config: ExternalMCPToolConfig = config):
            adapter = ExternalMCPToolAdapter(
                mcp_config,
                policy=policy,
                invoker=invoker_factory(mcp_config),
            )
            args_schema = _args_schema_for_mcp_tool(mcp_config)
            return StructuredTool.from_function(
                func=adapter.invoke,
                name=mcp_config.registry_tool_name,
                description=manifest.description,
                args_schema=args_schema,
            )

        registry.register(
            ToolDefinition(
                name=config.registry_tool_name,
                builder=_builder,
                manifest=manifest,
                use_cases=("agent", "plan_execute", "multi_agent", "mcp"),
                groups=("mcp.external",),
            )
        )
        registered.append(config.registry_tool_name)
    return registered


def _args_schema_for_mcp_tool(config: ExternalMCPToolConfig) -> type[BaseModel]:
    required = set(_required_parameter_names(config))
    properties = _schema_properties(config)
    fields: dict[str, Any] = {}

    for name in sorted(required | set(properties)):
        schema = properties.get(name, {})
        field_type = _python_type_for_json_schema(schema)
        description = _schema_description(name, schema)
        if name in required:
            fields[name] = (field_type, Field(..., description=description))
        else:
            fields[name] = (field_type | None, Field(default=None, description=description))

    if not fields:
        fields = {"payload": (dict[str, Any], Field(default_factory=dict, description="MCP tool payload."))}
    return create_model(f"{config.registry_tool_name.title().replace('_', '')}Args", **fields)


def _required_parameter_names(config: ExternalMCPToolConfig) -> tuple[str, ...]:
    raw_required = config.parameter_schema.get("required") if config.parameter_schema else None
    if isinstance(raw_required, list):
        return tuple(str(item) for item in raw_required if str(item).strip())
    return config.required_parameters


def _schema_properties(config: ExternalMCPToolConfig) -> dict[str, Any]:
    raw_properties = config.parameter_schema.get("properties") if config.parameter_schema else None
    if isinstance(raw_properties, dict):
        return raw_properties
    if config.parameter_schema and all(key not in config.parameter_schema for key in ("type", "required")):
        return config.parameter_schema
    return {}


def _python_type_for_json_schema(schema: Any) -> Any:
    raw_type = schema.get("type") if isinstance(schema, dict) else schema
    if isinstance(raw_type, list):
        raw_type = next((item for item in raw_type if item != "null"), "string")
    return _JSON_TYPE_MAP.get(str(raw_type or "string"), Any)


def _mcp_argument_type_errors(config: ExternalMCPToolConfig, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for name, schema in _schema_properties(config).items():
        if name not in arguments or arguments[name] is None:
            continue
        expected = _json_type_name(schema)
        if expected == "any" or _value_matches_json_type(arguments[name], expected):
            continue
        errors.append(
            {
                "loc": [name],
                "type": f"{expected}_type",
                "msg": f"Expected {expected}.",
                "input": arguments[name],
            }
        )
    return errors


def _json_type_name(schema: Any) -> str:
    raw_type = schema.get("type") if isinstance(schema, dict) else schema
    if isinstance(raw_type, list):
        raw_type = next((item for item in raw_type if item != "null"), "any")
    raw = str(raw_type or "any")
    return raw if raw in _JSON_TYPE_MAP else "any"


def _value_matches_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True


def _schema_description(name: str, schema: Any) -> str:
    if isinstance(schema, dict) and schema.get("description"):
        return str(schema["description"])
    return f"MCP argument: {name}"
