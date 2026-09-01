"""Read-only Tool Gateway for progressive business-context loading.

Planner-facing source tools are normal LangChain tools, but the workflow spec
requires them to pass through the same governance boundary: permission checks,
schema validation, timeout/retry, circuit breaking, output sanitization and
trace. This module provides that small application-facing entry point.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from app.agents.tools.contracts import (
    DEFAULT_AGENT_TOOL_PERMISSIONS,
    ToolRuntimeContext,
    make_agent_tool_context,
    tool_runtime_context,
)
from app.agents.tools.registry import ToolDefinition, ToolRegistry, ToolServiceBundle, get_tool_registry
from app.agents.tools.wrapper import wrap_tool_with_resilience
from app.observability.business_trace import add_trace_step


READ_ONLY_SOURCE_TOOL_NAMES = (
    "get_order_detail",
    "get_inventory_warehouse_detail",
    "get_shipping_detail",
    "get_supply_chain_detail",
    "get_product_constraints",
    "get_customer_case_context",
)


class ReadOnlyToolGateway:
    """Execute allow-listed source-system tools through shared tool governance."""

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        max_retries: int = 2,
        timeout_seconds: float = 10.0,
        failure_threshold: int = 3,
    ) -> None:
        self.registry = registry or get_tool_registry()
        self.max_retries = max(0, int(max_retries))
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.failure_threshold = max(1, int(failure_threshold))
        self._governed_tools: dict[tuple[int, str], BaseTool] = {}

    def build_tools(self, *, services: ToolServiceBundle) -> list[BaseTool]:
        """Build Agent-facing tools that execute through this gateway."""

        tools: list[BaseTool] = []
        for tool_name in READ_ONLY_SOURCE_TOOL_NAMES:
            definition = self.registry.get(tool_name)
            if definition is None:
                continue
            [raw_tool] = self.registry.build_tools(
                services=services,
                use_case="plan_execute",
                include_names=(tool_name,),
            )

            def _run_gateway_tool(*, _tool_name: str = tool_name, **kwargs) -> str:
                return self.execute(
                    tool_name=_tool_name,
                    args=kwargs,
                    services=services,
                )

            tools.append(
                StructuredTool.from_function(
                    func=_run_gateway_tool,
                    name=tool_name,
                    description=definition.manifest.description,
                    args_schema=getattr(raw_tool, "args_schema", None),
                    infer_schema=False,
                    handle_validation_error=True,
                    handle_tool_error=True,
                )
            )
        return tools

    def execute(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        services: ToolServiceBundle,
        runtime_context: ToolRuntimeContext | None = None,
    ) -> str:
        """Run a read-only source tool and return its standard JSON envelope."""

        definition = self.registry.get(tool_name)
        if tool_name not in READ_ONLY_SOURCE_TOOL_NAMES or definition is None:
            raise ValueError(f"只允许通过只读 Tool Gateway 调用 6 个业务源工具：{tool_name}")
        if definition.manifest.side_effects:
            raise ValueError(f"只读 Tool Gateway 拒绝 side_effects=True 的工具：{tool_name}")

        governed_tool = self._governed_tool(tool_name, definition, services)
        context = runtime_context or make_agent_tool_context(
            session_id=str(args.get("order_id") or "planner"),
            permissions=list(DEFAULT_AGENT_TOOL_PERMISSIONS),
        )
        idempotency_key = self._idempotency_key(tool_name, args, context)
        with tool_runtime_context(context):
            result = governed_tool.invoke(args)
        self._record_gateway_trace(tool_name, args, result, idempotency_key)
        return str(result)

    def execute_json(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        services: ToolServiceBundle,
        runtime_context: ToolRuntimeContext | None = None,
    ) -> dict[str, Any]:
        """Run a read-only source tool and parse its JSON envelope."""

        raw = self.execute(
            tool_name=tool_name,
            args=args,
            services=services,
            runtime_context=runtime_context,
        )
        return json.loads(raw)

    def _governed_tool(
        self,
        tool_name: str,
        definition: ToolDefinition,
        services: ToolServiceBundle,
    ) -> BaseTool:
        key = (id(services), tool_name)
        if key not in self._governed_tools:
            [tool] = self.registry.build_tools(
                services=services,
                use_case="plan_execute",
                include_names=(tool_name,),
            )
            self._governed_tools[key] = wrap_tool_with_resilience(
                tool,
                max_retries=self.max_retries,
                timeout_seconds=self.timeout_seconds,
                failure_threshold=self.failure_threshold,
                enable_cache=definition.cache_enabled,
            )
        return self._governed_tools[key]

    @staticmethod
    def _idempotency_key(tool_name: str, args: dict[str, Any], context: ToolRuntimeContext) -> str:
        payload = {
            "tool_name": tool_name,
            "args": args,
            "tenant_id": context.tenant_id,
            "request_id": context.request_id,
        }
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    @staticmethod
    def _record_gateway_trace(tool_name: str, args: dict[str, Any], result: str, idempotency_key: str) -> None:
        try:
            parsed = json.loads(result)
        except Exception:
            parsed = {"status": "unknown", "summary": str(result)[:300]}
        status = str(parsed.get("status") or "unknown")
        error = parsed.get("error") or {}
        add_trace_step(
            step_type="tool_gateway",
            name=tool_name,
            status="success" if status == "ok" else status,
            summary=f"只读工具 {tool_name} 已通过 Tool Gateway 执行。",
            error_code=error.get("code"),
            error_message=error.get("message"),
            input_summary={
                "order_id": args.get("order_id"),
                "intent": args.get("intent"),
                "question_present": bool(args.get("question")),
            },
            output_summary=parsed.get("summary"),
            metadata={
                "mode": "read_only",
                "side_effects": False,
                "idempotency_key": idempotency_key,
                "schema_validated_by": "langchain_structured_tool",
            },
        )
