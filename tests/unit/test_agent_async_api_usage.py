import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool

from app.agent.multi_agent import MultiAgentOrchestrator
from app.agent.parallel_tools import ParallelToolRunner, make_parallel_query_tool
from app.agent.tool_wrapper import wrap_tool_with_resilience
from app.repositories.in_memory_inventory_repository import InMemoryInventoryRepository
from app.repositories.in_memory_order_repository import InMemoryOrderRepository
from app.services.fulfillment_plan_service import FulfillmentPlanService
from app.services.inventory_analysis_service import InventoryAnalysisService
from app.services.order_analysis_service import OrderAnalysisService
from app.services.substitute_sku_service import SubstituteSkuService
from app.services.warehouse_service import WarehouseService


def _async_only_tool(name: str = "async_only") -> StructuredTool:
    def _sync(value: str = ""):
        raise AssertionError("sync invoke should not be used on async path")

    async def _async(value: str = ""):
        await asyncio.sleep(0)
        return f"{name}:{value}"

    return StructuredTool.from_function(
        func=_sync,
        coroutine=_async,
        name=name,
        description="async-only test tool",
    )


@pytest.mark.asyncio
async def test_resilient_tool_uses_langchain_async_tool_api():
    wrapped = wrap_tool_with_resilience(
        _async_only_tool(),
        max_retries=0,
        enable_cache=False,
        enable_circuit_breaker=False,
    )

    result = await wrapped.ainvoke({"value": "ok"})

    assert result == "async_only:ok"


@pytest.mark.asyncio
async def test_parallel_runner_uses_tool_ainvoke_in_async_path():
    runner = ParallelToolRunner(max_workers=2)
    first = _async_only_tool("first")
    second = _async_only_tool("second")

    results = await runner.run_parallel([
        (first, {"value": "1"}),
        (second, {"value": "2"}),
    ])

    assert results == [
        {"tool": "first", "result": "first:1", "error": None},
        {"tool": "second", "result": "second:2", "error": None},
    ]


@pytest.mark.asyncio
async def test_parallel_query_tool_exposes_async_coroutine_path():
    parallel_query = make_parallel_query_tool({"first": _async_only_tool("first")})

    result = await parallel_query.ainvoke({
        "queries": [{"tool": "first", "args": {"value": "1"}}],
    })

    assert "first:1" in result
    assert "并行执行了 1 个工具" in result


@pytest.mark.asyncio
async def test_multi_agent_orchestrator_runs_async_graph_with_default_checkpointer():
    class FakeKnowledgeService:
        def retrieve(self, **_kwargs):
            return SimpleNamespace(
                hits=[],
                matched_categories=[],
                answer_summary=SimpleNamespace(key_rules=[], suggested_actions=[]),
            )

    order_service = OrderAnalysisService(InMemoryOrderRepository())
    inventory_service = InventoryAnalysisService(
        inventory_repository=InMemoryInventoryRepository(),
        order_analysis_service=order_service,
    )
    knowledge_service = FakeKnowledgeService()
    warehouse_service = WarehouseService()
    substitute_service = SubstituteSkuService()
    fulfillment_service = FulfillmentPlanService(
        inventory_service=inventory_service,
        warehouse_service=warehouse_service,
        substitute_service=substitute_service,
    )
    orchestrator = MultiAgentOrchestrator(
        order_service=order_service,
        inventory_service=inventory_service,
        knowledge_service=knowledge_service,
        warehouse_service=warehouse_service,
        substitute_service=substitute_service,
        fulfillment_service=fulfillment_service,
        llm=None,
    )

    result = await orchestrator.arun(
        order_id="SO202502140001",
        question="这个订单库存是否足够，履约有什么风险？",
    )

    assert result["final_answer"]
    assert set(result["agents_called"]) == {"inventory_agent", "fulfillment_agent", "risk_agent"}
    assert result["rounds_per_agent"] == {
        "inventory_agent": 1,
        "fulfillment_agent": 1,
        "risk_agent": 1,
    }
