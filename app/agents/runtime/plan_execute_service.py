"""Plan-and-Execute Agent 外观服务（学习版注释）。

Plan-and-Execute 和普通 ReAct Agent 的区别：
- ReAct Agent 是边想边调用工具。
- Plan-and-Execute 是先拆计划，再按步骤执行，适合复杂任务。

这个文件只做服务封装：
1. 组装工具。
2. 组装短期/长期记忆。
3. 构建 LangGraph 图。
4. 对外提供 execute 方法。
"""

from __future__ import annotations

import uuid
import logging

from langchain_core.language_models import BaseChatModel

from app.agents.orchestration.plan_execute import PlanExecuteState, build_plan_execute_agent
from app.agents.runtime.checkpointer import create_checkpointer
from app.agents.tools.read_only_gateway import ReadOnlyToolGateway
from app.agents.tools.wrapper import wrap_all_tools
from app.agents.tools.contracts import make_agent_tool_context, tool_runtime_context
from app.agents.tools.registry import ToolServiceBundle, get_tool_registry
from app.application.routing.order_context_service import OrderContextService
from app.domain.fulfillment.plan_service import FulfillmentPlanService
from app.domain.inventory.analysis import InventoryAnalysisService
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
from app.domain.orders.analysis import OrderAnalysisService
from app.domain.fulfillment.substitute_sku import SubstituteSkuService
from app.domain.inventory.warehouse_service import WarehouseService

logger = logging.getLogger(__name__)


class PlanExecuteService:
    """Plan-and-Execute Agent 外观服务。

    使用方式：
        service = PlanExecuteService(order_svc, inv_svc, knowledge_svc, ..., llm=chat_model)
        result = await service.execute(
            order_id="SO202502140001",
            question="帮我完整分析这个订单的履约方案，包括库存、替代品和风险",
            session_id="user-123",
        )
        print(result["final_answer"])
        print(result["plan_steps"])      # 实际执行了哪些步骤
        print(result["step_results"])    # 每步的详细结果
    """

    def __init__(
        self,
        order_service: OrderAnalysisService,
        inventory_service: InventoryAnalysisService,
        knowledge_service: KnowledgeRetrievalService,
        warehouse_service: WarehouseService,
        substitute_service: SubstituteSkuService,
        fulfillment_service: FulfillmentPlanService,
        llm: BaseChatModel,
        max_iterations: int = 5,
    ) -> None:
        # 实时数据工具不缓存，目录数据工具可缓存（与 AgentService 策略一致）。
        # 订单/库存/仓库/履约方案都可能随时变，所以不缓存。
        context_service = OrderContextService(
            order_service=order_service,
            inventory_service=inventory_service,
        )
        tool_services = ToolServiceBundle(
            order_service=order_service,
            inventory_service=inventory_service,
            knowledge_service=knowledge_service,
            warehouse_service=warehouse_service,
            substitute_service=substitute_service,
            fulfillment_service=fulfillment_service,
            context_service=context_service,
        )
        tool_registry = get_tool_registry()
        _realtime_tools = tool_registry.build_tools(
            services=tool_services,
            use_case="plan_execute",
            include_names=(
                "analyze_order",
                "check_inventory",
                "search_warehouse_inventory",
                "generate_fulfillment_plan",
                "get_context_detail",
            ),
        )
        _source_context_tools = ReadOnlyToolGateway(registry=tool_registry).build_tools(
            services=tool_services,
        )
        # 知识库和替代 SKU 属于相对稳定数据，可以走工具缓存。
        _catalog_tools = tool_registry.build_tools(
            services=tool_services,
            use_case="plan_execute",
            include_names=("retrieve_knowledge", "find_substitute_sku"),
        )

        # 给所有工具套上弹性层：异常处理、超时、缓存等都在 wrapper 里做。
        resilient_tools = (
            wrap_all_tools(_realtime_tools, enable_cache=False)
            + _source_context_tools
            + wrap_all_tools(_catalog_tools, enable_cache=True)
        )

        from app.core.config import get_settings
        from app.memory import create_long_term_memory_store

        settings = get_settings()
        # checkpointer 是 LangGraph 的短期状态存储，用 thread_id 维持一次会话的执行状态。
        _checkpointer = create_checkpointer(
            settings.effective_database_url,
            ttl_seconds=settings.short_term_memory_ttl_seconds,
        )
        from app.infrastructure.llm.embedding_adapter import create_lazy_embed_model

        memory_embed_model = create_lazy_embed_model(settings)
        # 长期记忆如果使用向量后端，需要 embedding 模型把记忆文本转向量。
        # store 是 LangGraph Store，用于跨会话长期记忆。
        _store = create_long_term_memory_store(
            database_url=(
                settings.long_term_memory_database_url
                or settings.effective_database_url
            ),
            vector_store_type=settings.long_term_memory_vector_store_type,
            pgvector_table=settings.long_term_memory_pgvector_table,
            embedding_model=memory_embed_model,
            vector_dimension=settings.long_term_memory_vector_dimension,
            default_ttl_days=settings.long_term_memory_ttl_days,
        )
        # 构建真正的 Plan-and-Execute LangGraph。
        self._graph = build_plan_execute_agent(
            llm=llm,
            tools=resilient_tools,
            max_iterations=max_iterations,
            checkpointer=_checkpointer,
            store=_store,
        )

    async def execute(
        self,
        order_id: str,
        question: str,
        session_id: str | None = None,
        tenant_id: str = "default",
        user_id: str | None = None,
        roles: list[str] | None = None,
        permissions: list[str] | None = None,
        request_id: str = "",
    ) -> dict:
        """执行 Plan-and-Execute 工作流。

        Returns:
            {
              "final_answer":  str,          # 综合答复
              "plan_steps":    list[str],     # 实际执行的步骤列表
              "step_results":  list[dict],    # 每步 {step, result}
              "total_steps":   int,           # 执行步骤总数
            }
        """
        # 没有传 session_id 时自动生成一个 thread_id，保证 checkpointer 能区分不同会话。
        thread_id = session_id or f"pe-{order_id}-{uuid.uuid4().hex[:8]}"
        run_id = f"{thread_id}-{uuid.uuid4().hex[:8]}"
        # 初始状态必须和 PlanExecuteState 定义一致。
        initial_state: PlanExecuteState = {
            "input": question,
            "order_id": order_id,
            "run_id": run_id,
            "plan": [],
            "past_steps": [],
            "iterations": 0,
        }
        # LangGraph 的 thread_id 放在 configurable 里。
        config = {"configurable": {"thread_id": thread_id}}
        context = make_agent_tool_context(
            session_id=thread_id,
            tenant_id=tenant_id,
            user_id=user_id,
            roles=roles,
            permissions=permissions,
            request_id=request_id,
        )

        try:
            # ainvoke 会执行整张图，直到生成 final_answer 或达到最大迭代次数。
            with tool_runtime_context(context):
                final_state = await self._graph.ainvoke(initial_state, config)
        except Exception as exc:
            # 服务层兜底：图执行失败时返回结构化错误，而不是让 API 崩溃。
            logger.error("[PlanExecuteService] 执行失败：%s", exc)
            return {
                "final_answer": f"执行失败：{exc}",
                "plan_steps": [],
                "step_results": [],
                "total_steps": 0,
            }

        # past_steps 是 [(step, result), ...]，这里拆成前端更容易展示的结构。
        past = final_state.get("past_steps", [])
        return {
            "final_answer": final_state.get("final_answer", "未能生成最终答复"),
            "plan_steps": [step for step, _ in past],
            "step_results": [{"step": step, "result": result} for step, result in past],
            "total_steps": len(past),
        }
