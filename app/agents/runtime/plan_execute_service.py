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
from app.agents.tools.wrapper import wrap_all_tools
from app.agents.tools.factory import (
    make_fulfillment_plan_tool,
    make_inventory_tool,
    make_knowledge_tool,
    make_order_tool,
    make_substitute_tool,
    make_warehouse_tool,
)
from app.services.fulfillment_plan_service import FulfillmentPlanService
from app.services.inventory_analysis_service import InventoryAnalysisService
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
from app.services.order_analysis_service import OrderAnalysisService
from app.services.substitute_sku_service import SubstituteSkuService
from app.services.warehouse_service import WarehouseService

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
        _realtime_tools = [
            make_order_tool(order_service),
            make_inventory_tool(inventory_service),
            make_warehouse_tool(warehouse_service),
            make_fulfillment_plan_tool(fulfillment_service),
        ]
        # 知识库和替代 SKU 属于相对稳定数据，可以走工具缓存。
        _catalog_tools = [
            make_knowledge_tool(knowledge_service),
            make_substitute_tool(substitute_service),
        ]

        # 给所有工具套上弹性层：异常处理、超时、缓存等都在 wrapper 里做。
        resilient_tools = (
            wrap_all_tools(_realtime_tools, enable_cache=False)
            + wrap_all_tools(_catalog_tools, enable_cache=True)
        )

        from app.core.config import get_settings
        from app.memory import create_long_term_memory_store

        settings = get_settings()
        # checkpointer 是 LangGraph 的短期状态存储，用 thread_id 维持一次会话的执行状态。
        _checkpointer = create_checkpointer(
            settings.redis_url,
            ttl_seconds=settings.short_term_memory_ttl_seconds,
        )
        memory_embed_model = None
        # 长期记忆如果使用向量后端，需要 embedding 模型把记忆文本转向量。
        long_term_backend = settings.long_term_memory_backend.strip().lower()
        if long_term_backend in {"mysql_milvus", "mysql+milvus", "mysql", "milvus"}:
            from app.graph.embed_adapter import create_lazy_embed_model

            memory_embed_model = create_lazy_embed_model(settings)
        # store 是 LangGraph Store，用于跨会话长期记忆。
        _store = create_long_term_memory_store(
            db_path=settings.long_term_memory_db_path,
            backend=settings.long_term_memory_backend,
            mysql_url=settings.long_term_memory_mysql_url or settings.mysql_url,
            milvus_uri=settings.milvus_uri or f"http://{settings.milvus_host}:{settings.milvus_port}",
            milvus_token=settings.milvus_token,
            milvus_database=settings.milvus_database,
            milvus_collection=settings.long_term_memory_milvus_collection,
            milvus_alias=settings.long_term_memory_milvus_alias,
            milvus_timeout_seconds=settings.milvus_timeout_seconds,
            milvus_similarity_metric=settings.milvus_similarity_metric,
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

        try:
            # ainvoke 会执行整张图，直到生成 final_answer 或达到最大迭代次数。
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
