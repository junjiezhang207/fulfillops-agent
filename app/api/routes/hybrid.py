"""混合策略 API 路由 — Agent + Workflow + Multi-Agent 的智能路由。

文件作用摘要：
这个文件是项目里“多种智能链路并存”的展示入口。它把固定 Workflow、RAG 知识检索、
自由 ReAct Agent、多 Agent Supervisor、并行 Workflow、高级规则引擎都放到同一个 ``/hybrid`` 路由组下，
方便你对比不同方案的边界和效果。

核心端点：
  POST /api/v1/hybrid/run              — 自动选择 Workflow / RAG / Agent / Multi-Agent
  POST /api/v1/hybrid/resume           — HITL 人工审批后恢复工作流
  POST /api/v1/hybrid/advanced/run     — 高级业务规则 + 履约方案评分
  POST /api/v1/hybrid/parallel/run     — LangGraph fan-out/fan-in 并行工作流
  POST /api/v1/hybrid/multi-agent/run  — Supervisor 调度多个专家 Agent

学习重点：
1. Workflow：稳定、可审计、适合固定 SOP。
2. Agent：灵活、能开放追问、适合问题表达不固定。
3. Multi-Agent：适合库存、履约、风险等多领域并行分析。
4. Hybrid：不是一种 Agent，而是一个“路由策略”，决定什么时候用哪条链路。

面试官可能问：为什么不所有问题都交给 Agent？
回答：Agent 灵活但成本和不确定性更高。商家系统里大量问题其实是固定流程，用 Workflow
更快、更便宜、更可控；只有复杂开放问题才需要 Agent 或 Multi-Agent。
"""

import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Query, status

from app.agents.tools.factory import (
    make_fulfillment_plan_tool,
    make_substitute_tool,
    make_warehouse_tool,
)
from app.core.config import get_settings
from app.core.service_registry import (
    get_inventory_analysis_service,
    get_knowledge_retrieval_service,
    get_order_analysis_service,
)
from app.infrastructure.llm.chat_adapter import LLMFactory
from app.schemas.advanced_order import AdvancedOrderDetails, Location, WarehouseInfo
from app.schemas.common import ApiResponse
from app.schemas.workflow import WorkflowRunRequest
from app.agents.runtime.agent_service import AgentService
from app.application.routing.advanced_hybrid_service import AdvancedHybridService
from app.agents.runtime.multi_agent_service import MultiAgentService
from app.workflows.fulfillment.parallel_graph import (
    build_parallel_workflow,
    create_parallel_workflow_nodes,
)
from app.application.cache.client_cache_service import get_response_cache
from app.domain.fulfillment.plan_service import FulfillmentPlanService
from app.application.routing.hybrid_service import HybridService
from app.application.memory.session_memory_service import (
    get_session_service,
)
from app.domain.fulfillment.substitute_sku import SubstituteSkuService
from app.domain.inventory.warehouse_service import WarehouseService
from app.application.workflow.workflow_service import WorkflowService

router = APIRouter(prefix="/hybrid")
logger = logging.getLogger(__name__)

# ---- 模块级依赖装配 ----
# 这组基础服务是所有路径共享的业务底座。无论走 Workflow、Agent 还是 Multi-Agent，
# 最终都要回到订单、库存、知识库这些确定性服务。
_settings = get_settings()
_order_analysis_service = get_order_analysis_service()
_inventory_analysis_service = get_inventory_analysis_service()
_knowledge_retrieval_service = get_knowledge_retrieval_service()

# 会话缓存（短期记忆）。
# HybridService 会用它记录同一 thread_id 的历史，让多轮问题可以带着上下文继续。
_session_cache = get_session_service()

# Hybrid 响应缓存：缓存低风险 completed 结果，不等同于模型服务端 Prompt Caching。
# 它缓存的是业务响应结果，目的是减少重复点击或重复查询的后端开销。
_response_cache = get_response_cache()

# Workflow 服务
# 固定链路，适合“订单是否可发货”“库存是否不足”这种可流程化问题。
_workflow_service = WorkflowService(
    order_service=_order_analysis_service,
    inventory_service=_inventory_analysis_service,
    knowledge_service=_knowledge_retrieval_service,
)

# Agent 服务。
# 这里单独组装扩展工具，是因为 Hybrid 里的 Agent 路径也需要仓库、替代品、履约方案能力。
_warehouse_service = WarehouseService()
_substitute_service = SubstituteSkuService()
_fulfillment_service = FulfillmentPlanService(
    inventory_service=_inventory_analysis_service,
    warehouse_service=_warehouse_service,
    substitute_service=_substitute_service,
)

_extra_tools = [
    make_warehouse_tool(_warehouse_service),
    make_substitute_tool(_substitute_service),
    make_fulfillment_plan_tool(_fulfillment_service),
]

_chat_model = LLMFactory.create_chat_model(_settings, use_case="agent")
_supervisor_model = LLMFactory.create_chat_model(_settings, use_case="supervisor")
_workflow_finalize_model = LLMFactory.create_chat_model(_settings, use_case="workflow_finalize")
_agent_service: AgentService | None = None
if _chat_model is not None:
    # Agent 依赖模型，因此采用可选初始化。模型不可用时，Hybrid 仍可以保留部分非 Agent 能力。
    try:
        _agent_service = AgentService(
            order_service=_order_analysis_service,
            inventory_service=_inventory_analysis_service,
            knowledge_service=_knowledge_retrieval_service,
            chat_model=_chat_model,
            extra_tools=_extra_tools,
        )
    except Exception as exc:
        logger.exception("Hybrid Agent 初始化失败：%s", exc)

# 多 Agent 服务（Supervisor 模式）。
# 即使 supervisor_model 为 None，MultiAgentService 内部也有规则降级路径，方便本地无模型测试。
_multi_agent_service = MultiAgentService(
    order_service=_order_analysis_service,
    inventory_service=_inventory_analysis_service,
    knowledge_service=_knowledge_retrieval_service,
    warehouse_service=_warehouse_service,
    substitute_service=_substitute_service,
    fulfillment_service=_fulfillment_service,
    llm=_supervisor_model,
)

# 混合服务。
# 自动路由会优先使用小模型做意图识别，再分流到 Workflow / RAG / Agent / Multi-Agent。
# Agent 模型不可用时，workflow/rag 仍然可用；路由到 agent 时再降级到 workflow。
_hybrid_service: HybridService | None = HybridService(
    workflow_service=_workflow_service,
    agent_service=_agent_service,
    knowledge_service=_knowledge_retrieval_service,
    multi_agent_service=_multi_agent_service,
    intent_model=_workflow_finalize_model or _chat_model,
    session_cache=_session_cache,
    response_cache=_response_cache,
    simple_threshold=0.4,
    complex_threshold=0.7,
)

# 高级混合服务（处理真实业务复杂订单）。
# 下面的仓库是 demo 数据，但 schema 按真实仓储决策字段设计：容量、日处理量、成本、质量分等。
_demo_warehouses = [
    {
        "warehouse_id": "WH-SH",
        "name": "上海仓库",
        "location": {"province": "上海市", "city": "上海市", "district": "浦东新区"},
        "total_capacity": 50000,
        "used_capacity": 30000,
        "max_daily_shipments": 500,
        "processing_time_hours": 2,
        "quality_rating": 98.5,
        "storage_cost_per_unit_day": 0.5,
        "picking_cost_per_unit": 2.0,
        "packing_cost_per_shipment": 50,
        "handling_cost_per_unit": 1.0,
    },
    {
        "warehouse_id": "WH-GZ",
        "name": "广州仓库",
        "location": {"province": "广东省", "city": "广州市", "district": "番禺区"},
        "total_capacity": 40000,
        "used_capacity": 25000,
        "max_daily_shipments": 400,
        "processing_time_hours": 4,
        "quality_rating": 96.0,
        "storage_cost_per_unit_day": 0.4,
        "picking_cost_per_unit": 1.8,
        "packing_cost_per_shipment": 40,
        "handling_cost_per_unit": 0.9,
    },
    {
        "warehouse_id": "WH-CS",
        "name": "长沙仓库",
        "location": {"province": "湖南省", "city": "长沙市", "district": "天心区"},
        "total_capacity": 30000,
        "used_capacity": 18000,
        "max_daily_shipments": 300,
        "processing_time_hours": 6,
        "quality_rating": 92.0,
        "storage_cost_per_unit_day": 0.3,
        "picking_cost_per_unit": 1.5,
        "packing_cost_per_shipment": 30,
        "handling_cost_per_unit": 0.8,
    },
]

_warehouse_objects = [
    # 把 dict 转成 Pydantic schema，是为了让后续 AdvancedHybridService 处理的是稳定类型，
    # 而不是散落的原始字典。这样字段缺失或类型不对能更早暴露。
    WarehouseInfo(
        warehouse_id=w["warehouse_id"],
        name=w["name"],
        location=Location(
            province=w["location"]["province"],
            city=w["location"]["city"],
            district=w["location"]["district"],
        ),
        total_capacity=w["total_capacity"],
        used_capacity=w["used_capacity"],
        max_daily_shipments=w["max_daily_shipments"],
        processing_time_hours=w["processing_time_hours"],
        quality_rating=w["quality_rating"],
        storage_cost_per_unit_day=w["storage_cost_per_unit_day"],
        picking_cost_per_unit=w["picking_cost_per_unit"],
        packing_cost_per_shipment=w["packing_cost_per_shipment"],
        handling_cost_per_unit=w["handling_cost_per_unit"],
    )
    for w in _demo_warehouses
]

_advanced_hybrid_service = AdvancedHybridService(
    warehouses=_warehouse_objects,
    session_cache=_session_cache,
)

# 并行工作流服务（两阶段 fan-out/fan-in）。
# 它展示 LangGraph 的并行能力，适合互不依赖的节点同时执行，降低端到端延迟。
_parallel_nodes = create_parallel_workflow_nodes(
    order_service=_order_analysis_service,
    inventory_service=_inventory_analysis_service,
    knowledge_service=_knowledge_retrieval_service,
    chat_model=_workflow_finalize_model,
)
_parallel_graph = build_parallel_workflow(_parallel_nodes)

class HybridRunRequest:
    """混合服务请求模型。

    当前路由实际使用 Query 参数接收，保留这个类主要是为了学习说明：
    如果以后改成 JSON body，可以把它迁移成 Pydantic BaseModel。
    """

    order_id: str
    question: str = None
    filter_categories: list = None


@router.post("/run", response_model=ApiResponse)
def hybrid_run(
    order_id: str = Query(...),
    question: str = Query(None),
    filter_categories: list = Query(None),
    thread_id: str = Query(None),
):
    """运行混合策略 — 智能选择 Workflow / RAG / Agent / Multi-Agent。

    路由逻辑：
      1. 大模型意图识别，失败时规则兜底
      2. 固定履约判断 → Workflow
      3. 规则/SOP/政策依据 → RAG
      4. 开放式分析 → Agent
      5. 多领域综合问题 → Multi-Agent

    参数：
      - order_id: 订单 ID（必需）
      - question: 用户问题
      - filter_categories: 知识检索的类别过滤
      - thread_id: 会话 ID（用于短期记忆，可选）

    响应包含：
      - path_used: 实际使用的路径（workflow / rag / agent / multi_agent）
      - intent_level: 识别的意图类型（simple / rag / medium / complex / multi_domain）
      - confidence: 分类的置信度（0-1）
      - final_answer: 最终答复
      - tools_called: 调用的工具列表
      - execution_time_ms: 执行耗时
      - thread_id: 会话 ID
      - conversation_turns: 对话轮数（同一会话）
    """
    if _hybrid_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hybrid service unavailable.",
        )

    if not order_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="order_id is required",
        )

    try:
        # process 内部会完成：意图判断 -> 路径选择 -> 执行 -> 缓存/会话更新。
        # 路由层不关心具体选了 workflow 还是 agent，只负责统一响应。
        result = _hybrid_service.process(
            order_id=order_id,
            question=question,
            filter_categories=filter_categories or [],
            thread_id=thread_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {exc}",
        ) from exc

    response_data = asdict(result)

    # 生成响应消息
    if result.status == "interrupted":
        message = (
            f"Workflow interrupted — human decision required "
            f"(node: {result.interrupt.get('node') if result.interrupt else 'unknown'})"
        )
    elif result.from_cache:
        message = f"Cache hit! (path: {result.path_used}, execution: {result.execution_time_ms:.0f}ms)"
    else:
        message = (
            f"Hybrid processing complete "
            f"(path: {result.path_used}, confidence: {result.confidence:.1%}, "
            f"turns: {result.conversation_turns})"
        )

    return ApiResponse(
        success=result.status != "error",
        message=message,
        data=response_data,
    )


class ResumeRequest:
    """HITL 恢复请求模型。

    HITL = Human In The Loop。高风险订单可能被 Workflow 中断，等待人工批准或拒绝。
    """

    thread_id: str
    decision: str  # "approved" | "rejected"
    notes: str = ""


@router.post("/resume", response_model=ApiResponse)
def hybrid_resume(
    thread_id: str = Query(...),
    decision: str = Query(...),
    notes: str = Query(""),
):
    """恢复被中断的工作流。

    当中断事件返回后，调用此端点提供人工决策以继续执行。

    参数：
      - thread_id: 中断事件中返回的线程 ID（必填）
      - decision: 人工决策，可选 "approved" 或 "rejected"
      - notes: 可选备注

    响应：
      - 如果工作流已完成：返回最终结果
      - 如果再次中断：返回新的中断事件（例如 knowledge_retrieval 节点）
    """
    if _hybrid_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hybrid service unavailable.",
        )

    if not thread_id or not decision:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="thread_id and decision are required",
        )

    # 人工决策必须是白名单值，避免把任意字符串传进 Workflow 状态。
    if decision not in ("approved", "rejected"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="decision must be 'approved' or 'rejected'",
        )

    try:
        result = _hybrid_service.resume_workflow(
            thread_id=thread_id,
            decision=decision,
            notes=notes,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resuming workflow: {exc}",
        ) from exc

    response_data = asdict(result)

    if result.status == "interrupted":
        message = (
            f"Workflow re-interrupted — another human decision required "
            f"(node: {result.interrupt.get('node') if result.interrupt else 'unknown'})"
        )
    elif result.status == "completed":
        message = f"Workflow resumed and completed (thread: {thread_id})"
    else:
        message = f"Resume error: {result.final_answer}"

    return ApiResponse(
        success=result.status == "completed",
        message=message,
        data=response_data,
    )


@router.get("/stats", response_model=ApiResponse)
def hybrid_stats():
    """获取混合策略的路由配置和统计。

    这个接口适合前端展示当前阈值、各路径命中情况，也适合调试“为什么这次走了 Agent”。
    """
    if _hybrid_service is None:
        return ApiResponse(
            success=False,
            message="Hybrid service not available",
            data={},
        )

    stats = _hybrid_service.get_routing_stats()
    return ApiResponse(
        success=True,
        message="Hybrid routing configuration",
        data=stats,
    )


@router.get("/sessions", response_model=ApiResponse)
def session_stats():
    """获取当前活跃的会话统计信息。

    商家连续追问时，thread_id 对应的短期记忆会影响后续回答；这个接口用于观察会话缓存。
    """
    stats = _session_cache.get_stats()
    return ApiResponse(
        success=True,
        message=f"Active sessions: {stats['active_sessions']}",
        data=stats,
    )


@router.get("/cache", response_model=ApiResponse)
def cache_stats():
    """获取 Hybrid 响应缓存统计。

    注意：这是业务响应缓存，不是模型厂商的 prompt cache，也不是工具结果缓存。
    """
    stats = _response_cache.get_stats()
    return ApiResponse(
        success=True,
        message=f"Cache hits: {stats['cache_hits']}/{stats['total_requests']} ({stats['hit_rate']})",
        data=stats,
    )


@router.post("/cache/clear", response_model=ApiResponse)
def clear_cache():
    """清空 Hybrid 响应缓存。

    适合数据更新后手动清理，避免前端继续看到旧的低风险缓存结果。
    """
    cleared = _response_cache.clear_all()
    return ApiResponse(
        success=True,
        message=f"Cleared {cleared} cache entries",
        data={"cleared_entries": cleared},
    )


@router.post("/advanced/run", response_model=ApiResponse)
def advanced_hybrid_run(order: AdvancedOrderDetails, thread_id: str = Query(None)):
    """运行高级混合策略 — 处理真实业务复杂的订单。

    流程：
      1. 应用业务规则（根据客户分级和订单特征）
      2. 生成多个履约方案（快速、平衡、经济等）
      3. 对方案进行智能评分（成本、时间、质量）
      4. 推荐最优方案

    参数：
      - order: AdvancedOrderDetails（高级订单详情）
      - thread_id: 会话 ID（可选）

    响应包含：
      - rules_applied: 应用的业务规则列表
      - original_priority: 原始优先级
      - final_priority: 应用规则后的优先级
      - risk_level: 风险等级
      - risk_flags: 风险标签
      - fulfillment_options: 多个履约方案数组
      - recommended_option_id: 推荐方案的 ID
      - execution_time_ms: 执行耗时
    """
    try:
        # AdvancedHybridService 是纯业务规则/评分链路，不依赖 LLM。
        # 它适合展示“不是所有智能决策都要调用大模型”，规则和评分也很重要。
        result = _advanced_hybrid_service.process(
            order=order,
            inventory_snapshot={},  # 可选：从数据库加载实际库存
            thread_id=thread_id,
        )

        # 转换为字典便于 JSON 序列化。
        # 这里字段展开得比较详细，是为了前端能画方案对比表，也方便面试讲成本/时效/质量评分。
        response_data = {
            "order_id": result.order_id,
            "customer_id": result.customer_id,
            "customer_level": result.customer_level,
            "rules_applied": result.rules_applied,
            "rule_details": result.rule_details,
            "rule_conflicts": result.rule_conflicts,
            "original_priority": result.original_priority,
            "final_priority": result.final_priority,
            "risk_level": result.risk_level,
            "risk_flags": result.risk_flags,
            "pricing_adjustment": result.pricing_adjustment,
            "fulfillment_options": [
                {
                    "option_id": opt.option_id,
                    "strategy": opt.strategy.value,
                    "option_name": opt.option_name,
                    "description": opt.description,
                    "primary_warehouse_id": opt.primary_warehouse_id,
                    "backup_warehouse_id": opt.backup_warehouse_id,
                    "shipping_method": opt.shipping_method.value,
                    "warehouse_handling_cost": opt.warehouse_handling_cost,
                    "shipping_cost": opt.shipping_cost,
                    "packaging_cost": opt.packaging_cost,
                    "storage_cost": opt.storage_cost,
                    "insurance_cost": opt.insurance_cost,
                    "total_cost": opt.total_cost,
                    "processing_time_hours": opt.processing_time_hours,
                    "shipping_time_hours": opt.shipping_time_hours,
                    "total_time_hours": opt.total_time_hours,
                    "estimated_delivery_date": opt.estimated_delivery_date.isoformat()
                    if opt.estimated_delivery_date
                    else None,
                    "quality_risk": opt.quality_risk,
                    "defect_probability": opt.defect_probability,
                    "warehouse_quality_score": opt.warehouse_quality_score,
                    "shipping_quality_score": opt.shipping_quality_score,
                    "consolidation_opportunity": opt.consolidation_opportunity,
                    "inventory_optimization_benefit": opt.inventory_optimization_benefit,
                    "special_notes": opt.special_notes,
                    "cost_score": opt.cost_score,
                    "time_score": opt.time_score,
                    "quality_score": opt.quality_score,
                    "overall_score": opt.overall_score,
                }
                for opt in result.fulfillment_options
            ],
            "recommended_option_id": result.recommended_option_id,
            "execution_time_ms": result.execution_time_ms,
        }

        message = f"Advanced order processing complete (options: {len(result.fulfillment_options)}, execution: {result.execution_time_ms:.0f}ms)"

        return ApiResponse(
            success=True,
            message=message,
            data=response_data,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing advanced order: {exc}",
        ) from exc


@router.post("/parallel/run", response_model=ApiResponse)
def parallel_workflow_run(
    order_id: str = Query(...),
    question: str = Query(None),
    filter_categories: list = Query(None),
):
    """并行工作流 — 两阶段 fan-out/fan-in 执行。

    亮点：
      Stage-1：order_analysis ∥ inventory_analysis（同时触发，等全部完成）
      Stage-2：缺货路径用 Send API fan-out → knowledge ∥ warehouse（同时触发）

    与 /workflow/run 的区别：
      - 顺序工作流：order → inventory → knowledge = 串行等待
      - 并行工作流：max(order, inventory) + max(knowledge, warehouse) = 并发执行
      - 理论延迟降低 22-50%（取决于各节点实际 I/O 耗时）
    """
    import time

    if not order_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="order_id is required")

    start = time.time()
    try:
        # 并行图使用同步 invoke，是因为这个路由本身是同步函数；
        # 如果未来放进 async 路由，可以增加 graph.ainvoke 版本。
        initial_state = {
            "order_id": order_id,
            "question": question,
            "filter_categories": filter_categories or [],
            "trace": [],
            "errors": [],
        }
        final_state = _parallel_graph.invoke(initial_state)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Parallel workflow error: {exc}",
        ) from exc

    elapsed_ms = (time.time() - start) * 1000
    final_answer = final_state.get("final_answer")
    conclusion = final_answer.conclusion if final_answer else ""
    trace = [t.node for t in (final_state.get("trace") or [])]

    return ApiResponse(
        success=True,
        message=f"Parallel workflow complete ({elapsed_ms:.0f}ms, nodes: {trace})",
        data={
            "order_id": order_id,
            "final_answer": conclusion,
            "warehouse_search_result": final_state.get("warehouse_search_result"),
            "execution_nodes": trace,
            "execution_time_ms": elapsed_ms,
            "errors": [str(e) for e in (final_state.get("errors") or [])],
        },
    )


@router.post("/multi-agent/run", response_model=ApiResponse)
def multi_agent_run(
    order_id: str = Query(...),
    question: str = Query(...),
):
    """多 Agent 编排 — Supervisor 模式，专业化子 Agent 协同。

    调用链：
      Supervisor (LLM) 分析问题
        → InventoryAgent  (check_inventory + search_warehouse)
        → FulfillmentAgent (generate_plan + find_substitute)
        → RiskAgent        (analyze_order + retrieve_knowledge)
        → Synthesizer      (汇总所有 Agent 输出)

    与 /run 的区别：
      - /run：单个 ReAct Agent 自主决定用哪些工具
      - /multi-agent/run：Supervisor 按领域分发给专业 Agent，再汇总

    适用场景：
      跨领域复杂问题（既要库存、又要风险、又要规则）
    """
    if not order_id or not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="order_id and question are required")

    try:
        # MultiAgentService 内部会调用 MultiAgentOrchestrator：
        # 第一轮并行跑多个专家，再由 Supervisor 判断是否需要追问和汇总。
        result = _multi_agent_service.run(order_id=order_id, question=question)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-agent error: {exc}",
        ) from exc

    return ApiResponse(
        success=True,
        message=f"Multi-agent complete ({result.execution_time_ms:.0f}ms, agents: {result.agents_called})",
        data={
            "order_id": order_id,
            "question": question,
            "final_answer": result.final_answer,
            "agents_called": result.agents_called,
            "total_agents_called": result.total_agents_called,
            "agent_records": [
                {
                    "agent": r.agent_name,
                    "summary_length": r.summary_length,
                    "summary_preview": r.summary[:200],
                }
                for r in result.agent_records
            ],
            "supervisor_reasoning": result.supervisor_reasoning,
            "execution_log": result.execution_log,
            "execution_time_ms": result.execution_time_ms,
        },
    )
