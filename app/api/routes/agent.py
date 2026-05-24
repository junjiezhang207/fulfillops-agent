"""LangChain / LangGraph Agent 对话接口。

文件作用摘要：
这个文件是“自由问答型 Agent”的 HTTP 入口，主要承接商家运营、客服主管、仓储调度
这类用户的开放式问题。它把用户问题交给 ``AgentService``，再由 Agent 自主决定是否调用
订单、库存、知识库、仓库、替代品、履约方案等工具。

主要端点：
    POST /api/v1/agent/chat          — 普通一轮 Agent 对话
    POST /api/v1/agent/chat/stream   — NDJSON 流式 Agent 对话
    POST /api/v1/agent/plan-execute  — 先规划再执行的复杂任务 Agent
    GET/DELETE /api/v1/agent/cache   — 工具缓存观测和失效

学习重点：
1. 路由层负责限流、输入安全、模型选择、HTTP 异常映射。
2. Agent 的具体运行、记忆、工具包装、反思质量门都在 ``AgentService`` 和 ``app/agents``。
3. 没配 LLM 时不能让应用启动失败，而是在调用 Agent 接口时返回 503。
4. 同一 ``session_id`` 会共享短期上下文，适合多轮追问。

面试官可能问：为什么 Agent 初始化失败时不直接让 FastAPI 启动失败？
回答：真实系统里 Agent 可能依赖外部模型供应商，模型不可用不应该拖垮订单、库存、
健康检查、企业数据接入等基础接口。这里采用“服务可启动，能力按需 503”的降级方式。
"""

import logging

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.agents.quality.evaluation.langfuse_tracer import create_langfuse_tracer
from app.agents.tools.guardrails import InputGuardrails
from app.agents.tools.cache import GLOBAL_SCOPE, get_tool_cache
from app.agents.tools.factory import (
    make_fulfillment_plan_tool,
    make_substitute_tool,
    make_warehouse_tool,
)
from app.core.config import get_settings
from app.core.rate_limiter import check_agent_rate_limit
from app.core.service_registry import (
    get_inventory_analysis_service,
    get_knowledge_retrieval_service,
    get_order_analysis_service,
)
from app.infrastructure.llm.chat_adapter import LLMFactory
from app.schemas.agent import AgentChatRequest, AgentChatResponse, PlanExecuteRequest, PlanExecuteResponse
from app.schemas.common import ApiResponse
from app.agents.runtime.agent_service import AgentNotAvailableError, AgentService
from app.agents.runtime.plan_execute_service import PlanExecuteService
from app.domain.fulfillment.plan_service import FulfillmentPlanService
from app.domain.fulfillment.substitute_sku import SubstituteSkuService
from app.domain.inventory.warehouse_service import WarehouseService

router = APIRouter(prefix="/agent")
logger = logging.getLogger(__name__)

# ---- 模块级依赖装配 ----
# 这些 service 是 Agent 工具背后的业务能力。路由层只负责装配，不直接写业务规则。
_settings = get_settings()
_order_analysis_service = get_order_analysis_service()
_inventory_analysis_service = get_inventory_analysis_service()
_knowledge_retrieval_service = get_knowledge_retrieval_service()

# ---- 新工具的服务 ----
# 仓库、替代品、履约方案这三个服务是 Agent 的扩展工具能力：
# - 仓库工具：回答“哪个仓有货”
# - 替代品工具：回答“缺货 SKU 能不能替换”
# - 履约方案工具：回答“怎么组合发货最合理”
_warehouse_service = WarehouseService()
_substitute_service = SubstituteSkuService()
_fulfillment_service = FulfillmentPlanService(
    inventory_service=_inventory_analysis_service,
    warehouse_service=_warehouse_service,
    substitute_service=_substitute_service,
)

# ---- 构建扩展工具列表 ----
# AgentService 会把基础工具和这里的额外工具合并，再统一套上弹性包装。
_extra_tools = [
    make_warehouse_tool(_warehouse_service),
    make_substitute_tool(_substitute_service),
    make_fulfillment_plan_tool(_fulfillment_service),
]

# chat_model 为 None 时（未配置 LLM），各 service 保持 None，
# 接口在运行时返回 503，不阻断应用启动。
# 这种设计对学习项目也很有用：你即使没配模型，也能跑健康检查、Workflow、基础服务测试。
_agent_chat_model = LLMFactory.create_chat_model(_settings, use_case="agent")
_plan_execute_model = LLMFactory.create_chat_model(_settings, use_case="plan_execute")
_structured_extract_model = LLMFactory.create_chat_model(_settings, use_case="structured_extract")
_langfuse_tracer = create_langfuse_tracer(_settings)
_input_guard = InputGuardrails()
_agent_service: AgentService | None = None
_plan_execute_service: PlanExecuteService | None = None
_agent_services_by_model: dict[str, AgentService] = {}
_plan_execute_services_by_model: dict[str, PlanExecuteService] = {}


def _model_cache_key(model_id: str | None) -> str:
    """把模型 ID 规范成 service 缓存 key。

    为什么要缓存不同 model_id 对应的 AgentService？
    - 创建 AgentService 会组装工具、记忆、长短期存储和 tracing，重复创建没有必要。
    - 前端切换模型时，同一个模型可以复用同一个 service 实例。
    """
    return model_id.strip() if model_id and model_id.strip() else "__default__"


def _build_agent_service(chat_model: object, model_id: str | None = None) -> AgentService:
    """构建 ReAct Agent 服务。

    这里是 API 层和 service 层的装配点：把业务服务、模型、扩展工具、反思配置、Langfuse
    tracer 都传进去。真正 Agent 图的创建不在这里，而在 ``AgentService`` 内部。
    """
    return AgentService(
        order_service=_order_analysis_service,
        inventory_service=_inventory_analysis_service,
        knowledge_service=_knowledge_retrieval_service,
        chat_model=chat_model,
        extra_tools=_extra_tools,
        enable_reflection=_settings.agent_enable_reflection,
        reflection_threshold=_settings.agent_reflection_threshold,
        max_reflection_retries=_settings.agent_max_reflection_retries,
        structured_output_model=_structured_extract_model,
        langfuse_tracer=_langfuse_tracer,
    )


def _build_plan_execute_service(chat_model: object) -> PlanExecuteService:
    """构建 Plan-and-Execute 服务。

    Plan-and-Execute 复用同一批订单/库存/知识库/仓库/替代品/履约服务，但执行模式不同：
    它先让 Planner 生成步骤，再用 Executor 子 Agent 逐步执行。
    """
    return PlanExecuteService(
        order_service=_order_analysis_service,
        inventory_service=_inventory_analysis_service,
        knowledge_service=_knowledge_retrieval_service,
        warehouse_service=_warehouse_service,
        substitute_service=_substitute_service,
        fulfillment_service=_fulfillment_service,
        llm=chat_model,
    )


def _agent_service_for_model(model_id: str | None) -> AgentService | None:
    """按模型 ID 获取或懒加载 AgentService。

    学习重点：
    - 默认模型启动时会预构建。
    - 非默认模型只有前端真正选择时才创建，减少启动成本。
    - 如果创建失败返回 None，由接口层统一转成 503。
    """
    model_id = model_id.strip() if model_id and model_id.strip() else None
    cache_key = _model_cache_key(model_id)
    if cache_key in _agent_services_by_model:
        return _agent_services_by_model[cache_key]

    chat_model = (
        _agent_chat_model
        if model_id is None
        else LLMFactory.create_chat_model(_settings, use_case="agent", model_id=model_id)
    )
    if chat_model is None:
        return None
    try:
        service = _build_agent_service(chat_model, model_id=model_id)
    except AgentNotAvailableError as exc:
        logger.warning("Agent 初始化失败，接口将返回 503：%s", exc)
        return None
    except Exception as exc:
        logger.exception("Agent 初始化异常：%s", exc)
        return None
    _agent_services_by_model[cache_key] = service
    return service


def _plan_execute_service_for_model(model_id: str | None) -> PlanExecuteService | None:
    """按模型 ID 获取或懒加载 PlanExecuteService。"""
    model_id = model_id.strip() if model_id and model_id.strip() else None
    cache_key = _model_cache_key(model_id)
    if cache_key in _plan_execute_services_by_model:
        return _plan_execute_services_by_model[cache_key]

    chat_model = (
        _plan_execute_model
        if model_id is None
        else LLMFactory.create_chat_model(_settings, use_case="plan_execute", model_id=model_id)
    )
    if chat_model is None:
        return None
    try:
        service = _build_plan_execute_service(chat_model)
    except Exception as exc:
        logger.exception("PlanExecuteService 初始化异常：%s", exc)
        return None
    _plan_execute_services_by_model[cache_key] = service
    return service


if _agent_chat_model is not None:
    # 默认 Agent 尽量在模块加载时构建好，这样首次请求不会额外承担初始化成本。
    # 如果初始化失败，记录日志但不阻止 FastAPI 启动。
    try:
        _agent_service = _build_agent_service(_agent_chat_model)
        _agent_services_by_model[_model_cache_key(None)] = _agent_service
    except AgentNotAvailableError as exc:
        logger.warning("Agent 初始化失败，接口将返回 503：%s", exc)
    except Exception as exc:
        logger.exception("Agent 初始化异常：%s", exc)

if _plan_execute_model is not None:
    # Plan-and-Execute 单独用 use_case="plan_execute"，方便它走更适合规划/结构化输出的模型。
    try:
        _plan_execute_service = _build_plan_execute_service(_plan_execute_model)
        _plan_execute_services_by_model[_model_cache_key(None)] = _plan_execute_service
    except Exception as exc:
        logger.exception("PlanExecuteService 初始化异常：%s", exc)


@router.post("/chat", response_model=ApiResponse)
async def agent_chat(request: AgentChatRequest, http_request: Request) -> ApiResponse:
    """与 ReAct Agent 进行一轮对话（async，不阻塞事件循环）。

    Agent 会根据问题自主选择工具，可能调用：
        - analyze_order：查询订单详情
        - check_inventory：检查库存状态
        - retrieve_knowledge：检索履约规则

    同一 session_id 的连续请求共享对话历史，支持多轮追问。
    """
    # 顺序很重要：先限流，再做输入检查，再调模型。
    # 这样恶意高频请求不会消耗 Prompt Injection 检查之后的模型/工具资源。
    check_agent_rate_limit(http_request, request.session_id)
    agent_service = _agent_service_for_model(request.model_id)
    if agent_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent 不可用：请配置模型网关 config/model_gateway.yaml 和对应的后端 API Key。",
        )

    # ── 输入安全检查（Prompt Injection / 超长 / 黑名单）────────────────
    check = _input_guard.check(request.message)
    if not check.passed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"输入校验失败：{check.reason}",
        )

    try:
        result = await agent_service.chat(
            session_id=request.session_id,
            message=request.message,
            include_trace=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent 执行出错：{exc}",
        ) from exc

    # AgentService 已经负责 trace、反思、记忆和工具结果整理。
    # 路由层只把 service 结果转换成面向前端的响应 schema。
    safe_reply = result["reply"]

    response = AgentChatResponse(
        session_id=request.session_id,
        reply=safe_reply,
        tools_called=result["tools_called"],
        trace=result.get("trace"),
    )
    return ApiResponse(
        success=True,
        message="Agent 对话完成。",
        data=response.model_dump(mode="json"),
    )


@router.post("/chat/stream")
async def agent_chat_stream(request: AgentChatRequest, http_request: Request) -> StreamingResponse:
    """与 ReAct Agent 进行一轮流式对话。

    返回格式是 NDJSON：每行一个 JSON 事件，前端可以边读边渲染。
    常见事件：
      - status：链路状态
      - token：模型文本增量
      - tool_call：Agent 决定调用工具
      - tool_result：工具返回摘要
      - done：最终回复和指标
      - error：执行异常
    """
    check_agent_rate_limit(http_request, request.session_id)
    agent_service = _agent_service_for_model(request.model_id)
    if agent_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent 不可用：请配置模型网关 config/model_gateway.yaml 和对应的后端 API Key。",
        )

    check = _input_guard.check(request.message)
    if not check.passed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"输入校验失败：{check.reason}",
        )

    async def _event_generator():
        # 这里直接转发 service 生成的 NDJSON 行，避免路由层关心每个 token/tool 事件的细节。
        # 如果未来要做鉴权、审计或敏感信息过滤，可以在这里增加一个轻量事件过滤器。
        async for line in agent_service.stream_chat(
            session_id=request.session_id,
            message=request.message,
        ):
            yield line

    return StreamingResponse(
        _event_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/cache/stats", response_model=ApiResponse)
def get_cache_stats() -> ApiResponse:
    """查询工具结果缓存命中率统计。

    返回字段：
        cached_entries  — 当前缓存条目数
        total_hits      — 累计缓存命中次数
        total_misses    — 累计缓存未命中次数
        hit_rate        — 命中率（0.0 ~ 1.0）
    """
    return ApiResponse(
        success=True,
        message="缓存统计信息。",
        data=get_tool_cache().stats,
    )


@router.delete("/cache", response_model=ApiResponse)
def invalidate_cache(
    tool_name: str | None = Query(default=None, description="指定工具名称，留空则清理全部过期条目"),
    scope: str = Query(default=GLOBAL_SCOPE, description="缓存 scope，默认全局"),
) -> ApiResponse:
    """主动失效工具缓存。

    - 不传 tool_name：清理所有已过期条目（定期维护用）
    - 传 tool_name：立即删除该工具在指定 scope 下的全部缓存
      （数据更新后调用，强制下次重新查询）
    """
    cache = get_tool_cache()
    if tool_name:
        count = cache.invalidate_tool(tool_name, scope)
        msg = f"已删除工具 '{tool_name}' 在 scope='{scope}' 下的 {count} 条缓存。"
    else:
        count = cache.evict_expired()
        msg = f"已清理 {count} 条过期缓存条目。"
    return ApiResponse(success=True, message=msg, data={"evicted": count})


# ── Plan-and-Execute Agent ────────────────────────────────────────────────────

@router.post("/plan-execute", response_model=ApiResponse)
async def run_plan_execute(request: PlanExecuteRequest, http_request: Request) -> ApiResponse:
    """Plan-and-Execute Agent — 先制定全局计划，再逐步执行。

    与 /chat 的区别：
      /chat (ReAct)         — 每步局部决策，适合中等复杂度的单次对话
      /plan-execute         — 先规划所有步骤，再按序执行，适合步骤有明确依赖的复杂分析

    典型使用场景：
      - "帮我完整分析订单 SO123，包括库存状态、可用替代品和最优履约方案"
      - "SO123 无法全量发货，给我所有可能的解决路径"
      - 任何需要多个工具串联且顺序依赖明确的复杂问题

    返回字段：
      final_answer  — 综合所有步骤结果生成的最终答复
      plan_steps    — 实际执行的步骤列表（体现规划思路）
      step_results  — 每步的详细结果
      total_steps   — 执行步骤总数
    """
    check_agent_rate_limit(http_request, request.session_id or request.order_id)
    plan_execute_service = _plan_execute_service_for_model(request.model_id)
    if plan_execute_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Plan-and-Execute Agent 不可用：请配置模型网关 config/model_gateway.yaml 和对应的后端 API Key。",
        )

    # 输入安全检查
    check = _input_guard.check(request.question)
    if not check.passed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"输入校验失败：{check.reason}",
        )

    try:
        result = await plan_execute_service.execute(
            order_id=request.order_id,
            question=request.question,
            session_id=request.session_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plan-and-Execute 执行出错：{exc}",
        ) from exc

    response = PlanExecuteResponse(**result)
    return ApiResponse(
        success=True,
        message=f"Plan-and-Execute 完成，共执行 {result['total_steps']} 个步骤。",
        data=response.model_dump(mode="json"),
    )
