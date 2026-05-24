"""DeepEval 测试公共 fixtures。

设计说明：
  - 不依赖真实 LLM：AgentService 在 eval 环境下使用 NoopLLM，
    工具层直接调用 mock service，测试可在 CI 离线运行。
  - 需要真实 LLM 的 LLM-as-judge 测试用 @pytest.mark.slow 标记，
    CI 跳过，开发者本地跑时加 --slow 参数。
  - 每个 case 使用独立的 session_id（gc-<case_id>），避免会话历史污染。
"""

import pytest

from app.repositories.file_system_knowledge_repository import FileSystemKnowledgeRepository
from app.repositories.in_memory_inventory_repository import InMemoryInventoryRepository
from app.repositories.in_memory_order_repository import InMemoryOrderRepository
from app.services.fulfillment_plan_service import FulfillmentPlanService
from app.services.inventory_analysis_service import InventoryAnalysisService
from app.services.knowledge_retrieval_service import KnowledgeRetrievalService
from app.services.order_analysis_service import OrderAnalysisService
from app.services.substitute_sku_service import SubstituteSkuService
from app.services.warehouse_service import WarehouseService


def _build_services():
    """构建完整的 service 依赖树（使用内存 mock，无数据库依赖）。"""
    from app.core.config import get_settings
    settings = get_settings()

    order_repo = InMemoryOrderRepository()
    order_svc = OrderAnalysisService(order_repo)

    inv_repo = InMemoryInventoryRepository()
    inv_svc = InventoryAnalysisService(
        inventory_repository=inv_repo,
        order_analysis_service=order_svc,
    )

    knowledge_repo = FileSystemKnowledgeRepository(settings.knowledge_dir)
    knowledge_svc = KnowledgeRetrievalService(
        knowledge_repository=knowledge_repo,
        inventory_analysis_service=inv_svc,
    )

    warehouse_svc = WarehouseService()
    sub_svc = SubstituteSkuService()
    fulfillment_svc = FulfillmentPlanService(
        inventory_service=inv_svc,
        warehouse_service=warehouse_svc,
        substitute_service=sub_svc,
    )

    return order_svc, inv_svc, knowledge_svc, warehouse_svc, sub_svc, fulfillment_svc


@pytest.fixture(scope="session")
def agent_service_with_llm():
    """返回接入真实 LLM 的 AgentService（需要配置模型网关和对应 API Key）。

    若未配置，自动 skip 当前测试。
    """
    from app.agent.tools import (
        make_fulfillment_plan_tool,
        make_substitute_tool,
        make_warehouse_tool,
    )
    from app.core.config import get_settings
    from app.graph.llm_adapter import LLMFactory
    from app.services.agent_service import AgentNotAvailableError, AgentService

    settings = get_settings()
    chat_model = LLMFactory.create_chat_model(settings)
    if chat_model is None:
        pytest.skip("未配置模型网关或对应 API Key，跳过需要 LLM 的测试。")

    order_svc, inv_svc, knowledge_svc, warehouse_svc, sub_svc, fulfillment_svc = _build_services()

    extra_tools = [
        make_warehouse_tool(warehouse_svc),
        make_substitute_tool(sub_svc),
        make_fulfillment_plan_tool(fulfillment_svc),
    ]

    try:
        service = AgentService(
            order_service=order_svc,
            inventory_service=inv_svc,
            knowledge_service=knowledge_svc,
            chat_model=chat_model,
            extra_tools=extra_tools,
            enable_reflection=False,  # eval 时关闭反思，节省 token
            enable_structured_output=False,
        )
        return service
    except AgentNotAvailableError as exc:
        pytest.skip(f"AgentService 初始化失败：{exc}")


@pytest.fixture(scope="session")
def tool_services():
    """返回 service 元组，供纯工具层测试使用（无 LLM 依赖）。"""
    return _build_services()
