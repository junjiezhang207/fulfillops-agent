"""Agent 多轮对话 & 记忆测试。"""

import pytest

from app.core.config import Settings
from app.infrastructure.llm.chat_adapter import LLMFactory
from app.repositories.file_system_knowledge_repository import (
    FileSystemKnowledgeRepository,
)
from app.repositories.in_memory_inventory_repository import (
    InMemoryInventoryRepository,
)
from app.repositories.in_memory_order_repository import InMemoryOrderRepository
from app.agents.runtime.agent_service import AgentService
from app.domain.fulfillment.plan_service import FulfillmentPlanService
from app.domain.inventory.analysis import InventoryAnalysisService
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
from app.domain.orders.analysis import OrderAnalysisService
from app.domain.fulfillment.substitute_sku import SubstituteSkuService
from app.domain.inventory.warehouse_service import WarehouseService

# ---- Setup ----
@pytest.fixture
def settings():
    """获取当前系统配置。"""
    return Settings()


@pytest.fixture
def services(settings):
    """初始化所有后端服务。"""
    order_repo = InMemoryOrderRepository()
    order_service = OrderAnalysisService(order_repo)

    inventory_repo = InMemoryInventoryRepository()
    inventory_service = InventoryAnalysisService(
        inventory_repository=inventory_repo,
        order_analysis_service=order_service,
    )

    knowledge_repo = FileSystemKnowledgeRepository(settings.knowledge_dir)
    knowledge_service = KnowledgeRetrievalService(
        knowledge_repository=knowledge_repo,
        inventory_analysis_service=inventory_service,
    )

    warehouse_service = WarehouseService()
    substitute_service = SubstituteSkuService()
    fulfillment_service = FulfillmentPlanService(
        inventory_service=inventory_service,
        warehouse_service=warehouse_service,
        substitute_service=substitute_service,
    )

    return {
        "order": order_service,
        "inventory": inventory_service,
        "knowledge": knowledge_service,
        "warehouse": warehouse_service,
        "substitute": substitute_service,
        "fulfillment": fulfillment_service,
    }


@pytest.fixture
def agent_service(services, settings):
    """创建带工具的 Agent 服务。"""
    chat_model = LLMFactory.create_chat_model(settings)
    if chat_model is None:
        pytest.skip("未配置 LLM，跳过 Agent 测试")

    from app.agents.tools.factory import (
        make_fulfillment_plan_tool,
        make_substitute_tool,
        make_warehouse_tool,
    )

    extra_tools = [
        make_warehouse_tool(services["warehouse"]),
        make_substitute_tool(services["substitute"]),
        make_fulfillment_plan_tool(services["fulfillment"]),
    ]

    return AgentService(
        order_service=services["order"],
        inventory_service=services["inventory"],
        knowledge_service=services["knowledge"],
        chat_model=chat_model,
        extra_tools=extra_tools,
    )


# ---- 测试用例 ----
class TestAgentBasic:
    """Agent 基础功能测试。"""
    pytestmark = pytest.mark.slow

    @pytest.mark.asyncio
    async def test_single_turn_simple_question(self, agent_service):
        """测试单轮对话：简单问题。"""
        result = await agent_service.chat(
            session_id="test-001",
            message="订单 SO202502140001 是什么？",
            include_trace=True,
        )

        assert result["reply"]
        assert isinstance(result["reply"], str)
        assert len(result["reply"]) > 0
        assert result["trace"] is not None
        assert result["trace"].user_message == "订单 SO202502140001 是什么？"

    @pytest.mark.asyncio
    async def test_single_turn_with_tool_calls(self, agent_service):
        """测试单轮对话：触发工具调用。"""
        result = await agent_service.chat(
            session_id="test-002",
            message="订单 SO202502140001 的库存充足吗？",
            include_trace=True,
        )

        assert result["reply"]
        # 应该调用了 analyze_order 和 check_inventory
        assert len(result["tools_called"]) >= 2
        assert "analyze_order" in result["tools_called"]
        assert "check_inventory" in result["tools_called"]


class TestAgentMemory:
    """Agent 记忆机制测试。"""
    pytestmark = pytest.mark.slow

    @pytest.mark.asyncio
    async def test_multi_turn_same_session(self, agent_service):
        """测试多轮对话：同一 session_id 保留记忆。"""
        session_id = "test-memory-001"

        # 第一轮：询问订单
        result1 = await agent_service.chat(
            session_id=session_id,
            message="请分析订单 SO202502140001",
            include_trace=True,
        )
        assert result1["reply"]

        # 第二轮：追问，不重复说订单号（假设记忆工作）
        result2 = await agent_service.chat(
            session_id=session_id,
            message="如果库存不足，应该怎么办？",
            include_trace=True,
        )
        assert result2["reply"]
        assert result2["trace"].user_message == "如果库存不足，应该怎么办？"

        # 两轮对话应该有不同的 user_message
        assert result1["trace"].user_message != result2["trace"].user_message

    @pytest.mark.asyncio
    async def test_separate_sessions_independent(self, agent_service):
        """测试多轮对话：不同 session_id 相互独立。"""
        # 会话 A
        result_a1 = await agent_service.chat(
            session_id="session-a",
            message="订单 SO202502140001 的情况",
            include_trace=True,
        )

        # 会话 B
        result_b1 = await agent_service.chat(
            session_id="session-b",
            message="订单 SO202502140002 的情况",
            include_trace=True,
        )

        # 两个会话的回复应该不同（涉及不同订单）
        assert result_a1["reply"] != result_b1["reply"]


class TestAgentTools:
    """Agent 工具调用正确性测试。"""

    def test_warehouse_tool(self, services):
        """测试仓库库存查询工具。"""
        result = services["warehouse"].search_sku_inventory("SKU-IPHONE-CASE-001")
        assert result.sku_id == "SKU-IPHONE-CASE-001"
        assert result.total_available > 0
        assert len(result.warehouse_list) > 0
        assert result.summary

    def test_substitute_tool(self, services):
        """测试替代SKU查询工具。"""
        result = services["substitute"].search_substitutes("SKU-IPHONE-CASE-001")
        assert result.original_sku_id == "SKU-IPHONE-CASE-001"
        assert len(result.substitutes) > 0
        assert result.summary

    def test_fulfillment_plan_tool(self, services):
        """测试履约方案生成工具。"""
        result = services["fulfillment"].generate_plan("SO202502140001")
        assert result.order_id == "SO202502140001"
        assert result.plan_strategy in ["fast_track", "mixed", "backorder"]
        assert result.summary
        # 充足库存的订单应该是 fast_track
        assert result.plan_strategy == "fast_track"


class TestAgentTrace:
    """Agent 执行轨迹可观测性测试。"""
    pytestmark = pytest.mark.slow

    @pytest.mark.asyncio
    async def test_trace_structure(self, agent_service):
        """测试轨迹的完整结构。"""
        result = await agent_service.chat(
            session_id="test-trace",
            message="订单 SO202502140001 库存够吗？",
            include_trace=True,
        )

        trace = result["trace"]
        assert trace is not None
        assert trace.session_id == "test-trace"
        assert trace.user_message == "订单 SO202502140001 库存够吗？"
        assert trace.final_reply == result["reply"]
        assert trace.execution_steps >= 0
        # tools_called 应该是一个 ToolCallDetail 对象的列表
        assert isinstance(trace.tools_called, list)

    @pytest.mark.asyncio
    async def test_trace_tool_details(self, agent_service):
        """测试轨迹中的工具调用详情。"""
        result = await agent_service.chat(
            session_id="test-trace-detail",
            message="查询订单 SO202502140002 的库存分布",
            include_trace=True,
        )

        trace = result["trace"]
        if trace.tools_called:
            for tool_call in trace.tools_called:
                assert tool_call.tool_name
                assert tool_call.order >= 1
                # output 可能为空（如果工具没有输出），但不应该为 None
                assert tool_call.output is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
