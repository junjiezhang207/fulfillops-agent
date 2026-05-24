"""对比执行器：运行 Agent 和固定工作流，收集指标。"""

from app.comparison.framework import (
    ComparisonFramework,
    ComparisonResult,
    ComparisonScenario,
    ExecutionMetrics,
    PathType,
)
from app.core.config import get_settings
from app.graph.llm_adapter import LLMFactory
from app.repositories.file_system_knowledge_repository import (
    FileSystemKnowledgeRepository,
)
from app.repositories.in_memory_inventory_repository import (
    InMemoryInventoryRepository,
)
from app.repositories.in_memory_order_repository import InMemoryOrderRepository
from app.schemas.workflow import WorkflowRunRequest
from app.services.agent_service import AgentService
from app.services.fulfillment_plan_service import FulfillmentPlanService
from app.services.inventory_analysis_service import InventoryAnalysisService
from app.services.knowledge_retrieval_service import KnowledgeRetrievalService
from app.services.order_analysis_service import OrderAnalysisService
from app.services.substitute_sku_service import SubstituteSkuService
from app.services.warehouse_service import WarehouseService
from app.services.workflow_service import WorkflowService


class ComparisonExecutor:
    """对比执行器。"""

    def __init__(self):
        settings = get_settings()

        # 初始化所有服务
        order_repo = InMemoryOrderRepository()
        self.order_service = OrderAnalysisService(order_repo)

        inventory_repo = InMemoryInventoryRepository()
        self.inventory_service = InventoryAnalysisService(
            inventory_repository=inventory_repo,
            order_analysis_service=self.order_service,
        )

        knowledge_repo = FileSystemKnowledgeRepository(settings.knowledge_dir)
        self.knowledge_service = KnowledgeRetrievalService(
            knowledge_repository=knowledge_repo,
            inventory_analysis_service=self.inventory_service,
        )

        # 固定工作流
        self.workflow_service = WorkflowService(
            order_service=self.order_service,
            inventory_service=self.inventory_service,
            knowledge_service=self.knowledge_service,
        )

        # Agent
        chat_model = LLMFactory.create_chat_model(settings, use_case="agent")
        if chat_model is None:
            raise RuntimeError("Agent 需要配置 LLM，请检查 .env 文件")

        warehouse_service = WarehouseService()
        substitute_service = SubstituteSkuService()
        fulfillment_service = FulfillmentPlanService(
            inventory_service=self.inventory_service,
            warehouse_service=warehouse_service,
            substitute_service=substitute_service,
        )

        from app.agent.tools import (
            make_fulfillment_plan_tool,
            make_substitute_tool,
            make_warehouse_tool,
        )

        extra_tools = [
            make_warehouse_tool(warehouse_service),
            make_substitute_tool(substitute_service),
            make_fulfillment_plan_tool(fulfillment_service),
        ]

        self.agent_service = AgentService(
            order_service=self.order_service,
            inventory_service=self.inventory_service,
            knowledge_service=self.knowledge_service,
            chat_model=chat_model,
            extra_tools=extra_tools,
        )

    def run_agent_path(self, scenario: ComparisonScenario) -> ExecutionMetrics:
        """运行 Agent 路径。"""
        metrics = ExecutionMetrics(
            path_type=PathType.AGENT,
            scenario_name=scenario.name,
            user_input=scenario.user_input,
        )

        try:
            result = self.agent_service.chat(
                session_id=f"comparison-{scenario.name}",
                message=scenario.user_input,
                include_trace=True,
            )

            metrics.final_reply = result["reply"]
            metrics.tools_called = result["tools_called"]

        except Exception as exc:
            metrics.success = False
            metrics.error_message = str(exc)

        metrics.complete()
        return metrics

    def run_workflow_path(
        self, order_id: str, question: str, scenario: ComparisonScenario
    ) -> ExecutionMetrics:
        """运行固定工作流路径。"""
        metrics = ExecutionMetrics(
            path_type=PathType.WORKFLOW,
            scenario_name=scenario.name,
            user_input=scenario.user_input,
        )

        try:
            request = WorkflowRunRequest(
                order_id=order_id,
                question=question,
                filter_categories=[],
            )
            result = self.workflow_service.run(request)

            # 从工作流结果提取最终答复
            conclusion = result.final_answer.conclusion if result.final_answer else ""
            metrics.final_reply = conclusion

            # 工作流总是调用这些固定的步骤
            metrics.tools_called = ["dispatch", "order_analysis", "inventory_analysis"]
            if result.final_answer and result.final_answer.routing_path == "knowledge_path":
                metrics.tools_called.append("knowledge_retrieval")
            metrics.tools_called.append("finalize")

        except Exception as exc:
            metrics.success = False
            metrics.error_message = str(exc)

        metrics.complete()
        return metrics

    def run_comparison(
        self, scenarios: list[ComparisonScenario]
    ) -> ComparisonFramework:
        """运行完整对比。"""
        framework = ComparisonFramework()

        for scenario in scenarios:
            print(f"\n>> Running scenario: {scenario.name}")
            print(f"   User input: {scenario.user_input}")

            # 从场景中提取订单 ID（假设第一个匹配的订单）
            # 对于这个演示，我们使用固定的订单 ID
            order_id = "SO202502140001"
            if "SO202502140002" in scenario.user_input:
                order_id = "SO202502140002"
            elif "SO202502140003" in scenario.user_input:
                order_id = "SO202502140003"

            # 运行 Agent 路径
            print("   > Running Agent path...")
            agent_metrics = self.run_agent_path(scenario)
            print(f"     [OK] Agent took {agent_metrics.execution_time_ms:.0f}ms")

            # 运行工作流路径
            print("   > Running Workflow path...")
            workflow_metrics = self.run_workflow_path(order_id, scenario.user_input, scenario)
            print(f"     [OK] Workflow took {workflow_metrics.execution_time_ms:.0f}ms")

            # 记录对比结果
            result = ComparisonResult(
                scenario_name=scenario.name,
                user_input=scenario.user_input,
                agent_metrics=agent_metrics,
                workflow_metrics=workflow_metrics,
            )
            framework.record_result(result)

        return framework
