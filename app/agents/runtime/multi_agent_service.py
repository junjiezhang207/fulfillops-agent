"""多 Agent 服务（学习版注释）— MultiAgentOrchestrator 的外观层。

职责：
  1. 依赖注入：组装 MultiAgentOrchestrator 所需的所有 service
  2. 格式转换：MultiAgentState → 标准化的 MultiAgentResult
  3. 计时和可观测：记录总耗时、各 Agent 调用轨迹

为什么需要这一层：
- app.agents.orchestration.multi_agent 更偏“Agent 编排实现”。
- app.agents.runtime.multi_agent_service 更偏“业务服务入口”。
- API 层只依赖这个 Service，避免直接知道 Agent 内部状态结构。
"""

import time
from dataclasses import dataclass, field

from app.agents.orchestration.multi_agent import MultiAgentOrchestrator
from app.domain.fulfillment.plan_service import FulfillmentPlanService
from app.domain.inventory.analysis import InventoryAnalysisService
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
from app.domain.orders.analysis import OrderAnalysisService
from app.domain.fulfillment.substitute_sku import SubstituteSkuService
from app.domain.inventory.warehouse_service import WarehouseService


@dataclass
class AgentCallRecord:
    """单个专业 Agent 的调用记录。

    这不是 Agent 的全部内部状态，而是给前端/面试演示看的摘要记录。
    """

    # 被调用的专业 Agent 名称，例如 inventory_agent / fulfillment_agent。
    agent_name: str
    # Supervisor 分配给这个 Agent 的具体问题。
    question: str
    # 该 Agent 返回的摘要。
    summary: str
    # 该 Agent 调用过的工具名。
    tools_called: list[str] = field(default_factory=list)
    # 摘要长度，便于前端展示或做简单统计。
    summary_length: int = 0

    def __post_init__(self):
        # dataclass 初始化后自动计算 summary_length，避免外部手动传错。
        self.summary_length = len(self.summary)


@dataclass
class MultiAgentResult:
    """多 Agent 服务的标准化输出。

    Orchestrator 返回的是 dict，这里转成 dataclass 是为了字段更清楚。
    """

    order_id: str
    question: str

    # 最终答案
    final_answer: str

    # 调用了哪些 Agent（按顺序）
    agents_called: list[str] = field(default_factory=list)

    # 每个 Agent 的详细输出
    agent_records: list[AgentCallRecord] = field(default_factory=list)

    # Supervisor 的推理过程
    supervisor_reasoning: list[str] = field(default_factory=list)

    # 执行日志
    execution_log: list[str] = field(default_factory=list)

    # 统计
    total_agents_called: int = 0
    execution_time_ms: float = 0.0

    def __post_init__(self):
        # 保持统计字段和 agents_called 列表一致。
        self.total_agents_called = len(self.agents_called)


class MultiAgentService:
    """多 Agent 系统外观服务。

    与 AgentService 的区别：
      AgentService:    单个 ReAct Agent，自主选择工具，适合中等复杂问题
      MultiAgentService: 多专业 Agent 协同，Supervisor 统筹调度，适合跨域复杂问题

    典型场景对比：
      "这个订单的库存够吗？" → AgentService（单域问题）
      "订单缺货，评估风险，给出最优履约方案，并检查相关规则" → MultiAgentService（跨域）
    """

    def __init__(
        self,
        order_service: OrderAnalysisService,
        inventory_service: InventoryAnalysisService,
        knowledge_service: KnowledgeRetrievalService,
        warehouse_service: WarehouseService,
        substitute_service: SubstituteSkuService,
        fulfillment_service: FulfillmentPlanService,
        llm=None,
    ):
        # 这里集中组装 MultiAgentOrchestrator 的依赖。
        # 以后新增专业 Agent，通常也是从这里把新的 service 注入进去。
        self._orchestrator = MultiAgentOrchestrator(
            order_service=order_service,
            inventory_service=inventory_service,
            knowledge_service=knowledge_service,
            warehouse_service=warehouse_service,
            substitute_service=substitute_service,
            fulfillment_service=fulfillment_service,
            llm=llm,
        )

    def run(self, order_id: str, question: str) -> MultiAgentResult:
        """执行多 Agent 编排并返回标准化结果。

        Args:
            order_id: 订单 ID
            question: 用户问题

        Returns:
            MultiAgentResult 包含最终答案、Agent 调用链、执行日志
        """
        start_time = time.time()

        # 真正的多 Agent 调度发生在 orchestrator 内部。
        raw = self._orchestrator.run(order_id=order_id, question=question)

        # 记录端到端耗时，不包含前端渲染时间。
        execution_time_ms = (time.time() - start_time) * 1000

        # 转换 agent_results 为 AgentCallRecord。
        # raw 里可能缺字段，所以用 get 做容错。
        agent_records = [
            AgentCallRecord(
                agent_name=r["agent"],
                question=r.get("question", question),
                summary=r.get("summary", ""),
                tools_called=list(r.get("tools_called", [])),
            )
            for r in raw.get("agent_results", [])
        ]

        # 返回统一结构，屏蔽 orchestrator 内部字典细节。
        return MultiAgentResult(
            order_id=order_id,
            question=question,
            final_answer=raw.get("final_answer", ""),
            agents_called=raw.get("agents_called", []),
            agent_records=agent_records,
            supervisor_reasoning=raw.get("supervisor_reasoning", []),
            execution_log=raw.get("execution_log", []),
            execution_time_ms=execution_time_ms,
        )
