"""并行工作流 — 两阶段 fan-out/fan-in，使用 LangGraph Send API。

这是普通 workflow 的性能优化版本。建议先读懂 ``workflow.py`` 的串行图，
再读这里：这里不是改业务判断，而是把互不依赖的节点并行执行。

核心思想：
  - 可以同时做的事就 fan-out。
  - 需要等多个分支都回来时，就放 barrier/merge 节点 fan-in。

亮点（面试重点）：
  ┌──────────────────────────────────────────────────────────────┐
  │  Stage-1 并行（两个独立数据源同时查询）                         │
  │    dispatch ──→ order_analysis     ─┐                        │
  │             └→ inventory_analysis ─→ stage1_barrier          │
  │                                                              │
  │  Stage-2 并行（仅缺货路径，Send API 条件 fan-out）              │
  │    stage1_barrier → [Send] knowledge_retrieval ─┐            │
  │                           [Send] warehouse_search ─→ merge   │
  └──────────────────────────────────────────────────────────────┘

LangGraph 并行机制说明：
  fan-out ：同一节点有多条出边 → 并行触发所有下游
  fan-in  ：同一节点有多条入边 → 等所有上游完成后才触发
  Send API：conditional_edge 返回 list[Send] → 动态 fan-out
             每个 Send(节点名, state副本) 独立发送，互不阻塞

延迟对比（理论值）：
  顺序：order(200ms) + inventory(300ms) + knowledge(400ms) = 900ms
  并行：max(200,300)(300ms) + max(400, warehouse)(400ms) = 700ms
  提升约 22%（I/O 密集场景效果更显著）
"""

from datetime import datetime
from typing import Union

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from langchain_core.language_models import BaseChatModel

from app.workflows.fulfillment.nodes import WorkflowNodes
from app.workflows.fulfillment.state import GraphState
from app.workflows.fulfillment.trace import ErrorEvent, build_trace_event
from app.services.inventory_analysis_service import InventoryAnalysisService
from app.services.knowledge_retrieval_service import KnowledgeRetrievalService
from app.services.order_analysis_service import OrderAnalysisService


class ParallelWorkflowNodes(WorkflowNodes):
    """扩展 WorkflowNodes：增加并行执行所需的三个节点。

    新增节点：
      stage1_barrier  — Stage-1 fan-in 同步点（纯等待，无计算）
      warehouse_search — 跨仓库备货查询（与 knowledge_retrieval 并行）
      merge_stockout  — Stage-2 fan-in 同步点（纯等待，无计算）
    """

    # ------------------------------------------------------------------ #
    # barrier 节点：fan-in 同步点
    # ------------------------------------------------------------------ #

    def stage1_barrier(self, state: GraphState) -> dict:
        """Stage-1 fan-in barrier。

        设计理由：
          LangGraph 的 fan-in 语义依赖"多条边汇入同一节点"。
          order_analysis 和 inventory_analysis 各向本节点发送一条边，
          LangGraph 会等两者都完成后才执行本节点，从而实现真正的同步等待。

          本节点不写任何 state，仅充当"门闩"。
        """
        return {}

    def merge_stockout(self, state: GraphState) -> dict:
        """Stage-2 fan-in barrier — knowledge + warehouse 并行结果汇聚点。

        knowledge_retrieval 和 warehouse_search 各向本节点发送一条边，
        确保 finalize 只在两者都完成后才执行。
        """
        return {}

    # ------------------------------------------------------------------ #
    # warehouse_search：缺货路径的并行 I/O 节点
    # ------------------------------------------------------------------ #

    def warehouse_search(self, state: GraphState) -> dict:
        """跨仓库备货查询 — 与 knowledge_retrieval 并行执行。

        业务价值：
          缺货时，在等待知识库规则检索的同时，
          提前查询其他仓库是否有可调配库存，
          为 finalize 节点提供"跨仓调配"这一额外决策维度。

        并行收益：
          knowledge_retrieval 约 400ms（向量检索 + LLM）
          warehouse_search    约 200ms（纯数据库查询）
          串行 = 600ms，并行 = max(400, 200) = 400ms，节省 33%
        """
        start_ts = datetime.now()
        order_id = state.get("order_id", "")

        try:
            # 复用 inventory_service 做跨仓库分析
            # 实际生产中应调用专门的 WarehouseService.search_all_warehouses()
            result = self.inventory_service.analyze_inventory(order_id)
            end_ts = datetime.now()

            summary = result.summary if result else "无跨仓库数据"
            insufficient = getattr(result, "insufficient_skus", [])
            note = f"跨仓查询完成，缺货 SKU：{insufficient or '无'}"

            return {
                "warehouse_search_result": summary,
                "trace": [
                    build_trace_event(
                        node="warehouse_search",
                        start_ts=start_ts,
                        end_ts=end_ts,
                        status="ok",
                        note=note,
                    )
                ],
            }
        except Exception as exc:
            end_ts = datetime.now()
            return {
                "warehouse_search_result": None,
                "errors": [
                    ErrorEvent(
                        node="warehouse_search",
                        message=str(exc),
                        exception_type=type(exc).__name__,
                    )
                ],
                "trace": [
                    build_trace_event(
                        node="warehouse_search",
                        start_ts=start_ts,
                        end_ts=end_ts,
                        status="error",
                    )
                ],
            }


# ============================================================================
# 路由函数
# ============================================================================

def route_after_stage1(state: GraphState) -> Union[str, list]:
    """Stage-1 barrier 之后的分支路由。

    返回值可以是：
      str        → 普通单路由，mapping 映射到节点名
      list[Send] → 动态 fan-out，Send API 同时触发多个节点

    Send API 核心语义：
      Send(node_name, state_dict) 向目标节点发送一份 state 副本，
      多个 Send 同时发出 = 并行 fan-out，目标节点独立运行、互不阻塞。
      这是 LangGraph 中实现"条件并行"的标准模式。
    """
    if state.get("fulfillment_branch") == "fulfillable":
        # 快速路径：库存充足，直接 finalize
        return "finalize"

    # 缺货路径：同时启动知识检索 + 跨仓查询（Send API fan-out）
    return [
        Send("knowledge_retrieval", dict(state)),
        Send("warehouse_search", dict(state)),
    ]


# ============================================================================
# 工厂函数
# ============================================================================

def build_parallel_workflow(nodes: ParallelWorkflowNodes) -> CompiledStateGraph:
    """构建两阶段并行工作流。

    图拓扑（→ 顺序边，⇉ 并行边）：

      START
        → dispatch
        ⇉ order_analysis       ↘
        ⇉ inventory_analysis    → stage1_barrier → route_after_stage1
                                                     │
                                          ┌──────────┴──────────┐
                                     fast_path              stockout [Send]
                                          │               ⇉ knowledge_retrieval ↘
                                          │               ⇉ warehouse_search    → merge_stockout
                                          │                                           │
                                          └───────────────────────────────────────────┘
                                                                                      ↓
                                                                                  finalize
      END

    关键设计决策：
      1. stage1_barrier 是纯 fan-in 节点（无业务逻辑），只做同步等待
      2. Send API 在 route_after_stage1 中实现条件 fan-out
      3. merge_stockout 同样是纯 fan-in，确保 finalize 获得完整数据

    注意：
      这个函数目前用于演示并行图结构，未像 workflow.build_workflow 那样暴露
      checkpointer/store 参数。如果要把它提升为生产主链路，应补齐 Redis
      checkpointer 和长期记忆 store 的 compile 参数，避免并行版本和串行版本
      在会话恢复能力上不一致。
    """
    graph: StateGraph = StateGraph(GraphState)

    # 注册全部节点
    graph.add_node("dispatch", nodes.dispatch)
    graph.add_node("order_analysis", nodes.order_analysis)
    graph.add_node("inventory_analysis", nodes.inventory_analysis)
    graph.add_node("stage1_barrier", nodes.stage1_barrier)       # fan-in barrier
    graph.add_node("knowledge_retrieval", nodes.knowledge_retrieval)
    graph.add_node("warehouse_search", nodes.warehouse_search)   # 新增并行节点
    graph.add_node("merge_stockout", nodes.merge_stockout)       # fan-in barrier
    graph.add_node("finalize", nodes.finalize)

    # ── Stage-1 fan-out ──────────────────────────────────────────────
    # dispatch 有两条出边 → LangGraph 并行触发 order_analysis 和 inventory_analysis
    graph.add_edge(START, "dispatch")
    graph.add_edge("dispatch", "order_analysis")       # 并行分支 1
    graph.add_edge("dispatch", "inventory_analysis")   # 并行分支 2

    # ── Stage-1 fan-in ───────────────────────────────────────────────
    # stage1_barrier 有两条入边 → LangGraph 等两个分支都完成后才触发
    graph.add_edge("order_analysis", "stage1_barrier")
    graph.add_edge("inventory_analysis", "stage1_barrier")

    # ── Stage-1 → 条件路由（可能触发 Stage-2 并行）───────────────────
    # 当 route_after_stage1 返回 list[Send] 时，mapping 中的 Send 目标必须列出
    graph.add_conditional_edges(
        "stage1_barrier",
        route_after_stage1,
        {
            "finalize": "finalize",                         # 快速路径
            "knowledge_retrieval": "knowledge_retrieval",   # Send 目标 1
            "warehouse_search": "warehouse_search",         # Send 目标 2
        },
    )

    # ── Stage-2 fan-in ───────────────────────────────────────────────
    # merge_stockout 有两条入边 → 等 knowledge + warehouse 都完成
    graph.add_edge("knowledge_retrieval", "merge_stockout")
    graph.add_edge("warehouse_search", "merge_stockout")
    graph.add_edge("merge_stockout", "finalize")

    graph.add_edge("finalize", END)

    return graph.compile()


def create_parallel_workflow_nodes(
    order_service: OrderAnalysisService,
    inventory_service: InventoryAnalysisService,
    knowledge_service: KnowledgeRetrievalService,
    chat_model: BaseChatModel | None = None,
) -> ParallelWorkflowNodes:
    """工厂函数：创建并行工作流节点实例。"""
    return ParallelWorkflowNodes(
        order_service=order_service,
        inventory_service=inventory_service,
        knowledge_service=knowledge_service,
        chat_model=chat_model,
    )
