"""LangGraph 工作流状态对象（GraphState）。

为什么用 TypedDict 而不是 Pydantic BaseModel？
    1. LangGraph 1.0+ 官方推荐 TypedDict 做状态。
    2. 节点返回的是"增量 dict"，TypedDict 天然契合这种写法。
    3. TypedDict 的字段可以用 Annotated 注册 reducer，Pydantic 不行。
    4. 避免每次节点返回时触发一次完整 Pydantic 校验，性能更合理。

字段分三类：
    入参字段：order_id / question / filter_categories
    中间结果字段：*_result + fulfillment_branch
    可观察字段：trace / errors / final_answer

学习重点：
    LangGraph 的节点通常不是“返回最终结果”，而是“返回一小块 state 更新”。
    比如 order_analysis 节点只返回 {"order_result": ...}，inventory_analysis 节点
    再返回 {"inventory_result": ..., "fulfillment_branch": ...}。
    框架会把这些增量合并成完整 GraphState。
"""

from operator import add
from typing import Annotated, Literal, TypedDict

from app.graph.trace import ErrorEvent, TraceEvent
from app.schemas.inventory import InventoryAnalysisResult
from app.schemas.knowledge import KnowledgeRetrieveResult
from app.schemas.orders import OrderAnalysisResult
from app.schemas.workflow import FinalAnswer


# Literal 类型限定分支取值，避免条件边返回非法值。
# 如果将来新增路径，比如 "manual_review"，这里和 router 都要一起扩展。
FulfillmentBranch = Literal["fulfillable", "stockout"]


class GraphState(TypedDict, total=False):
    """固定 workflow 的共享状态。

    total=False 的含义：所有字段都是"可选"的。
    这非常重要，因为 state 是被节点"渐进式填充"的：
        invoke 时只传 order_id / question / filter_categories 三个入参，
        其余字段在对应节点执行后才会出现。
    """

    # ----- 入参字段（invoke 时由调用方传入） -----
    # order_id 是主键，几乎所有节点都会用它查业务数据。
    order_id: str
    # question 是用户的自然语言问题，主要给知识检索和最终总结使用。
    question: str | None
    # filter_categories 用于限制 RAG 只查某些规则类别。
    filter_categories: list[str]

    # ----- 中间结果字段（由各节点填充，单次写入） -----
    # 这三个 *_result 是 workflow 的核心事实来源。
    # finalize 不重新查业务系统，只阅读这些节点已经产出的结果。
    order_result: OrderAnalysisResult | None
    inventory_result: InventoryAnalysisResult | None
    knowledge_result: KnowledgeRetrieveResult | None

    # 库存判断后的分支决策结果。只由 inventory_analysis_node 写一次，
    # 被 router.route_after_inventory 读取用于条件边路由。
    fulfillment_branch: FulfillmentBranch | None

    # ----- 并行 Stage-2 额外字段 -----
    # 由 warehouse_search 节点写入（并行工作流专用）。
    # 串行 workflow 不会写这个字段，但 TypedDict total=False 允许它缺省。
    warehouse_search_result: str | None

    # ----- 出口与可观察字段 -----
    # finalize 节点唯一负责写 final_answer。调用方通常只需要读取这个字段和 trace。
    final_answer: FinalAnswer | None

    # ----- Human-in-the-Loop 字段 -----
    # 风险评估结果（由 inventory_analysis 节点写入）
    risk_level: str | None                 # "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
    risk_signals: list[str] | None         # 触发的规则名称列表，如 ["high_value_order"]
    # 人工决策结果，由 LangGraph interrupt/resume 注入。
    # 例如审批员选择 approved 后，resume 会把这个决策带回中断点继续执行。
    human_decision: dict | None
    # 中断信息，由节点在调用 interrupt() 前写入 state，也会返回给前端展示。
    interrupt_info: dict | None

    # trace / errors 使用 operator.add 作为 reducer：
    # 每个节点返回的 list 会被"累加"到 state 里，而不是覆盖。
    # 这是 LangGraph 状态合并机制最核心的一个用法，要特别理解。
    trace: Annotated[list[TraceEvent], add]
    errors: Annotated[list[ErrorEvent], add]
