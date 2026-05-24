"""工作流模块数据模型。

这里的模型对应 POST /api/v1/workflow/run 的请求与响应，
也被 graph.state.GraphState 引用（FinalAnswer）。

设计理念："结构化输入 + 结构化输出 + 全程 trace 可见"。

学习时可以按数据流理解：
  WorkflowRunRequest  -> API 接收用户请求
  GraphState          -> LangGraph 节点之间传递中间状态
  FinalAnswer         -> finalize 节点生成最终业务结论
  WorkflowRunResult   -> API 返回给调用方的完整结果
"""

from pydantic import BaseModel, Field

from app.graph.trace import ErrorEvent, TraceEvent
from app.schemas.inventory import InventoryAnalysisResult
from app.schemas.knowledge import KnowledgeRetrieveResult
from app.schemas.orders import OrderAnalysisResult


class WorkflowRunRequest(BaseModel):
    """主链路运行请求模型。

    这是 workflow 的入口参数。它不会包含中间结果；
    中间结果由 graph 节点逐步写入 GraphState。
    """

    order_id: str = Field(..., description="待分析的订单编号。")
    question: str | None = Field(
        default=None,
        description=(
            "可选的业务问题。提供时会进入知识检索节点；"
            "不提供且库存充足时会走快速路径，跳过知识检索。"
        ),
    )
    filter_categories: list[str] = Field(
        default_factory=list,
        description=(
            "可选的知识类别过滤条件，透传给知识检索服务。"
            "合法值参见 KnowledgeRetrieveRequest。"
        ),
    )


class FinalAnswer(BaseModel):
    """主链路最终结论模型。

    finalize_node 会根据已有的订单/库存/知识结果拼出这个对象。
    它是 workflow 最重要的“对用户可读结果”，但仍然保持结构化：
    结论、依据、建议动作、路径标识分开存，方便前端展示和自动评测。
    """

    conclusion: str = Field(
        ..., description="对整条链路的一句话结论（由 LLM 或规则模板生成）。"
    )
    key_evidences: list[str] = Field(
        default_factory=list,
        description="支撑结论的关键事实（订单/库存/知识中摘出的最有价值的短句）。",
    )
    suggested_actions: list[str] = Field(
        default_factory=list,
        description="基于当前状态给出的行动建议。",
    )
    routing_path: str = Field(
        ...,
        description=(
            "主链路实际走的路径标识，方便调用方判断。"
            "取值：fast_path（库存充足直通）/ knowledge_path（经过知识检索）。"
        ),
    )


class InterruptEvent(BaseModel):
    """Human-in-the-Loop 中断事件（增强版）。

    当节点调用 ``interrupt(...)`` 时，WorkflowService 会把中断信息转成这个模型。
    前端审批台拿到它后，展示风险原因和选项；审批完成后再用 thread_id 调 resume。
    """
    type: str = Field(..., description="中断类型：standard_approval / critical_approval")
    node: str = Field(..., description="触发中断的节点名")
    prompt: str = Field(..., description="给审批员的完整提示（含风险说明和建议动作）")
    context: dict = Field(default_factory=dict, description="订单摘要、库存摘要、风险详情")
    options: list[str] = Field(
        default_factory=lambda: ["approved", "rejected", "escalate"],
        description="可选决策：approved=确认 / rejected=拒绝 / escalate=上报",
    )
    thread_id: str = Field(..., description="用于 resume 的线程 ID")
    # 新增：风险信息
    risk_level: str = Field(default="HIGH", description="风险等级：LOW/MEDIUM/HIGH/CRITICAL")
    risk_signals: list[str] = Field(default_factory=list, description="触发的风险规则名称")
    timeout_seconds: int = Field(default=1800, description="超时秒数，到期后自动降级处理")


class ApprovalRequest(BaseModel):
    """审批员提交决策时的请求体（resume 接口使用）。

    这是 resume 接口的请求体。
    decision 会作为 LangGraph ``Command(resume=decision)`` 的值回到中断节点；
    reason 和 approver_id 则用于审计，不参与路由判断。
    """
    decision: str = Field(..., description="决策：approved / rejected / escalate")
    reason: str = Field(..., min_length=1, description="审批理由（必填，写入审计日志）")
    approver_id: str = Field(..., min_length=1, description="审批人 ID（工号或用户名）")


class ApprovalAuditEntry(BaseModel):
    """单条审批记录（审计日志中的一条）。"""
    thread_id: str
    order_id: str
    interrupt_type: str
    risk_level: str
    risk_signals: list[str] = Field(default_factory=list)
    decision: str
    reason: str
    approver_id: str
    requested_at: str
    decided_at: str


class WorkflowRunResult(BaseModel):
    """主链路运行结果模型。

    这是 API 响应 data 字段的完整结构，汇聚：
      1. 入参回显
      2. 三个模块的中间结果
      3. 最终结论
      4. 执行轨迹（trace）与错误列表

    注意：快速路径下 knowledge_result 可以为 None；
    出错时 final_answer 也可能为 None，但 trace/errors 会说明发生了什么。
    """

    order_id: str = Field(..., description="订单编号。")
    question: str | None = Field(default=None, description="原始业务问题。")
    filter_categories: list[str] = Field(
        default_factory=list,
        description="实际生效的知识类别过滤条件。",
    )
    order_result: OrderAnalysisResult | None = Field(
        default=None, description="订单分析节点结果。"
    )
    inventory_result: InventoryAnalysisResult | None = Field(
        default=None, description="库存判断节点结果。"
    )
    knowledge_result: KnowledgeRetrieveResult | None = Field(
        default=None,
        description="知识检索节点结果。当走快速路径时为 null。",
    )
    final_answer: FinalAnswer | None = Field(
        default=None, description="finalize 节点产出的最终结论。"
    )
    trace: list[TraceEvent] = Field(
        default_factory=list,
        description="按执行顺序记录的所有节点轨迹。",
    )
    errors: list[ErrorEvent] = Field(
        default_factory=list,
        description="节点执行过程中产生的错误事件列表。",
    )
