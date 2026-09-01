"""工作流模块数据模型。

这里的模型对应 POST /api/v1/workflow/run 的请求与响应，
也被 graph.state.GraphState 引用（FinalAnswer）。

设计理念："结构化输入 + 结构化输出 + 全程 trace 可见"。

数据流：
  WorkflowRunRequest  -> API 接收用户请求
  GraphState          -> LangGraph 节点之间传递中间状态
  FinalAnswer         -> finalize 节点生成最终业务结论
  WorkflowRunResult   -> API 返回给调用方的完整结果
"""

from pydantic import BaseModel, Field

from app.workflows.fulfillment.trace import ErrorEvent, TraceEvent
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
    session_memory: dict = Field(
        default_factory=dict,
        description="短期结构化会话记忆快照；不包含实时库存、订单状态或物流 ETA。",
    )


class ProposalAction(BaseModel):
    """可审批的单个履约动作。"""

    action_id: str = Field(..., description="动作唯一编号。")
    action_type: str = Field(
        ...,
        description=(
            "动作类型：switch_warehouse / split_order / merge_order / change_carrier / "
            "inventory_transfer / stockout_resolution / ship_from_warehouse"
        ),
    )
    sku_id: str | None = Field(default=None, description="本动作关联的 SKU。")
    quantity: int = Field(default=0, ge=0, description="动作处理数量。")
    from_warehouse: str | None = Field(default=None, description="来源仓。")
    to_warehouse: str | None = Field(default=None, description="目标仓或调入仓。")
    carrier: str | None = Field(default=None, description="物流渠道。")
    cost_delta: float = Field(default=0.0, description="相对默认履约的成本变化。")
    eta_hours: int = Field(default=0, ge=0, description="预计完成/送达小时数。")
    reason: str = Field(..., description="动作原因。")
    reversible: bool = Field(default=True, description="执行后是否可回滚。")
    depends_on: list[str] = Field(default_factory=list, description="Action DAG 中的前置动作 ID。")
    responsibility_domain: str | None = Field(default=None, description="责任域：WMS / TMS / ERP / CRM。")
    business_evidence: list[str] = Field(default_factory=list, description="该动作引用的关键业务依据。")


class ExecutionProposal(BaseModel):
    """Agent 给前端 Action Card 使用的结构化执行提案。"""

    proposal_id: str = Field(..., description="提案 ID，用于审批和幂等追踪。")
    order_id: str = Field(..., description="订单编号。")
    title: str = Field(..., description="操作卡片标题。")
    summary: str = Field(..., description="提案摘要。")
    status: str = Field(default="pending_approval", description="pending_approval / ready / invalidated / rejected")
    capabilities: list[str] = Field(default_factory=list, description="本提案覆盖的 Agent 动作能力。")
    actions: list[ProposalAction] = Field(default_factory=list, description="可执行动作列表。")
    decision_context: dict = Field(default_factory=dict, description="程序固定加载的订单履约决策上下文 JSON。")
    inventory_snapshot: list[dict] = Field(default_factory=list, description="审批时库存快照。")
    cost_breakdown: dict = Field(default_factory=dict, description="成本、价差和物流费用拆解。")
    eta: dict = Field(default_factory=dict, description="时效估算。")
    rule_citations: list[str] = Field(default_factory=list, description="引用规则/SOP。")
    data_fingerprint: str = Field(..., description="订单、库存、物流报价快照指纹。")
    freshness: dict = Field(default_factory=dict, description="订单/库存/物流数据新鲜度。")
    approval_required: bool = Field(default=True, description="是否必须人工审批。")
    preflight_checks: list[str] = Field(default_factory=list, description="执行前必须重新校验的项目。")
    invalidation_reason: str | None = Field(default=None, description="旧提案失效原因。")
    goal_type: str = Field(default="fulfillment_resolution", description="处理目标类型，用于匹配成功条件模板。")
    action_dag: dict = Field(default_factory=dict, description="Action DAG：nodes / edges / scheduling_hint。")
    success_criteria: dict = Field(default_factory=dict, description="执行后 VERIFYING 阶段必须满足的成功条件。")
    plan_version: int = Field(default=1, ge=1, description="计划版本，Replan 后递增。")
    context_version: str = Field(default="", description="生成计划时对应的业务上下文版本。")
    expires_at: str | None = Field(default=None, description="计划过期时间，过期后禁止直接执行。")


class PreflightValidation(BaseModel):
    """人工批准后的执行前二次校验结果。"""

    status: str = Field(..., description="valid / invalidated / rejected / skipped")
    checked_at: str = Field(..., description="校验时间。")
    checks: list[dict] = Field(default_factory=list, description="订单状态、库存和物流报价校验明细。")
    old_fingerprint: str | None = Field(default=None, description="审批前提案快照指纹。")
    new_fingerprint: str | None = Field(default=None, description="审批后实时数据快照指纹。")
    message: str = Field(..., description="校验结论。")
    replacement_proposal: ExecutionProposal | None = Field(
        default=None,
        description="旧提案失效时重新生成的新提案。",
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
    execution_proposal: ExecutionProposal | None = Field(
        default=None,
        description="如果本次生成了可审批执行提案，前端可用它渲染 Action Card。",
    )
    preflight_validation: PreflightValidation | None = Field(
        default=None,
        description="人工审批后的执行前二次校验结果。",
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
    timeout_seconds: int = Field(default=1800, description="审批超时秒数，到期后进入人工升级或告警")


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
    session_memory: dict = Field(
        default_factory=dict,
        description="本轮使用的短期结构化会话记忆快照。",
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
    execution_proposal: ExecutionProposal | None = Field(
        default=None,
        description="可审批的履约执行提案。",
    )
    preflight_validation: PreflightValidation | None = Field(
        default=None,
        description="执行前二次校验结果。",
    )
    trace: list[TraceEvent] = Field(
        default_factory=list,
        description="按执行顺序记录的所有节点轨迹。",
    )
    errors: list[ErrorEvent] = Field(
        default_factory=list,
        description="节点执行过程中产生的错误事件列表。",
    )
