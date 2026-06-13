"""Agent API and structured-output schemas.

Learning notes:
- Defines the frontend/backend data contract for Agent chat.
- Tool traces and structured fulfillment decisions are kept here for UI display and audit.
"""

from pydantic import BaseModel, Field


class ToolCallDetail(BaseModel):
    """单个工具调用的详细记录。"""
    tool_name: str = Field(..., description="工具名称")
    input_args: dict = Field(..., description="工具输入参数")
    output: str = Field(..., description="工具输出结果")
    order: int = Field(..., description="本轮调用顺序（从1开始）")


class ReActCycle(BaseModel):
    """ReAct 单个推理-行动-观察循环的轨迹记录。

    大厂 Agent 系统的核心可解释性指标：
      - thought:      LLM 在调用工具前的推理文本（解释为什么要调用这个工具）
      - action:       调用的工具名 + 参数
      - observation:  工具返回结果的摘要（前 300 字符）
      - cycle_index:  本 cycle 在当前 turn 中的序号（从 0 开始）

    一次 Agent turn 可能包含多个 ReActCycle（多轮工具调用）。
    """
    cycle_index: int = Field(..., description="本 cycle 在当前 turn 的序号")
    thought: str = Field(default="", description="工具调用前的推理文本")
    action: str = Field(default="", description="调用的工具名")
    action_input: dict = Field(default_factory=dict, description="工具调用参数")
    observation: str = Field(default="", description="工具返回结果摘要（前 300 字）")


class AgentExecutionTrace(BaseModel):
    """Agent 单轮执行的完整轨迹。"""
    session_id: str = Field(..., description="会话 ID")
    user_message: str = Field(..., description="用户输入消息")
    tools_called: list[ToolCallDetail] = Field(default_factory=list)
    react_cycles: list[ReActCycle] = Field(
        default_factory=list,
        description="完整的 ReAct 推理-行动-观察循环序列",
    )
    model_reasoning: str = Field(default="")
    final_reply: str = Field(..., description="Agent 最终回复")
    execution_steps: int = Field(..., description="执行步骤数")


# ============================================================================
# 结构化输出模型（Issue 2）
# 用 llm.with_structured_output(FulfillmentDecision) 替代自由文本输出
# ============================================================================

class FulfillmentDecision(BaseModel):
    """Agent 对履约问题的结构化决策结果。

    为什么用结构化输出：
      自由文本下游系统无法可靠解析（缺了哪些 SKU、推荐哪个仓库）。
      用 with_structured_output() 让 LLM 直接输出 Pydantic 对象，
      上游系统可以强类型访问所有字段，无需二次解析。
    """
    can_fulfill: bool = Field(description="是否可以完整履约当前订单")
    missing_skus: list[str] = Field(default_factory=list, description="缺货的 SKU 列表，若无则为空")
    recommended_action: str = Field(description="推荐的下一步处理动作（一句话）")
    key_risks: list[str] = Field(default_factory=list, description="关键风险点列表")
    confidence: float = Field(ge=0.0, le=1.0, description="决策置信度，0.0 到 1.0")
    reasoning: str = Field(description="决策依据和推理过程摘要")


class ReflectionInfo(BaseModel):
    """自反思评估结果（集成进 AgentChatResponse 后对外可见）。"""
    score: float = Field(description="质量综合分 0-1")
    passed: bool = Field(description="是否通过质量门")
    retry_count: int = Field(description="实际重试次数")
    reason: str = Field(default="", description="评估理由")


class AgentChatRequest(BaseModel):
    """Agent 对话请求。"""
    session_id: str = Field(..., description="会话 ID，相同 ID 共享对话历史")
    message: str = Field(..., description="用户本轮消息")
    model_id: str | None = Field(default=None, description="可选模型 ID，由后端模型网关解析")
    tenant_id: str = Field(default="default", description="租户 ID，用于工具权限和缓存隔离")
    user_id: str | None = Field(default=None, description="用户 ID，用于工具审计和权限上下文")
    roles: list[str] | None = Field(default=None, description="用户角色列表")
    permissions: list[str] | None = Field(default=None, description="用户显式工具权限列表")


class PlanExecuteRequest(BaseModel):
    """Plan-and-Execute Agent 请求。"""
    order_id: str = Field(..., description="订单 ID，如 SO202502140001")
    question: str = Field(..., description="复杂问题，如'帮我完整分析履约方案'")
    session_id: str | None = Field(default=None, description="会话 ID（可选，留空自动生成）")
    model_id: str | None = Field(default=None, description="可选模型 ID，由后端模型网关解析")
    tenant_id: str = Field(default="default", description="租户 ID，用于工具权限和缓存隔离")
    user_id: str | None = Field(default=None, description="用户 ID，用于工具审计和权限上下文")
    roles: list[str] | None = Field(default=None, description="用户角色列表")
    permissions: list[str] | None = Field(default=None, description="用户显式工具权限列表")


class PlanExecuteResponse(BaseModel):
    """Plan-and-Execute Agent 响应。"""
    final_answer: str = Field(..., description="综合最终答复")
    plan_steps: list[str] = Field(default_factory=list, description="实际执行的步骤列表")
    step_results: list[dict] = Field(default_factory=list, description="每步 {step, result}")
    total_steps: int = Field(..., description="执行步骤总数")


class AgentChatResponse(BaseModel):
    """Agent 对话响应（增强版，含结构化输出和反思信息）。"""
    session_id: str
    reply: str = Field(..., description="Agent 的自然语言最终回复")
    tools_called: list[str] = Field(default_factory=list)
    trace: AgentExecutionTrace | None = Field(default=None)
    # 新增：结构化决策（Issue 2）
    decision: FulfillmentDecision | None = Field(
        default=None, description="从回复中提取的结构化履约决策，LLM 不可用时为 None"
    )
    # 新增：反思信息（Issue 3）
    reflection: ReflectionInfo | None = Field(
        default=None, description="自反思评估结果，未启用反思时为 None"
    )
