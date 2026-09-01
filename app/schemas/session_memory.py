"""短期结构化会话记忆 schema。

Session Memory 只保存运营在对话中明确表达的偏好、约束、反馈和指代关系。
实时订单状态、库存、物流 ETA、仓库状态等业务事实必须每轮从源系统重新读取。
"""

from pydantic import BaseModel, Field


class RecentMessage(BaseModel):
    role: str = Field(..., description="user / assistant")
    content: str = Field(..., description="短句摘要，不保存长回答。")
    created_at: str = Field(..., description="ISO 时间。")


class StructuredSessionMemory(BaseModel):
    current_topic: str = Field(default="", description="当前讨论主题。")
    user_preferences: dict[str, object] = Field(default_factory=dict, description="用户偏好。")
    confirmed_constraints: dict[str, object] = Field(default_factory=dict, description="用户明确确认的约束。")
    plan_feedback: dict[str, object] = Field(default_factory=dict, description="对历史方案的接受/拒绝/修改反馈。")
    references: dict[str, object] = Field(default_factory=dict, description="那个仓、第二个方案等指代关系。")


class SessionMemoryPatch(BaseModel):
    current_topic: str | None = None
    user_preferences: dict[str, object] = Field(default_factory=dict)
    confirmed_constraints: dict[str, object] = Field(default_factory=dict)
    plan_feedback: dict[str, object] = Field(default_factory=dict)
    references: dict[str, object] = Field(default_factory=dict)
    extracted_signals: list[str] = Field(default_factory=list)


class SessionMemorySnapshot(BaseModel):
    thread_id: str
    order_id: str = ""
    recent_messages: list[RecentMessage] = Field(default_factory=list)
    structured: StructuredSessionMemory = Field(default_factory=StructuredSessionMemory)
    memory_use_case: str = "memory_extraction"
    token_budget: int = 800
    updated_at: str | None = None
