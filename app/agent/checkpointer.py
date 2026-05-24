"""文件作用摘要：创建 Agent 会话短期记忆用的 LangGraph checkpointer。

这个文件很小，但它是 Agent 多轮对话能“记住上下文”的入口。LangGraph
通过 checkpointer 按 ``thread_id`` 保存消息历史；同一个 session_id 会映射到
同一个 thread_id，因此用户追问“那这个订单呢”时，Agent 能接上前文。

主要做的事：
1. 暴露 ``create_checkpointer`` 这个统一入口，供 ``AgentService`` 调用。
2. 真实创建逻辑委托给 ``app.memory.short_term.create_redis_checkpointer``。
3. Redis 可用时使用 Redis checkpointer，适合多实例和服务重启恢复。
4. Redis 不可用时由底层逻辑降级到 MemorySaver，保证本地演示不崩。

这个文件不负责长期记忆。长期记忆在 ``app/memory/long_term.py``，保存的是
跨会话可复用的偏好、摘要和订单处理结论；这里保存的是当前会话消息历史。

学习时只需要看 ``create_checkpointer``，理解它是短期记忆的兼容入口。
"""

from __future__ import annotations

from app.memory.short_term import create_redis_checkpointer


# 面试官可能问：短期记忆为什么用 checkpointer，而不是自己维护 messages 列表？
# 回答：LangGraph 原生通过 checkpointer 按 thread_id 保存消息历史，能和 Agent 图
# 的执行状态天然集成。自己维护列表容易漏掉工具消息、AIMessage、恢复逻辑和多实例问题。
def create_checkpointer(redis_url: str = "", ttl_seconds: int = 86400):
    """创建 LangGraph checkpointer，作为会话短期记忆。"""
    return create_redis_checkpointer(redis_url=redis_url or None, ttl_seconds=ttl_seconds)
