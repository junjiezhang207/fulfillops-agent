"""Agent 会话短期记忆 checkpointer。

本模块是 Agent 多轮对话保存上下文的入口。LangGraph
通过 checkpointer 按 ``thread_id`` 保存消息历史；同一个 session_id 会映射到
同一个 thread_id，因此用户追问“那这个订单呢”时，Agent 能接上前文。

主要做的事：
1. 暴露 ``create_checkpointer`` 这个统一入口，供 ``AgentService`` 调用。
2. 真实创建逻辑委托给 ``app.memory.short_term.create_redis_checkpointer``。
3. 固定使用 Redis checkpointer，适合多实例和服务重启恢复。
4. Redis 不可用时直接启动失败，避免短期记忆在多实例下分叉。

这个文件不负责长期记忆。长期记忆在 ``app/memory/long_term.py``，保存的是
跨会话可复用的偏好、摘要和订单处理结论；这里保存的是当前会话消息历史。
"""

from __future__ import annotations

from app.memory.short_term import create_redis_checkpointer


# 使用 LangGraph checkpointer 保存短期记忆，避免手动维护消息列表时遗漏工具消息、
# AIMessage、恢复逻辑或多实例一致性问题。
def create_checkpointer(redis_url: str = "", ttl_seconds: int = 86400):
    """创建 LangGraph checkpointer，作为会话短期记忆。"""
    return create_redis_checkpointer(redis_url=redis_url or None, ttl_seconds=ttl_seconds)
