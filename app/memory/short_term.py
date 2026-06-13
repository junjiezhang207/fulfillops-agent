"""短期记忆 — Redis Checkpointer。

短期记忆 = 会话内的对话历史（多轮对话上下文）。

技术选型：
  RedisSaver（Redis 持久化，重启仍在，多实例共享）

接入方式：
  graph.compile(checkpointer=create_redis_checkpointer())

生产策略：
  Redis 不可用时直接抛异常。短期记忆、工具缓存和限流都属于跨实例一致性能力，
  不能在生产里静默降级为进程内存，否则多实例下会出现状态不一致。
"""

import logging
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)

# 默认 Redis URL（与 app/core/config.py 的 redis_url 保持一致）
_DEFAULT_REDIS_URL = "redis://localhost:6379"


def create_redis_checkpointer(
    redis_url: Optional[str] = None,
    ttl_seconds: int = 86400,  # 24 小时，超时会话自动清理
) -> BaseCheckpointSaver:
    """创建 Redis 短期记忆 Checkpointer。

    Args:
        redis_url:   Redis 连接地址，默认 redis://localhost:6379
        ttl_seconds: 会话 TTL（秒），超时自动清理，默认 24 小时

    Returns:
        RedisSaver。Redis 不可用时抛出异常。

    接入示例：
        checkpointer = create_redis_checkpointer("redis://localhost:6379")
        agent = build_agent(model, tools, checkpointer=checkpointer)

    """
    if not redis_url:
        raise RuntimeError("短期记忆必须配置 REDIS_URL，生产模式不允许降级到 MemorySaver。")
    url = redis_url or _DEFAULT_REDIS_URL

    try:
        from langgraph.checkpoint.redis import RedisSaver

        # 验证连接是否可用
        import redis as redis_client
        r = redis_client.from_url(url, socket_connect_timeout=2)
        r.ping()

        saver = RedisSaver(
            redis_url=url,
            connection_args={"socket_connect_timeout": 2},
            ttl={"default_ttl": ttl_seconds, "refresh_on_read": True},
        )
        saver.setup()
        logger.info("短期记忆：已连接 Redis Checkpointer (%s)", url)
        return saver

    except Exception as exc:
        raise RuntimeError(f"短期记忆 Redis 不可用，已拒绝降级：{exc}") from exc


def get_session_history_from_redis(
    session_id: str,
    redis_url: Optional[str] = None,
):
    """从 Redis 获取 LangChain 格式的对话历史（供非 LangGraph 场景使用）。

    使用场景：
      LangChain Chain（非 LangGraph）需要会话历史时，
      用 RedisChatMessageHistory 代替 InMemoryChatMessageHistory。

    Args:
        session_id: 会话 ID
        redis_url:  Redis 连接地址

    Returns:
        RedisChatMessageHistory。Redis 不可用时抛出异常。

    接入示例：
        from langchain_core.runnables.history import RunnableWithMessageHistory
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_session_history=get_session_history_from_redis,
        )

    """
    if not redis_url:
        raise RuntimeError("会话历史必须配置 REDIS_URL，生产模式不允许降级到内存历史。")
    url = redis_url or _DEFAULT_REDIS_URL

    try:
        from langchain_community.chat_message_histories import RedisChatMessageHistory
        return RedisChatMessageHistory(
            session_id=session_id,
            url=url,
            ttl=86400,  # 24 小时
            key_prefix="multiship:chat:",
        )
    except Exception as exc:
        raise RuntimeError(f"Redis 会话历史不可用，已拒绝降级：{exc}") from exc
