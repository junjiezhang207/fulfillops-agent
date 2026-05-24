"""记忆系统 — 短期（Redis）+ 长期（SQLite / MySQL + Milvus）。

使用方式：
    from app.memory import create_redis_checkpointer, create_long_term_memory_store

    # 短期记忆：替换 MemorySaver
    checkpointer = create_redis_checkpointer(redis_url="redis://localhost:6379")

    # 长期记忆：传给 graph.compile(store=...)
    store = create_long_term_memory_store()

    # 接入 Agent
    agent = build_agent(model, tools, checkpointer=checkpointer)
    graph = graph.compile(checkpointer=checkpointer, store=store)
"""

from app.memory.long_term import (
    LongTermMemoryStore,
    MemoryPolicy,
    MySQLMilvusLongTermMemoryStore,
    SQLiteLongTermMemoryStore,
    create_long_term_memory_store,
)
from app.memory.governance import GovernanceDecision, MemoryGovernanceService
from app.memory.short_term import create_redis_checkpointer, get_session_history_from_redis

__all__ = [
    # 短期记忆
    "create_redis_checkpointer",
    "get_session_history_from_redis",
    # 长期记忆
    "create_long_term_memory_store",
    "LongTermMemoryStore",
    "SQLiteLongTermMemoryStore",
    "MySQLMilvusLongTermMemoryStore",
    "MemoryPolicy",
    "GovernanceDecision",
    "MemoryGovernanceService",
]
