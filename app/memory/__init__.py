"""记忆系统 — 短期（PostgreSQL）+ 长期（PostgreSQL + PGVector）。

使用方式：
    from app.memory import create_postgres_checkpointer, create_long_term_memory_store

    # 短期记忆：替换 MemorySaver
    checkpointer = create_postgres_checkpointer(database_url="postgresql+psycopg://...")

    # 长期记忆：传给 graph.compile(store=...)
    store = create_long_term_memory_store(database_url="postgresql+psycopg://...")

    # 接入 Agent
    agent = build_agent(model, tools, checkpointer=checkpointer)
    graph = graph.compile(checkpointer=checkpointer, store=store)
"""

from app.memory.long_term import (
    LongTermMemoryStore,
    MemoryPolicy,
    PostgreSQLPGVectorLongTermMemoryStore,
    create_long_term_memory_store,
)
from app.memory.governance import GovernanceDecision, MemoryGovernanceService
from app.memory.postgres_short_term import PostgreSQLCheckpointSaver, create_postgres_checkpointer

__all__ = [
    # 短期记忆
    "PostgreSQLCheckpointSaver",
    "create_postgres_checkpointer",
    # 长期记忆
    "create_long_term_memory_store",
    "LongTermMemoryStore",
    "PostgreSQLPGVectorLongTermMemoryStore",
    "MemoryPolicy",
    "GovernanceDecision",
    "MemoryGovernanceService",
]
