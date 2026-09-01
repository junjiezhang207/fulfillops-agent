"""短期记忆 PostgreSQL 兼容入口。

项目已统一使用 PostgreSQL 保存 LangGraph checkpoint。这个模块保留短期记忆
入口文件名，实际实现转发到 ``app.memory.postgres_short_term``。
"""

from __future__ import annotations

from .postgres_short_term import PostgreSQLCheckpointSaver, create_postgres_checkpointer

ShortTermCheckpointSaver = PostgreSQLCheckpointSaver
create_short_term_checkpointer = create_postgres_checkpointer

__all__ = [
    "PostgreSQLCheckpointSaver",
    "ShortTermCheckpointSaver",
    "create_postgres_checkpointer",
    "create_short_term_checkpointer",
]
