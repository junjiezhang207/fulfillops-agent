"""PostgreSQL-backed short-term LangGraph checkpointer.

Short-term memory keeps the current conversation and graph execution state by
``thread_id``. PostgreSQL is the single shared state service for this project,
so checkpoints, channel blobs and pending writes are persisted in normal tables
with an ``expires_at`` column for TTL-style cleanup.
"""

from __future__ import annotations

import logging
import pickle
import random
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def _pack(value: Any) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _unpack(payload: bytes | memoryview) -> Any:
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    return pickle.loads(payload)


class PostgreSQLCheckpointSaver(BaseCheckpointSaver[str]):
    """LangGraph checkpointer stored in PostgreSQL tables."""

    def __init__(
        self,
        database_url: str,
        *,
        ttl_seconds: int = 86400,
        table_prefix: str = "lg",
    ) -> None:
        if not database_url:
            raise RuntimeError("短期记忆必须配置 DATABASE_URL/POSTGRES_URL，生产模式不允许降级到 MemorySaver。")
        super().__init__()
        self.database_url = database_url
        self.ttl_seconds = int(ttl_seconds)
        self.table_prefix = table_prefix.strip("_") or "lg"
        self._engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=1800, future=True)

    def setup(self) -> None:
        checkpoints = f"{self.table_prefix}_checkpoints"
        blobs = f"{self.table_prefix}_checkpoint_blobs"
        writes = f"{self.table_prefix}_checkpoint_writes"
        with self._engine.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {checkpoints} (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    checkpoint BYTEA NOT NULL,
                    metadata BYTEA NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
            """))
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{checkpoints}_expires
                ON {checkpoints} (expires_at)
            """))
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {blobs} (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    version TEXT NOT NULL,
                    typed_payload BYTEA NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
                )
            """))
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{blobs}_expires
                ON {blobs} (expires_at)
            """))
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {writes} (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    write_index INT NOT NULL,
                    channel TEXT NOT NULL,
                    typed_payload BYTEA NOT NULL,
                    task_path TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, write_index)
                )
            """))
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{writes}_expires
                ON {writes} (expires_at)
            """))

    @property
    def _checkpoints_table(self) -> str:
        return f"{self.table_prefix}_checkpoints"

    @property
    def _blobs_table(self) -> str:
        return f"{self.table_prefix}_checkpoint_blobs"

    @property
    def _writes_table(self) -> str:
        return f"{self.table_prefix}_checkpoint_writes"

    def _expires_at(self) -> datetime | None:
        if self.ttl_seconds <= 0:
            return None
        return _now() + timedelta(seconds=self.ttl_seconds)

    def _delete_expired(self, conn) -> None:
        now = _now()
        for table in (self._writes_table, self._blobs_table, self._checkpoints_table):
            conn.execute(
                text(f"DELETE FROM {table} WHERE expires_at IS NOT NULL AND expires_at <= :now"),
                {"now": now},
            )

    def _load_blobs(self, conn, thread_id: str, checkpoint_ns: str, versions: ChannelVersions) -> dict[str, Any]:
        channel_values: dict[str, Any] = {}
        for channel, version in versions.items():
            row = conn.execute(
                text(f"""
                    SELECT typed_payload
                    FROM {self._blobs_table}
                    WHERE thread_id=:thread_id
                      AND checkpoint_ns=:checkpoint_ns
                      AND channel=:channel
                      AND version=:version
                      AND (expires_at IS NULL OR expires_at > :now)
                """),
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "channel": channel,
                    "version": str(version),
                    "now": _now(),
                },
            ).mappings().first()
            if row is None:
                continue
            typed = _unpack(row["typed_payload"])
            if typed[0] != "empty":
                channel_values[channel] = self.serde.loads_typed(typed)
        return channel_values

    def _load_writes(self, conn, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> list[tuple[str, str, Any]]:
        rows = conn.execute(
            text(f"""
                SELECT task_id, channel, typed_payload, write_index
                FROM {self._writes_table}
                WHERE thread_id=:thread_id
                  AND checkpoint_ns=:checkpoint_ns
                  AND checkpoint_id=:checkpoint_id
                  AND (expires_at IS NULL OR expires_at > :now)
                ORDER BY task_id ASC, write_index ASC
            """),
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "now": _now(),
            },
        ).mappings().all()
        return [
            (row["task_id"], row["channel"], self.serde.loads_typed(_unpack(row["typed_payload"])))
            for row in rows
        ]

    def _tuple_from_row(self, conn, row, config: RunnableConfig | None = None) -> CheckpointTuple:
        checkpoint: Checkpoint = self.serde.loads_typed(_unpack(row["checkpoint"]))
        metadata = self.serde.loads_typed(_unpack(row["metadata"]))
        thread_id = row["thread_id"]
        checkpoint_ns = row["checkpoint_ns"]
        checkpoint_id = row["checkpoint_id"]
        checkpoint_config = config or {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
        return CheckpointTuple(
            config=checkpoint_config,
            checkpoint={
                **checkpoint,
                "channel_values": self._load_blobs(conn, thread_id, checkpoint_ns, checkpoint["channel_versions"]),
            },
            metadata=metadata,
            pending_writes=self._load_writes(conn, thread_id, checkpoint_ns, checkpoint_id),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row["parent_checkpoint_id"],
                    }
                }
                if row["parent_checkpoint_id"]
                else None
            ),
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        params = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "now": _now()}
        if checkpoint_id is None:
            where = "thread_id=:thread_id AND checkpoint_ns=:checkpoint_ns"
        else:
            where = "thread_id=:thread_id AND checkpoint_ns=:checkpoint_ns AND checkpoint_id=:checkpoint_id"
            params["checkpoint_id"] = checkpoint_id

        with self._engine.begin() as conn:
            self._delete_expired(conn)
            row = conn.execute(
                text(f"""
                    SELECT *
                    FROM {self._checkpoints_table}
                    WHERE {where}
                      AND (expires_at IS NULL OR expires_at > :now)
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                """),
                params,
            ).mappings().first()
            if row is None:
                return None
            return self._tuple_from_row(conn, row, config=config if checkpoint_id else None)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        clauses = ["(expires_at IS NULL OR expires_at > :now)"]
        params: dict[str, Any] = {"now": _now()}
        if config:
            clauses.append("thread_id=:thread_id")
            clauses.append("checkpoint_ns=:checkpoint_ns")
            params["thread_id"] = config["configurable"]["thread_id"]
            params["checkpoint_ns"] = config["configurable"].get("checkpoint_ns", "")
            config_checkpoint_id = get_checkpoint_id(config)
            if config_checkpoint_id:
                clauses.append("checkpoint_id=:checkpoint_id")
                params["checkpoint_id"] = config_checkpoint_id
        before_checkpoint_id = get_checkpoint_id(before) if before else None
        if before_checkpoint_id:
            clauses.append("checkpoint_id < :before_checkpoint_id")
            params["before_checkpoint_id"] = before_checkpoint_id
        sql_limit = "" if limit is None else " LIMIT :limit"
        if limit is not None:
            params["limit"] = int(limit)

        with self._engine.begin() as conn:
            self._delete_expired(conn)
            rows = conn.execute(
                text(f"""
                    SELECT *
                    FROM {self._checkpoints_table}
                    WHERE {' AND '.join(clauses)}
                    ORDER BY checkpoint_id DESC
                    {sql_limit}
                """),
                params,
            ).mappings().all()
            for row in rows:
                item = self._tuple_from_row(conn, row)
                if filter and not all(value == item.metadata.get(key) for key, value in filter.items()):
                    continue
                yield item

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        c = checkpoint.copy()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]
        expires_at = self._expires_at()

        with self._engine.begin() as conn:
            self._delete_expired(conn)
            for channel, version in new_versions.items():
                typed = self.serde.dumps_typed(values[channel]) if channel in values else ("empty", b"")
                conn.execute(
                    text(f"""
                        INSERT INTO {self._blobs_table} (
                            thread_id, checkpoint_ns, channel, version, typed_payload, expires_at
                        ) VALUES (
                            :thread_id, :checkpoint_ns, :channel, :version, :typed_payload, :expires_at
                        )
                        ON CONFLICT (thread_id, checkpoint_ns, channel, version) DO UPDATE SET
                            typed_payload=EXCLUDED.typed_payload,
                            expires_at=EXCLUDED.expires_at
                    """),
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "channel": channel,
                        "version": str(version),
                        "typed_payload": _pack(typed),
                        "expires_at": expires_at,
                    },
                )
            conn.execute(
                text(f"""
                    INSERT INTO {self._checkpoints_table} (
                        thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                        checkpoint, metadata, expires_at
                    ) VALUES (
                        :thread_id, :checkpoint_ns, :checkpoint_id, :parent_checkpoint_id,
                        :checkpoint, :metadata, :expires_at
                    )
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) DO UPDATE SET
                        parent_checkpoint_id=EXCLUDED.parent_checkpoint_id,
                        checkpoint=EXCLUDED.checkpoint,
                        metadata=EXCLUDED.metadata,
                        expires_at=EXCLUDED.expires_at
                """),
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                    "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
                    "checkpoint": _pack(self.serde.dumps_typed(c)),
                    "metadata": _pack(self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))),
                    "expires_at": expires_at,
                },
            )
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        expires_at = self._expires_at()
        with self._engine.begin() as conn:
            self._delete_expired(conn)
            for idx, (channel, value) in enumerate(writes):
                write_index = WRITES_IDX_MAP.get(channel, idx)
                conn.execute(
                    text(f"""
                        INSERT INTO {self._writes_table} (
                            thread_id, checkpoint_ns, checkpoint_id, task_id, write_index,
                            channel, typed_payload, task_path, expires_at
                        ) VALUES (
                            :thread_id, :checkpoint_ns, :checkpoint_id, :task_id, :write_index,
                            :channel, :typed_payload, :task_path, :expires_at
                        )
                        ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id, task_id, write_index) DO NOTHING
                    """),
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "write_index": write_index,
                        "channel": channel,
                        "typed_payload": _pack(self.serde.dumps_typed(value)),
                        "task_path": task_path,
                        "expires_at": expires_at,
                    },
                )

    def delete_thread(self, thread_id: str) -> None:
        with self._engine.begin() as conn:
            for table in (self._writes_table, self._blobs_table, self._checkpoints_table):
                conn.execute(text(f"DELETE FROM {table} WHERE thread_id=:thread_id"), {"thread_id": thread_id})

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ):
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        return f"{current_v + 1:032}.{random.random():016}"


def create_postgres_checkpointer(database_url: str | None = None, ttl_seconds: int = 86400) -> BaseCheckpointSaver:
    """Create the PostgreSQL short-term memory checkpointer."""
    from app.core.config import get_settings

    settings = get_settings()
    url = database_url or settings.effective_database_url
    saver = PostgreSQLCheckpointSaver(url, ttl_seconds=ttl_seconds)
    saver.setup()
    logger.info("短期记忆：已连接 PostgreSQL Checkpointer。")
    return saver

