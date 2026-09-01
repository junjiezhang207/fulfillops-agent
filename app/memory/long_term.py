"""长期记忆 Store。

长期记忆保存跨会话仍然有价值的信息，例如：
  - 客户偏好：是否接受替代品、偏好的仓库/物流。
  - 历史决策：某个订单最终为什么走了缺货流程。
  - 会话摘要：用户前几次问过什么、系统给过什么承诺。

生产架构使用同一套 PostgreSQL：
  - 普通表保存精确字段、TTL、审计信息。
  - PGVector 表保存向量索引，支持相似案例检索。

这个项目现在固定使用 PostgreSQL + PGVector：
  - PostgreSQL 是权威数据源，保存结构化 value、TTL、审计信息。
  - PGVector 是可重建的语义索引。
  - 向量库或 embedding 不可用时直接失败，不允许退回文本检索。
"""

from __future__ import annotations

import hashlib
import logging
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchItem,
    SearchOp,
)


logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _from_iso(value: str | None) -> datetime:
    if not value:
        return _now()
    return datetime.fromisoformat(value)


def _from_db_datetime(value: datetime | str | None) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    return _from_iso(value)


def _namespace_key(namespace: tuple[str, ...]) -> str:
    return json.dumps(list(namespace), ensure_ascii=False)


def _parse_namespace(raw: str) -> tuple[str, ...]:
    return tuple(json.loads(raw))


def _namespace_path(namespace: tuple[str, ...]) -> str:
    return "/".join(quote(part, safe="") for part in namespace)


def _namespace_hash(namespace: tuple[str, ...]) -> str:
    return hashlib.sha256(_namespace_key(namespace).encode("utf-8")).hexdigest()


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _vector_id(namespace: tuple[str, ...], key: str) -> str:
    digest = hashlib.sha256(f"{_namespace_key(namespace)}:{key}".encode("utf-8")).hexdigest()
    return f"mem_{digest}"


def _utc_naive(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _epoch_millis(value: datetime | None) -> int:
    if value is None:
        return 0
    return int(value.astimezone(timezone.utc).timestamp() * 1000)


def _safe_identifier(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"非法 SQL 标识符：{name!r}")
    return name


def _text_for_search(value: dict[str, Any]) -> str:
    """把结构化记忆压成可检索文本。

    这里不做复杂 embedding，是为了本地演示稳定可跑；生产替换成向量库时，
    这个函数对应“生成 embedding 前的文档文本”。
    """
    parts: list[str] = []
    for key, item in value.items():
        if key.startswith("_"):
            continue
        if isinstance(item, (str, int, float, bool)):
            parts.append(f"{key}:{item}")
        elif isinstance(item, list):
            parts.append(f"{key}:{' '.join(map(str, item))}")
        elif isinstance(item, dict):
            parts.append(f"{key}:{json.dumps(item, ensure_ascii=False)}")
    return "\n".join(parts)


def _tokens(text: str) -> set[str]:
    words = {w.lower() for w in re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]", text)}
    # 中文单字检索召回偏宽，所以额外加入连续 bigram，让相似度更稳一点。
    chinese = re.findall(r"[\u4e00-\u9fff]+", text)
    for chunk in chinese:
        words.update(chunk[i : i + 2] for i in range(max(len(chunk) - 1, 0)))
    return {w for w in words if w}


def _matches_filter(value: dict[str, Any], filter_: dict[str, Any] | None) -> bool:
    if not filter_:
        return True
    for key, expected in filter_.items():
        actual = value.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _decode_json_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return dict(value or {})


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(item):.8f}" for item in vector) + "]"


def _embed_text(embed_model: object | None, text: str) -> list[float] | None:
    if embed_model is None or not text:
        return None
    try:
        if hasattr(embed_model, "get_text_embedding"):
            return [float(item) for item in embed_model.get_text_embedding(text)]
        if hasattr(embed_model, "embed_query"):
            return [float(item) for item in embed_model.embed_query(text)]
        if callable(embed_model):
            return [float(item) for item in embed_model(text)]
    except Exception as exc:
        raise RuntimeError(f"长期记忆 embedding 生成失败，生产模式拒绝降级：{exc}") from exc
    raise RuntimeError("长期记忆未配置可用 embedding 模型，生产模式拒绝降级为文本检索。")


@dataclass
class MemoryPolicy:
    """长期记忆写入策略。

    default_ttl_days=None 表示默认不过期。
    importance 影响 search 排序：相似度相同的情况下，更重要的记忆排在前面。
    """

    default_ttl_days: int | None = 180
    default_importance: float = 0.5


class PostgreSQLPGVectorLongTermMemoryStore(BaseStore):
    """PostgreSQL + PGVector 长期记忆实现。

    企业级拆分原则：
    - PostgreSQL 是权威数据源，保存结构化 value、namespace、TTL、importance、访问热度和审计。
    - PGVector 是可重建的语义索引，只保存向量和少量检索元数据。
    - 写入先落 PostgreSQL 记忆表，再同步 PGVector；PGVector 不可用时直接失败。
    """

    def __init__(
        self,
        database_url: str,
        *,
        pgvector_table: str = "long_term_memory_vectors",
        embedding_model: object | None = None,
        vector_dimension: int = 1024,
        policy: MemoryPolicy | None = None,
        table_name: str = "long_term_memory",
        audit_table_name: str = "long_term_memory_audit",
    ) -> None:
        if not database_url:
            raise ValueError("使用 PostgreSQL + PGVector 长期记忆时必须配置 DATABASE_URL 或 LONG_TERM_MEMORY_DATABASE_URL")
        self.database_url = database_url
        self.embedding_model = embedding_model
        self.vector_dimension = int(vector_dimension)
        self.policy = policy or MemoryPolicy()
        self.table_name = _safe_identifier(table_name)
        self.audit_table_name = _safe_identifier(audit_table_name)
        self.pgvector_table = _safe_identifier(pgvector_table)

        self._engine = self._create_engine()
        self._init_schema()
        self._init_pgvector()

    def _create_engine(self):
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise RuntimeError("PostgreSQL 长期记忆需要安装 sqlalchemy 和 psycopg。") from exc
        return create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_recycle=1800,
            future=True,
        )

    def _init_schema(self) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    namespace_hash CHAR(64) NOT NULL,
                    key_hash CHAR(64) NOT NULL,
                    namespace TEXT NOT NULL,
                    namespace_path TEXT NOT NULL,
                    memory_key VARCHAR(512) NOT NULL,
                    value_json JSONB NOT NULL,
                    search_text TEXT NOT NULL,
                    importance DOUBLE PRECISION NOT NULL,
                    vector_id VARCHAR(96) NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NULL,
                    access_count INT NOT NULL DEFAULT 0,
                    version BIGINT NOT NULL DEFAULT 1,
                    deleted_at TIMESTAMP NULL,
                    CONSTRAINT uq_ltm_namespace_key UNIQUE (namespace_hash, key_hash)
                )
            """))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_ltm_namespace_path ON {self.table_name} (namespace_path)"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_ltm_vector_id ON {self.table_name} (vector_id)"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_ltm_expires ON {self.table_name} (expires_at)"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_ltm_updated ON {self.table_name} (updated_at)"))
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.audit_table_name} (
                    id BIGSERIAL PRIMARY KEY,
                    namespace_hash CHAR(64) NOT NULL,
                    key_hash CHAR(64) NOT NULL,
                    namespace TEXT NOT NULL,
                    memory_key VARCHAR(512) NOT NULL,
                    event_type VARCHAR(32) NOT NULL,
                    value_json JSONB NULL,
                    created_at TIMESTAMP NOT NULL
                )
            """))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_ltm_audit_memory ON {self.audit_table_name} (namespace_hash, key_hash)"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_ltm_audit_created ON {self.audit_table_name} (created_at)"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_ltm_search_text ON {self.table_name} USING GIN (to_tsvector('simple', search_text))"))

    def _init_pgvector(self) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.pgvector_table} (
                    id VARCHAR(96) PRIMARY KEY,
                    namespace_path TEXT NOT NULL,
                    memory_key VARCHAR(512) NOT NULL,
                    search_text TEXT NOT NULL,
                    embedding vector({self.vector_dimension}) NOT NULL,
                    importance DOUBLE PRECISION NOT NULL,
                    updated_at BIGINT NOT NULL,
                    expires_at BIGINT NOT NULL
                )
            """))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{self.pgvector_table}_namespace ON {self.pgvector_table} (namespace_path)"))
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pgvector_table}_embedding
                ON {self.pgvector_table} USING hnsw (embedding vector_cosine_ops)
            """))

    def _delete_expired(self, conn) -> None:
        from sqlalchemy import text

        now = _utc_naive(_now())
        conn.execute(
            text(f"""
                UPDATE {self.table_name}
                SET deleted_at=:now
                WHERE deleted_at IS NULL AND expires_at IS NOT NULL AND expires_at <= :now
            """),
            {"now": now},
        )

    def _embedding_for(self, text: str) -> list[float] | None:
        embedding = _embed_text(self.embedding_model, text)
        if embedding and len(embedding) != self.vector_dimension:
            raise RuntimeError(
                f"长期记忆 embedding 维度不匹配：实际 {len(embedding)}，配置 {self.vector_dimension}。"
            )
        return embedding

    def _write_audit(
        self,
        conn,
        namespace: tuple[str, ...],
        key: str,
        event_type: str,
        value: dict[str, Any] | None,
    ) -> None:
        from sqlalchemy import text

        conn.execute(
            text(f"""
                INSERT INTO {self.audit_table_name} (
                    namespace_hash, key_hash, namespace, memory_key, event_type, value_json, created_at
                )
                VALUES (
                    :namespace_hash, :key_hash, :namespace, :memory_key, :event_type, :value_json, :created_at
                )
            """),
            {
                "namespace_hash": _namespace_hash(namespace),
                "key_hash": _key_hash(key),
                "namespace": _namespace_key(namespace),
                "memory_key": key,
                "event_type": event_type,
                "value_json": json.dumps(value, ensure_ascii=False, sort_keys=True) if value is not None else None,
                "created_at": _utc_naive(_now()),
            },
        )

    def _upsert_vector(
        self,
        vector_id: str,
        namespace: tuple[str, ...],
        key: str,
        search_text: str,
        embedding: list[float] | None,
        importance: float,
        updated_at: datetime,
        expires_at: datetime | None,
    ) -> None:
        if embedding is None:
            raise RuntimeError("长期记忆缺少 embedding，拒绝写入不完整记忆。")
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {self.pgvector_table} (
                        id, namespace_path, memory_key, search_text, embedding,
                        importance, updated_at, expires_at
                    ) VALUES (
                        :id, :namespace_path, :memory_key, :search_text,
                        CAST(:embedding AS vector), :importance, :updated_at, :expires_at
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        namespace_path=EXCLUDED.namespace_path,
                        memory_key=EXCLUDED.memory_key,
                        search_text=EXCLUDED.search_text,
                        embedding=EXCLUDED.embedding,
                        importance=EXCLUDED.importance,
                        updated_at=EXCLUDED.updated_at,
                        expires_at=EXCLUDED.expires_at
                """),
                {
                    "id": vector_id,
                    "namespace_path": _namespace_path(namespace),
                    "memory_key": key[:512],
                    "search_text": search_text,
                    "embedding": _vector_literal(embedding),
                    "importance": float(importance),
                    "updated_at": _epoch_millis(updated_at),
                    "expires_at": _epoch_millis(expires_at),
                },
            )

    def _delete_vector(self, vector_id: str) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(text(f"DELETE FROM {self.pgvector_table} WHERE id=:id"), {"id": vector_id})

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index=None,
        *,
        ttl=None,
    ) -> None:
        from sqlalchemy import text

        now = _now()
        ttl_days = value.get("_ttl_days", self.policy.default_ttl_days)
        expires_at = None if ttl_days is None else now + timedelta(days=float(ttl_days))
        if ttl is not None and not isinstance(ttl, bool):
            expires_at = now + timedelta(seconds=float(ttl))

        importance = float(value.get("_importance", self.policy.default_importance))
        importance = max(0.0, min(1.0, importance))
        search_text = _text_for_search(value)
        embedding = self._embedding_for(search_text)
        vector_id = _vector_id(namespace, key)

        params = {
            "namespace_hash": _namespace_hash(namespace),
            "key_hash": _key_hash(key),
            "namespace": _namespace_key(namespace),
            "namespace_path": _namespace_path(namespace),
            "memory_key": key,
            "value_json": json.dumps(value, ensure_ascii=False, sort_keys=True),
            "search_text": search_text,
            "importance": importance,
            "vector_id": vector_id,
            "created_at": _utc_naive(now),
            "updated_at": _utc_naive(now),
            "expires_at": _utc_naive(expires_at),
        }
        with self._engine.begin() as conn:
            self._delete_expired(conn)
            conn.execute(
                text(f"""
                    INSERT INTO {self.table_name} (
                        namespace_hash, key_hash, namespace, namespace_path, memory_key,
                        value_json, search_text, importance, vector_id,
                        created_at, updated_at, expires_at, access_count, version, deleted_at
                    )
                    VALUES (
                        :namespace_hash, :key_hash, :namespace, :namespace_path, :memory_key,
                        :value_json, :search_text, :importance, :vector_id,
                        :created_at, :updated_at, :expires_at, 0, 1, NULL
                    )
                    ON CONFLICT (namespace_hash, key_hash) DO UPDATE SET
                        namespace=EXCLUDED.namespace,
                        namespace_path=EXCLUDED.namespace_path,
                        memory_key=EXCLUDED.memory_key,
                        value_json=EXCLUDED.value_json,
                        search_text=EXCLUDED.search_text,
                        importance=EXCLUDED.importance,
                        vector_id=EXCLUDED.vector_id,
                        updated_at=EXCLUDED.updated_at,
                        expires_at=EXCLUDED.expires_at,
                        version=version+1,
                        deleted_at=NULL
                """),
                params,
            )
            self._write_audit(conn, namespace, key, "put", value)
        self._upsert_vector(vector_id, namespace, key, search_text, embedding, importance, now, expires_at)

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        from sqlalchemy import text

        params = {"namespace_hash": _namespace_hash(namespace), "key_hash": _key_hash(key), "now": _utc_naive(_now())}
        with self._engine.begin() as conn:
            self._delete_expired(conn)
            row = conn.execute(
                text(f"""
                    SELECT *
                    FROM {self.table_name}
                    WHERE namespace_hash=:namespace_hash
                      AND key_hash=:key_hash
                      AND deleted_at IS NULL
                      AND (expires_at IS NULL OR expires_at > :now)
                """),
                params,
            ).mappings().fetchone()
            if row is None:
                return None
            conn.execute(
                text(f"""
                    UPDATE {self.table_name}
                    SET access_count=access_count+1
                    WHERE namespace_hash=:namespace_hash AND key_hash=:key_hash
                """),
                params,
            )
        return self._row_to_item(row)

    def _row_to_item(self, row: dict[str, Any]) -> Item:
        namespace = _parse_namespace(row["namespace"])
        return Item(
            namespace=namespace,
            key=row["memory_key"],
            value=_decode_json_value(row["value_json"]),
            created_at=_from_db_datetime(row["created_at"]),
            updated_at=_from_db_datetime(row["updated_at"]),
        )

    def _rows_for_prefix(self, conn, namespace_prefix: tuple[str, ...]) -> list[dict[str, Any]]:
        from sqlalchemy import text

        now = _utc_naive(_now())
        if namespace_prefix:
            prefix = _namespace_path(namespace_prefix)
            rows = conn.execute(
                text(f"""
                    SELECT *
                    FROM {self.table_name}
                    WHERE deleted_at IS NULL
                      AND (expires_at IS NULL OR expires_at > :now)
                      AND (namespace_path = :prefix OR namespace_path LIKE :prefix_like)
                """),
                {"now": now, "prefix": prefix, "prefix_like": f"{prefix}/%"},
            ).mappings().all()
        else:
            rows = conn.execute(
                text(f"""
                    SELECT *
                    FROM {self.table_name}
                    WHERE deleted_at IS NULL
                      AND (expires_at IS NULL OR expires_at > :now)
                """),
                {"now": now},
            ).mappings().all()
        return [dict(row) for row in rows]

    def _row_by_vector_id(self, conn, vector_id: str) -> dict[str, Any] | None:
        from sqlalchemy import text

        row = conn.execute(
            text(f"""
                SELECT *
                FROM {self.table_name}
                WHERE vector_id=:vector_id
                  AND deleted_at IS NULL
                  AND (expires_at IS NULL OR expires_at > :now)
            """),
            {"vector_id": vector_id, "now": _utc_naive(_now())},
        ).mappings().fetchone()
        return dict(row) if row else None

    def _search_vector_ids(
        self,
        namespace_prefix: tuple[str, ...],
        query_vector: list[float] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        if query_vector is None:
            raise RuntimeError("长期记忆缺少 query embedding，无法执行语义检索。")
        now_ms = _epoch_millis(_now())
        clauses = ["(expires_at = 0 OR expires_at > :now_ms)"]
        params: dict[str, Any] = {
            "embedding": _vector_literal(query_vector),
            "limit": max(limit * 20, 50),
            "now_ms": now_ms,
        }
        if namespace_prefix:
            prefix = _namespace_path(namespace_prefix)
            clauses.append("(namespace_path = :prefix OR namespace_path LIKE :prefix_like)")
            params["prefix"] = prefix
            params["prefix_like"] = f"{prefix}/%"
        from sqlalchemy import text

        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT id, embedding <=> CAST(:embedding AS vector) AS distance
                    FROM {self.pgvector_table}
                    WHERE {' AND '.join(clauses)}
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :limit
                """),
                params,
            ).mappings().all()
        results: list[tuple[str, float]] = []
        for row in rows:
            results.append((str(row["id"]), self._pgvector_similarity(float(row["distance"]))))
        return results

    @staticmethod
    def _pgvector_similarity(distance: float) -> float:
        return max(0.0, min(1.0, 1.0 - distance))

    def _legacy_similarity(self, distance: float) -> float:
        return self._pgvector_similarity(distance)

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        query_vector = self._embedding_for(query or "")
        query_tokens = _tokens(query or "")
        vector_hits = self._search_vector_ids(namespace_prefix, query_vector, max(limit + offset, limit))

        with self._engine.begin() as conn:
            self._delete_expired(conn)
            if vector_hits:
                rows: list[dict[str, Any]] = []
                vector_scores: dict[str, float] = {}
                for vector_id, score in vector_hits:
                    row = self._row_by_vector_id(conn, vector_id)
                    if row is None:
                        continue
                    rows.append(row)
                    vector_scores[vector_id] = score
                if query_tokens:
                    seen_ids = {row["id"] for row in rows}
                    for row in self._rows_for_prefix(conn, namespace_prefix):
                        if row["id"] not in seen_ids:
                            rows.append(row)
                            seen_ids.add(row["id"])
            else:
                rows = self._rows_for_prefix(conn, namespace_prefix)
                vector_scores = {}

        scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for row in rows:
            value = _decode_json_value(row["value_json"])
            if not _matches_filter(value, filter):
                continue
            namespace = _parse_namespace(row["namespace"])
            if namespace[: len(namespace_prefix)] != namespace_prefix:
                continue

            if row["vector_id"] in vector_scores:
                relevance = vector_scores[row["vector_id"]]
            elif query_tokens:
                memory_tokens = _tokens(row["search_text"])
                relevance = len(query_tokens & memory_tokens) / max(len(query_tokens), 1)
                if relevance <= 0:
                    continue
            else:
                relevance = 0.0

            score = relevance * 0.75 + float(row["importance"]) * 0.20 + min(int(row["access_count"]), 10) * 0.005
            scored.append((score, row, value))

        scored.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
        selected = scored[offset : offset + limit]
        if selected:
            from sqlalchemy import text

            with self._engine.begin() as conn:
                for _, row, _ in selected:
                    conn.execute(
                        text(f"UPDATE {self.table_name} SET access_count=access_count+1 WHERE id=:id"),
                        {"id": row["id"]},
                    )

        return [
            SearchItem(
                namespace=_parse_namespace(row["namespace"]),
                key=row["memory_key"],
                value=value,
                created_at=_from_db_datetime(row["created_at"]),
                updated_at=_from_db_datetime(row["updated_at"]),
                score=round(score, 4),
            )
            for score, row, value in selected
        ]

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        from sqlalchemy import text

        vector_id = _vector_id(namespace, key)
        with self._engine.begin() as conn:
            conn.execute(
                text(f"""
                    UPDATE {self.table_name}
                    SET deleted_at=:now, updated_at=:now
                    WHERE namespace_hash=:namespace_hash AND key_hash=:key_hash
                """),
                {
                    "now": _utc_naive(_now()),
                    "namespace_hash": _namespace_hash(namespace),
                    "key_hash": _key_hash(key),
                },
            )
            self._write_audit(conn, namespace, key, "delete", None)
        self._delete_vector(vector_id)

    def list_namespaces(
        self,
        *,
        prefix=None,
        suffix=None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        with self._engine.begin() as conn:
            rows = self._rows_for_prefix(conn, tuple(prefix or ()))
        namespaces = [_parse_namespace(row["namespace"]) for row in rows]
        if suffix:
            suffix_tuple = tuple(suffix)
            namespaces = [ns for ns in namespaces if ns[-len(suffix_tuple) :] == suffix_tuple]
        if max_depth is not None:
            namespaces = [ns[:max_depth] for ns in namespaces]
        namespaces = sorted(set(namespaces))
        return namespaces[offset : offset + limit]

    def batch(self, ops: Iterable[GetOp | SearchOp | PutOp | ListNamespacesOp]) -> list[Any]:
        results: list[Any] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self.get(op.namespace, op.key, refresh_ttl=op.refresh_ttl))
            elif isinstance(op, SearchOp):
                results.append(
                    self.search(
                        op.namespace_prefix,
                        query=op.query,
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                        refresh_ttl=op.refresh_ttl,
                    )
                )
            elif isinstance(op, PutOp):
                if op.value is None:
                    self.delete(op.namespace, op.key)
                else:
                    self.put(op.namespace, op.key, op.value, index=op.index, ttl=op.ttl)
                results.append(None)
            elif isinstance(op, ListNamespacesOp):
                results.append(self.list_namespaces(limit=op.limit, offset=op.offset, max_depth=op.max_depth))
            else:
                raise TypeError(f"Unsupported store op: {type(op)!r}")
        return results

    async def abatch(self, ops: Sequence[GetOp | SearchOp | PutOp | ListNamespacesOp]) -> list[Any]:
        return self.batch(ops)


def create_long_term_memory_store(
    *,
    database_url: str = "",
    vector_store_type: str = "pgvector",
    pgvector_table: str = "long_term_memory_vectors",
    embedding_model: object | None = None,
    vector_dimension: int = 1024,
    default_ttl_days: int | None = 180,
    default_importance: float = 0.5,
) -> PostgreSQLPGVectorLongTermMemoryStore:
    """创建长期记忆 Store。

    长期记忆固定使用 PostgreSQL 保存权威记录和审计。
    向量索引固定使用同库 PGVector。
    """
    if not database_url:
        raise ValueError("使用长期记忆时必须配置 DATABASE_URL 或 LONG_TERM_MEMORY_DATABASE_URL")
    policy = MemoryPolicy(
        default_ttl_days=default_ttl_days,
        default_importance=default_importance,
    )
    store_type = (vector_store_type or "pgvector").strip().lower()
    try:
        if store_type != "pgvector":
            raise ValueError(f"不支持的长期记忆向量后端：{vector_store_type!r}")
        return PostgreSQLPGVectorLongTermMemoryStore(
            database_url=database_url,
            pgvector_table=pgvector_table,
            embedding_model=embedding_model,
            vector_dimension=vector_dimension,
            policy=policy,
        )
    except Exception as exc:
        # 生产模式下长期记忆必须落到 PostgreSQL + PGVector。
        # 如果这里降级到 InMemoryStore，多实例会话会出现不可解释的不一致，
        # 也无法满足审计、TTL、冲突治理和语义召回要求。
        raise RuntimeError(f"长期记忆 PostgreSQL/{store_type} 初始化失败：{exc}") from exc


LongTermMemoryStore = PostgreSQLPGVectorLongTermMemoryStore

