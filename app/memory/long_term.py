"""长期记忆 Store。

长期记忆保存跨会话仍然有价值的信息，例如：
  - 客户偏好：是否接受替代品、偏好的仓库/物流。
  - 历史决策：某个订单最终为什么走了缺货流程。
  - 会话摘要：用户前几次问过什么、系统给过什么承诺。

生产架构通常是“结构化库 + 向量库”：
  - MySQL 保存精确字段、TTL、审计信息。
  - Milvus 保存向量索引，支持相似案例检索。

这个项目为了本地可运行，先提供一个 SQLite 实现：
  - 真的落盘，不是进程内存，服务重启后仍可读取。
  - search() 支持 namespace 前缀、filter、TTL、importance、轻量文本相似度。
  - 接口完全遵循 LangGraph BaseStore，后续切换 MySQL + Milvus 不影响节点代码。
"""

from __future__ import annotations

import hashlib
import logging
import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
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


DEFAULT_MEMORY_DB = Path("storage") / "long_term_memory.sqlite3"
logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _to_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


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


def _milvus_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


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
        logger.warning("长期记忆 embedding 生成失败，降级为文本检索：%s", exc)
    return None


@dataclass
class MemoryPolicy:
    """长期记忆写入策略。

    default_ttl_days=None 表示默认不过期。
    importance 影响 search 排序：相似度相同的情况下，更重要的记忆排在前面。
    """

    default_ttl_days: int | None = 180
    default_importance: float = 0.5


class SQLiteLongTermMemoryStore(BaseStore):
    """可落盘的 LangGraph BaseStore 实现。

    namespace 采用 tuple，例如：
      ("customers", "C001")
      ("orders", "SO202502140001")
      ("sessions", "user-123")

    key 是该 namespace 下的稳定 ID；value 必须是 dict。
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_MEMORY_DB,
        policy: MemoryPolicy | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.policy = policy or MemoryPolicy()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ltm_namespace ON long_term_memory(namespace)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ltm_expires ON long_term_memory(expires_at)")

    def _delete_expired(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "DELETE FROM long_term_memory WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (_to_iso(_now()),),
        )

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index=None,
        *,
        ttl=None,
    ) -> None:
        """写入或更新一条长期记忆。

        value 可选元字段：
          _importance: 0~1，影响 search 排序。
          _ttl_days: 单条记忆 TTL；None 表示使用默认策略。
        """
        now = _now()
        ttl_days = value.get("_ttl_days", self.policy.default_ttl_days)
        expires_at = None if ttl_days is None else now + timedelta(days=float(ttl_days))
        if ttl is not None and not isinstance(ttl, bool):
            expires_at = now + timedelta(seconds=float(ttl))

        importance = float(value.get("_importance", self.policy.default_importance))
        importance = max(0.0, min(1.0, importance))

        namespace_raw = _namespace_key(namespace)
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
        search_text = _text_for_search(value)

        with self._connect() as conn:
            self._delete_expired(conn)
            existing = conn.execute(
                "SELECT created_at FROM long_term_memory WHERE namespace=? AND key=?",
                (namespace_raw, key),
            ).fetchone()
            created_at = existing["created_at"] if existing else _to_iso(now)
            conn.execute(
                """
                INSERT INTO long_term_memory (
                    namespace, key, value_json, search_text, importance,
                    created_at, updated_at, expires_at, access_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(
                    (SELECT access_count FROM long_term_memory WHERE namespace=? AND key=?), 0
                ))
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value_json=excluded.value_json,
                    search_text=excluded.search_text,
                    importance=excluded.importance,
                    updated_at=excluded.updated_at,
                    expires_at=excluded.expires_at
                """,
                (
                    namespace_raw,
                    key,
                    payload,
                    search_text,
                    importance,
                    created_at,
                    _to_iso(now),
                    _to_iso(expires_at),
                    namespace_raw,
                    key,
                ),
            )

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        namespace_raw = _namespace_key(namespace)
        with self._connect() as conn:
            self._delete_expired(conn)
            row = conn.execute(
                "SELECT * FROM long_term_memory WHERE namespace=? AND key=?",
                (namespace_raw, key),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE long_term_memory SET access_count=access_count+1 WHERE namespace=? AND key=?",
                (namespace_raw, key),
            )
        return Item(
            namespace=namespace,
            key=key,
            value=json.loads(row["value_json"]),
            created_at=_from_iso(row["created_at"]),
            updated_at=_from_iso(row["updated_at"]),
        )

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
        prefix = list(namespace_prefix)
        query_tokens = _tokens(query or "")
        rows: list[sqlite3.Row]

        with self._connect() as conn:
            self._delete_expired(conn)
            rows = list(conn.execute("SELECT * FROM long_term_memory"))

        scored: list[tuple[float, sqlite3.Row, dict[str, Any]]] = []
        for row in rows:
            namespace = _parse_namespace(row["namespace"])
            if tuple(namespace[: len(prefix)]) != tuple(prefix):
                continue
            value = json.loads(row["value_json"])
            if not _matches_filter(value, filter):
                continue

            if query_tokens:
                memory_tokens = _tokens(row["search_text"])
                overlap = len(query_tokens & memory_tokens)
                lexical = overlap / max(len(query_tokens), 1)
            else:
                lexical = 0.0

            if query_tokens and lexical <= 0:
                continue

            # 排序分 = 文本相关性 + 重要性 + 访问热度的轻量加权。
            score = lexical * 0.75 + float(row["importance"]) * 0.20 + min(row["access_count"], 10) * 0.005
            scored.append((score, row, value))

        scored.sort(key=lambda item: (item[0], item[1]["updated_at"]), reverse=True)
        selected = scored[offset : offset + limit]
        if selected:
            with self._connect() as conn:
                for _, row, _ in selected:
                    conn.execute(
                        "UPDATE long_term_memory SET access_count=access_count+1 WHERE namespace=? AND key=?",
                        (row["namespace"], row["key"]),
                    )

        return [
            SearchItem(
                namespace=_parse_namespace(row["namespace"]),
                key=row["key"],
                value=value,
                created_at=_from_iso(row["created_at"]),
                updated_at=_from_iso(row["updated_at"]),
                score=round(score, 4),
            )
            for score, row, value in selected
        ]

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM long_term_memory WHERE namespace=? AND key=?",
                (_namespace_key(namespace), key),
            )

    def list_namespaces(
        self,
        *,
        prefix=None,
        suffix=None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        with self._connect() as conn:
            self._delete_expired(conn)
            rows = conn.execute("SELECT DISTINCT namespace FROM long_term_memory").fetchall()

        namespaces = [_parse_namespace(row["namespace"]) for row in rows]
        if prefix:
            prefix_tuple = tuple(prefix)
            namespaces = [ns for ns in namespaces if ns[: len(prefix_tuple)] == prefix_tuple]
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


class MySQLMilvusLongTermMemoryStore(BaseStore):
    """MySQL + Milvus 长期记忆实现。

    企业级拆分原则：
    - MySQL 是权威数据源，保存结构化 value、namespace、TTL、importance、访问热度和审计。
    - Milvus 是可重建的语义索引，只保存向量和少量检索元数据。
    - 写入先落 MySQL，再尽力同步 Milvus；Milvus 不可用时降级为 MySQL 文本检索。
    """

    def __init__(
        self,
        mysql_url: str,
        *,
        milvus_uri: str = "http://localhost:19530",
        milvus_token: str = "",
        milvus_database: str = "default",
        milvus_collection: str = "long_term_memory_vectors",
        milvus_alias: str = "ltm_milvus",
        milvus_timeout_seconds: float = 10.0,
        milvus_similarity_metric: str = "COSINE",
        embedding_model: object | None = None,
        vector_dimension: int = 512,
        policy: MemoryPolicy | None = None,
        table_name: str = "long_term_memory",
        audit_table_name: str = "long_term_memory_audit",
    ) -> None:
        if not mysql_url:
            raise ValueError("使用 MySQL + Milvus 长期记忆时必须配置 MYSQL_URL 或 LONG_TERM_MEMORY_MYSQL_URL")
        self.mysql_url = mysql_url
        self.embedding_model = embedding_model
        self.vector_dimension = int(vector_dimension)
        self.policy = policy or MemoryPolicy()
        self.table_name = _safe_identifier(table_name)
        self.audit_table_name = _safe_identifier(audit_table_name)
        self.milvus_uri = milvus_uri
        self.milvus_token = milvus_token
        self.milvus_database = milvus_database or "default"
        self.milvus_collection = milvus_collection
        self.milvus_alias = re.sub(r"[^a-zA-Z0-9_]", "_", milvus_alias).strip("_") or "ltm_milvus"
        self.milvus_timeout_seconds = float(milvus_timeout_seconds)
        self.milvus_similarity_metric = (milvus_similarity_metric or "COSINE").upper()
        self._milvus = None

        self._engine = self._create_engine()
        self._init_schema()
        self._init_milvus()

    def _create_engine(self):
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise RuntimeError("MySQL 长期记忆需要安装 sqlalchemy 和 pymysql。") from exc
        return create_engine(
            self.mysql_url,
            pool_pre_ping=True,
            pool_recycle=1800,
            future=True,
        )

    def _init_schema(self) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id BIGINT NOT NULL AUTO_INCREMENT,
                    namespace_hash CHAR(64) NOT NULL,
                    key_hash CHAR(64) NOT NULL,
                    namespace TEXT NOT NULL,
                    namespace_path VARCHAR(1024) NOT NULL,
                    memory_key VARCHAR(512) NOT NULL,
                    value_json JSON NOT NULL,
                    search_text TEXT NOT NULL,
                    importance DOUBLE NOT NULL,
                    vector_id VARCHAR(96) NOT NULL,
                    created_at DATETIME(6) NOT NULL,
                    updated_at DATETIME(6) NOT NULL,
                    expires_at DATETIME(6) NULL,
                    access_count INT NOT NULL DEFAULT 0,
                    version BIGINT NOT NULL DEFAULT 1,
                    deleted_at DATETIME(6) NULL,
                    PRIMARY KEY (id),
                    UNIQUE KEY uq_ltm_namespace_key (namespace_hash, key_hash),
                    KEY idx_ltm_namespace_path (namespace_path(255)),
                    KEY idx_ltm_vector_id (vector_id),
                    KEY idx_ltm_expires (expires_at),
                    KEY idx_ltm_updated (updated_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """))
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.audit_table_name} (
                    id BIGINT NOT NULL AUTO_INCREMENT,
                    namespace_hash CHAR(64) NOT NULL,
                    key_hash CHAR(64) NOT NULL,
                    namespace TEXT NOT NULL,
                    memory_key VARCHAR(512) NOT NULL,
                    event_type VARCHAR(32) NOT NULL,
                    value_json JSON NULL,
                    created_at DATETIME(6) NOT NULL,
                    PRIMARY KEY (id),
                    KEY idx_ltm_audit_memory (namespace_hash, key_hash),
                    KEY idx_ltm_audit_created (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """))
            try:
                conn.execute(text(f"CREATE FULLTEXT INDEX idx_ltm_search_text ON {self.table_name} (search_text)"))
            except Exception:
                # MySQL 没有启用全文索引或索引已存在时，不影响主流程；search 仍有词元召回兜底。
                pass

    def _init_milvus(self) -> None:
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, db, utility
        except ImportError as exc:
            logger.warning("pymilvus 未安装，长期记忆将降级为 MySQL 文本检索：%s", exc)
            return

        try:
            connection_kwargs = {
                "alias": self.milvus_alias,
                "uri": self.milvus_uri,
                "token": self.milvus_token,
                "timeout": self.milvus_timeout_seconds,
            }
            if self.milvus_database != "default":
                connections.connect(**connection_kwargs, db_name="default")
                try:
                    databases = db.list_database(using=self.milvus_alias, timeout=self.milvus_timeout_seconds)
                    if self.milvus_database not in databases:
                        db.create_database(
                            self.milvus_database,
                            using=self.milvus_alias,
                            timeout=self.milvus_timeout_seconds,
                        )
                finally:
                    connections.disconnect(self.milvus_alias)

            connections.connect(**connection_kwargs, db_name=self.milvus_database)
            if not utility.has_collection(
                self.milvus_collection,
                using=self.milvus_alias,
                timeout=self.milvus_timeout_seconds,
            ):
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=96),
                    FieldSchema(name="namespace_path", dtype=DataType.VARCHAR, max_length=1024),
                    FieldSchema(name="memory_key", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="search_text", dtype=DataType.VARCHAR, max_length=8192),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dimension),
                    FieldSchema(name="importance", dtype=DataType.FLOAT),
                    FieldSchema(name="updated_at", dtype=DataType.INT64),
                    FieldSchema(name="expires_at", dtype=DataType.INT64),
                ]
                schema = CollectionSchema(fields, description="Long-term memory semantic index")
                self._milvus = Collection(
                    self.milvus_collection,
                    schema=schema,
                    using=self.milvus_alias,
                )
                self._milvus.create_index(
                    "embedding",
                    {
                        "metric_type": self.milvus_similarity_metric,
                        "index_type": "HNSW",
                        "params": {"M": 16, "efConstruction": 200},
                    },
                    timeout=self.milvus_timeout_seconds,
                )
            else:
                self._milvus = Collection(self.milvus_collection, using=self.milvus_alias)
                existing_dim = self._milvus_vector_dim(self._milvus)
                if existing_dim and existing_dim != self.vector_dimension:
                    raise RuntimeError(
                        f"Milvus collection '{self.milvus_collection}' 向量维度是 {existing_dim}，"
                        f"当前长期记忆 embedding 维度是 {self.vector_dimension}。"
                    )
            self._milvus.load(timeout=self.milvus_timeout_seconds)
        except Exception as exc:
            self._milvus = None
            logger.warning("Milvus 长期记忆索引初始化失败，降级为 MySQL 文本检索：%s", exc)

    @staticmethod
    def _milvus_vector_dim(collection: object) -> int | None:
        try:
            for field in collection.schema.fields:
                params = getattr(field, "params", {}) or {}
                if "dim" in params:
                    return int(params["dim"])
        except Exception:
            return None
        return None

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
            logger.warning(
                "长期记忆 embedding 维度不匹配：实际 %s，配置 %s。本次降级为文本检索。",
                len(embedding),
                self.vector_dimension,
            )
            return None
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
        if self._milvus is None or embedding is None:
            return
        try:
            row = [
                [vector_id],
                [_namespace_path(namespace)],
                [key[:512]],
                [search_text[:8192]],
                [embedding],
                [float(importance)],
                [_epoch_millis(updated_at)],
                [_epoch_millis(expires_at)],
            ]
            if hasattr(self._milvus, "upsert"):
                self._milvus.upsert(row, timeout=self.milvus_timeout_seconds)
            else:
                self._milvus.delete(f'id == "{_milvus_quote(vector_id)}"')
                self._milvus.insert(row, timeout=self.milvus_timeout_seconds)
            if hasattr(self._milvus, "flush"):
                self._milvus.flush(timeout=self.milvus_timeout_seconds)
        except Exception as exc:
            logger.warning("长期记忆 Milvus upsert 失败，MySQL 已保留权威记录：%s", exc)

    def _delete_vector(self, vector_id: str) -> None:
        if self._milvus is None:
            return
        try:
            self._milvus.delete(f'id == "{_milvus_quote(vector_id)}"', timeout=self.milvus_timeout_seconds)
        except Exception as exc:
            logger.warning("长期记忆 Milvus delete 失败：%s", exc)

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
                    ON DUPLICATE KEY UPDATE
                        namespace=VALUES(namespace),
                        namespace_path=VALUES(namespace_path),
                        memory_key=VALUES(memory_key),
                        value_json=VALUES(value_json),
                        search_text=VALUES(search_text),
                        importance=VALUES(importance),
                        vector_id=VALUES(vector_id),
                        updated_at=VALUES(updated_at),
                        expires_at=VALUES(expires_at),
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
        if self._milvus is None or query_vector is None:
            return []
        now_ms = _epoch_millis(_now())
        expr = f"(expires_at == 0 or expires_at > {now_ms})"
        if namespace_prefix:
            prefix = _milvus_quote(_namespace_path(namespace_prefix))
            expr = f'namespace_path like "{prefix}%" and {expr}'
        try:
            raw = self._milvus.search(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": self.milvus_similarity_metric, "params": {"ef": 64}},
                limit=max(limit * 20, 50),
                expr=expr,
                output_fields=["id"],
                timeout=self.milvus_timeout_seconds,
            )
        except Exception as exc:
            logger.warning("长期记忆 Milvus 搜索失败，降级 MySQL 文本检索：%s", exc)
            return []

        hits = raw[0] if raw else []
        results: list[tuple[str, float]] = []
        for hit in hits:
            vector_id = str(getattr(hit, "id", "") or getattr(hit, "pk", "") or "")
            distance = float(getattr(hit, "score", getattr(hit, "distance", 0.0)))
            if vector_id:
                results.append((vector_id, self._milvus_similarity(distance)))
        return results

    def _milvus_similarity(self, distance: float) -> float:
        if self.milvus_similarity_metric == "L2":
            return 1.0 / (1.0 + max(distance, 0.0))
        return max(0.0, min(1.0, distance))

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
    db_path: str | Path | None = None,
    *,
    backend: str = "sqlite",
    mysql_url: str = "",
    milvus_uri: str = "http://localhost:19530",
    milvus_token: str = "",
    milvus_database: str = "default",
    milvus_collection: str = "long_term_memory_vectors",
    milvus_alias: str = "ltm_milvus",
    milvus_timeout_seconds: float = 10.0,
    milvus_similarity_metric: str = "COSINE",
    embedding_model: object | None = None,
    vector_dimension: int = 512,
    default_ttl_days: int | None = 180,
    default_importance: float = 0.5,
) -> SQLiteLongTermMemoryStore | MySQLMilvusLongTermMemoryStore:
    """创建长期记忆 Store。

    默认使用 SQLite，保证本地开发开箱即用。
    配置 ``backend="mysql_milvus"`` 后切换到 MySQL + Milvus 企业级拆分存储。
    """
    policy = MemoryPolicy(
        default_ttl_days=default_ttl_days,
        default_importance=default_importance,
    )
    normalized_backend = backend.strip().lower()
    if normalized_backend in {"mysql_milvus", "mysql+milvus", "mysql", "milvus"}:
        return MySQLMilvusLongTermMemoryStore(
            mysql_url=mysql_url,
            milvus_uri=milvus_uri,
            milvus_token=milvus_token,
            milvus_database=milvus_database,
            milvus_collection=milvus_collection,
            milvus_alias=milvus_alias,
            milvus_timeout_seconds=milvus_timeout_seconds,
            milvus_similarity_metric=milvus_similarity_metric,
            embedding_model=embedding_model,
            vector_dimension=vector_dimension,
            policy=policy,
        )
    return SQLiteLongTermMemoryStore(
        db_path=db_path or DEFAULT_MEMORY_DB,
        policy=policy,
    )


# 兼容旧导入名：之前 README/代码里提到 LongTermMemoryStore。
LongTermMemoryStore = SQLiteLongTermMemoryStore

