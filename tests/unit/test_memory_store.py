from datetime import datetime, timezone
import json

import pytest

from app.memory.long_term import (
    MySQLMilvusLongTermMemoryStore,
    create_long_term_memory_store,
)


def test_mysql_milvus_long_term_memory_requires_mysql_url():
    with pytest.raises(ValueError, match="MYSQL_URL|LONG_TERM_MEMORY_MYSQL_URL"):
        create_long_term_memory_store()


def test_mysql_milvus_long_term_memory_factory_wires_enterprise_options(monkeypatch):
    captured = {}

    def fake_init(self, mysql_url, **kwargs):
        captured["mysql_url"] = mysql_url
        captured.update(kwargs)

    monkeypatch.setattr(MySQLMilvusLongTermMemoryStore, "__init__", fake_init)

    store = create_long_term_memory_store(
        mysql_url="mysql+pymysql://root:root@mysql:3306/multiship_agent",
        milvus_uri="http://milvus:19530",
        milvus_token="token",
        milvus_database="memory_prod",
        milvus_collection="ltm_vectors",
        milvus_alias="ltm-prod",
        milvus_timeout_seconds=3,
        milvus_similarity_metric="COSINE",
        vector_dimension=1024,
        default_ttl_days=365,
    )

    assert isinstance(store, MySQLMilvusLongTermMemoryStore)
    assert captured["mysql_url"] == "mysql+pymysql://root:root@mysql:3306/multiship_agent"
    assert captured["milvus_uri"] == "http://milvus:19530"
    assert captured["milvus_database"] == "memory_prod"
    assert captured["milvus_collection"] == "ltm_vectors"
    assert captured["vector_dimension"] == 1024
    assert captured["policy"].default_ttl_days == 365


def test_mysql_milvus_long_term_memory_exported():
    assert MySQLMilvusLongTermMemoryStore.__name__ == "MySQLMilvusLongTermMemoryStore"


def test_mysql_milvus_embedding_dimension_mismatch_falls_back_to_text_search():
    store = object.__new__(MySQLMilvusLongTermMemoryStore)
    store.embedding_model = lambda text: [0.1, 0.2]
    store.vector_dimension = 3

    assert store._embedding_for("stockout manual review") is None

    store.vector_dimension = 2
    assert store._embedding_for("stockout manual review") == [0.1, 0.2]


class _FakeBegin:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeEngine:
    def __init__(self):
        self.conn = _FakeConn()

    def begin(self):
        return _FakeBegin(self.conn)


class _FakeConn:
    def __init__(self):
        self.executed = []

    def execute(self, statement, params=None):
        self.executed.append((str(statement), params or {}))
        return self


def _mysql_memory_row(
    *,
    row_id: int,
    vector_id: str,
    key: str,
    namespace: tuple[str, ...] = ("orders", "SO1"),
    summary: str,
    memory_type: str = "order_decision",
    importance: float = 0.5,
    access_count: int = 0,
):
    return {
        "id": row_id,
        "namespace": json.dumps(list(namespace), ensure_ascii=False),
        "memory_key": key,
        "value_json": json.dumps(
            {"memory_type": memory_type, "summary": summary, "_importance": importance},
            ensure_ascii=False,
        ),
        "search_text": f"summary:{summary}",
        "importance": importance,
        "access_count": access_count,
        "vector_id": vector_id,
        "created_at": datetime.now(tz=timezone.utc),
        "updated_at": datetime.now(tz=timezone.utc),
    }


def test_mysql_milvus_search_keeps_mysql_text_fallback_when_vector_hits_are_filtered(monkeypatch):
    store = object.__new__(MySQLMilvusLongTermMemoryStore)
    store._engine = _FakeEngine()
    store.table_name = "long_term_memory"
    store.milvus_similarity_metric = "COSINE"

    vector_row = _mysql_memory_row(
        row_id=1,
        vector_id="v1",
        key="vector-only",
        summary="customer prefers blue packaging",
        memory_type="user_preference",
        importance=0.95,
    )
    lexical_row = _mysql_memory_row(
        row_id=2,
        vector_id="v2",
        key="lexical-match",
        summary="order stockout needs manual review",
        memory_type="order_decision",
        importance=0.4,
    )

    monkeypatch.setattr(store, "_embedding_for", lambda text: [0.1, 0.2])
    monkeypatch.setattr(store, "_search_vector_ids", lambda namespace, vector, limit: [("v1", 0.99)])
    monkeypatch.setattr(store, "_row_by_vector_id", lambda conn, vector_id: vector_row)
    monkeypatch.setattr(store, "_rows_for_prefix", lambda conn, namespace: [vector_row, lexical_row])
    monkeypatch.setattr(store, "_delete_expired", lambda conn: None)

    results = store.search(
        ("orders", "SO1"),
        query="stockout manual review",
        filter={"memory_type": "order_decision"},
        limit=5,
    )

    assert [item.key for item in results] == ["lexical-match"]
    assert results[0].value["summary"] == "order stockout needs manual review"
    assert any("access_count=access_count+1" in sql for sql, _ in store._engine.conn.executed)


def test_mysql_milvus_similarity_normalizes_metric_scores():
    store = object.__new__(MySQLMilvusLongTermMemoryStore)

    store.milvus_similarity_metric = "COSINE"
    assert store._milvus_similarity(1.2) == 1.0
    assert store._milvus_similarity(-0.5) == 0.0

    store.milvus_similarity_metric = "L2"
    assert store._milvus_similarity(0.0) == 1.0
    assert store._milvus_similarity(3.0) == 0.25
