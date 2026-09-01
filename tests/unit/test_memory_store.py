from datetime import datetime, timezone
import json

import pytest

from app.memory.long_term import (
    PostgreSQLPGVectorLongTermMemoryStore,
    create_long_term_memory_store,
)


def test_pgvector_long_term_memory_requires_database_url():
    with pytest.raises(ValueError, match="DATABASE_URL|LONG_TERM_MEMORY_DATABASE_URL"):
        create_long_term_memory_store()


def test_long_term_memory_factory_defaults_to_pgvector(monkeypatch):
    captured = {}

    def fake_init(self, database_url, **kwargs):
        captured["database_url"] = database_url
        captured.update(kwargs)

    monkeypatch.setattr(PostgreSQLPGVectorLongTermMemoryStore, "__init__", fake_init)

    store = create_long_term_memory_store(
        database_url="postgresql+psycopg://fulfillops:secret@postgres:5432/fulfillops_agent",
        pgvector_table="ltm_vectors",
        vector_dimension=1024,
        default_ttl_days=365,
    )

    assert isinstance(store, PostgreSQLPGVectorLongTermMemoryStore)
    assert captured["database_url"] == "postgresql+psycopg://fulfillops:secret@postgres:5432/fulfillops_agent"
    assert captured["pgvector_table"] == "ltm_vectors"
    assert captured["vector_dimension"] == 1024
    assert captured["policy"].default_ttl_days == 365


def test_long_term_memory_factory_rejects_non_pgvector_backend(monkeypatch):
    captured = {}

    def fake_init(self, database_url, **kwargs):
        captured["database_url"] = database_url
        captured.update(kwargs)

    monkeypatch.setattr(PostgreSQLPGVectorLongTermMemoryStore, "__init__", fake_init)

    with pytest.raises(RuntimeError, match="不支持的长期记忆向量后端"):
        create_long_term_memory_store(
            database_url="postgresql+psycopg://fulfillops:secret@postgres:5432/fulfillops_agent",
            vector_store_type="legacy_vector_store",
        )


def test_pgvector_long_term_memory_exported():
    assert PostgreSQLPGVectorLongTermMemoryStore.__name__ == "PostgreSQLPGVectorLongTermMemoryStore"


def test_pgvector_embedding_dimension_mismatch_fails_closed():
    store = object.__new__(PostgreSQLPGVectorLongTermMemoryStore)
    store.embedding_model = lambda text: [0.1, 0.2]
    store.vector_dimension = 3

    with pytest.raises(RuntimeError, match="维度不匹配"):
        store._embedding_for("stockout manual review")

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


def _postgres_memory_row(
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


def test_pgvector_search_keeps_postgres_text_fallback_when_vector_hits_are_filtered(monkeypatch):
    store = object.__new__(PostgreSQLPGVectorLongTermMemoryStore)
    store._engine = _FakeEngine()
    store.table_name = "long_term_memory"

    vector_row = _postgres_memory_row(
        row_id=1,
        vector_id="v1",
        key="vector-only",
        summary="customer prefers blue packaging",
        memory_type="user_preference",
        importance=0.95,
    )
    lexical_row = _postgres_memory_row(
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


def test_pgvector_similarity_normalizes_cosine_distance():
    store = object.__new__(PostgreSQLPGVectorLongTermMemoryStore)

    assert store._pgvector_similarity(0.0) == 1.0
    assert store._pgvector_similarity(0.4) == 0.6
    assert store._pgvector_similarity(1.2) == 0.0
