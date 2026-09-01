from types import SimpleNamespace

import pytest

from app.repositories.vector_store_factory import create_vector_store


def test_vector_store_factory_uses_local_fallback_when_configured():
    settings = SimpleNamespace(vector_store_type="local")

    assert create_vector_store(settings) is None


def test_vector_store_factory_builds_pgvector_store(monkeypatch):
    captured = {}

    class FakePGVectorStore:
        @classmethod
        def from_params(cls, **kwargs):
            captured.update(kwargs)
            return cls()

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "llama_index.vector_stores.postgres":
            return SimpleNamespace(PGVectorStore=FakePGVectorStore)
        return real_import(name, globals, locals, fromlist, level)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)
    settings = SimpleNamespace(
        vector_store_type="pgvector",
        vector_store_fallback_to_local=False,
        pgvector_database="rag_prod",
        pgvector_host="postgres.internal",
        pgvector_port=5432,
        pgvector_user="fulfillops",
        pgvector_password="secret",
        pgvector_table="knowledge_vectors",
        pgvector_dim=1024,
    )

    store = create_vector_store(settings)

    assert isinstance(store, FakePGVectorStore)
    assert captured == {
        "database": "rag_prod",
        "host": "postgres.internal",
        "password": "secret",
        "port": 5432,
        "user": "fulfillops",
        "table_name": "knowledge_vectors",
        "embed_dim": 1024,
    }


def test_vector_store_factory_raises_when_pgvector_import_missing_and_strict(monkeypatch):
    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "llama_index.vector_stores.postgres":
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)
    settings = SimpleNamespace(
        vector_store_type="pgvector",
        vector_store_fallback_to_local=False,
        pgvector_database="rag_prod",
        pgvector_host="postgres.internal",
        pgvector_port=5432,
        pgvector_user="fulfillops",
        pgvector_password="secret",
        pgvector_table="knowledge_vectors",
        pgvector_dim=1024,
    )

    with pytest.raises(RuntimeError, match="PGVector"):
        create_vector_store(settings)


def test_vector_store_factory_can_fall_back_to_local_when_pgvector_unavailable(monkeypatch):
    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "llama_index.vector_stores.postgres":
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)
    settings = SimpleNamespace(
        vector_store_type="pgvector",
        vector_store_fallback="local",
        vector_store_fallback_to_local=True,
        pgvector_database="rag_prod",
        pgvector_host="postgres.internal",
        pgvector_port=5432,
        pgvector_user="fulfillops",
        pgvector_password="secret",
        pgvector_table="knowledge_vectors",
        pgvector_dim=1024,
    )

    assert create_vector_store(settings) is None
