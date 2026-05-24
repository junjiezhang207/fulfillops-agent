from types import SimpleNamespace

import pytest

from app.repositories.vector_store_factory import create_vector_store


def test_vector_store_factory_uses_local_fallback_when_configured():
    settings = SimpleNamespace(vector_store_type="local")

    assert create_vector_store(settings) is None


def test_vector_store_factory_builds_milvus_store_with_enterprise_defaults(monkeypatch):
    captured = {}
    pymilvus_calls = []

    class FakeMilvusVectorStore:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeConnections:
        def connect(self, **kwargs):
            pymilvus_calls.append(("connect", kwargs))

        def disconnect(self, alias):
            pymilvus_calls.append(("disconnect", alias))

    class FakeDb:
        def list_database(self, using, timeout):
            pymilvus_calls.append(("list_database", using, timeout))
            return ["default"]

        def create_database(self, db_name, using, timeout):
            pymilvus_calls.append(("create_database", db_name, using, timeout))

    class FakeUtility:
        def has_collection(self, collection_name, using, timeout):
            pymilvus_calls.append(("has_collection", collection_name, using, timeout))
            return False

    class FakeCollection:
        def __init__(self, name, using):
            pymilvus_calls.append(("collection", name, using))

        def load(self, timeout):
            pymilvus_calls.append(("load", timeout))

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "llama_index.vector_stores.milvus":
            return SimpleNamespace(MilvusVectorStore=FakeMilvusVectorStore)
        if name == "pymilvus":
            return SimpleNamespace(
                connections=FakeConnections(),
                db=FakeDb(),
                utility=FakeUtility(),
                Collection=FakeCollection,
            )
        return real_import(name, globals, locals, fromlist, level)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)
    settings = SimpleNamespace(
        vector_store_type="milvus",
        vector_store_fallback_to_local=False,
        milvus_uri="http://milvus.internal:19530",
        milvus_collection="test_collection",
        milvus_dim=512,
        milvus_overwrite=False,
        milvus_upsert_mode=True,
        milvus_batch_size=10,
        milvus_token="token",
        milvus_database="rag_prod",
        milvus_alias="rag-test",
        milvus_timeout_seconds=3,
        milvus_similarity_metric="COSINE",
        milvus_consistency_level="Session",
    )

    store = create_vector_store(settings)

    assert isinstance(store, FakeMilvusVectorStore)
    assert captured["uri"] == "http://milvus.internal:19530"
    assert captured["collection_name"] == "test_collection"
    assert captured["dim"] == 512
    assert captured["upsert_mode"] is True
    assert captured["overwrite"] is False
    assert captured["similarity_metric"] == "COSINE"
    assert captured["db_name"] == "rag_prod"
    assert ("create_database", "rag_prod", "rag_test", 3.0) in pymilvus_calls
    assert any(call[0] == "has_collection" for call in pymilvus_calls)


def test_vector_store_factory_raises_when_milvus_import_missing_and_strict(monkeypatch):
    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pymilvus":
            return SimpleNamespace(
                connections=SimpleNamespace(connect=lambda **kwargs: None),
                db=SimpleNamespace(),
                utility=SimpleNamespace(has_collection=lambda *args, **kwargs: False),
                Collection=object,
            )
        if name == "llama_index.vector_stores.milvus":
            raise ImportError("missing")
        return real_import(name, globals, locals, fromlist, level)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)
    settings = SimpleNamespace(
        vector_store_type="milvus",
        vector_store_fallback_to_local=False,
        milvus_uri="http://milvus.internal:19530",
        milvus_collection="test_collection",
        milvus_dim=512,
        milvus_overwrite=False,
        milvus_upsert_mode=True,
        milvus_batch_size=10,
        milvus_token="",
        milvus_database="default",
        milvus_timeout_seconds=3,
    )

    with pytest.raises(RuntimeError, match="Milvus"):
        create_vector_store(settings)


def test_vector_store_factory_uses_pymilvus_to_load_existing_collection(monkeypatch):
    captured = {"loaded": False}

    class FakeMilvusVectorStore:
        def __init__(self, **kwargs):
            captured["vector_store"] = kwargs

    class FakeCollection:
        def __init__(self, name, using):
            captured["collection"] = (name, using)

        def load(self, timeout):
            captured["loaded"] = True
            captured["timeout"] = timeout

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pymilvus":
            return SimpleNamespace(
                connections=SimpleNamespace(connect=lambda **kwargs: captured.setdefault("connect", kwargs)),
                db=SimpleNamespace(),
                utility=SimpleNamespace(has_collection=lambda *args, **kwargs: True),
                Collection=FakeCollection,
            )
        if name == "llama_index.vector_stores.milvus":
            return SimpleNamespace(MilvusVectorStore=FakeMilvusVectorStore)
        return real_import(name, globals, locals, fromlist, level)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)
    settings = SimpleNamespace(
        vector_store_type="milvus",
        vector_store_fallback_to_local=False,
        milvus_uri="http://localhost:19530",
        milvus_collection="knowledge_base",
        milvus_dim=512,
        milvus_overwrite=False,
        milvus_upsert_mode=True,
        milvus_batch_size=10,
        milvus_token="",
        milvus_database="default",
        milvus_alias="rag_knowledge_base",
        milvus_timeout_seconds=5,
    )

    create_vector_store(settings)

    assert captured["loaded"] is True
    assert captured["collection"] == ("knowledge_base", "rag_knowledge_base")
