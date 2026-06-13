import os
import time

import pytest

from app.memory import create_long_term_memory_store

pytestmark = pytest.mark.slow


def _mysql_milvus_store_or_skip():
    mysql_url = os.getenv("LONG_TERM_MEMORY_MYSQL_URL") or os.getenv("MYSQL_URL")
    milvus_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    if os.getenv("RUN_MYSQL_MILVUS_PERF") != "1" or not mysql_url:
        pytest.skip("Set RUN_MYSQL_MILVUS_PERF=1 and MySQL/Milvus env vars to run this smoke test.")
    return create_long_term_memory_store(
        mysql_url=mysql_url,
        milvus_uri=milvus_uri,
        milvus_collection=os.getenv("LONG_TERM_MEMORY_MILVUS_COLLECTION", "long_term_memory_vectors_1024"),
        milvus_alias=os.getenv("LONG_TERM_MEMORY_MILVUS_ALIAS", "ltm_milvus_perf"),
        vector_dimension=int(os.getenv("LONG_TERM_MEMORY_VECTOR_DIMENSION", "1024")),
        default_ttl_days=None,
    )


def test_mysql_milvus_long_term_memory_search_smoke_under_small_business_dataset():
    store = _mysql_milvus_store_or_skip()
    for idx in range(100):
        keyword = "stockout manual review" if idx == 42 else "ordinary order sufficient inventory"
        store.put(
            ("orders", f"SO{idx:06d}"),
            "decision",
            {
                "memory_type": "order_decision",
                "summary": f"{keyword}, order index {idx}",
                "_importance": 0.9 if idx == 42 else 0.2,
            },
        )

    start = time.perf_counter()
    results = store.search(("orders",), query="stockout manual review", limit=5)
    elapsed = time.perf_counter() - start

    assert results
    assert elapsed < 5.0


def test_mysql_milvus_long_term_memory_write_smoke_under_small_business_dataset():
    store = _mysql_milvus_store_or_skip()

    start = time.perf_counter()
    for idx in range(100):
        store.put(
            ("sessions", "merchant-A"),
            f"summary-{idx}",
            {"summary": f"session summary {idx}", "_importance": 0.5},
        )
    elapsed = time.perf_counter() - start

    assert store.get(("sessions", "merchant-A"), "summary-99") is not None
    assert elapsed < 10.0
