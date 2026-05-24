import time

import pytest

from app.memory import create_long_term_memory_store

pytestmark = pytest.mark.slow


def test_sqlite_long_term_memory_search_smoke_under_small_business_dataset(tmp_path):
    """小型商家数据集下的长期记忆检索烟测。

    这个测试不是严格压测，只用来防止搜索逻辑出现明显退化：
    例如一次普通 namespace 查询在几百条记忆下突然变成数秒级。
    真正 P95/并发压测应使用 Locust/k6 针对运行中的服务执行。
    """
    store = create_long_term_memory_store(db_path=tmp_path / "memory.sqlite3", default_ttl_days=None)
    for idx in range(600):
        keyword = "缺货 人工复核" if idx == 420 else "普通订单 库存充足"
        store.put(
            ("orders", f"SO{idx:06d}"),
            "decision",
            {
                "memory_type": "order_decision",
                "summary": f"{keyword}，订单序号 {idx}",
                "_importance": 0.9 if idx == 420 else 0.2,
            },
        )

    start = time.perf_counter()
    results = store.search(("orders",), query="缺货 人工复核", limit=5)
    elapsed = time.perf_counter() - start

    assert results
    assert results[0].namespace == ("orders", "SO000420")
    assert elapsed < 2.0


def test_sqlite_long_term_memory_write_smoke_under_small_business_dataset(tmp_path):
    store = create_long_term_memory_store(db_path=tmp_path / "memory.sqlite3", default_ttl_days=None)

    start = time.perf_counter()
    for idx in range(300):
        store.put(
            ("sessions", "merchant-A"),
            f"summary-{idx}",
            {"summary": f"会话摘要 {idx}", "_importance": 0.5},
        )
    elapsed = time.perf_counter() - start

    assert store.get(("sessions", "merchant-A"), "summary-299") is not None
    assert elapsed < 3.0
