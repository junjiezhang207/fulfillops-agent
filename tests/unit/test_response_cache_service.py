from app.application.cache.client_cache_service import (
    ClientCacheService,
    ResponseCacheService,
    get_client_cache,
    get_response_cache,
)


def test_response_cache_hits_same_order_question_and_context():
    cache = ResponseCacheService(ttl_hours=1)

    written = cache.set(
        order_id="SO1",
        question="这个订单状态是什么？",
        answer="订单正常履约中。",
        path_used="workflow",
        tools_called=["order_analysis"],
        execution_time_ms=1234,
        cache_context="orders:v1",
    )
    hit = cache.get("SO1", "这个订单状态是什么？", cache_context="orders:v1")

    assert written is True
    assert hit is not None
    assert hit.answer == "订单正常履约中。"
    stats = cache.get_stats()
    assert stats["cache_hits"] == 1
    assert stats["estimated_latency_saved_ms"] == 1234
    assert "tokens_saved" not in stats


def test_response_cache_context_prevents_stale_reuse():
    cache = ResponseCacheService(ttl_hours=1)
    cache.set(
        order_id="SO1",
        question="这个订单状态是什么？",
        answer="订单正常履约中。",
        path_used="workflow",
        cache_context="orders:v1",
    )

    assert cache.get("SO1", "这个订单状态是什么？", cache_context="orders:v2") is None


def test_response_cache_skips_high_risk_business_decisions():
    cache = ResponseCacheService(ttl_hours=1)

    written = cache.set(
        order_id="SO1",
        question="这个订单缺货了要不要人工审批？",
        answer="建议人工审批。",
        path_used="agent",
        tools_called=["check_inventory"],
    )

    assert written is False
    assert cache.get_stats()["cache_skips"] == 1
    assert cache.get("SO1", "这个订单缺货了要不要人工审批？") is None


def test_legacy_client_cache_name_still_points_to_response_cache():
    assert ClientCacheService is ResponseCacheService
    assert get_client_cache() is get_response_cache()
