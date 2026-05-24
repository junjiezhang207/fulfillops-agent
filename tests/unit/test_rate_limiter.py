from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status

from app.core import rate_limiter
from app.core.rate_limiter import RateLimiter, RedisRateLimiter, _WindowRecord


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}

    def incr(self, key):
        self.values[key] = int(self.values.get(key, 0)) + 1
        return self.values[key]

    def expire(self, key, ttl):
        self.ttls[key] = ttl
        return True

    def get(self, key):
        return self.values.get(key)


def test_rate_limiter_allows_until_window_quota_is_exhausted():
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    assert limiter.is_allowed("merchant-A") is True
    assert limiter.remaining("merchant-A") == 1
    assert limiter.is_allowed("merchant-A") is True
    assert limiter.remaining("merchant-A") == 0
    assert limiter.is_allowed("merchant-A") is False


def test_rate_limiter_resets_after_window(monkeypatch):
    now = {"value": 100.0}
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: now["value"])
    limiter = RateLimiter(max_requests=1, window_seconds=10)

    assert limiter.is_allowed("session-1") is True
    assert limiter.is_allowed("session-1") is False

    now["value"] = 111.0

    assert limiter.is_allowed("session-1") is True
    assert limiter.remaining("session-1") == 0


def test_rate_limiter_evicts_only_expired_records(monkeypatch):
    now = {"value": 50.0}
    monkeypatch.setattr(rate_limiter.time, "monotonic", lambda: now["value"])
    limiter = RateLimiter(max_requests=3, window_seconds=10)
    limiter._store = {
        "old": _WindowRecord(count=1, window_start=30.0),
        "fresh": _WindowRecord(count=1, window_start=45.0),
    }

    assert limiter.evict_expired() == 1
    assert "old" not in limiter._store
    assert "fresh" in limiter._store


def test_redis_rate_limiter_uses_redis_fixed_window(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(RedisRateLimiter, "_create_client", staticmethod(lambda redis_url: fake))
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 120.0)
    limiter = RedisRateLimiter("redis://localhost:6379/0", max_requests=2, window_seconds=60, namespace="test")

    assert limiter.is_allowed("session-1") is True
    assert limiter.remaining("session-1") == 1
    assert limiter.is_allowed("session-1") is True
    assert limiter.remaining("session-1") == 0
    assert limiter.is_allowed("session-1") is False
    assert fake.ttls["multiship:rate_limit:test:session-1:2"] == 61


def test_redis_rate_limiter_falls_back_to_memory_when_redis_fails(monkeypatch):
    class BrokenRedis(FakeRedis):
        def incr(self, key):
            raise RuntimeError("redis down")

    monkeypatch.setattr(RedisRateLimiter, "_create_client", staticmethod(lambda redis_url: BrokenRedis()))
    limiter = RedisRateLimiter("redis://localhost:6379/0", max_requests=1, window_seconds=60, namespace="test")

    assert limiter.is_allowed("session-1") is True
    assert limiter.is_allowed("session-1") is False


def test_check_agent_rate_limit_raises_429_when_session_exceeds_quota(monkeypatch):
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    monkeypatch.setattr(rate_limiter, "agent_limiter", limiter)
    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))

    rate_limiter.check_agent_rate_limit(request, "s1")

    with pytest.raises(HTTPException) as exc:
        rate_limiter.check_agent_rate_limit(request, "s1")

    assert exc.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert exc.value.headers == {"Retry-After": "60"}


def test_check_workflow_rate_limit_falls_back_to_client_host(monkeypatch):
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    monkeypatch.setattr(rate_limiter, "workflow_limiter", limiter)
    request = SimpleNamespace(client=SimpleNamespace(host="10.0.0.8"))

    rate_limiter.check_workflow_rate_limit(request)

    with pytest.raises(HTTPException) as exc:
        rate_limiter.check_workflow_rate_limit(request)

    assert exc.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
