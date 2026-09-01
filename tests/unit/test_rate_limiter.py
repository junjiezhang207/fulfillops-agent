from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status

from app.core import rate_limiter
from app.core.rate_limiter import PostgreSQLRateLimiter


class FakeScalarResult:
    def __init__(self, value=0, rowcount=0):
        self.value = value
        self.rowcount = rowcount

    def scalar_one(self):
        return self.value

    def scalar(self):
        return self.value


class FakeConn:
    def __init__(self):
        self.counts = {}

    def execute(self, statement, params=None):
        sql = str(statement)
        params = params or {}
        if "INSERT INTO rate_limit_windows" in sql:
            key = (params["namespace"], params["rate_key"], params["window_id"])
            self.counts[key] = self.counts.get(key, 0) + 1
            return FakeScalarResult(self.counts[key])
        if "SELECT request_count" in sql:
            key = (params["namespace"], params["rate_key"], params["window_id"])
            return FakeScalarResult(self.counts.get(key, 0))
        return FakeScalarResult(rowcount=0)


class FakeBegin:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeEngine:
    def __init__(self):
        self.conn = FakeConn()

    def begin(self):
        return FakeBegin(self.conn)

    def connect(self):
        return FakeBegin(self.conn)


class FixedLimiter:
    def __init__(self, max_requests: int):
        self._max = max_requests
        self._window = 60
        self.calls: dict[str, int] = {}

    def is_allowed(self, key: str) -> bool:
        self.calls[key] = self.calls.get(key, 0) + 1
        return self.calls[key] <= self._max

    def remaining(self, key: str) -> int:
        return max(0, self._max - self.calls.get(key, 0))

    def evict_expired(self) -> int:
        return 0


def test_create_rate_limiter_requires_database_url(monkeypatch):
    monkeypatch.setattr(rate_limiter, "get_settings", lambda: SimpleNamespace(effective_database_url=""))

    with pytest.raises(RuntimeError, match="必须配置 DATABASE_URL"):
        rate_limiter.create_rate_limiter(namespace="test", max_requests=1, window_seconds=60)


def test_postgres_rate_limiter_uses_fixed_window(monkeypatch):
    fake_engine = FakeEngine()
    monkeypatch.setattr(rate_limiter, "create_engine", lambda *args, **kwargs: fake_engine)
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 120.0)
    limiter = PostgreSQLRateLimiter("postgresql+psycopg://unit-test", max_requests=2, window_seconds=60, namespace="test")

    assert limiter.is_allowed("session-1") is True
    assert limiter.remaining("session-1") == 1
    assert limiter.is_allowed("session-1") is True
    assert limiter.remaining("session-1") == 0
    assert limiter.is_allowed("session-1") is False


def test_check_agent_rate_limit_raises_429_when_session_exceeds_quota(monkeypatch):
    limiter = FixedLimiter(max_requests=1)
    monkeypatch.setattr(rate_limiter, "agent_limiter", limiter)
    request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))

    rate_limiter.check_agent_rate_limit(request, "s1")

    with pytest.raises(HTTPException) as exc:
        rate_limiter.check_agent_rate_limit(request, "s1")

    assert exc.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert exc.value.headers == {"Retry-After": "60"}


def test_check_workflow_rate_limit_falls_back_to_client_host(monkeypatch):
    limiter = FixedLimiter(max_requests=1)
    monkeypatch.setattr(rate_limiter, "workflow_limiter", limiter)
    request = SimpleNamespace(client=SimpleNamespace(host="10.0.0.8"))

    rate_limiter.check_workflow_rate_limit(request)

    with pytest.raises(HTTPException) as exc:
        rate_limiter.check_workflow_rate_limit(request)

    assert exc.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
