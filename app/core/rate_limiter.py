"""PostgreSQL 固定窗口限流器。

生产策略：
- 所有限流状态必须写入 PostgreSQL。
- PostgreSQL 不可用时直接抛错，不允许降级到进程内存。
- 这样可以保证多进程、多实例部署时看到同一份限流状态。
"""

from __future__ import annotations

import time
from typing import Protocol

from fastapi import HTTPException, Request, status
from sqlalchemy import create_engine, text

from app.core.config import get_settings


class RateLimiterBackend(Protocol):
    _max: int
    _window: float

    def is_allowed(self, key: str) -> bool:
        ...

    def remaining(self, key: str) -> int:
        ...

    def evict_expired(self) -> int:
        ...


class PostgreSQLRateLimiter:
    """PostgreSQL 固定窗口限流器。

    窗口内每次请求通过 upsert 递增计数；窗口结束后按 expires_at 清理。
    """

    def __init__(
        self,
        database_url: str,
        max_requests: int = 20,
        window_seconds: float = 60.0,
        namespace: str = "default",
    ) -> None:
        if not database_url:
            raise RuntimeError("速率限制必须配置 DATABASE_URL/POSTGRES_URL，生产模式不允许降级到内存限流。")
        self._database_url = database_url
        self._max = max_requests
        self._window = window_seconds
        self._namespace = namespace
        self._engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=1800, future=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS rate_limit_windows (
                    namespace VARCHAR(64) NOT NULL,
                    rate_key VARCHAR(256) NOT NULL,
                    window_id BIGINT NOT NULL,
                    request_count INT NOT NULL DEFAULT 0,
                    expires_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (namespace, rate_key, window_id)
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_rate_limit_expires ON rate_limit_windows (expires_at)"))

    def _window_id(self) -> int:
        return int(time.time() // self._window)

    def _ttl_seconds(self) -> int:
        return max(1, int(self._window) + 1)

    def _expires_at(self):
        from datetime import datetime, timedelta, timezone

        return datetime.now(tz=timezone.utc).replace(tzinfo=None) + timedelta(seconds=self._ttl_seconds())

    def is_allowed(self, key: str) -> bool:
        window_id = self._window_id()
        with self._engine.begin() as conn:
            conn.execute(text("DELETE FROM rate_limit_windows WHERE expires_at <= CURRENT_TIMESTAMP"))
            count = conn.execute(
                text("""
                    INSERT INTO rate_limit_windows (
                        namespace, rate_key, window_id, request_count, expires_at
                    ) VALUES (
                        :namespace, :rate_key, :window_id, 1, :expires_at
                    )
                    ON CONFLICT (namespace, rate_key, window_id) DO UPDATE SET
                        request_count=rate_limit_windows.request_count + 1
                    RETURNING request_count
                """),
                {
                    "namespace": self._namespace,
                    "rate_key": key,
                    "window_id": window_id,
                    "expires_at": self._expires_at(),
                },
            ).scalar_one()
        return int(count) <= self._max

    def remaining(self, key: str) -> int:
        with self._engine.connect() as conn:
            value = conn.execute(
                text("""
                    SELECT request_count
                    FROM rate_limit_windows
                    WHERE namespace=:namespace
                      AND rate_key=:rate_key
                      AND window_id=:window_id
                      AND expires_at > CURRENT_TIMESTAMP
                """),
                {
                    "namespace": self._namespace,
                    "rate_key": key,
                    "window_id": self._window_id(),
                },
            ).scalar()
        count = int(value or 0)
        return max(0, self._max - count)

    def evict_expired(self) -> int:
        with self._engine.begin() as conn:
            result = conn.execute(text("DELETE FROM rate_limit_windows WHERE expires_at <= CURRENT_TIMESTAMP"))
        return int(result.rowcount or 0)


def create_rate_limiter(
    *,
    namespace: str,
    max_requests: int,
    window_seconds: float,
) -> RateLimiterBackend:
    """创建 PostgreSQL 限流器。生产环境 PostgreSQL 不可用时直接失败。"""

    database_url = get_settings().effective_database_url.strip()
    if not database_url:
        raise RuntimeError("速率限制必须配置 DATABASE_URL/POSTGRES_URL，生产模式不允许降级到内存限流。")
    return PostgreSQLRateLimiter(
        database_url=database_url,
        max_requests=max_requests,
        window_seconds=window_seconds,
        namespace=namespace,
    )


agent_limiter: RateLimiterBackend | None = None
workflow_limiter: RateLimiterBackend | None = None


def _agent_limiter() -> RateLimiterBackend:
    global agent_limiter
    if agent_limiter is None:
        agent_limiter = create_rate_limiter(namespace="agent", max_requests=20, window_seconds=60.0)
    return agent_limiter


def _workflow_limiter() -> RateLimiterBackend:
    global workflow_limiter
    if workflow_limiter is None:
        workflow_limiter = create_rate_limiter(namespace="workflow", max_requests=10, window_seconds=60.0)
    return workflow_limiter


def check_agent_rate_limit(request: Request, session_id: str) -> None:
    """FastAPI 依赖项：检查 Agent 接口限流，超限抛 429。"""
    key = f"agent:{session_id}"
    limiter = _agent_limiter()
    if not limiter.is_allowed(key):
        remaining = limiter.remaining(key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"请求过于频繁，每分钟最多 {limiter._max} 次。剩余：{remaining}",
            headers={"Retry-After": "60"},
        )

def check_workflow_rate_limit(request: Request, session_id: str | None = None) -> None:
    """FastAPI 依赖项：检查 Workflow 接口限流，超限抛 429。"""
    key = f"workflow:{session_id or request.client.host}"
    limiter = _workflow_limiter()
    if not limiter.is_allowed(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"请求过于频繁，每分钟最多 {limiter._max} 次。",
            headers={"Retry-After": "60"},
        )
