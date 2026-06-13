"""Redis 固定窗口限流器。

生产策略：
- 所有限流状态必须写入 Redis。
- Redis 不可用时直接抛错，不允许降级到进程内存。
- 这样可以保证多进程、多实例部署时看到同一份限流状态。
"""

from __future__ import annotations

import time
from typing import Protocol

from fastapi import HTTPException, Request, status

from app.core.config import get_settings


_REDIS_KEY_PREFIX = "multiship:rate_limit"


class RateLimiterBackend(Protocol):
    _max: int
    _window: float

    def is_allowed(self, key: str) -> bool:
        ...

    def remaining(self, key: str) -> int:
        ...

    def evict_expired(self) -> int:
        ...


class RedisRateLimiter:
    """Redis 固定窗口限流器。

    Redis key 形如：
        multiship:rate_limit:agent:session-1:29384711

    最后一段是按窗口大小切分出的窗口 ID。窗口内每次请求执行 INCR，
    首次写入时设置 EXPIRE；窗口结束后 Redis 自动清理。
    """

    def __init__(
        self,
        redis_url: str,
        max_requests: int = 20,
        window_seconds: float = 60.0,
        namespace: str = "default",
    ) -> None:
        self._redis_url = redis_url
        self._max = max_requests
        self._window = window_seconds
        self._namespace = namespace
        self._client = self._create_client(redis_url)

    @staticmethod
    def _create_client(redis_url: str):
        import redis

        client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        return client

    def _window_id(self) -> int:
        return int(time.time() // self._window)

    def _make_key(self, key: str) -> str:
        return f"{_REDIS_KEY_PREFIX}:{self._namespace}:{key}:{self._window_id()}"

    def _ttl_seconds(self) -> int:
        return max(1, int(self._window) + 1)

    def _redis(self, operation):
        try:
            return operation(self._client)
        except Exception as exc:
            raise RuntimeError(f"限流 Redis 操作失败，生产模式拒绝降级：{exc}") from exc

    def is_allowed(self, key: str) -> bool:
        redis_key = self._make_key(key)

        def _operation(client) -> int:
            count = int(client.incr(redis_key))
            if count == 1:
                client.expire(redis_key, self._ttl_seconds())
            return count

        return self._redis(_operation) <= self._max

    def remaining(self, key: str) -> int:
        redis_key = self._make_key(key)

        def _operation(client) -> int:
            value = client.get(redis_key)
            count = int(value) if value is not None else 0
            return max(0, self._max - count)

        return self._redis(_operation)

    def evict_expired(self) -> int:
        """Redis 依赖 key TTL 自动过期。"""
        return 0


def create_rate_limiter(
    *,
    namespace: str,
    max_requests: int,
    window_seconds: float,
) -> RateLimiterBackend:
    """创建 Redis 限流器。生产环境 Redis 不可用时直接失败。"""

    redis_url = get_settings().redis_url.strip()
    if not redis_url:
        raise RuntimeError("速率限制必须配置 REDIS_URL，生产模式不允许降级到内存限流。")
    return RedisRateLimiter(
        redis_url=redis_url,
        max_requests=max_requests,
        window_seconds=window_seconds,
        namespace=namespace,
    )


# Agent 接口：每 session 每分钟 20 次。
agent_limiter = create_rate_limiter(namespace="agent", max_requests=20, window_seconds=60.0)
# Workflow 接口：每 session 每分钟 10 次。
workflow_limiter = create_rate_limiter(namespace="workflow", max_requests=10, window_seconds=60.0)


def check_agent_rate_limit(request: Request, session_id: str) -> None:
    """FastAPI 依赖项：检查 Agent 接口限流，超限抛 429。"""
    key = f"agent:{session_id}"
    if not agent_limiter.is_allowed(key):
        remaining = agent_limiter.remaining(key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"请求过于频繁，每分钟最多 {agent_limiter._max} 次。剩余：{remaining}",
            headers={"Retry-After": "60"},
        )


def check_workflow_rate_limit(request: Request, session_id: str | None = None) -> None:
    """FastAPI 依赖项：检查 Workflow 接口限流，超限抛 429。"""
    key = f"workflow:{session_id or request.client.host}"
    if not workflow_limiter.is_allowed(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"请求过于频繁，每分钟最多 {workflow_limiter._max} 次。",
            headers={"Retry-After": "60"},
        )
