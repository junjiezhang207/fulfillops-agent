"""Redis 固定窗口限流器 — 按 session_id 或 IP 限制请求频率。

算法：Redis 固定窗口计数器（INCR + EXPIRE），Redis 不可用时降级内存计数器。

大厂标准：
  - AI 接口每次调用都可能花费数角到数元，必须有限流保护
  - 超限返回 429 Too Many Requests，不触发 LLM 调用
  - 默认：每个 session 每分钟最多 20 次请求
  - Redis 让多进程 / 多实例共享同一份限流状态
"""

from __future__ import annotations

import logging
import time
from threading import RLock
from dataclasses import dataclass, field
from typing import Protocol

from fastapi import HTTPException, Request, status

from app.core.config import get_settings


logger = logging.getLogger(__name__)
_REDIS_KEY_PREFIX = "multiship:rate_limit"
_REDIS_RETRY_INTERVAL_SECONDS = 30.0


class RateLimiterBackend(Protocol):
    _max: int
    _window: float

    def is_allowed(self, key: str) -> bool:
        ...

    def remaining(self, key: str) -> int:
        ...

    def evict_expired(self) -> int:
        ...


@dataclass
class _WindowRecord:
    count: int = 0
    window_start: float = field(default_factory=time.monotonic)


class RateLimiter:
    """固定窗口计数器限流器（线程安全）。

    Args:
        max_requests: 窗口内最大请求数
        window_seconds: 窗口大小（秒）
    """

    def __init__(self, max_requests: int = 20, window_seconds: float = 60.0) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._store: dict[str, _WindowRecord] = {}
        self._lock = RLock()

    def is_allowed(self, key: str) -> bool:
        """检查指定 key 是否在限流范围内。"""
        now = time.monotonic()
        with self._lock:
            rec = self._store.get(key)
            if rec is None or now - rec.window_start >= self._window:
                self._store[key] = _WindowRecord(count=1, window_start=now)
                return True
            if rec.count >= self._max:
                return False
            rec.count += 1
            return True

    def remaining(self, key: str) -> int:
        """返回当前窗口内剩余可用次数。"""
        with self._lock:
            rec = self._store.get(key)
            if rec is None:
                return self._max
            now = time.monotonic()
            if now - rec.window_start >= self._window:
                return self._max
            return max(0, self._max - rec.count)

    def evict_expired(self) -> int:
        """清理过期记录（可由定时任务调用）。"""
        now = time.monotonic()
        with self._lock:
            expired = [k for k, v in self._store.items()
                       if now - v.window_start >= self._window]
            for k in expired:
                del self._store[k]
            return len(expired)


class RedisRateLimiter:
    """Redis 固定窗口限流器，Redis 故障时自动降级为内存限流。

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
        fallback: RateLimiter | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._max = max_requests
        self._window = window_seconds
        self._namespace = namespace
        self._fallback = fallback or RateLimiter(max_requests=max_requests, window_seconds=window_seconds)
        self._redis_healthy = True
        self._next_retry_at = 0.0
        self._client = self._create_client(redis_url)

    @staticmethod
    def _create_client(redis_url: str):
        import redis

        client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        client.ping()
        return client

    def _window_id(self) -> int:
        return int(time.time() // self._window)

    def _make_key(self, key: str) -> str:
        return f"{_REDIS_KEY_PREFIX}:{self._namespace}:{key}:{self._window_id()}"

    def _ttl_seconds(self) -> int:
        return max(1, int(self._window) + 1)

    def _can_try_redis(self) -> bool:
        return self._redis_healthy or time.monotonic() >= self._next_retry_at

    def _mark_redis_success(self) -> None:
        self._redis_healthy = True
        self._next_retry_at = 0.0

    def _mark_redis_failure(self, exc: Exception) -> None:
        if self._redis_healthy:
            logger.warning("速率限制 Redis 不可用，临时降级到内存限流：%s", exc)
        self._redis_healthy = False
        self._next_retry_at = time.monotonic() + _REDIS_RETRY_INTERVAL_SECONDS

    def _try_redis(self, operation):
        if not self._can_try_redis():
            return None
        try:
            value = operation(self._client)
            self._mark_redis_success()
            return value
        except Exception as exc:
            self._mark_redis_failure(exc)
            return None

    def is_allowed(self, key: str) -> bool:
        redis_key = self._make_key(key)

        def _operation(client) -> int:
            count = int(client.incr(redis_key))
            if count == 1:
                client.expire(redis_key, self._ttl_seconds())
            return count

        count = self._try_redis(_operation)
        if count is None:
            return self._fallback.is_allowed(key)
        return count <= self._max

    def remaining(self, key: str) -> int:
        redis_key = self._make_key(key)

        def _operation(client) -> int:
            value = client.get(redis_key)
            count = int(value) if value is not None else 0
            return max(0, self._max - count)

        remaining = self._try_redis(_operation)
        if remaining is None:
            return self._fallback.remaining(key)
        return remaining

    def evict_expired(self) -> int:
        """Redis 依赖 key TTL 自动过期；这里只清理内存兜底记录。"""
        return self._fallback.evict_expired()


def _safe_redis_label(redis_url: str) -> str:
    """生成适合日志展示的 Redis 地址，避免泄露密码。"""
    try:
        from urllib.parse import urlsplit

        parts = urlsplit(redis_url)
        host = parts.hostname or "localhost"
        port = f":{parts.port}" if parts.port else ""
        db = parts.path or ""
        return f"{parts.scheme}://{host}{port}{db}"
    except Exception:
        return "<redis-url>"


def create_rate_limiter(
    *,
    namespace: str,
    max_requests: int,
    window_seconds: float,
) -> RateLimiterBackend:
    """创建限流器：优先 Redis，失败时降级内存。"""
    redis_url = get_settings().redis_url.strip()
    if not redis_url:
        logger.info("速率限制：未配置 Redis URL，使用内存限流。")
        return RateLimiter(max_requests=max_requests, window_seconds=window_seconds)

    try:
        limiter = RedisRateLimiter(
            redis_url=redis_url,
            max_requests=max_requests,
            window_seconds=window_seconds,
            namespace=namespace,
        )
        logger.info("速率限制：已启用 Redis (%s)，namespace=%s", _safe_redis_label(redis_url), namespace)
        return limiter
    except Exception as exc:
        logger.warning("速率限制：Redis 不可用，使用内存限流：%s", exc)
        return RateLimiter(max_requests=max_requests, window_seconds=window_seconds)


# ── 全局实例 ──────────────────────────────────────────────────────────────────

# Agent 接口：每 session 每分钟 20 次（防止脚本刷接口）
agent_limiter = create_rate_limiter(namespace="agent", max_requests=20, window_seconds=60.0)
# Workflow 接口：每 session 每分钟 10 次（workflow 调用更重）
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
