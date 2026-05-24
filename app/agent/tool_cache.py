"""文件作用摘要：Agent 工具结果的 TTL 缓存。

这个文件负责缓存“稳定、可复用”的工具结果，减少 Agent 在同一轮或短时间内
反复调用下游服务。它不是业务数据库，也不应该缓存强实时数据。是否启用缓存
由 ``AgentService`` 和 ``tool_wrapper.py`` 决定：订单、库存、履约方案这类实时
数据默认不缓存；知识规则、替代 SKU 这类目录数据可以短时间缓存。

主要做的事：
1. ``ToolResultCache``：线程安全的内存 TTL 兜底缓存。
2. ``RedisToolResultCache``：优先使用 Redis 的 TTL 缓存，Redis 不可用时自动降级内存。
3. ``GLOBAL_SCOPE``：跨会话共享稳定数据的 scope。
4. ``TOOL_TTLS``：为不同工具设置不同过期时间。
5. ``get`` / ``set``：按 scope、工具名、参数 hash 读写缓存。
6. ``invalidate_scope`` / ``invalidate_tool``：支持按会话或工具清理缓存。
7. ``stats``：返回缓存命中、条目数等简单统计。

缓存 key 设计：
``(scope, tool_name, args_hash)``。
这样同一个工具、同一组参数才会命中，避免不同订单或不同 SKU 串数据。

学习时先看：
1. ``TOOL_TTLS``：哪些工具允许缓存。
2. ``_make_key``：缓存 key 怎么生成。
3. ``get`` / ``set``：命中和写入逻辑。
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

# 跨会话共享的全局缓存 scope 标识
GLOBAL_SCOPE = "__global__"

# 缓存仅用于目录/规则类工具（稳定数据），实时数据工具在 agent_service 里已禁用缓存
# 实时工具（check_inventory / search_warehouse_inventory / analyze_order /
#           generate_fulfillment_plan）不在此表中，它们的 enable_cache=False
TOOL_TTLS: dict[str, float] = {
    "retrieve_knowledge":  600,   # 10 min — 履约规则极少变动
    "find_substitute_sku": 300,   # 5 min  — 替代关系是目录数据（不含库存量）
}
_DEFAULT_TTL: float = 120
_REDIS_KEY_PREFIX = "multiship:tool-cache:v1"
_REDIS_RETRY_INTERVAL_SECONDS = 30.0


def _safe_redis_label(redis_url: str) -> str:
    """生成适合日志展示的 Redis 地址，避免泄露密码。"""
    try:
        parts = urlsplit(redis_url)
        netloc = parts.hostname or ""
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        if parts.username:
            netloc = f"{parts.username}:***@{netloc}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except Exception:
        return "<redis-url>"


# 面试官可能问：为什么工具缓存 key 要包含参数 hash？
# 回答：同一个工具传不同订单号/SKU 时结果完全不同。把参数参与 key 计算，
# 可以避免“查 A 订单却命中 B 订单结果”的严重业务串数据问题。
@dataclass
class CacheEntry:
    value: str
    created_at: float
    ttl: float
    hits: int = field(default=0)

    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl

    def touch(self) -> str:
        self.hits += 1
        return self.value


# 面试官可能问：为什么这里只缓存目录/规则类工具，不缓存库存？
# 回答：库存和订单状态变化快，缓存可能让 Agent 基于旧数据做错误决策；
# 知识规则、替代 SKU 这类低频变化数据短时间缓存更安全，能减少重复调用。
class ToolResultCache:
    """线程安全的 TTL 缓存，供 wrap_tool_with_resilience 使用。

    典型场景：
      同一会话内 Agent 多次调用 check_inventory("SO123")，
      第一次真实查询，之后 60s 内直接命中缓存，节省 LLM → 工具 往返。

    使用方式：
        cache = get_tool_cache()
        hit = cache.get(GLOBAL_SCOPE, "check_inventory", {"order_id": "SO123"})
        if hit is None:
            result = call_real_service(...)
            cache.set(GLOBAL_SCOPE, "check_inventory", {"order_id": "SO123"}, result)
    """

    def __init__(self, tool_ttls: dict[str, float] | None = None):
        self._store: dict[str, CacheEntry] = {}
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._ttls = tool_ttls or TOOL_TTLS

    # ── key 生成 ────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(scope: str, tool_name: str, kwargs: dict) -> str:
        payload = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
        args_hash = hashlib.sha256(payload.encode()).hexdigest()[:12]
        return f"{scope}:{tool_name}:{args_hash}"

    # ── 读写 ────────────────────────────────────────────────────────────────

    def get(self, scope: str, tool_name: str, kwargs: dict) -> str | None:
        """命中返回缓存值，未命中或已过期返回 None。"""
        key = self._make_key(scope, tool_name, kwargs)
        with self._lock:
            entry = self._store.get(key)
            if entry is None or entry.is_expired():
                if entry is not None:
                    del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return entry.touch()

    def set(self, scope: str, tool_name: str, kwargs: dict, value: str) -> None:
        """写入缓存，TTL 由 TOOL_TTLS 决定。"""
        key = self._make_key(scope, tool_name, kwargs)
        ttl = self._ttls.get(tool_name, _DEFAULT_TTL)
        with self._lock:
            self._store[key] = CacheEntry(
                value=value, created_at=time.time(), ttl=ttl
            )

    # ── 失效 ────────────────────────────────────────────────────────────────

    def invalidate_scope(self, scope: str) -> int:
        """删除某个会话的全部缓存条目（会话结束时调用）。"""
        prefix = f"{scope}:"
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]
            return len(keys)

    def invalidate_tool(self, tool_name: str, scope: str = GLOBAL_SCOPE) -> int:
        """删除某工具的全部缓存（数据发生更新时调用）。"""
        prefix = f"{scope}:{tool_name}:"
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]
            return len(keys)

    def evict_expired(self) -> int:
        """清理过期条目，可由后台定时任务调用。"""
        with self._lock:
            expired = [k for k, v in self._store.items() if v.is_expired()]
            for k in expired:
                del self._store[k]
            return len(expired)

    # ── 统计 ────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "backend": "memory",
                "cached_entries": len(self._store),
                "total_hits": self._hits,
                "total_misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }


class RedisToolResultCache:
    """Redis TTL 缓存，Redis 不可用时自动降级到内存缓存。

    这里不使用 RediSearch，只使用普通 Redis 的 GET/SETEX/SCAN/DEL。
    因此它和 LangGraph RedisSaver 不同：工具缓存只需要普通 Redis 即可。
    """

    def __init__(
        self,
        redis_url: str,
        tool_ttls: dict[str, float] | None = None,
        fallback: ToolResultCache | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._ttls = tool_ttls or TOOL_TTLS
        self._fallback = fallback or ToolResultCache(tool_ttls=self._ttls)
        self._hits = 0
        self._misses = 0
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

    @staticmethod
    def _args_hash(kwargs: dict) -> str:
        payload = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def _make_key(self, scope: str, tool_name: str, kwargs: dict) -> str:
        return f"{_REDIS_KEY_PREFIX}:{scope}:{tool_name}:{self._args_hash(kwargs)}"

    def _scope_prefix(self, scope: str) -> str:
        return f"{_REDIS_KEY_PREFIX}:{scope}:"

    def _tool_prefix(self, scope: str, tool_name: str) -> str:
        return f"{_REDIS_KEY_PREFIX}:{scope}:{tool_name}:"

    def _ttl_for(self, tool_name: str) -> int:
        return max(1, int(self._ttls.get(tool_name, _DEFAULT_TTL)))

    def _can_try_redis(self) -> bool:
        return self._redis_healthy or time.monotonic() >= self._next_retry_at

    def _mark_redis_success(self) -> None:
        self._redis_healthy = True
        self._next_retry_at = 0.0

    def _mark_redis_failure(self, exc: Exception) -> None:
        if self._redis_healthy:
            logger.warning("工具缓存 Redis 不可用，临时降级到内存缓存：%s", exc)
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

    def _delete_by_prefix(self, prefix: str) -> int:
        def _operation(client) -> int:
            deleted = 0
            batch: list[str] = []
            for key in client.scan_iter(match=f"{prefix}*", count=100):
                batch.append(key)
                if len(batch) >= 100:
                    deleted += int(client.delete(*batch))
                    batch.clear()
            if batch:
                deleted += int(client.delete(*batch))
            return deleted

        return self._try_redis(_operation) or 0

    def _count_redis_entries(self) -> int | None:
        def _operation(client) -> int:
            return sum(1 for _ in client.scan_iter(match=f"{_REDIS_KEY_PREFIX}:*", count=100))

        return self._try_redis(_operation)

    def get(self, scope: str, tool_name: str, kwargs: dict) -> str | None:
        """优先从 Redis 读取；Redis 不可用时读内存兜底缓存。"""
        key = self._make_key(scope, tool_name, kwargs)

        sentinel = object()

        def _operation(client) -> str | object:
            value = client.get(key)
            return value if value is not None else sentinel

        redis_value = self._try_redis(_operation)
        if redis_value is not None:
            if redis_value is sentinel:
                self._misses += 1
                return None
            self._hits += 1
            return str(redis_value)

        return self._fallback.get(scope, tool_name, kwargs)

    def set(self, scope: str, tool_name: str, kwargs: dict, value: str) -> None:
        """优先写 Redis；Redis 不可用时写内存兜底缓存。"""
        key = self._make_key(scope, tool_name, kwargs)
        ttl = self._ttl_for(tool_name)

        wrote_to_redis = self._try_redis(
            lambda client: bool(client.setex(key, ttl, value))
        )
        if not wrote_to_redis:
            self._fallback.set(scope, tool_name, kwargs, value)

    def invalidate_scope(self, scope: str) -> int:
        redis_deleted = self._delete_by_prefix(self._scope_prefix(scope))
        fallback_deleted = self._fallback.invalidate_scope(scope)
        return redis_deleted + fallback_deleted

    def invalidate_tool(self, tool_name: str, scope: str = GLOBAL_SCOPE) -> int:
        redis_deleted = self._delete_by_prefix(self._tool_prefix(scope, tool_name))
        fallback_deleted = self._fallback.invalidate_tool(tool_name, scope)
        return redis_deleted + fallback_deleted

    def evict_expired(self) -> int:
        """Redis 依赖 key TTL 自动过期；这里只清理内存兜底缓存。"""
        return self._fallback.evict_expired()

    @property
    def stats(self) -> dict:
        fallback_stats = self._fallback.stats
        redis_entries = self._count_redis_entries()
        total_hits = self._hits + fallback_stats["total_hits"]
        total_misses = self._misses + fallback_stats["total_misses"]
        total = total_hits + total_misses
        return {
            "backend": "redis" if self._redis_healthy else "memory_fallback",
            "cached_entries": (redis_entries or 0) + fallback_stats["cached_entries"],
            "redis_entries": redis_entries or 0,
            "fallback_entries": fallback_stats["cached_entries"],
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": round(total_hits / total, 3) if total else 0.0,
        }


# ── 全局单例 ─────────────────────────────────────────────────────────────────

_cache: Any | None = None


# 面试官可能问：为什么缓存用全局单例？
# 回答：复用同一个对象才能让不同工具包装器共享命中。当前优先使用 Redis，
# 这样多实例可以共享缓存；Redis 不可用时降级到内存，保证本地开发不崩。
def get_tool_cache():
    """获取全局工具缓存单例。"""
    global _cache
    if _cache is None:
        try:
            from app.core.config import get_settings

            redis_url = get_settings().redis_url.strip()
            if not redis_url:
                raise ValueError("未配置 REDIS_URL")
            _cache = RedisToolResultCache(redis_url)
            logger.info("工具缓存：已启用 Redis (%s)", _safe_redis_label(redis_url))
        except Exception as exc:
            logger.warning("工具缓存：Redis 不可用，使用内存缓存：%s", exc)
            _cache = ToolResultCache()
    return _cache
