"""Agent 工具结果 TTL 缓存。

本模块负责缓存“稳定、可复用”的工具结果，减少 Agent 在同一轮或短时间内
反复调用下游服务。它不是业务数据库，也不应该缓存强实时数据。是否启用缓存
由 ``AgentService`` 和 ``tool_wrapper.py`` 决定：订单、库存、履约方案这类实时
数据默认不缓存；知识规则、替代 SKU 这类目录数据可以短时间缓存。

主要组成：
1. ``RedisToolResultCache``：使用 Redis 的 TTL 缓存，Redis 不可用时直接失败。
3. ``GLOBAL_SCOPE``：跨会话共享稳定数据的 scope。
4. ``TOOL_TTLS``：为不同工具设置不同过期时间。
5. ``get`` / ``set``：按 scope、工具名、参数 hash 读写缓存。
6. ``invalidate_scope`` / ``invalidate_tool``：支持按会话或工具清理缓存。
7. ``stats``：返回缓存命中、条目数等简单统计。

缓存 key 设计：
``(scope, tool_name, args_hash)``。
这样同一个工具、同一组参数才会命中，避免不同订单或不同 SKU 串数据。
"""

import hashlib
import json
import logging
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from app.agents.tools.contracts import TOOL_MANIFESTS

logger = logging.getLogger(__name__)

# 跨会话共享的全局缓存 scope 标识
GLOBAL_SCOPE = "__global__"

# 缓存仅用于目录/规则类工具（稳定数据），实时数据工具在 agent_service 里已禁用缓存
# 实时工具（check_inventory / search_warehouse_inventory / analyze_order /
#           generate_fulfillment_plan）不在此表中，它们的 enable_cache=False
TOOL_TTLS: dict[str, float] = {
    "retrieve_knowledge":  TOOL_MANIFESTS["retrieve_knowledge"].ttl_seconds or 600,
    "find_substitute_sku": TOOL_MANIFESTS["find_substitute_sku"].ttl_seconds or 300,
}
_DEFAULT_TTL: float = 120
_REDIS_KEY_PREFIX = "multiship:tool-cache:v1"

CACHE_INVALIDATION_EVENTS: dict[str, tuple[str, ...]] = {
    "knowledge_updated": ("retrieve_knowledge",),
    "knowledge_reindexed": ("retrieve_knowledge",),
    "catalog_updated": ("find_substitute_sku",),
    "substitute_sku_updated": ("find_substitute_sku",),
    "all_reference_data_updated": ("retrieve_knowledge", "find_substitute_sku"),
}


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


class RedisToolResultCache:
    """Redis TTL 缓存。

    这里不使用 RediSearch，只使用普通 Redis 的 GET/SETEX/SCAN/DEL。
    因此它和 LangGraph RedisSaver 不同：工具缓存只需要普通 Redis 即可。
    生产模式下 Redis 是强依赖，不能降级到进程内存，否则多实例缓存不一致。
    """

    def __init__(
        self,
        redis_url: str,
        tool_ttls: dict[str, float] | None = None,
    ) -> None:
        self._redis_url = redis_url
        self._ttls = tool_ttls or TOOL_TTLS
        self._hits = 0
        self._misses = 0
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

    def _try_redis(self, operation):
        try:
            return operation(self._client)
        except Exception as exc:
            raise RuntimeError(f"工具缓存 Redis 操作失败，生产模式拒绝降级：{exc}") from exc

    def _delete_by_prefix(self, prefix: str) -> int:
        return self._delete_by_pattern(f"{prefix}*")

    def _delete_by_pattern(self, pattern: str) -> int:
        def _operation(client) -> int:
            deleted = 0
            batch: list[str] = []
            for key in client.scan_iter(match=pattern, count=100):
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
        """从 Redis 读取工具结果；Redis 不可用时直接失败。"""
        key = self._make_key(scope, tool_name, kwargs)

        sentinel = object()

        def _operation(client) -> str | object:
            value = client.get(key)
            return value if value is not None else sentinel

        redis_value = self._try_redis(_operation)
        if redis_value is sentinel:
            self._misses += 1
            return None
        self._hits += 1
        return str(redis_value)

    def set(self, scope: str, tool_name: str, kwargs: dict, value: str) -> None:
        """写入 Redis；Redis 不可用时直接失败。"""
        key = self._make_key(scope, tool_name, kwargs)
        ttl = self._ttl_for(tool_name)

        self._try_redis(lambda client: bool(client.setex(key, ttl, value)))

    def invalidate_scope(self, scope: str) -> int:
        redis_deleted = self._delete_by_prefix(self._scope_prefix(scope))
        return redis_deleted

    def invalidate_tool(self, tool_name: str, scope: str | None = GLOBAL_SCOPE) -> int:
        if scope is None:
            redis_deleted = self._delete_by_pattern(f"{_REDIS_KEY_PREFIX}:*:{tool_name}:*")
        else:
            redis_deleted = self._delete_by_prefix(self._tool_prefix(scope, tool_name))
        return redis_deleted

    def invalidate_event(self, event_type: str, scope: str | None = GLOBAL_SCOPE) -> dict[str, int]:
        tools = CACHE_INVALIDATION_EVENTS.get(event_type, ())
        return {tool: self.invalidate_tool(tool, scope) for tool in tools}

    def evict_expired(self) -> int:
        """Redis 依赖 key TTL 自动过期，无需本地清理。"""
        return 0

    @property
    def stats(self) -> dict:
        redis_entries = self._count_redis_entries()
        total = self._hits + self._misses
        return {
            "backend": "redis",
            "cached_entries": redis_entries or 0,
            "redis_entries": redis_entries or 0,
            "total_hits": self._hits,
            "total_misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }


# ── 全局单例 ─────────────────────────────────────────────────────────────────

_cache: Any | None = None


# 全局单例让不同工具包装器共享命中；生产模式下 Redis 不可用直接失败。
def get_tool_cache():
    """获取全局工具缓存单例。"""
    global _cache
    if _cache is None:
        from app.core.config import get_settings

        redis_url = get_settings().redis_url.strip()
        if not redis_url:
            raise RuntimeError("工具缓存必须配置 REDIS_URL，生产模式不允许降级到内存缓存。")
        _cache = RedisToolResultCache(redis_url)
        logger.info("工具缓存：已启用 Redis (%s)", _safe_redis_label(redis_url))
    return _cache
