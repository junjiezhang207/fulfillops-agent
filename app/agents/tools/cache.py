"""Agent 工具结果 TTL 缓存。

本模块负责缓存“稳定、可复用”的工具结果，减少 Agent 在同一轮或短时间内
反复调用下游服务。它不是业务数据库，也不应该缓存强实时数据。是否启用缓存
由 ``AgentService`` 和 ``tool_wrapper.py`` 决定：订单、库存、履约方案这类实时
数据默认不缓存；知识规则、替代 SKU 这类目录数据可以短时间缓存。

主要组成：
1. ``PostgreSQLToolResultCache``：使用 PostgreSQL 的 TTL 缓存，PostgreSQL 不可用时直接失败。
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

from app.agents.tools.contracts import TOOL_MANIFESTS
from sqlalchemy import create_engine, text

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
_CACHE_SCHEMA_VERSION = "v1"

CACHE_INVALIDATION_EVENTS: dict[str, tuple[str, ...]] = {
    "knowledge_updated": ("retrieve_knowledge",),
    "knowledge_reindexed": ("retrieve_knowledge",),
    "catalog_updated": ("find_substitute_sku",),
    "substitute_sku_updated": ("find_substitute_sku",),
    "all_reference_data_updated": ("retrieve_knowledge", "find_substitute_sku"),
}


class PostgreSQLToolResultCache:
    """PostgreSQL TTL 缓存。

    生产模式下 PostgreSQL 是强依赖，不能降级到进程内存，否则多实例缓存不一致。
    """

    def __init__(
        self,
        database_url: str,
        tool_ttls: dict[str, float] | None = None,
    ) -> None:
        if not database_url:
            raise RuntimeError("工具缓存必须配置 DATABASE_URL/POSTGRES_URL，生产模式不允许降级到内存缓存。")
        self._database_url = database_url
        self._ttls = tool_ttls or TOOL_TTLS
        self._hits = 0
        self._misses = 0
        self._engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=1800, future=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tool_result_cache (
                    cache_key VARCHAR(192) PRIMARY KEY,
                    schema_version VARCHAR(16) NOT NULL,
                    scope VARCHAR(128) NOT NULL,
                    tool_name VARCHAR(128) NOT NULL,
                    args_hash VARCHAR(32) NOT NULL,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tool_cache_scope ON tool_result_cache (scope)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tool_cache_tool ON tool_result_cache (tool_name)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tool_cache_expires ON tool_result_cache (expires_at)"))

    @staticmethod
    def _args_hash(kwargs: dict) -> str:
        payload = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def _make_key(self, scope: str, tool_name: str, kwargs: dict) -> str:
        return f"{_CACHE_SCHEMA_VERSION}:{scope}:{tool_name}:{self._args_hash(kwargs)}"

    def _ttl_for(self, tool_name: str) -> int:
        return max(1, int(self._ttls.get(tool_name, _DEFAULT_TTL)))

    def _delete_expired(self, conn) -> int:
        from datetime import datetime, timezone

        result = conn.execute(
            text("DELETE FROM tool_result_cache WHERE expires_at <= :now"),
            {"now": datetime.now(tz=timezone.utc).replace(tzinfo=None)},
        )
        return int(result.rowcount or 0)

    def get(self, scope: str, tool_name: str, kwargs: dict) -> str | None:
        """从 PostgreSQL 读取工具结果；PostgreSQL 不可用时直接失败。"""
        key = self._make_key(scope, tool_name, kwargs)
        from datetime import datetime, timezone

        with self._engine.begin() as conn:
            self._delete_expired(conn)
            row = conn.execute(
                text("""
                    SELECT value
                    FROM tool_result_cache
                    WHERE cache_key=:cache_key AND expires_at > :now
                """),
                {"cache_key": key, "now": datetime.now(tz=timezone.utc).replace(tzinfo=None)},
            ).mappings().first()
        if row is None:
            self._misses += 1
            return None
        self._hits += 1
        return str(row["value"])

    def set(self, scope: str, tool_name: str, kwargs: dict, value: str) -> None:
        """写入 PostgreSQL；PostgreSQL 不可用时直接失败。"""
        key = self._make_key(scope, tool_name, kwargs)
        ttl = self._ttl_for(tool_name)
        from datetime import datetime, timedelta, timezone

        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        with self._engine.begin() as conn:
            self._delete_expired(conn)
            conn.execute(
                text("""
                    INSERT INTO tool_result_cache (
                        cache_key, schema_version, scope, tool_name, args_hash,
                        value, created_at, expires_at
                    ) VALUES (
                        :cache_key, :schema_version, :scope, :tool_name, :args_hash,
                        :value, :created_at, :expires_at
                    )
                    ON CONFLICT (cache_key) DO UPDATE SET
                        value=EXCLUDED.value,
                        created_at=EXCLUDED.created_at,
                        expires_at=EXCLUDED.expires_at
                """),
                {
                    "cache_key": key,
                    "schema_version": _CACHE_SCHEMA_VERSION,
                    "scope": scope,
                    "tool_name": tool_name,
                    "args_hash": self._args_hash(kwargs),
                    "value": value,
                    "created_at": now,
                    "expires_at": now + timedelta(seconds=ttl),
                },
            )

    def invalidate_scope(self, scope: str) -> int:
        with self._engine.begin() as conn:
            result = conn.execute(text("DELETE FROM tool_result_cache WHERE scope=:scope"), {"scope": scope})
        return int(result.rowcount or 0)

    def invalidate_tool(self, tool_name: str, scope: str | None = GLOBAL_SCOPE) -> int:
        params: dict[str, Any] = {"tool_name": tool_name}
        where = "tool_name=:tool_name"
        if scope is None:
            pass
        else:
            where += " AND scope=:scope"
            params["scope"] = scope
        with self._engine.begin() as conn:
            result = conn.execute(text(f"DELETE FROM tool_result_cache WHERE {where}"), params)
        return int(result.rowcount or 0)

    def invalidate_event(self, event_type: str, scope: str | None = GLOBAL_SCOPE) -> dict[str, int]:
        tools = CACHE_INVALIDATION_EVENTS.get(event_type, ())
        return {tool: self.invalidate_tool(tool, scope) for tool in tools}

    def evict_expired(self) -> int:
        with self._engine.begin() as conn:
            return self._delete_expired(conn)

    @property
    def stats(self) -> dict:
        with self._engine.begin() as conn:
            self._delete_expired(conn)
            cached_entries = conn.execute(text("SELECT COUNT(*) FROM tool_result_cache")).scalar_one()
        total = self._hits + self._misses
        return {
            "backend": "postgresql",
            "cached_entries": int(cached_entries or 0),
            "total_hits": self._hits,
            "total_misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }


# ── 全局单例 ─────────────────────────────────────────────────────────────────

_cache: Any | None = None


# 全局单例让不同工具包装器共享命中；生产模式下 PostgreSQL 不可用直接失败。
def get_tool_cache():
    """获取全局工具缓存单例。"""
    global _cache
    if _cache is None:
        from app.core.config import get_settings

        database_url = get_settings().effective_database_url.strip()
        if not database_url:
            raise RuntimeError("工具缓存必须配置 DATABASE_URL/POSTGRES_URL，生产模式不允许降级到内存缓存。")
        _cache = PostgreSQLToolResultCache(database_url)
        logger.info("工具缓存：已启用 PostgreSQL。")
    return _cache
