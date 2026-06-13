"""生产基础设施启动检查。

这里不做降级，只做快速失败：
- Redis 负责短期记忆、缓存、限流和幂等。
- MySQL 负责企业数据、HITL、Trace、长期记忆元数据。
- Milvus 负责 RAG 和长期记忆向量检索。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InfrastructureCheckResult:
    name: str
    ok: bool
    detail: str


def verify_required_infrastructure(settings) -> list[InfrastructureCheckResult]:
    """检查生产必需依赖，不可用时抛出 RuntimeError。"""
    results = [
        _check_redis(settings.redis_url),
        _check_mysql(settings.mysql_url),
        _check_milvus(settings),
    ]
    failed = [item for item in results if not item.ok]
    if failed:
        details = "；".join(f"{item.name}: {item.detail}" for item in failed)
        raise RuntimeError(f"基础设施检查失败，生产模式拒绝启动：{details}")
    return results


def _check_redis(redis_url: str) -> InfrastructureCheckResult:
    if not redis_url:
        return InfrastructureCheckResult("Redis", False, "未配置 REDIS_URL")
    try:
        import redis

        client = redis.from_url(redis_url, socket_connect_timeout=1.0, socket_timeout=1.0)
        client.ping()
        return InfrastructureCheckResult("Redis", True, "连接正常")
    except Exception as exc:
        return InfrastructureCheckResult("Redis", False, str(exc))


def _check_mysql(mysql_url: str) -> InfrastructureCheckResult:
    if not mysql_url:
        return InfrastructureCheckResult("MySQL", False, "未配置 MYSQL_URL")
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(mysql_url, pool_pre_ping=True, future=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return InfrastructureCheckResult("MySQL", True, "连接正常")
    except Exception as exc:
        return InfrastructureCheckResult("MySQL", False, str(exc))


def _check_milvus(settings) -> InfrastructureCheckResult:
    uri = settings.milvus_uri or f"http://{settings.milvus_host}:{settings.milvus_port}"
    try:
        from pymilvus import connections, utility

        alias = "startup_check"
        connections.connect(
            alias=alias,
            uri=uri,
            token=settings.milvus_token,
            timeout=settings.milvus_timeout_seconds,
            db_name=settings.milvus_database or "default",
        )
        utility.list_collections(using=alias, timeout=settings.milvus_timeout_seconds)
        connections.disconnect(alias)
        return InfrastructureCheckResult("Milvus", True, "连接正常")
    except Exception as exc:
        return InfrastructureCheckResult("Milvus", False, str(exc))
