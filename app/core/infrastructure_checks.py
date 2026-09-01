"""生产基础设施启动检查。

这里不做降级，只做快速失败：
- PostgreSQL 负责企业数据、短期记忆、缓存、限流、幂等、HITL、Trace 和长期记忆元数据。
- PGVector 负责 RAG 与长期记忆向量索引。
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
    results: list[InfrastructureCheckResult] = []
    if bool(getattr(settings, "require_postgres", True)):
        results.append(_check_postgres(settings.effective_database_url))
    if bool(getattr(settings, "require_pgvector", True)) or _requires_pgvector(settings):
        results.append(_check_pgvector(settings.effective_database_url))
    failed = [item for item in results if not item.ok]
    if failed:
        details = "；".join(f"{item.name}: {item.detail}" for item in failed)
        raise RuntimeError(f"基础设施检查失败，生产模式拒绝启动：{details}")
    return results


def _requires_pgvector(settings) -> bool:
    store_type = str(getattr(settings, "vector_store_type", "") or "").strip().lower()
    memory_store_type = str(getattr(settings, "long_term_memory_vector_store_type", "") or "").strip().lower()
    return store_type == "pgvector" or memory_store_type == "pgvector"


def _check_postgres(database_url: str) -> InfrastructureCheckResult:
    if not database_url:
        return InfrastructureCheckResult("PostgreSQL", False, "未配置 DATABASE_URL/POSTGRES_URL")
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(database_url, pool_pre_ping=True, future=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return InfrastructureCheckResult("PostgreSQL", True, "连接正常")
    except Exception as exc:
        return InfrastructureCheckResult("PostgreSQL", False, str(exc))


def _check_pgvector(database_url: str) -> InfrastructureCheckResult:
    if not database_url:
        return InfrastructureCheckResult("PGVector", False, "未配置 DATABASE_URL/POSTGRES_URL")
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(database_url, pool_pre_ping=True, future=True)
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        engine.dispose()
        return InfrastructureCheckResult("PGVector", True, "vector 扩展可用")
    except Exception as exc:
        return InfrastructureCheckResult("PGVector", False, str(exc))
