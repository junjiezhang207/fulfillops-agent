"""RAG 向量存储工厂。

这个文件只负责“根据配置创建向量库适配器”，不负责文档切片、embedding 或检索。

当前策略：
  - 默认使用 PostgreSQL 的 PGVector 扩展。
  - 仅测试或极简本地运行可显式设置为 local。

为什么要用工厂函数？
  KnowledgeRetrievalService 只关心 LlamaIndex StorageContext，不应该知道 PGVector
  连接参数、表名、fallback 策略这些基础设施细节。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PGVectorRuntimeConfig:
    """PGVector 运行时连接配置。"""

    database: str
    host: str
    port: int
    user: str
    password: str
    table: str
    dim: int


def create_vector_store(settings: object):
    """根据配置创建 LlamaIndex 向量存储后端。

    这是 RAG 索引层的唯一入口。
    settings.vector_store_type 决定返回哪种后端：
      - "pgvector"：返回 PGVectorStore。
      - "local"：返回 None，使用 LlamaIndex 默认本地索引。

    Returns:
        PGVectorStore；local 时返回 None。

    KnowledgeRetrievalService 接收此返回值，传入 StorageContext：
        if vector_store:
            ctx = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes, storage_context=ctx)
        else:
            index = VectorStoreIndex(nodes)  # 本地存储
    """
    store_type = getattr(settings, "vector_store_type", "pgvector").strip().lower()

    if store_type == "pgvector":
        try:
            return _create_pgvector_store(settings)
        except RuntimeError:
            if not bool(getattr(settings, "vector_store_fallback_to_local", False)):
                raise
            fallback = str(getattr(settings, "vector_store_fallback", "local") or "local").strip().lower()
            logger.warning("PGVector 不可用，按配置降级到 %s。", fallback)
            if fallback == "local":
                return None
            raise

    if store_type == "local":
        return None

    raise RuntimeError(f"不支持的 VECTOR_STORE_TYPE={store_type!r}，可选：pgvector / local。")


def _create_pgvector_store(settings: object):
    """创建 PGVector 向量存储实例。"""
    config = _pgvector_runtime_config(settings)
    try:
        from llama_index.vector_stores.postgres import PGVectorStore  # type: ignore[import]

        vector_store = PGVectorStore.from_params(
            database=config.database,
            host=config.host,
            password=config.password,
            port=config.port,
            user=config.user,
            table_name=config.table,
            embed_dim=config.dim,
        )
        logger.info(
            "PGVector 向量存储初始化成功：host=%s database=%s table=%s dim=%d",
            config.host,
            config.database,
            config.table,
            config.dim,
        )
        return vector_store
    except ImportError as exc:
        raise RuntimeError(
            "llama-index-vector-stores-postgres 未安装，无法使用 PGVector 向量库。"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"PGVector 初始化失败（{config.host}:{config.port}/{config.database}）：{exc}") from exc


def _pgvector_runtime_config(settings: object) -> PGVectorRuntimeConfig:
    """从 settings 中解析 PGVector 连接参数。"""
    configured_dim = int(getattr(settings, "pgvector_dim", 1024))
    gateway_dim = _embedding_dimension_from_gateway(settings)
    dim = gateway_dim or configured_dim
    if gateway_dim and gateway_dim != configured_dim:
        logger.warning(
            "PGVECTOR_DIM=%d 与模型网关 Embedding 维度=%d 不一致，已按模型网关维度创建/校验 PGVector。",
            configured_dim,
            gateway_dim,
        )
    return PGVectorRuntimeConfig(
        database=str(getattr(settings, "pgvector_database", "") or "fulfillops_agent"),
        host=str(getattr(settings, "pgvector_host", "") or "localhost"),
        port=int(getattr(settings, "pgvector_port", 5432) or 5432),
        user=str(getattr(settings, "pgvector_user", "") or "fulfillops"),
        password=str(getattr(settings, "pgvector_password", "") or ""),
        table=str(getattr(settings, "pgvector_table", "") or "knowledge_base_vectors"),
        dim=dim,
    )


def _embedding_dimension_from_gateway(settings: object) -> int | None:
    """优先从模型网关读取当前 Embedding 维度，避免 .env 配错导致维度不一致。"""
    try:
        from app.infrastructure.llm.model_gateway import ModelGateway

        profile = ModelGateway(settings).resolve_profile(use_case="embedding", model_type="embedding")
        if profile and profile.dimension:
            return int(profile.dimension)
    except Exception as exc:
        logger.warning("读取模型网关 Embedding 维度失败，继续使用 PGVECTOR_DIM：%s", exc)
    return None

