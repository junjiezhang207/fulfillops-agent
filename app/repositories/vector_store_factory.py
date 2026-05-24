"""RAG 向量存储工厂。

这个文件只负责“根据配置创建向量库适配器”，不负责文档切片、embedding 或检索。

当前策略：
  - 生产/企业级主路径：Milvus / Zilliz。
  - 本地开发兜底：返回 None，让 LlamaIndex 使用本地默认向量存储。

为什么要用工厂函数？
  KnowledgeRetrievalService 只关心 LlamaIndex StorageContext，不应该知道 Milvus
  连接参数、upsert 策略、fallback 策略这些基础设施细节。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MilvusRuntimeConfig:
    """Milvus 运行时连接配置。

    这个 dataclass 专门服务 pymilvus 连接层。
    LlamaIndex 的 MilvusVectorStore 仍负责真正写入/检索向量，pymilvus 负责：
    - 建立连接
    - 确保 database 存在
    - 按 overwrite 策略删除旧 collection
    - 对已存在 collection 做健康检查和 load
    """

    uri: str
    token: str
    database: str
    collection: str
    alias: str
    dim: int
    overwrite: bool
    upsert_mode: bool
    batch_size: int
    timeout: float
    similarity_metric: str
    consistency_level: str


def create_vector_store(settings: object):
    """根据配置创建 LlamaIndex 向量存储后端。

    这是 RAG 索引层的唯一入口。
    settings.vector_store_type 决定返回哪种后端：
      - "milvus" / "zilliz"：返回 MilvusVectorStore。
      - 其他值：返回 None，由 LlamaIndex 使用本地存储。

    Returns:
        None                  — 使用 LlamaIndex 默认本地存储
        MilvusVectorStore     — Milvus 向量数据库

    KnowledgeRetrievalService 接收此返回值，传入 StorageContext：
        if vector_store:
            ctx = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes, storage_context=ctx)
        else:
            index = VectorStoreIndex(nodes)  # 本地存储
    """
    store_type = getattr(settings, "vector_store_type", "local").strip().lower()

    if store_type in {"milvus", "zilliz"}:
        return _create_milvus_store(settings)

    logger.info("RAG 向量存储使用本地文件系统（VECTOR_STORE_TYPE=local）")
    return None


def _create_milvus_store(settings: object):
    """创建 Milvus 向量存储实例。

    集合不存在时自动创建；集合存在时默认 upsert，不重复插入同一个 chunk。
    需要强制重建可临时设置 MILVUS_OVERWRITE=true，或删除集合后调用知识库重建接口。

    关键参数说明：
      uri: Milvus/Zilliz 连接地址。
      collection_name: RAG 知识库集合名。
      dim: embedding 维度，必须与实际 embedding 模型输出一致。
      upsert_mode: 使用稳定 chunk_id 覆盖旧记录，避免重复入库。
      overwrite: 强制覆盖集合，通常只在本地重建或测试时打开。
    """
    config = _milvus_runtime_config(settings)

    try:
        _prepare_milvus_with_pymilvus(config)

        from llama_index.vector_stores.milvus import MilvusVectorStore  # type: ignore[import]

        vector_store = MilvusVectorStore(
            uri=config.uri,
            token=config.token,
            collection_name=config.collection,
            dim=config.dim,
            overwrite=config.overwrite,
            upsert_mode=config.upsert_mode,
            batch_size=config.batch_size,
            similarity_metric=config.similarity_metric,
            consistency_level=config.consistency_level,
            db_name=config.database,
        )
        logger.info(
            "Milvus 向量存储初始化成功：uri=%s database=%s collection=%s dim=%d upsert=%s overwrite=%s",
            config.uri,
            config.database,
            config.collection,
            config.dim,
            config.upsert_mode,
            config.overwrite,
        )
        return vector_store

    except ImportError:
        return _fallback_or_raise(
            settings,
            "pymilvus 或 llama-index-vector-stores-milvus 未安装，无法使用 Milvus 向量库。",
        )
    except Exception as exc:
        return _fallback_or_raise(settings, f"Milvus 连接失败（{config.uri}）：{exc}")


def _milvus_runtime_config(settings: object) -> MilvusRuntimeConfig:
    """从 settings 中解析 Milvus 连接参数。"""
    host = getattr(settings, "milvus_host", "localhost")
    port = getattr(settings, "milvus_port", 19530)
    uri = getattr(settings, "milvus_uri", "") or f"http://{host}:{port}"
    collection = getattr(settings, "milvus_collection", "knowledge_base")
    alias = getattr(settings, "milvus_alias", "") or f"rag_{collection}"
    configured_dim = int(getattr(settings, "milvus_dim", 512))
    gateway_dim = _embedding_dimension_from_gateway(settings)
    dim = gateway_dim or configured_dim
    if gateway_dim and gateway_dim != configured_dim:
        logger.warning(
            "MILVUS_DIM=%d 与模型网关 Embedding 维度=%d 不一致，已按模型网关维度创建/校验 Milvus。",
            configured_dim,
            gateway_dim,
        )
    return MilvusRuntimeConfig(
        uri=uri,
        token=getattr(settings, "milvus_token", ""),
        database=getattr(settings, "milvus_database", "default") or "default",
        collection=collection,
        alias=_safe_milvus_alias(alias),
        dim=dim,
        overwrite=bool(getattr(settings, "milvus_overwrite", False)),
        upsert_mode=bool(getattr(settings, "milvus_upsert_mode", True)),
        batch_size=int(getattr(settings, "milvus_batch_size", 100)),
        timeout=float(getattr(settings, "milvus_timeout_seconds", 10.0)),
        similarity_metric=str(getattr(settings, "milvus_similarity_metric", "COSINE") or "COSINE").upper(),
        consistency_level=str(getattr(settings, "milvus_consistency_level", "Session") or "Session"),
    )


def _embedding_dimension_from_gateway(settings: object) -> int | None:
    """优先从模型网关读取当前 Embedding 维度，避免 .env 配错导致维度不一致。"""
    try:
        from app.services.model_gateway import ModelGateway

        profile = ModelGateway(settings).resolve_profile(use_case="embedding", model_type="embedding")
        if profile and profile.dimension:
            return int(profile.dimension)
    except Exception as exc:
        logger.warning("读取模型网关 Embedding 维度失败，继续使用 MILVUS_DIM：%s", exc)
    return None


def _safe_milvus_alias(alias: str) -> str:
    """把 collection 名称转成 pymilvus 连接 alias 可用的安全字符串。"""
    return re.sub(r"[^a-zA-Z0-9_]", "_", alias).strip("_") or "rag_milvus"


def _prepare_milvus_with_pymilvus(config: MilvusRuntimeConfig) -> None:
    """使用 pymilvus 管理连接、database 和 collection 生命周期。

    为什么这里还要用 pymilvus？
    LlamaIndex 的 MilvusVectorStore 很适合接入索引写入，但企业项目里通常还需要
    显式管理连接、数据库、覆盖策略、健康检查和 collection load。
    这些属于基础设施生命周期，用 pymilvus 表达更清楚，也更容易被面试官理解。
    """
    from pymilvus import Collection, connections, db, utility

    connection_kwargs = {
        "alias": config.alias,
        "uri": config.uri,
        "token": config.token,
        "timeout": config.timeout,
    }
    if config.database != "default":
        connections.connect(**connection_kwargs, db_name="default")
        try:
            databases = db.list_database(using=config.alias, timeout=config.timeout)
            if config.database not in databases:
                db.create_database(config.database, using=config.alias, timeout=config.timeout)
        finally:
            connections.disconnect(config.alias)

    connections.connect(**connection_kwargs, db_name=config.database)

    if config.overwrite and utility.has_collection(
        config.collection,
        using=config.alias,
        timeout=config.timeout,
    ):
        utility.drop_collection(
            config.collection,
            using=config.alias,
            timeout=config.timeout,
        )
        logger.info("Milvus collection 已按 overwrite 策略删除：%s", config.collection)
        return

    if utility.has_collection(config.collection, using=config.alias, timeout=config.timeout):
        collection = Collection(config.collection, using=config.alias)
        existing_dim = _collection_vector_dim(collection)
        if existing_dim and existing_dim != config.dim:
            raise RuntimeError(
                f"Milvus collection '{config.collection}' 向量维度是 {existing_dim}，"
                f"当前 Embedding 维度是 {config.dim}。请更换 MILVUS_COLLECTION，"
                "或临时设置 MILVUS_OVERWRITE=true 重建集合。"
            )
        collection.load(timeout=config.timeout)
        logger.info(
            "Milvus collection 健康检查通过并已 load：database=%s collection=%s",
            config.database,
            config.collection,
        )
        return

    logger.info(
        "Milvus collection 尚不存在，将由 LlamaIndex 首次写入时创建：database=%s collection=%s dim=%d",
        config.database,
        config.collection,
        config.dim,
    )


def _collection_vector_dim(collection: object) -> int | None:
    """读取已有 Milvus collection 的向量字段维度。"""
    try:
        for field in collection.schema.fields:
            params = getattr(field, "params", {}) or {}
            if "dim" in params:
                return int(params["dim"])
    except Exception as exc:
        logger.warning("读取 Milvus collection 维度失败：%s", exc)
    return None


def _fallback_or_raise(settings: object, message: str):
    """Milvus 不可用时的降级策略。

    本地开发默认允许 fallback 到本地向量存储，保证项目能跑起来。
    生产环境建议设置 VECTOR_STORE_FALLBACK_TO_LOCAL=false，
    避免 Milvus 挂了以后系统静默降级，导致检索结果和线上预期不一致。
    """
    if bool(getattr(settings, "vector_store_fallback_to_local", True)):
        logger.warning("%s 已回退本地向量存储。", message)
        return None
    raise RuntimeError(message)
