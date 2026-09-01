"""Backend service registry.

生产装配原则：
- 订单和库存只从企业数据仓储读取。
- 不再挂载 InMemory demo fallback，避免真实数据缺失时被假数据掩盖。
- Workflow、Agent 和 RAG query planning 复用同一套生产仓储。
"""

from functools import lru_cache

from app.core.config import get_settings
from app.repositories.enterprise_data_repository import EnterpriseDataRepository
from app.repositories.file_system_knowledge_repository import FileSystemKnowledgeRepository
from app.infrastructure.llm.chat_adapter import LLMFactory
from app.domain.inventory.analysis import InventoryAnalysisService
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
from app.domain.orders.analysis import OrderAnalysisService
from app.rag.query_rewriter import QueryRewriter


@lru_cache
def get_enterprise_data_repository() -> EnterpriseDataRepository:
    """全局企业数据仓库。

    多个路由必须共用这一份实例，否则后台刚导入的数据，Agent/Workflow 那边
    可能仍然查询不到。生产版使用 PostgreSQL，初始化失败直接暴露。
    """

    return EnterpriseDataRepository(get_settings().effective_database_url)


@lru_cache
def get_order_analysis_service() -> OrderAnalysisService:
    return OrderAnalysisService(get_enterprise_data_repository())


@lru_cache
def get_inventory_analysis_service() -> InventoryAnalysisService:
    enterprise_repo = get_enterprise_data_repository()
    return InventoryAnalysisService(
        inventory_repository=enterprise_repo,
        order_analysis_service=get_order_analysis_service(),
    )


@lru_cache
def get_knowledge_retrieval_service() -> KnowledgeRetrievalService:
    settings = get_settings()
    extra_dirs = [
        item.strip()
        for item in getattr(settings, "knowledge_extra_dirs", "").split(",")
        if item.strip()
    ]
    knowledge_repository = FileSystemKnowledgeRepository(
        settings.knowledge_dir,
        extra_dirs=extra_dirs,
        recursive=bool(getattr(settings, "knowledge_recursive", True)),
    )
    rewrite_model = LLMFactory.create_chat_model(settings, use_case="query_rewrite")
    intent_model = LLMFactory.create_chat_model(settings, use_case="intent_classification")
    return KnowledgeRetrievalService(
        knowledge_repository=knowledge_repository,
        inventory_analysis_service=get_inventory_analysis_service(),
        query_rewriter=QueryRewriter(rewrite_model),
        intent_classifier_model=intent_model,
        # Cloud rerank/compression are useful quality knobs, but they add several
        # external HTTPS calls per RAG request. Keep the default demo path fast
        # and stable; retrieval still uses vector + BM25 + business reranking.
        cross_encoder=None,
        compressor=None,
    )
