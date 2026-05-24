"""Backend service registry.

Learning notes:
- lru_cache is used as a lightweight singleton mechanism for repositories and services.
- Enterprise data is the primary source; demo repositories are fallbacks.
- Workflow, Agent and RAG reuse these services, so imported data reaches all chains.
"""

from functools import lru_cache

from app.core.config import get_settings
from app.repositories.composite_repositories import (
    CompositeInventoryRepository,
    CompositeOrderRepository,
)
from app.repositories.enterprise_data_repository import EnterpriseDataRepository
from app.repositories.file_system_knowledge_repository import FileSystemKnowledgeRepository
from app.repositories.in_memory_inventory_repository import InMemoryInventoryRepository
from app.repositories.in_memory_order_repository import InMemoryOrderRepository
from app.graph.llm_adapter import LLMFactory
from app.services.inventory_analysis_service import InventoryAnalysisService
from app.services.knowledge_retrieval_service import KnowledgeRetrievalService
from app.services.order_analysis_service import OrderAnalysisService
from app.services.query_rewriter import QueryRewriter
from app.services.reranker import ContextualCompressor, create_reranker


@lru_cache
def get_enterprise_data_repository() -> EnterpriseDataRepository:
    """全局企业数据仓库。

    多个路由必须共用这一份实例，否则后台刚导入的数据，Agent/Workflow 那边
    可能仍然查询不到。用 lru_cache 做轻量单例，既直观，也方便测试时清缓存。
    """

    return EnterpriseDataRepository(get_settings().enterprise_data_dir)


@lru_cache
def get_order_analysis_service() -> OrderAnalysisService:
    enterprise_repo = get_enterprise_data_repository()
    repository = CompositeOrderRepository(
        primary=enterprise_repo,
        fallback=InMemoryOrderRepository(),
    )
    return OrderAnalysisService(repository)


@lru_cache
def get_inventory_analysis_service() -> InventoryAnalysisService:
    enterprise_repo = get_enterprise_data_repository()
    repository = CompositeInventoryRepository(
        primary=enterprise_repo,
        fallback=InMemoryInventoryRepository(),
    )
    return InventoryAnalysisService(
        inventory_repository=repository,
        order_analysis_service=get_order_analysis_service(),
    )


@lru_cache
def get_knowledge_retrieval_service() -> KnowledgeRetrievalService:
    settings = get_settings()
    knowledge_repository = FileSystemKnowledgeRepository(settings.knowledge_dir)
    rewrite_model = LLMFactory.create_chat_model(settings, use_case="rag_rewrite")
    compress_model = LLMFactory.create_chat_model(settings, use_case="rag_compress")
    return KnowledgeRetrievalService(
        knowledge_repository=knowledge_repository,
        inventory_analysis_service=get_inventory_analysis_service(),
        query_rewriter=QueryRewriter(rewrite_model),
        cross_encoder=create_reranker(settings),
        compressor=ContextualCompressor(compress_model),
    )
