"""Compatibility export for the RAG service.

The implementation has moved to ``app.rag``. New code should import from:

    from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
"""

from app.rag.knowledge_retrieval_service import *  # noqa: F401,F403
