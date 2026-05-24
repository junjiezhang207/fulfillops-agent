"""RAG capability package.

Real RAG implementations live in this package: query rewrite, retrieval
planning, vector store creation, reranking, answer building, and the main
knowledge retrieval service.

Keep this module lightweight. Import concrete objects from their submodules,
for example:

    from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
"""
