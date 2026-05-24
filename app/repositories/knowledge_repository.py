"""Knowledge repository interface.

Learning notes:
- RAG should not care whether documents come from local files, object storage or a database.
- This protocol defines the document access boundary.
"""

from typing import Protocol


class KnowledgeRepository(Protocol):
    """知识仓储接口协议。"""

    def list_knowledge_paths(self) -> list[str]:
        """返回所有知识文档路径。"""

    def rebuild_index_required(self) -> bool:
        """是否需要重建索引。"""
