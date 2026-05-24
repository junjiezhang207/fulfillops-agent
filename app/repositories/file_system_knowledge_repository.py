"""File-system knowledge repository.

Learning notes:
- Reads local Markdown/text knowledge files such as rules, SOPs and policies.
- It only loads documents; embedding, indexing and reranking are handled by RAG services.
"""

from pathlib import Path

from app.repositories.knowledge_repository import KnowledgeRepository


class FileSystemKnowledgeRepository(KnowledgeRepository):
    """基于本地文件系统的知识文档仓储实现。"""

    def __init__(self, knowledge_dir: str) -> None:
        self.knowledge_dir = Path(knowledge_dir)

    def list_knowledge_paths(self) -> list[str]:
        """返回知识目录下所有 Markdown 文档路径。"""

        return [
            str(path)
            for path in sorted(self.knowledge_dir.glob("*.md"))
            if path.is_file()
        ]

    def rebuild_index_required(self) -> bool:
        """当前文件系统知识源默认允许手动重建索引。"""

        return True
