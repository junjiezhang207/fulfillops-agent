"""File-system knowledge repository.

Learning notes:
- Reads local Markdown/text knowledge files such as rules, SOPs and policies.
- It only loads documents; embedding, indexing and reranking are handled by RAG services.
"""

from pathlib import Path

from app.repositories.knowledge_repository import KnowledgeRepository


class FileSystemKnowledgeRepository(KnowledgeRepository):
    """基于本地文件系统的知识文档仓储实现。"""

    def __init__(
        self,
        knowledge_dir: str,
        extra_dirs: list[str] | None = None,
        recursive: bool = True,
        extensions: tuple[str, ...] = (".md", ".txt"),
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.extra_dirs = [Path(item) for item in (extra_dirs or []) if item]
        self.recursive = recursive
        self.extensions = tuple(ext.lower() for ext in extensions)

    def list_knowledge_paths(self) -> list[str]:
        """返回知识目录下所有 Markdown/text 文档路径。

        主目录仍然用于知识库管理接口上传/删除；extra_dirs 用于把已有的
        ``knowledge_base`` 这类只读知识资产纳入同一套 RAG 索引。
        """

        paths: dict[str, Path] = {}
        for root in [self.knowledge_dir, *self.extra_dirs]:
            if not root.exists():
                continue
            pattern = "**/*" if self.recursive else "*"
            for path in root.glob(pattern):
                if path.is_file() and path.suffix.lower() in self.extensions:
                    paths[str(path.resolve())] = path
        return [str(paths[key]) for key in sorted(paths)]

    def rebuild_index_required(self) -> bool:
        """当前文件系统知识源默认允许手动重建索引。"""

        return True
