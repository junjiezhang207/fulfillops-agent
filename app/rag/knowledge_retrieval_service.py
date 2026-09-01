"""供应链履约主 RAG 服务（学习版注释）。

这个文件是项目里“知识库问答/规则检索”的主链路，不是普通的工具类。
你可以把它按下面 6 个阶段理解：

1. 文档入库：从 ``app/data/knowledge`` 读取 Markdown 文档。
2. 文档切片：用 LlamaIndex 的 ``SentenceSplitter`` 把长文档切成 TextNode。
3. 补元数据：给每个切片补 ``category/document_id/chunk_id/title`` 等字段。
4. 混合召回：同时走向量检索和 BM25 关键词检索，再用 RRF 融合。
5. 业务重排：根据订单问题识别出来的业务意图，给相关知识类别加权。
6. 结果组装：转成项目自己的 ``KnowledgeRetrieveResult``，供 Agent/Workflow 使用。

为什么这里不像一个简单的 ``search(query)``：
- 面试展示时，RAG 不能只说“向量库查一下”，还要体现企业里常见的混合召回、
  metadata 过滤、重排、缓存失效、可替换向量库等设计。
- 这个类只负责“检索编排”，意图识别和答案摘要已经拆到
  ``RAGQueryPlanner`` / ``RAGAnswerBuilder``，避免一个类什么都做。
"""

import asyncio
import hashlib
import json
import logging
import re
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor.node import BaseNodePostprocessor
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import (
    BaseNode,
    NodeWithScore,
    QueryBundle,
    TextNode,
    TransformComponent,
)
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    FilterCondition,
    MetadataFilters,
)
from llama_index.retrievers.bm25 import BM25Retriever

from app.core.config import get_settings
from app.infrastructure.llm.embedding_adapter import create_embed_model
from app.repositories.knowledge_repository import KnowledgeRepository
from app.rag.vector_store_factory import create_vector_store
from app.schemas.knowledge import (
    HybridScoreDetail,
    KnowledgeHit,
    KnowledgeMetadata,
    KnowledgeRetrieveResult,
    QueryIntent,
    QueryIntentType,
)
from app.schemas.inventory import InventoryAnalysisResult
from app.domain.inventory.analysis import InventoryAnalysisService
from app.rag.query_rewriter import QueryRewriter
from app.rag.rag_answer_builder import RAGAnswerBuilder
from app.rag.rag_query_planner import RAGQueryPlanner, RetrievalInputs
from app.rag.reranker import ContextualCompressor, PassageReranker
from app.observability.business_trace import add_trace_step


logger = logging.getLogger(__name__)

# 每个知识片段的目标长度。太短会丢上下文，太长会让召回不够精准。
CHUNK_SIZE = 220
# 相邻切片保留一点重叠，避免规则条款刚好被切断后检索不到完整语义。
CHUNK_OVERLAP = 40
# 每一路检索最多召回多少个候选片段；后面还会去重、重排、截断。
RETRIEVAL_TOP_K = 8
# 最终返回给前端/Agent 的知识片段数量，控制回答上下文不要过长。
FINAL_HIT_COUNT = 5
# 无订单检索时使用的占位订单号。
# 这个值不会真的去查 OMS/WMS，只是为了让返回结构里的 order_id 字段保持稳定，
# 这样上层 Agent、Trace Center 和前端展示逻辑不用因为“有没有订单号”再写两套分支。
KNOWLEDGE_ONLY_ORDER_ID = "ADHOC-KNOWLEDGE"
# Cross-Encoder 精排时只处理前几个候选，避免本地小模型或云端 reranker 成本过高。
RERANK_TOP_K = 8
# 主意图对应知识类别的业务加权分。
PRIMARY_INTENT_BOOST = 0.35
# 次意图对应知识类别的业务加权分。
SECONDARY_INTENT_BOOST = 0.12
# 关键词重合分进入最终排序时的权重。
KEYWORD_SCORE_WEIGHT = 0.08
# 用于解释关键词命中的业务词表。这里和排序只做轻量相关性补充，
# 主要召回仍然依赖向量 + BM25。
RAG_KEYWORD_TERMS = (
    "缺货",
    "库存不足",
    "高优先级",
    "区域",
    "跨仓",
    "跨区域",
    "调拨",
    "延迟发货",
    "履约",
    "人工介入",
    "仓配",
    "拆单",
    "合单",
    "补发",
    "退货",
    "售后",
    "SOP",
    "SKU",
    "优秀案例",
    "历史案例",
    "相似案例",
    "案例",
    "复盘",
)

# 文件名关键字 -> 业务知识分类。
# 这里不是为了“硬编码业务”，而是给 Markdown 知识文件建立最基础的可过滤标签。
CATEGORY_BY_FILE_KEYWORD = {
    "excellent_case": "excellent_case",
    "excellent-case": "excellent_case",
    "excellent_cases": "excellent_case",
    "stockout": "stockout_rule",
    "priority": "priority_rule",
    "regional": "regional_strategy",
    "split_merge": "split_merge_rule",
    "after_sales": "after_sales_rule",
}

# 意图识别结果 -> 知识分类。
# RAGQueryPlanner 会判断用户问的是缺货、优先级、区域策略等；
# 这里把意图映射到知识库 category，方便后面做 business boost。
CATEGORY_BY_INTENT = {
    QueryIntentType.STOCKOUT_HANDLING: "stockout_rule",
    QueryIntentType.PRIORITY_FULFILLMENT: "priority_rule",
    QueryIntentType.REGIONAL_STRATEGY: "regional_strategy",
    QueryIntentType.SPLIT_MERGE: "split_merge_rule",
    QueryIntentType.AFTER_SALES: "after_sales_rule",
    QueryIntentType.GENERAL: "general",
}

CATEGORY_ALIASES = {
    "excellent_case": "excellent_case",
    "excellent-case": "excellent_case",
    "excellent_cases": "excellent_case",
    "优秀案例": "excellent_case",
    "历史案例": "excellent_case",
    "相似案例": "excellent_case",
    "案例": "excellent_case",
    "stockout": "stockout_rule",
    "缺货": "stockout_rule",
    "缺货处理": "stockout_rule",
    "priority": "priority_rule",
    "优先级": "priority_rule",
    "高优先级": "priority_rule",
    "regional": "regional_strategy",
    "区域履约": "regional_strategy",
    "仓配策略": "regional_strategy",
    "split": "split_merge_rule",
    "split_merge": "split_merge_rule",
    "拆单合单": "split_merge_rule",
    "after_sales": "after_sales_rule",
    "售后": "after_sales_rule",
    "通用": "general",
    "general": "general",
}

EXCELLENT_CASE_TERMS = ("优秀案例", "历史案例", "相似案例", "案例", "复盘", "best practice")
EXCELLENT_CASE_BOOST = 0.18


def normalize_knowledge_category(value: object) -> str:
    """Normalize user/file/front-matter category names to stable RAG categories."""

    normalized = str(value or "").strip()
    return CATEGORY_ALIASES.get(normalized, CATEGORY_ALIASES.get(normalized.lower(), normalized or "general"))


class KnowledgeFrontMatterExtractor(TransformComponent):
    """读取 Markdown front matter，把显式业务 metadata 写入文档。

    知识库文档可以在开头写类似这样的元信息：

    ``category: priority_rule``
    ``tags: [高优先级, 直发]``
    ``expires_at: 2026-12-31``

    这些字段不会直接参与正文切片，但会被写进 node.metadata，后面可以用于：
    - 向量库 metadata filter，例如只检索某个 category。
    - 前端展示来源、版本、负责人、失效时间。
    - 注册表对比和旧 chunk 清理。
    """

    @classmethod
    def class_name(cls) -> str:
        return "KnowledgeFrontMatterExtractor"

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        for node in nodes:
            # LlamaIndex 传进来的 node 此时还是整篇文档级别。
            # 如果文档没有 front matter，就保持原样进入后续 Markdown 切片流程。
            text = node.get_content()
            parsed = self._parse_front_matter(text)
            if parsed is None:
                continue
            metadata, body = parsed
            # front matter 字段写入 metadata，正文只保留 body。
            # 这样 embedding 时不会把 category/version 这类治理字段当作正文语义。
            node.metadata.update(metadata)
            node.metadata["frontmatter"] = metadata
            node.set_content(body.strip())
        return nodes

    @classmethod
    def _parse_front_matter(cls, text: str) -> tuple[dict[str, object], str] | None:
        """解析 Markdown 开头的 ``---`` front matter。

        返回值是 ``(metadata, body)``：
        - metadata：解析出来的键值对。
        - body：去掉 front matter 后的正文。

        这里没有引入 PyYAML，是因为当前知识库只需要很轻量的 key/value 和列表解析；
        保持纯文本解析能减少依赖，也方便面试时解释实现边界。
        """
        if not text.startswith("---"):
            return None
        match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)(.*)\Z", text, flags=re.DOTALL)
        if not match:
            return None
        raw_meta, body = match.groups()
        metadata: dict[str, object] = {}
        for raw_line in raw_meta.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if not key:
                continue
            metadata[key] = cls._parse_front_matter_value(key, value)
        return metadata, body

    @staticmethod
    def _parse_front_matter_value(key: str, value: str) -> object:
        """把 front matter 字符串值转成更适合 metadata 的 Python 对象。"""
        list_like_keys = {"tags", "aliases", "categories", "regions", "business_scope"}
        if value.startswith("[") and value.endswith("]"):
            raw_items = value[1:-1].split(",")
            return [item.strip().strip("'\"") for item in raw_items if item.strip()]
        if key in list_like_keys and "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


class MarkdownSectionSplitter(TransformComponent):
    """按 Markdown 标题优先切片，再用 SentenceSplitter 兜底处理长章节。

    直接对整篇文档做固定长度切片的问题是：一个 chunk 可能跨越多个业务章节，
    召回后很难解释“这段内容属于哪个 SOP 小节”。这里先按 Markdown 标题切出章节，
    再对过长章节做 SentenceSplitter，可以同时保留章节路径和合理 chunk 长度。
    """

    @classmethod
    def class_name(cls) -> str:
        return "MarkdownSectionSplitter"

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        # SentenceSplitter 只负责把过长章节切短；章节识别由本类自己完成。
        # include_metadata=False 很关键：切片长度只按正文计算，避免 metadata 太长影响切片。
        splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            include_metadata=False,
        )
        output: list[BaseNode] = []
        for node in nodes:
            text = node.get_content()
            if not text.strip():
                continue
            metadata = dict(node.metadata)
            sections = self._split_markdown_sections(node.get_content())
            if not sections:
                # 没有 Markdown 标题时退回普通切片，保证纯文本/简单文档也能入库。
                parent_metadata = {
                    **metadata,
                    "_parent_section_path": [],
                    "_parent_context": self._parent_context(text),
                    "_business_unit_type": self._infer_business_unit_type([], text),
                }
                output.extend(self._split_with_metadata(splitter, text, parent_metadata))
                continue
            for section_path, section_text in sections:
                section_metadata = dict(metadata)
                if section_path:
                    # _markdown_title 和 _markdown_section_path 是临时字段，
                    # 后面 BusinessMetadataEnricher 会转换成稳定的 title/section_path。
                    section_metadata["_markdown_title"] = section_path[0]
                    section_metadata["_markdown_section_path"] = section_path
                section_metadata["_parent_section_path"] = section_path
                section_metadata["_parent_context"] = self._parent_context(section_text)
                section_metadata["_business_unit_type"] = self._infer_business_unit_type(section_path, section_text)
                output.extend(self._split_with_metadata(splitter, section_text.strip(), section_metadata))
        return output

    @staticmethod
    def _split_with_metadata(
        splitter: SentenceSplitter,
        text: str,
        metadata: dict[str, object],
    ) -> list[BaseNode]:
        """先按正文切片，再把 metadata 补回每个 chunk。

        LlamaIndex 的 splitter 如果带着 metadata 一起算长度，较长的 tags/section_path
        会挤占 chunk 预算，导致正文被切得过碎。这里先用空 metadata 切正文，
        再把业务 metadata 复制回去，切片效果更稳定。
        """

        chunks = list(splitter([TextNode(text=text, metadata={})]))
        for index, chunk in enumerate(chunks):
            chunk.metadata.update(metadata)
            chunk.metadata["_child_index_in_parent"] = index
            chunk.metadata["_child_count_in_parent"] = len(chunks)
        return chunks

    @staticmethod
    def _parent_context(text: str, max_chars: int = 4000) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        return normalized if len(normalized) <= max_chars else f"{normalized[:max_chars]}...<truncated>"

    @staticmethod
    def _infer_business_unit_type(section_path: list[str], text: str) -> str:
        haystack = " ".join([*section_path, text[:400]]).lower()
        patterns = (
            ("scope", ("scope", "范围", "适用", "业务范围")),
            ("prerequisite", ("prerequisite", "前置", "前提", "先决条件", "触发条件")),
            ("exception", ("exception", "异常", "例外", "失败", "冲突")),
            ("escalation", ("escalation", "升级", "人工", "主管", "审批", "转交")),
            ("procedure", ("procedure", "流程", "步骤", "操作", "处理", "执行")),
            ("rule", ("rule", "规则", "不得", "必须", "应当", "标准")),
        )
        for unit_type, markers in patterns:
            if any(marker in haystack for marker in markers):
                return unit_type
        return "general"

    @staticmethod
    def _split_markdown_sections(text: str) -> list[tuple[list[str], str]]:
        """按 Markdown 标题层级拆出章节文本。

        返回的每一项是 ``(section_path, section_text)``：
        - section_path：例如 ``["售后政策", "退货限制"]``。
        - section_text：该标题下的正文，包含标题行本身。

        这里保留标题行，是为了让 chunk 即使脱离原文，也带有最基本的语义提示。
        """
        heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
        sections: list[tuple[list[str], str]] = []
        current_lines: list[str] = []
        current_path: list[str] = []
        heading_stack: list[str] = []

        def flush() -> None:
            # 遇到下一个标题或文档结束时，把当前章节写入 sections。
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((list(current_path), content))

        for line in text.splitlines():
            match = heading_re.match(line.strip())
            if match:
                flush()
                level = len(match.group(1))
                title = match.group(2).strip()
                # 根据标题层级维护一个栈：
                # # A -> ["A"]
                # ## B -> ["A", "B"]
                # 新的同级/上级标题会覆盖对应层级之后的旧路径。
                heading_stack[:] = heading_stack[: level - 1] + [title]
                current_path = list(heading_stack)
                current_lines = [line]
                continue
            current_lines.append(line)
        flush()
        return sections


@dataclass
class RetrievedNodes:
    """RAG 召回阶段的中间结果。

    这个对象只在服务内部流转，不直接返回给 API。
    单独建一个 dataclass 是为了让“扩展后的 query”和“召回到的 nodes”
    绑在一起，后续构建 score_detail / matched_terms 时还能知道用了哪些 query。
    """

    # 经过意图识别、订单上下文补充、Query Rewrite 后的查询列表。
    queries: list[str]
    # LlamaIndex 返回的候选切片，里面包含 node 本体和召回分数。
    nodes: list[NodeWithScore]


class BusinessMetadataEnricher(TransformComponent):
    """给切片补齐过滤、去重和溯源 metadata。

    LlamaIndex 的切片默认只知道文本和一些原始文件信息。
    但企业 RAG 一般需要：
    - 按知识类别过滤，比如只查“缺货规则”。
    - 能定位来源，比如来自哪个文件、哪个章节。
    - 多路召回后能稳定去重，比如用 chunk_id 作为唯一键。

    所以这个 TransformComponent 会插在 IngestionPipeline 里，
    在切片进入向量库/BM25 之前统一补齐业务元数据。
    """

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """LlamaIndex pipeline 调用入口。

        ``nodes`` 是 SentenceSplitter 切出来的片段列表。
        这里逐个检查并补 metadata，最后仍然返回原来的 node 列表。
        """
        chunk_counters: dict[str, int] = {}
        for node in nodes:
            # 只处理文本切片。理论上 SimpleDirectoryReader 也可能产生其他 BaseNode，
            # 这里跳过，避免访问 TextNode 专有方法时报错。
            if not isinstance(node, TextNode):
                continue

            # 优先取 file_path，其次取 filename；都没有时给 unknown。
            # Path(...).name 只保留文件名，避免不同机器上的绝对路径污染 metadata。
            source_path = str(node.metadata.get("file_path") or node.metadata.get("filename") or "unknown")
            source_file = Path(source_path).name
            # 通过文件名推断知识类别，例如 stockout_rules.md -> stockout_rule。
            category = normalize_knowledge_category(
                str(node.metadata.get("category") or self._infer_category(source_file))
            )
            # document_id 用文件名去掉 .md，方便前端或日志展示。
            document_id = str(
                node.metadata.get("document_id")
                or node.metadata.get("doc_id")
                or self._infer_document_id(source_path, source_file)
            )

            # chunk_id 必须稳定：同一个文档同一个切片顺序生成相同 id。
            # 后面向量 + BM25 融合去重时，就靠这个字段判断是不是同一段知识。
            # 注意这里必须按 document_id 独立计数，不能用整批节点的 enumerate。
            # 否则 A 文档切片数量变化后，B 文档的 chunk 编号也会漂移，PGVector 旧数据就很难清理干净。
            order = chunk_counters.get(document_id, 0)
            chunk_counters[document_id] = order + 1
            chunk_id = f"{document_id}::chunk-{order:04d}"
            node.id_ = chunk_id

            # title/section_path/tags 都是为了“可解释召回”。
            # 面试时可以说：不是只返回一段文本，还能说清楚它来自哪个业务章节。
            title = str(
                node.metadata.get("title")
                or node.metadata.get("_markdown_title")
                or self._extract_title(node.get_content(), source_file)
            )
            section_path = list(
                node.metadata.get("_markdown_section_path")
                or self._extract_section_path(node.get_content(), title)
            )
            parent_section_path = list(node.metadata.get("_parent_section_path") or section_path)
            parent_basis = "::".join(parent_section_path or [title, document_id])
            parent_hash = hashlib.sha1(parent_basis.encode("utf-8")).hexdigest()[:10]
            parent_chunk_id = str(node.metadata.get("parent_chunk_id") or f"{document_id}::parent-{parent_hash}")
            explicit_tags = node.metadata.get("tags", [])
            if isinstance(explicit_tags, str):
                explicit_tags = [explicit_tags]
            business_scope = node.metadata.get("business_scope", [])
            if isinstance(business_scope, str):
                business_scope = [business_scope]
            tags = sorted(set([category, *section_path, *explicit_tags, *business_scope]))

            # metadata 会随 node 一起写入向量库。PGVector/本地索引都可以用这些字段过滤或展示。
            node.metadata.update(
                {
                    "category": category,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "parent_chunk_id": parent_chunk_id,
                    "parent_section_path": parent_section_path,
                    "parent_context_excerpt": str(node.metadata.get("_parent_context", "")),
                    "child_index_in_parent": int(node.metadata.get("_child_index_in_parent", order)),
                    "child_count_in_parent": int(node.metadata.get("_child_count_in_parent", 1)),
                    "business_unit_type": str(node.metadata.get("_business_unit_type", "general")),
                    "title": title,
                    "section_path": section_path,
                    "tags": tags,
                    "source_file": source_file,
                    "source_path": source_path,
                    "version": str(node.metadata.get("version", "")),
                    "owner": str(node.metadata.get("owner", "")),
                    "effective_date": str(node.metadata.get("effective_date", "")),
                    # expires_at 是知识治理字段：运营可以在上传时声明规则失效日期。
                    # 检索阶段会再次检查这个字段，过期 chunk 不参与最终排序，避免旧 SOP 误导 Agent。
                    "expires_at": str(node.metadata.get("expires_at", "")),
                    "region": str(node.metadata.get("region", "")),
                    "business_scope": list(business_scope),
                    "knowledge_source": str(node.metadata.get("knowledge_source", "")),
                    "collection_name": str(node.metadata.get("collection_name", "")),
                    "version_status": str(node.metadata.get("version_status", "")),
                    "is_active": str(node.metadata.get("is_active", "")),
                    "vector_backend": str(node.metadata.get("vector_backend", "")),
                    "source_case_id": str(node.metadata.get("source_case_id", "")),
                }
            )
        return nodes

    def _infer_category(self, source_file: str) -> str:
        """根据文件名推断业务类别。

        这里保持简单可控：知识库文件命名规范越稳定，规则越可靠。
        真实企业也常用“目录/文件名/文档标签”作为第一层 metadata。
        """
        fname = source_file.lower()
        for keyword, category in CATEGORY_BY_FILE_KEYWORD.items():
            if keyword in fname:
                return category
        return "general"

    def _infer_document_id(self, source_path: str, source_file: str) -> str:
        """从路径生成稳定 document_id，避免递归知识库里同名文件互相覆盖。"""
        path = Path(source_path)
        stem_path = path.with_suffix("")
        parts = list(stem_path.parts)
        for marker in ("knowledge_base", "knowledge"):
            if marker in parts:
                relative_parts = parts[parts.index(marker) + 1 :]
                if relative_parts:
                    return self._safe_document_id("__".join(relative_parts))
        return self._safe_document_id(Path(source_file).stem)

    @staticmethod
    def _safe_document_id(value: str) -> str:
        return re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._-") or "unknown"

    def _extract_title(self, text: str, source_file: str) -> str:
        """从 Markdown 一级标题中提取文档标题。

        如果文档没有 ``# 标题``，就退回文件名，保证 title 字段永远有值。
        """
        for line in text.splitlines():
            if line.strip().startswith("# "):
                return line.strip().removeprefix("# ").strip()
        return source_file.removesuffix(".md")

    def _extract_section_path(self, text: str, title: str) -> list[str]:
        """提取简化版章节路径。

        这里只取一级标题和最多两个二级标题，原因是：
        - 太长的 section_path 对前端展示没意义。
        - tags 太多会让 metadata 变乱。
        """
        sections = [title]
        for line in text.splitlines():
            if line.strip().startswith("## "):
                sections.append(line.strip().removeprefix("## ").strip())
        return sections[:3]


class BusinessRulePostprocessor(BaseNodePostprocessor):
    """按业务意图给候选片段做轻量排序修正。

    向量检索/BM25 只知道“文本相似不相似”，不一定知道业务优先级。
    例如用户问“库存不足怎么处理”，stockout_rule 类文档应该天然更靠前。
    这个 postprocessor 就是在通用召回分数上叠加一点业务分。
    """

    # LlamaIndex 的 postprocessor 接口不直接传业务 intent，所以这里在每次调用前临时赋值。
    # 注意：外层每次请求都会创建新的 BusinessRulePostprocessor，避免并发请求互相污染。
    current_intent: QueryIntent | None = None

    @classmethod
    def class_name(cls) -> str:
        """LlamaIndex 需要的组件名称，用于日志/序列化识别。"""
        return "BusinessRulePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """按业务规则调整 score 并重排序。

        分数设计：
        - primary intent 对应的知识类别加 PRIMARY_INTENT_BOOST。
        - secondary intent 对应的知识类别加 SECONDARY_INTENT_BOOST。
        - 其他类别不加分。

        这不是最终答案评分，只是召回片段的排序辅助。
        """
        if not nodes or self.current_intent is None:
            return nodes

        intent = self.current_intent
        query_text = query_bundle.query_str if query_bundle is not None else ""
        # 主意图只会有一个，比如 STOCKOUT_HANDLING。
        primary_category = CATEGORY_BY_INTENT.get(intent.primary_intent)
        # 次意图可能有多个，比如同时涉及区域策略和拆单规则。
        secondary_categories = {
            CATEGORY_BY_INTENT.get(secondary)
            for secondary in intent.secondary_intents
        }

        reranked = []
        for node_with_score in nodes:
            category = node_with_score.node.metadata.get("category", "general")
            rule_boost = 0.0
            # 主意图加权更大，因为它最能代表用户当前问题。
            if category == primary_category:
                rule_boost = PRIMARY_INTENT_BOOST
            # 次意图只做轻微加权，避免把真正高相似文本挤掉。
            elif category in secondary_categories:
                rule_boost = SECONDARY_INTENT_BOOST
            if category == "excellent_case" and self._should_boost_excellent_case(intent, query_text):
                rule_boost = max(rule_boost, EXCELLENT_CASE_BOOST)

            # node_with_score.score 可能为 None，所以用 0.0 兜底。
            base_score = float(node_with_score.score or 0.0)
            adjusted_score = base_score + rule_boost
            node_with_score.node.metadata.update(
                {
                    "_rag_base_score": round(base_score, 4),
                    "_rag_business_rule_score": round(rule_boost, 4),
                    "_rag_pre_rerank_score": round(adjusted_score, 4),
                }
            )
            # NodeWithScore 通常不直接原地改分数，新建对象更清楚。
            reranked.append(NodeWithScore(node=node_with_score.node, score=adjusted_score))

        return sorted(reranked, key=lambda n: n.score or 0.0, reverse=True)

    @staticmethod
    def _should_boost_excellent_case(intent: QueryIntent, query_text: str) -> bool:
        normalized_query = query_text.lower()
        if any(term.lower() in normalized_query for term in EXCELLENT_CASE_TERMS):
            return True
        return intent.primary_intent in {
            QueryIntentType.STOCKOUT_HANDLING,
            QueryIntentType.PRIORITY_FULFILLMENT,
            QueryIntentType.REGIONAL_STRATEGY,
            QueryIntentType.SPLIT_MERGE,
            QueryIntentType.AFTER_SALES,
        }


class KnowledgeRetrievalService:
    """面向履约场景的 RAG 检索服务。

    这个类是 RAG 的“编排层”，它不自己写 embedding 算法，也不自己写 BM25 算法，
    而是把 LlamaIndex、模型网关、知识仓库、业务意图识别组合起来。

    这里刻意把职责拆成几个清晰阶段：
    - 输入准备：判断是否有订单上下文，并调用 QueryPlanner 做意图识别和 query 扩展。
    - 索引准备：懒加载或预热向量索引，同时准备 BM25 关键词索引。
    - 混合召回：向量召回负责语义相似，BM25 负责关键词精确命中。
    - 业务排序：根据意图、关键词和可选 reranker 调整顺序。
    - 结果封装：把框架内部 NodeWithScore 转成稳定的业务 schema。

    这样写的好处是：面试时可以把它讲成一条完整 RAG 工程链路，
    而不是“调一个向量数据库接口”。

    依赖关系：
    - KnowledgeRepository：告诉服务知识文件在哪里、是否支持重建索引。
    - InventoryAnalysisService：给 QueryPlanner 用，用订单上下文增强 query。
    - QueryRewriter：可选，用小模型/云模型做 query 扩展。
    - PassageReranker：可选，用 reranker 做更细的相关性排序。
    - ContextualCompressor：可选，把长 chunk 压缩成更短的可引用文本。
    """

    def __init__(
        self,
        knowledge_repository: KnowledgeRepository,
        inventory_analysis_service: InventoryAnalysisService,
        query_rewriter: QueryRewriter | None = None,
        intent_classifier_model=None,
        cross_encoder: PassageReranker | None = None,
        compressor: ContextualCompressor | None = None,
    ) -> None:
        # 知识文件访问层。这个服务不直接写死 app/data/knowledge，方便未来换存储。
        # 例如现在是本地文件系统，后续可以换成对象存储、后台上传目录或数据库。
        self.knowledge_repository = knowledge_repository

        # 向量索引懒加载缓存。第一次检索才构建，避免项目启动时加载模型很慢。
        # 如果调用 warmup_index()，也可以在启动阶段主动构建，避免用户第一次提问超时。
        self._index: VectorStoreIndex | None = None
        # 当前知识库切出来的 TextNode。BM25 必须基于内存节点构建。
        # 即使用 PGVector 做外部向量库，BM25 仍然需要这些本地文本节点。
        self._nodes: list[TextNode] = []
        # BM25 关键词检索器懒加载缓存。
        self._bm25_retriever: BM25Retriever | None = None

        # reranker/compressor 都是可选增强能力：
        # 没配模型时 RAG 仍然能跑，只是少了精排/压缩。
        self._cross_encoder = cross_encoder
        self._compressor = compressor
        # QueryPlanner 负责“理解问题 + 扩展查询”，避免检索服务越来越胖。
        self._query_planner = RAGQueryPlanner(
            inventory_analysis_service=inventory_analysis_service,
            query_rewriter=query_rewriter,
            intent_classifier_model=intent_classifier_model,
        )
        # AnswerBuilder 负责“把命中的片段组装成结果/摘要”，和召回逻辑分离。
        self._answer_builder = RAGAnswerBuilder()

        self._settings = get_settings()
        # Embedding/RAG 向量库可能会加载本地模型或连接 PGVector，放到第一次检索时懒初始化，
        # 避免 FastAPI 启动阶段被模型权重加载卡住。
        self._embed_model = None
        self._vector_store = None
        # 多个请求同时第一次进入 RAG 时，只允许一个线程真正构建索引。
        # 否则本地开发会出现重复 embedding、重复写 PGVector、接口一起超时的问题。
        self._index_build_lock = threading.Lock()
        # 最近一次重建时的文档变化摘要，用于 API 返回和排查“旧文档是否已清理”。
        self._last_registry_changes: list[dict[str, object]] = []
        # IngestionPipeline 是 LlamaIndex 的文档处理流水线：
        # 1. KnowledgeFrontMatterExtractor：抽取文档开头的 category/tags/version 等元数据。
        # 2. MarkdownSectionSplitter：按标题层级切分，再用 SentenceSplitter 控制 chunk 大小。
        # 3. BusinessMetadataEnricher：补齐 document_id/chunk_id/source_file 等检索治理字段。
        self._pipeline = IngestionPipeline(
            transformations=[
                KnowledgeFrontMatterExtractor(),
                MarkdownSectionSplitter(),
                BusinessMetadataEnricher(),
            ]
        )

    def retrieve(
        self,
        order_id: str | None = None,
        question: str = "",
        filter_categories: list[str] | None = None,
        session_memory: dict[str, Any] | None = None,
    ) -> KnowledgeRetrieveResult:
        """同步检索入口。

        同步链路一般给普通 FastAPI service 或单元测试使用。
        流程固定为：
        1. prepare：识别意图、整理订单上下文、确定过滤类别。
        2. retrieve：混合召回候选知识片段。
        3. rank/build：业务重排、可选 rerank、转 KnowledgeHit。
        4. build_result：组装最终结果。

        ``order_id`` 允许为空：
        - 有订单号：使用订单、库存、履约状态增强 query。
        - 无订单号：直接走知识库问答，适合运营人员查 SOP、规则和上传文档内容。
        """
        started_at = time.monotonic()
        # 统一把 None、空格等输入归一化成空字符串，后面只判断 truthy/falsy。
        normalized_order_id = self._normalize_order_id(order_id)
        try:
            # 这一层会根据是否有订单号，选择“订单上下文检索”或“纯知识库检索”。
            prepared = self._prepare_retrieval_inputs(normalized_order_id, question, filter_categories, session_memory)
            # retrieved 仍然是框架内部的 NodeWithScore，不急着转业务 schema。
            retrieved = self._retrieve_nodes(question, prepared)
            # 排序阶段会写入 score_detail 需要的 metadata，再统一转 KnowledgeHit。
            hits = self._rank_and_build_hits(question, retrieved, prepared.intent)
            result = self._build_result(
                normalized_order_id or KNOWLEDGE_ONLY_ORDER_ID,
                question,
                prepared,
                retrieved.queries,
                hits,
            )
            add_trace_step(
                # Trace Center 记录 RAG 链路的输入、输出、证据片段和耗时。
                # 这里不是业务逻辑必需，但对排查“为什么没检索到/为什么慢”非常重要。
                step_type="rag",
                name="retrieve_knowledge",
                status="success",
                duration_ms=(time.monotonic() - started_at) * 1000,
                summary="RAG 检索完成",
                input_summary={
                    "order_id": normalized_order_id or None,
                    "query": question,
                    "filter_categories": filter_categories or [],
                    "retrieval_top_k": RETRIEVAL_TOP_K,
                },
                output_summary={"hit_count": len(hits), "expanded_query_count": len(retrieved.queries)},
                metadata={
                    "expanded_queries": retrieved.queries,
                    "matched_categories": self._hit_categories(hits),
                    "empty_result": len(hits) == 0,
                    "low_confidence": bool(hits and max(hit.score for hit in hits) < 0.35),
                    "rerank_enabled": self._cross_encoder is not None,
                    "has_order_context": getattr(prepared, "has_order_context", bool(normalized_order_id)),
                },
                evidence=[
                    {
                        "source_file": hit.metadata.source_file,
                        "chunk_id": hit.metadata.chunk_id,
                        "parent_chunk_id": getattr(hit.metadata, "parent_chunk_id", ""),
                        "business_unit_type": getattr(hit.metadata, "business_unit_type", ""),
                        "title": hit.metadata.title,
                        "score": hit.score,
                    }
                    for hit in hits[:5]
                    if hit.metadata
                ],
            )
            return result
        except Exception as exc:
            add_trace_step(
                # 失败也写 trace，这样前端看到超时或 SSL 错误时，Trace Center 里还能定位到 RAG 阶段。
                step_type="rag",
                name="retrieve_knowledge",
                status="error",
                duration_ms=(time.monotonic() - started_at) * 1000,
                summary="RAG 检索失败",
                error_code=exc.__class__.__name__,
                error_message=str(exc),
                input_summary={
                    "order_id": normalized_order_id or None,
                    "query": question,
                    "filter_categories": filter_categories or [],
                },
            )
            raise

    async def aretrieve(
        self,
        order_id: str | None = None,
        question: str = "",
        filter_categories: list[str] | None = None,
        session_memory: dict[str, Any] | None = None,
    ) -> "KnowledgeRetrieveResult":
        """异步检索入口。

        给异步 API / Agent 调用使用。它和同步版本返回结构一致，
        只是 query rewrite、检索、rerank、compress 尽量走异步方法。

        注意：向量库和 reranker 的底层实现不一定是真异步。
        所以这里的异步主要是为了不阻塞 FastAPI 事件循环，并让多条 expanded query 可以并发等待。
        """
        started_at = time.monotonic()
        normalized_order_id = self._normalize_order_id(order_id)
        try:
            prepared = self._prepare_retrieval_inputs(normalized_order_id, question, filter_categories, session_memory)
            retrieved = await self._aretrieve_nodes(question, prepared)
            hits = await self._arank_and_build_hits(question, retrieved, prepared.intent)
            result = self._build_result(
                normalized_order_id or KNOWLEDGE_ONLY_ORDER_ID,
                question,
                prepared,
                retrieved.queries,
                hits,
            )
            add_trace_step(
                # 异步入口和同步入口记录同样的 trace 字段，便于横向对比。
                step_type="rag",
                name="retrieve_knowledge",
                status="success",
                duration_ms=(time.monotonic() - started_at) * 1000,
                summary="异步 RAG 检索完成",
                input_summary={
                    "order_id": normalized_order_id or None,
                    "query": question,
                    "filter_categories": filter_categories or [],
                    "retrieval_top_k": RETRIEVAL_TOP_K,
                },
                output_summary={"hit_count": len(hits), "expanded_query_count": len(retrieved.queries)},
                metadata={
                    "expanded_queries": retrieved.queries,
                    "matched_categories": self._hit_categories(hits),
                    "empty_result": len(hits) == 0,
                    "low_confidence": bool(hits and max(hit.score for hit in hits) < 0.35),
                    "rerank_enabled": self._cross_encoder is not None,
                    "has_order_context": getattr(prepared, "has_order_context", bool(normalized_order_id)),
                },
                evidence=[
                    {
                        "source_file": hit.metadata.source_file,
                        "chunk_id": hit.metadata.chunk_id,
                        "parent_chunk_id": getattr(hit.metadata, "parent_chunk_id", ""),
                        "business_unit_type": getattr(hit.metadata, "business_unit_type", ""),
                        "title": hit.metadata.title,
                        "score": hit.score,
                    }
                    for hit in hits[:5]
                    if hit.metadata
                ],
            )
            return result
        except Exception as exc:
            add_trace_step(
                # 异步失败也保留输入摘要，避免只看到前端“请求超时”但不知道 query 内容。
                step_type="rag",
                name="retrieve_knowledge",
                status="error",
                duration_ms=(time.monotonic() - started_at) * 1000,
                summary="异步 RAG 检索失败",
                error_code=exc.__class__.__name__,
                error_message=str(exc),
                input_summary={
                    "order_id": normalized_order_id or None,
                    "query": question,
                    "filter_categories": filter_categories or [],
                },
            )
            raise

    @staticmethod
    def _normalize_order_id(order_id: str | None) -> str:
        """清洗订单号输入。

        这里不做订单格式校验，只负责把 None/空白字符串统一成空字符串。
        订单是否存在由 QueryPlanner/InventoryAnalysisService 负责判断。
        """
        return (order_id or "").strip()

    def _prepare_retrieval_inputs(
        self,
        order_id: str,
        question: str,
        filter_categories: list[str] | None,
        session_memory: dict[str, Any] | None = None,
    ) -> RetrievalInputs:
        """准备检索需要的上下文对象。

        ``RetrievalInputs`` 是 QueryPlanner 给检索层的标准上下文：
        - active_filters：需要过滤的知识类别。
        - inventory_result：订单/库存分析结果，或者无订单时的占位结果。
        - intent：用户问题对应的业务意图。
        - retrieval_context：用于 query rewrite 的上下文文本。
        """
        if order_id:
            # 有订单号时走完整业务链路：查订单、查库存、识别风险，再扩展 query。
            return self._query_planner.prepare(
                order_id,
                question,
                self._normalize_filter_categories(filter_categories),
                session_memory=session_memory,
            )
        # 没有订单号时，不再强行查订单，直接构造“知识库检索”上下文。
        return self._prepare_without_order(
            question,
            self._normalize_filter_categories(filter_categories),
            session_memory=session_memory,
        )

    @staticmethod
    def _normalize_filter_categories(filter_categories: list[str] | None) -> list[str]:
        return list(dict.fromkeys(
            normalize_knowledge_category(category)
            for category in (filter_categories or [])
            if str(category or "").strip()
        ))

    def _prepare_without_order(
        self,
        question: str,
        filter_categories: list[str] | None,
        session_memory: dict[str, Any] | None = None,
    ) -> RetrievalInputs:
        """构造无订单知识库检索上下文。

        这个方法解决之前项目过度围绕订单的问题：
        运营人员可以直接问“哪些情况不允许直接补发”“高优先级订单规则是什么”，
        系统仍然会做意图识别、query 扩展、RAG 检索，只是不再依赖订单和库存数据。
        """
        # 无订单时仍然保留意图识别。
        # 这样“补发/退货/高优先级/跨仓”之类问题仍能映射到对应知识 category。
        intent = self._query_planner.recognize_intent(
            question=question,
            insufficient_skus=[],
            fulfillment_ready=False,
        )
        # 构造一个占位 InventoryAnalysisResult，是为了复用下游 RetrievalInputs schema。
        # 它明确声明“没有订单上下文”，避免 AnswerBuilder/Trace 误以为真的查过订单。
        inventory_result = InventoryAnalysisResult(
            order_id=KNOWLEDGE_ONLY_ORDER_ID,
            fulfillment_ready=False,
            insufficient_skus=[],
            sku_checks=[],
            order_summary="未提供订单号，本次请求只检索知识库文档。",
            summary="无订单上下文，已跳过订单和库存分析。",
        )
        # retrieval_context 会进入 QueryPlanner 的 query rewrite/扩展逻辑。
        # 这里强调“规则、SOP、制度、上传文档”，防止无订单问题被错误改写成订单分析问题。
        retrieval_context = (
            f"用户问题：{question}\n"
            "检索模式：无订单知识库检索。\n"
            "请检索最相关的规则、SOP、制度、表格或上传文档内容。"
        )
        return RetrievalInputs(
            active_filters=filter_categories or [],
            inventory_result=inventory_result,
            intent=intent,
            retrieval_context=retrieval_context,
            has_order_context=False,
            session_memory_context=self._query_planner.session_memory_context(session_memory),
        )

    def _retrieve_nodes(self, question: str, prepared: RetrievalInputs) -> RetrievedNodes:
        """同步召回候选节点。

        这个方法只做“拿到候选 NodeWithScore”，不做最终结果转换。
        拆出来是为了让召回、排序、结果构建三个步骤更好读。
        """
        # 根据原问题 + 订单上下文 + 意图，生成多条查询语句。
        queries = self._query_planner.build_queries(question, prepared)
        # 构建混合检索器，然后用多条 query 分别检索并合并去重。
        nodes = self._retrieve_with_expanded_queries(
            self._build_hybrid_retriever(prepared.active_filters),
            queries,
            prepared.active_filters,
        )
        return RetrievedNodes(queries=queries, nodes=nodes)

    async def _aretrieve_nodes(self, question: str, prepared: RetrievalInputs) -> RetrievedNodes:
        """异步召回候选节点。

        和 ``_retrieve_nodes`` 的职责完全一致，只是 query 扩展和检索使用异步能力。
        """
        queries = await self._query_planner.abuild_queries(question, prepared)
        nodes = await self._aretrieve_with_expanded_queries(
            self._build_hybrid_retriever(prepared.active_filters),
            queries,
            prepared.active_filters,
        )
        return RetrievedNodes(queries=queries, nodes=nodes)

    def _rank_and_build_hits(
        self,
        question: str,
        retrieved: RetrievedNodes,
        intent: QueryIntent,
    ) -> list[KnowledgeHit]:
        """同步完成“排序 + 转 KnowledgeHit”。

        输入还是 LlamaIndex 的 NodeWithScore，输出变成项目自己的 KnowledgeHit。
        """
        ranked_nodes = self._rank_nodes(question, retrieved.queries, retrieved.nodes, intent)
        compressed_texts = self._compress_hits(question, ranked_nodes)
        return self._hits_from_nodes(ranked_nodes, retrieved.queries, intent, compressed_texts)

    async def _arank_and_build_hits(
        self,
        question: str,
        retrieved: RetrievedNodes,
        intent: QueryIntent,
    ) -> list[KnowledgeHit]:
        """异步完成“排序 + 转 KnowledgeHit”。"""
        ranked_nodes = await self._arank_nodes(question, retrieved.queries, retrieved.nodes, intent)
        compressed_texts = await self._acompress_hits(question, ranked_nodes)
        return self._hits_from_nodes(ranked_nodes, retrieved.queries, intent, compressed_texts)

    def _rank_nodes(
        self,
        question: str,
        expanded_queries: list[str],
        nodes: list[NodeWithScore],
        intent: QueryIntent,
    ) -> list[NodeWithScore]:
        """对召回结果做业务重排和可选 Cross-Encoder 精排。

        排序分两层：
        - 先做业务规则加权，让“问题意图对应的知识类别”更靠前。
        - 如果配置了 reranker，再用 Cross-Encoder/重排模型做语义精排。

        这里没有把 reranker 当成必需依赖：
        - 本地开发可以不配 reranker，避免额外模型调用导致超时。
        - 面试展示时可以说明这是一个“可插拔增强点”，不是硬耦合。
        """
        # 第一步：给每个 node 写入 keyword_score。
        # 这一步不是重新召回，而是为排序和可解释性补一个轻量信号。
        keyword_scored = self._annotate_keyword_scores(nodes, expanded_queries)
        # 第二步：按业务意图加权，例如“高优先级”问题优先展示 priority_rule。
        ranked = self._postprocess_by_business_rules(keyword_scored, expanded_queries, intent)
        if self._cross_encoder is None or not ranked:
            return ranked
        # reranker 通常输入纯文本列表，返回最相关的 passage 文本和相关性分数。
        # 只取前 RERANK_TOP_K 个候选，避免精排阶段成为主要耗时来源。
        top_passages = self._cross_encoder.rerank_with_scores(
            question,
            [node.node.get_content() for node in ranked],
            top_k=RERANK_TOP_K,
        )
        return self._keep_passage_order_with_scores(ranked, top_passages)

    async def _arank_nodes(
        self,
        question: str,
        expanded_queries: list[str],
        nodes: list[NodeWithScore],
        intent: QueryIntent,
    ) -> list[NodeWithScore]:
        """异步版本重排。

        适合接云端 reranker，避免阻塞事件循环。
        """
        keyword_scored = self._annotate_keyword_scores(nodes, expanded_queries)
        ranked = self._postprocess_by_business_rules(keyword_scored, expanded_queries, intent)
        if self._cross_encoder is None or not ranked:
            return ranked
        top_passages = await asyncio.to_thread(
            self._cross_encoder.rerank_with_scores,
            question,
            [node.node.get_content() for node in ranked],
            RERANK_TOP_K,
        )
        return self._keep_passage_order_with_scores(ranked, top_passages)

    def _postprocess_by_business_rules(
        self,
        nodes: list[NodeWithScore],
        expanded_queries: list[str],
        intent: QueryIntent,
    ) -> list[NodeWithScore]:
        """执行业务意图加权。

        为什么不直接在 QueryFusionRetriever 里做？
        因为融合检索器只关心检索分数，不了解本项目的履约业务意图。
        """
        # postprocessor 带有本次请求的 intent，不能跨请求复用。
        postprocessor = BusinessRulePostprocessor()
        postprocessor.current_intent = intent
        # QueryBundle 是 LlamaIndex postprocessor 的标准入参。
        # 这里把多条扩展 query 拼起来，便于未来 postprocessor 读取 query 文本。
        query_bundle = QueryBundle(query_str=" ".join(expanded_queries))
        return postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)

    def _annotate_keyword_scores(
        self,
        nodes: list[NodeWithScore],
        expanded_queries: list[str],
    ) -> list[NodeWithScore]:
        """计算轻量关键词分，并把它纳入排序。

        QueryFusionRetriever 已经融合了向量和 BM25 排名，但它返回给我们的
        NodeWithScore 是一个融合后分数，无法直接拆出 BM25 原始分。
        所以这里额外计算“query 业务词和 chunk 文本的重合度”，作为可解释的
        keyword_score，并用很小权重参与最终排序。
        """
        merged_query = " ".join(expanded_queries)
        scored: list[NodeWithScore] = []
        for node_with_score in nodes:
            # semantic_score 这里代表 QueryFusionRetriever 融合后的基础分。
            # 它已经包含向量/BM25 的 RRF 排名信号，但不方便解释具体命中了哪些业务词。
            semantic_score = float(node_with_score.score or 0.0)
            keyword_score = self._keyword_overlap_score(
                node_with_score.node.get_content(),
                merged_query,
            )
            keyword_boost = keyword_score * KEYWORD_SCORE_WEIGHT
            node_with_score.node.metadata.update(
                {
                    # 这些 _rag_* 字段只在本次检索链路内部使用，
                    # 最终会被 _node_to_hit 读取并写入 HybridScoreDetail。
                    "_rag_semantic_score": round(semantic_score, 4),
                    "_rag_keyword_score": round(keyword_score, 4),
                    "_rag_keyword_boost": round(keyword_boost, 4),
                    "_rag_retrieval_channels": [
                        "semantic",
                        *(["keyword"] if keyword_score > 0 else []),
                    ],
                }
            )
            scored.append(
                NodeWithScore(
                    node=node_with_score.node,
                    score=semantic_score + keyword_boost,
                )
            )
        return scored

    def _keep_passage_order(
        self,
        nodes: list[NodeWithScore],
        ordered_passages: list[str],
    ) -> list[NodeWithScore]:
        """按 Cross-Encoder 返回的文本顺序重排 NodeWithScore。

        CrossEncoderReranker 返回的是 passage 文本，不是 node 对象。
        所以这里用文本内容把排序结果映射回原来的 NodeWithScore。
        """
        # 建一个 passage -> 排名 的字典，排序时按 reranker 返回顺序排列。
        passage_order = {passage: i for i, passage in enumerate(ordered_passages)}
        return sorted(
            # 只保留 reranker 返回的 passage；没进 reranker top_k 的候选会被截掉。
            [node for node in nodes if node.node.get_content() in passage_order],
            key=lambda node: passage_order.get(node.node.get_content(), 999),
        )

    def _keep_passage_order_with_scores(
        self,
        nodes: list[NodeWithScore],
        ranked_passages,
    ) -> list[NodeWithScore]:
        """按 reranker 返回顺序重排，并把 rerank 分写入 metadata。

        ``PassageReranker.rerank_with_scores`` 返回的是 RankedPassage 列表。
        主 RAG 仍然需要 NodeWithScore，所以这里做一次映射：
        - 用 passage 文本找到原始 NodeWithScore。
        - 把 rerank_score 写进 node.metadata，供 _node_to_hit 生成 score_detail。
        - 最终 score 使用“业务加权分 + rerank 归一化分”的轻量融合。
        """
        passage_rank = {item.text: (rank, item.rerank_score) for rank, item in enumerate(ranked_passages)}
        output: list[NodeWithScore] = []
        max_rank_score = max((abs(float(item.rerank_score)) for item in ranked_passages), default=0.0) or 1.0
        for node in nodes:
            text = node.node.get_content()
            if text not in passage_rank:
                continue
            rank, rerank_score = passage_rank[text]
            normalized_rerank = float(rerank_score) / max_rank_score
            pre_rerank_score = float(node.node.metadata.get("_rag_pre_rerank_score", node.score or 0.0))
            final_score = pre_rerank_score + normalized_rerank
            node.node.metadata.update(
                {
                    "_rag_rerank_score": round(float(rerank_score), 4),
                    "_rag_rerank_rank": rank + 1,
                    "_rag_rerank_normalized_score": round(normalized_rerank, 4),
                    "_rag_final_score": round(final_score, 4),
                }
            )
            output.append(NodeWithScore(node=node.node, score=final_score))
        return sorted(output, key=lambda node: node.score or 0.0, reverse=True)

    def _compress_hits(self, question: str, ranked_nodes: list[NodeWithScore]) -> list[str] | None:
        """同步压缩 top chunk 文本。

        compressor 的作用是把命中的长片段压成更短的上下文。
        没配置 compressor 时返回 None，后面会保留原始 chunk 文本。
        """
        if self._compressor is None or not ranked_nodes:
            return None
        return self._compressor.compress(question, self._top_passages(ranked_nodes))

    async def _acompress_hits(
        self,
        question: str,
        ranked_nodes: list[NodeWithScore],
    ) -> list[str] | None:
        """异步压缩 top chunk 文本。"""
        if self._compressor is None or not ranked_nodes:
            return None
        return await self._compressor.acompress(question, self._top_passages(ranked_nodes))

    @staticmethod
    def _top_passages(nodes: list[NodeWithScore], limit: int = FINAL_HIT_COUNT) -> list[str]:
        """取前 N 个候选的纯文本，给 reranker/compressor 使用。"""
        return [node.node.get_content() for node in nodes[:limit]]

    def _hits_from_nodes(
        self,
        ranked_nodes: list[NodeWithScore],
        expanded_queries: list[str],
        intent: QueryIntent,
        compressed_texts: list[str] | None = None,
    ) -> list[KnowledgeHit]:
        """把排序后的 NodeWithScore 转成对外返回的 KnowledgeHit。"""
        # 多条扩展 query 合并后用于提取 matched_terms，让前端能展示“命中了哪些词”。
        merged_query = " ".join(expanded_queries)
        # 只取最终需要返回的前几个节点，避免把所有候选都暴露给上层。
        hits = [
            self._node_to_hit(node, merged_query, intent)
            for node in ranked_nodes[:FINAL_HIT_COUNT]
        ]
        # 没启用 compressor 时，KnowledgeHit.text 保留原始 chunk。
        if not compressed_texts:
            return hits
        # 启用 compressor 时，只替换 text，不改分数、来源、metadata。
        return [
            hit.model_copy(update={"text": compressed_texts[i]})
            if i < len(compressed_texts) and compressed_texts[i]
            else hit
            for i, hit in enumerate(hits)
        ]

    def _build_result(
        self,
        order_id: str,
        question: str,
        prepared: RetrievalInputs,
        expanded_queries: list[str],
        hits: list[KnowledgeHit],
    ) -> KnowledgeRetrieveResult:
        """组装最终 RAG 输出。

        这里除了 hits，还会生成：
        - matched_categories：这次覆盖了哪些知识域。
        - answer_summary：给 Agent / Workflow 使用的结构化摘要。
        - summary：给日志或 API 展示的一句话说明。
        """
        # 结果组装委托给 RAGAnswerBuilder，保持本文件聚焦“检索链路”。
        return self._answer_builder.build_result(
            order_id=order_id,
            question=question,
            prepared=prepared,
            expanded_queries=expanded_queries,
            hits=hits,
        )

    @staticmethod
    def _hit_categories(hits: list[KnowledgeHit]) -> list[str]:
        categories: set[str] = set()
        for hit in hits:
            category = getattr(hit, "category", None)
            if not category and getattr(hit, "metadata", None):
                category = getattr(hit.metadata, "category", None)
            if category:
                categories.add(str(category))
        return sorted(categories)

    async def _aretrieve_with_expanded_queries(
        self,
        hybrid_retriever: "QueryFusionRetriever",
        expanded_queries: list[str],
        filter_categories: list[str] | None = None,
    ) -> list[NodeWithScore]:
        """并行执行多个扩展查询。

        一个用户问题会扩展成多条 query。
        同步版本是一条一条查；异步版本可以把这些查询放进线程池并发跑。

        这里直接使用 LlamaIndex retriever 的 ``aretrieve()``，不再手写
        ``asyncio.to_thread(hybrid_retriever.retrieve, ...)`` 这一层胶水。
        """
        # 每条扩展 query 都是一次独立检索，gather 会并发等待所有检索结果。
        tasks = [
            hybrid_retriever.aretrieve(query)
            for query in expanded_queries
        ]
        results = await asyncio.gather(*tasks)

        # results 是 list[list[NodeWithScore]]，先拍平，再做统一过滤和去重。
        return self._dedupe_and_filter_nodes(
            [node for nodes in results for node in nodes],
            filter_categories,
        )

    def rebuild_index(self) -> str:
        """手动重建知识索引。

        用于知识库文档更新后刷新向量索引和 BM25 索引。
        FastAPI 的 knowledge-mgmt rebuild 接口最终会走到这里。

        重建不是简单地“再插入一遍向量”：
        外部向量库不会自动删除旧 chunk，所以这里会结合 document registry
        先找出已经删除或变化的旧 chunk，再清理，再写入新索引。
        """
        if not self.knowledge_repository.rebuild_index_required():
            return "当前知识源不支持手动重建索引。"
        if not self.knowledge_repository.list_knowledge_paths():
            # 全量删除文档也是一种合法的知识库变更。
            # 这时没有新 index 可构建，但仍然要根据 registry 清理外部向量库里的旧 chunk。
            previous_registry = self._read_document_registry()
            changes = self._build_document_change_plan(
                current_documents={},
                previous_documents=previous_registry.get("documents", {}),
            )
            vector_store = self._get_vector_store()
            if vector_store is not None:
                self._delete_stale_vector_chunks(vector_store, changes)
            self._write_document_registry({}, changes)
            self._index = None
            self._nodes = []
            self._bm25_retriever = None
            deleted = sum(1 for change in changes if change["change_type"] == "deleted")
            return f"知识目录为空，已清理 {deleted} 个已删除文档的旧向量记录。"
        # 重新构建向量索引；本地模式会写缓存，PGVector 模式会写外部向量库。
        self._index = self._build_index(persist=True)
        # BM25 是内存索引，向量索引重建后也要同步刷新。
        self._bm25_retriever = BM25Retriever.from_defaults(
            nodes=self._nodes, similarity_top_k=self._retrieval_top_k()
        )
        replaced = sum(1 for change in self._last_registry_changes if change["change_type"] == "updated")
        deleted = sum(1 for change in self._last_registry_changes if change["change_type"] == "deleted")
        created = sum(1 for change in self._last_registry_changes if change["change_type"] == "created")
        return (
            "知识索引已完成重建，并同步刷新 BM25Retriever。"
            f"文档变更：新增 {created}，替换 {replaced}，删除 {deleted}。"
        )

    def _get_or_build_index(self) -> VectorStoreIndex:
        """懒加载向量索引。

        第一次检索或 warmup_index 会触发这里。
        加锁的原因是：多个用户请求同时到达时，只允许一个请求执行建库，
        其他请求等待建库完成后直接复用同一个 ``self._index``。
        """
        # 第一次访问时加载/构建；后续复用，避免重复 embedding。
        if self._index is None:
            with self._index_build_lock:
                if self._index is None:
                    self._index = self._load_or_build_index()
        return self._index

    def _load_or_build_index(self) -> VectorStoreIndex:
        """优先加载缓存索引，缓存不可用时重新构建。

        根据向量后端不同分三种情况：
        - PGVector 外部向量库：优先判断表里是否已有向量。
        - 配置了外部向量库但连接失败并降级：不复用旧本地缓存，避免维度不一致。
        - 纯本地模式：通过文档指纹判断 ``storage/knowledge_index`` 是否还能复用。
        """
        # LlamaIndex 的索引构建会读取全局 Settings.embed_model。
        Settings.embed_model = self._get_embed_model()

        vector_store = self._get_vector_store()
        if vector_store is not None:
            # 外部向量库由后端自己管理持久化，不依赖本地 doc_fingerprint。
            # 如果 collection 里已经有向量，只需要创建一个指向该 collection 的 VectorStoreIndex。
            # 这样服务重启不会重复 embedding 已入库文档。
            loaded = self._load_existing_external_index(vector_store)
            if loaded is not None:
                return loaded
            return self._build_index(persist=False)

        if self._configured_external_vector_store():
            # 配置期望使用 PGVector，但连接失败后按开发策略回退到了本地存储。
            # 这种情况下不能复用 storage/knowledge_index 里的旧缓存，因为它可能是
            # 历史低维 embedding（例如旧 BGE 512 维）生成的，容易和当前阿里云 1024 维 query 冲突。
            return self._build_index(persist=False)

        # 本地缓存目录和指纹文件。指纹一致才允许复用旧索引。
        cache_dir = Path(self._settings.knowledge_index_cache_dir)
        marker_file = cache_dir / "doc_fingerprint.txt"
        current_fp = self._build_document_fingerprint()

        if marker_file.exists() and marker_file.read_text(encoding="utf-8") == current_fp:
            try:
                # 指纹一致：优先从磁盘加载，启动速度更快。
                storage_context = StorageContext.from_defaults(persist_dir=str(cache_dir))
                index = load_index_from_storage(storage_context)
                # 本地向量索引可以加载，但 BM25 需要内存中的 TextNode。
                self._nodes = self._run_ingestion_pipeline()
                return index
            except Exception:
                # 缓存文件损坏或版本不兼容时，直接重建索引。
                pass

        # 没缓存、指纹不一致、缓存坏了，都会走重建。
        return self._build_index(persist=True)

    def warmup_index(self) -> str:
        """预热 RAG 索引，避免用户第一次提问时承担建库成本。

        本地开发时最容易超时的点通常不是检索本身，而是第一次请求顺便触发：
        1. 加载 embedding 模型或连接云端 embedding。
        2. 读取所有 Markdown 文档并切片。
        3. 写入 PGVector 向量库。
        4. 构建 BM25 内存索引。

        预热把这些工作提前到启动/手动调用阶段完成，用户真正提问时只做检索和排序。
        """
        # 先确保向量索引可用；当前配置为 PGVector，这里会完成连接或建表。
        self._get_or_build_index()
        if self._bm25_retriever is None and self._nodes:
            # 向量索引预热后，顺手把 BM25 也建好。
            # 否则第一次混合召回仍然会在用户请求里构建 BM25。
            self._bm25_retriever = BM25Retriever.from_defaults(
                nodes=self._nodes,
                similarity_top_k=self._retrieval_top_k(),
            )
        return f"知识索引预热完成：{len(self._nodes)} 个 chunk。"

    def _load_existing_external_index(self, vector_store: object) -> VectorStoreIndex | None:
        """外部向量库已有数据时直接加载，避免每次重启都全量 embedding。

        对 PGVector 这类持久化向量库来说，服务重启后表里的向量还在。
        如果这里仍然全量构建，就会造成：
        - 启动/首次提问耗时很长。
        - 旧 chunk 没清理干净时可能重复写入。
        - 云端 embedding 产生额外成本。

        因此只要能判断 collection 已有数据，就直接从 vector_store 创建索引对象。
        """
        if not self._external_vector_store_has_data(vector_store):
            return None
        # BM25 不是外部持久化的，仍然需要跑一次 ingestion 拿到内存 TextNode。
        self._nodes = self._run_ingestion_pipeline()
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self._get_embed_model(),
        )

    @staticmethod
    def _external_vector_store_has_data(vector_store: object) -> bool:
        """尽量用各后端公开能力判断外部向量库是否已有向量。

        不同 LlamaIndex VectorStore 暴露的 collection 属性名不完全一致：
        - LlamaIndex 向量库适配器通常会暴露 table/collection 相关内部属性。
        - 具体后端的实现也可能包一层内部对象。

        这里不强依赖具体类型，只要对象上有 count() 能力，就用它判断是否已有数据。
        判断失败时返回 False，让后续走安全的重建逻辑。
        """
        collection = getattr(vector_store, "_collection", None)
        if collection is not None and hasattr(collection, "count"):
            try:
                return int(collection.count()) > 0
            except Exception:
                return False
        return False

    def _build_index(self, persist: bool) -> VectorStoreIndex:
        """构建向量索引，并按配置选择本地持久化或外部向量库。

        VectorStoreIndex 是 LlamaIndex 的核心索引对象。
        它负责把 TextNode 通过 embedding 模型转成向量，并写入对应的存储后端。

        ``persist`` 只影响本地 LlamaIndex 存储：
        - 本地模式：persist=True 时写入 knowledge_index_cache_dir。
        - 外部向量库模式：向量库自己持久化，persist 参数不参与。
        """
        # 构建索引前先重新跑文档流水线，拿到最新 TextNode。
        self._nodes = self._run_ingestion_pipeline()
        # current_documents 记录“当前文件 -> 当前 chunk_ids”，用于和上一次注册表对比。
        current_documents = self._build_current_document_records(self._nodes)
        previous_registry = self._read_document_registry()
        # changes 里包含 created/updated/deleted/unchanged，以及需要删除的旧 chunk_id。
        changes = self._build_document_change_plan(
            current_documents=current_documents,
            previous_documents=previous_registry.get("documents", {}),
        )

        vector_store = self._get_vector_store()
        if vector_store is not None:
            # 外部向量库不会自动知道“某个文档删了/变短了”。
            # 所以重建前先按旧 registry 里的 chunk_id 删除过期向量，再写入新切片。
            self._delete_stale_vector_chunks(vector_store, changes)
            self._sanitize_node_metadata_for_vector_store(self._nodes)
            # 外部向量库模式：把 PGVector vector_store 注入 LlamaIndex StorageContext。
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            index = VectorStoreIndex(self._nodes, storage_context=storage_context)
        else:
            # 本地模式：使用 LlamaIndex 默认存储，并按需持久化到磁盘。
            index = VectorStoreIndex(self._nodes)
            if persist:
                cache_dir = Path(self._settings.knowledge_index_cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                index.storage_context.persist(persist_dir=str(cache_dir))
                # 保存本次构建对应的指纹，下次可判断缓存是否过期。
                (cache_dir / "doc_fingerprint.txt").write_text(
                    self._build_document_fingerprint(), encoding="utf-8"
                )

        self._write_document_registry(current_documents, changes)
        return index

    @staticmethod
    def _sanitize_node_metadata_for_vector_store(nodes: list[TextNode]) -> None:
        """写入外部向量库前清洗 metadata。

        外部向量库通常只接受 str/int/float/None 这类标量 metadata。
        但项目里的 ``tags``、``section_path``、``business_scope`` 是 list，
        如果直接写入会报错。因此这里把复杂结构序列化成 JSON 字符串。

        读取命中结果时会通过 ``_metadata_list`` 再把 JSON 字符串还原成 list 展示。
        """
        allowed = (str, int, float, type(None))
        for node in nodes:
            cleaned: dict[str, object] = {}
            for key, value in node.metadata.items():
                if isinstance(value, bool):
                    cleaned[key] = str(value).lower()
                elif isinstance(value, allowed):
                    cleaned[key] = value
                else:
                    cleaned[key] = json.dumps(value, ensure_ascii=False)
            node.metadata = cleaned

    def list_document_statuses(self) -> list[dict[str, object]]:
        """返回知识文档与索引注册表的对比状态。

        这个方法给知识库管理 API 使用，用来判断：
        - 文件是否已进入索引。
        - 当前文件内容是否和上一次索引时一致。
        - 是否存在已经从目录删除、但注册表里仍有记录的文档。

        sync_status 含义：
        - new：当前目录有文件，但还没有写入索引注册表。
        - changed：文件 hash 和注册表不一致，需要重建索引。
        - indexed：文件和注册表一致，说明已入库。
        - deleted：注册表里有记录，但当前目录已找不到文件，需要清理旧向量。
        """

        registry = self._read_document_registry()
        indexed_docs = registry.get("documents", {})
        current_files = self._scan_knowledge_files()
        rows: list[dict[str, object]] = []

        for document_id, file_info in current_files.items():
            indexed = indexed_docs.get(document_id)
            if indexed is None:
                # 新文件：还没有任何 chunk 写入过索引。
                sync_status = "new"
                version = 1
                indexed_hash = ""
                chunk_count = 0
            elif indexed.get("content_hash") != file_info["content_hash"]:
                # 文件存在但内容变了：需要重建并替换旧 chunk。
                sync_status = "changed"
                version = int(indexed.get("version", 1))
                indexed_hash = str(indexed.get("content_hash", ""))
                chunk_count = len(indexed.get("chunk_ids", []))
            else:
                # hash 一致：说明当前文件和上次建索引时一致。
                sync_status = "indexed"
                version = int(indexed.get("version", 1))
                indexed_hash = str(indexed.get("content_hash", ""))
                chunk_count = len(indexed.get("chunk_ids", []))

            rows.append({
                **file_info,
                "version": version,
                "indexed_hash": indexed_hash,
                "chunk_count": chunk_count,
                "sync_status": sync_status,
            })

        for document_id, indexed in indexed_docs.items():
            if document_id in current_files:
                continue
            # 注册表里有、当前目录没有，说明用户删除了知识文件。
            # 这种状态必须暴露出来，否则外部向量库里的旧 chunk 会继续被命中。
            rows.append({
                "document_id": document_id,
                "filename": indexed.get("filename", f"{document_id}.md"),
                "source_path": indexed.get("source_path", ""),
                "category": indexed.get("category", "general"),
                "size_bytes": 0,
                "last_modified": None,
                "content_hash": "",
                "version": int(indexed.get("version", 1)),
                "indexed_hash": indexed.get("content_hash", ""),
                "chunk_count": len(indexed.get("chunk_ids", [])),
                "sync_status": "deleted",
            })

        return sorted(rows, key=lambda item: str(item["document_id"]))

    def document_registry_snapshot(self) -> dict[str, object]:
        """返回知识文档注册表快照，供状态页展示和面试讲解。

        这个快照不是业务知识内容，而是“索引状态账本”：
        它记录每个文档的 hash、版本、chunk_id 列表以及最近一次变更摘要。
        """

        registry = self._read_document_registry()
        return {
            "registry_path": str(self._document_registry_path()),
            "schema_version": registry.get("schema_version", 1),
            "updated_at": registry.get("updated_at"),
            "documents": registry.get("documents", {}),
            "last_changes": registry.get("last_changes", []),
        }

    def _document_registry_path(self) -> Path:
        """知识文档注册表路径。

        注册表不放在知识文件目录里，而是放在索引缓存目录里。
        原因是它描述的是“当前索引里有哪些文档版本”，属于索引状态，不属于业务原始文档。
        """

        return Path(self._settings.knowledge_index_cache_dir) / "knowledge_document_registry.json"

    def _read_document_registry(self) -> dict[str, object]:
        """读取知识文档注册表；不存在或损坏时返回空注册表。

        这里选择容错读取：注册表坏了不会让应用完全启动失败。
        后续重建索引时会按空注册表处理，相当于重新建立一份索引状态账本。
        """

        path = self._document_registry_path()
        if not path.exists():
            return {"schema_version": 1, "documents": {}, "last_changes": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("知识文档注册表读取失败，将按空注册表处理：%s", exc)
            return {"schema_version": 1, "documents": {}, "last_changes": []}

    def _write_document_registry(
        self,
        documents: dict[str, dict[str, object]],
        changes: list[dict[str, object]],
    ) -> None:
        """写入知识文档注册表。

        registry 的核心作用是解决“旧文档污染检索”：
        下次重建索引时，可以知道旧版本有哪些 chunk_id，从而在 PGVector 里先删旧 chunk。

        last_changes 只保留摘要，不保存完整 stale_chunk_ids，
        是为了让 API 展示足够清晰，同时避免注册表越来越臃肿。
        """

        payload = {
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "documents": documents,
            "last_changes": [
                {
                    "document_id": change["document_id"],
                    "change_type": change["change_type"],
                    "stale_chunk_count": len(change.get("stale_chunk_ids", [])),
                }
                for change in changes
            ],
        }
        path = self._document_registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._last_registry_changes = payload["last_changes"]

    def _scan_knowledge_files(self) -> dict[str, dict[str, object]]:
        """扫描当前知识目录中的 Markdown 文档，并计算内容 hash。

        这一步只看“文件级别”的状态，不做切片、不做 embedding。
        它的作用是快速得到当前知识源的 document_id/category/hash，
        用来和注册表比较是否需要重建索引。
        """

        files: dict[str, dict[str, object]] = {}
        # 复用 BusinessMetadataEnricher 的 document_id/category 推断规则，
        # 保证状态页看到的 document_id 和真正入库时的 document_id 一致。
        enricher = BusinessMetadataEnricher()
        for path_text in self.knowledge_repository.list_knowledge_paths():
            path = Path(path_text)
            if not path.exists() or not path.is_file():
                continue
            content = path.read_bytes()
            try:
                # front matter 只支持 UTF-8 文档；无法解码时仍然可以用文件名推断元数据。
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = ""
            parsed_frontmatter = KnowledgeFrontMatterExtractor._parse_front_matter(text)
            frontmatter = parsed_frontmatter[0] if parsed_frontmatter else {}
            # document_id 优先使用文档显式声明，没声明时按路径生成稳定 ID。
            # 这样递归目录里出现同名文件时，不会互相覆盖。
            document_id = str(
                frontmatter.get("document_id")
                or frontmatter.get("doc_id")
                or enricher._infer_document_id(str(path), path.name)
            )
            category = enricher._normalize_category(
                str(frontmatter.get("category") or enricher._infer_category(path.name))
            )
            files[document_id] = {
                "document_id": document_id,
                "filename": path.name,
                "source_path": str(path),
                "category": category,
                "size_bytes": len(content),
                "last_modified": path.stat().st_mtime,
                "content_hash": hashlib.sha256(content).hexdigest(),
            }
        return files

    def _build_current_document_records(
        self,
        nodes: list[TextNode],
    ) -> dict[str, dict[str, object]]:
        """根据当前文件和切片结果生成新的文档注册表 documents 字段。

        ``_scan_knowledge_files`` 只能知道文件 hash；
        这个方法会额外把每个文档对应的 chunk_ids 记录下来。
        外部向量库清理旧数据时，真正删除的是 chunk_id，而不是文件名。
        """

        old_documents = self._read_document_registry().get("documents", {})
        current_files = self._scan_knowledge_files()
        chunk_ids_by_doc: dict[str, list[str]] = {}
        for node in nodes:
            # BusinessMetadataEnricher 已经给每个 node 写入 document_id/chunk_id。
            # 这里按文档聚合，形成 document -> chunk_ids 的映射。
            document_id = str(node.metadata.get("document_id", "unknown"))
            chunk_id = str(node.metadata.get("chunk_id", node.node_id))
            chunk_ids_by_doc.setdefault(document_id, []).append(chunk_id)

        documents: dict[str, dict[str, object]] = {}
        for document_id, file_info in current_files.items():
            old = old_documents.get(document_id, {})
            content_changed = old.get("content_hash") != file_info["content_hash"]
            old_version = int(old.get("version", 0) or 0)
            # 只有内容变化时版本号递增；纯重启/重复预热不会让版本号乱涨。
            version = old_version + 1 if content_changed else max(old_version, 1)
            documents[document_id] = {
                **file_info,
                "version": version,
                "status": "active",
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "chunk_ids": chunk_ids_by_doc.get(document_id, []),
            }
        return documents

    def _build_document_change_plan(
        self,
        *,
        current_documents: dict[str, dict[str, object]],
        previous_documents: dict[str, dict[str, object]],
    ) -> list[dict[str, object]]:
        """对比新旧注册表，得到本次重建需要清理的旧 chunk。

        返回中的 stale_chunk_ids 会交给 PGVector delete_nodes。
        - created：新文档，没有旧 chunk 需要删。
        - updated：同 document_id 内容或切片发生变化，先删旧 chunk 再写新 chunk。
        - deleted：文件已不存在，删除旧 chunk。
        - unchanged：没有变化，不需要清理。

        这里同时比较 chunk_size/chunk_overlap，是因为切片参数变化即使文件内容不变，
        chunk_id 列表也可能变化；旧切片必须清理，否则新旧切片会同时被召回。
        """

        changes: list[dict[str, object]] = []
        all_document_ids = sorted(set(current_documents) | set(previous_documents))
        for document_id in all_document_ids:
            current = current_documents.get(document_id)
            previous = previous_documents.get(document_id)
            if previous is None and current is not None:
                change_type = "created"
                stale_chunk_ids: list[str] = []
            elif current is None and previous is not None:
                change_type = "deleted"
                stale_chunk_ids = list(previous.get("chunk_ids", []))
            elif current is not None and previous is not None:
                old_chunk_ids = list(previous.get("chunk_ids", []))
                new_chunk_ids = list(current.get("chunk_ids", []))
                changed = (
                    # 内容变更、切片参数变更、chunk_id 列表变更，都视为需要替换。
                    previous.get("content_hash") != current.get("content_hash")
                    or previous.get("chunk_size") != current.get("chunk_size")
                    or previous.get("chunk_overlap") != current.get("chunk_overlap")
                    or old_chunk_ids != new_chunk_ids
                )
                change_type = "updated" if changed else "unchanged"
                stale_chunk_ids = old_chunk_ids if changed else []
            else:
                continue

            changes.append({
                "document_id": document_id,
                "change_type": change_type,
                "stale_chunk_ids": stale_chunk_ids,
            })
        return changes

    def _delete_stale_vector_chunks(
        self,
        vector_store: object,
        changes: list[dict[str, object]],
    ) -> None:
        """在外部向量库中删除过期 chunk。

        LlamaIndex 的 PGVectorStore 支持 delete_nodes(node_ids=[...])。
        这里按 registry 里记录的旧 chunk_id 删除，可以覆盖三类企业常见场景：
        1. 同名文档替换：删旧 chunk，再写新 chunk。
        2. 文档变短：旧版本多出来的 chunk 不会残留。
        3. 文档删除：旧文档不会继续被 RAG 命中。

        这是外部向量库模式最关键的治理逻辑之一。
        没有这一步，用户删除或更新文档后，PGVector 里仍可能保留旧向量，
        RAG 就会回答已经失效的规则。
        """

        stale_chunk_ids = sorted({
            # changes 里只有 updated/deleted 才会带 stale_chunk_ids；
            # created/unchanged 会自然被过滤掉。
            chunk_id
            for change in changes
            for chunk_id in change.get("stale_chunk_ids", [])
            if chunk_id
        })
        if not stale_chunk_ids:
            return
        if not hasattr(vector_store, "delete_nodes"):
            logger.warning("当前向量库不支持 delete_nodes，无法清理旧知识 chunk。")
            return
        try:
            vector_store.delete_nodes(node_ids=stale_chunk_ids)
            logger.info("已从外部向量库清理 %d 个旧知识 chunk。", len(stale_chunk_ids))
        except Exception as exc:
            # 这里选择抛出异常，而不是静默忽略。
            # 企业知识库宁愿重建失败，也不能带着旧规则继续提供答案。
            raise RuntimeError(f"清理旧知识向量失败：{exc}") from exc

    def _run_ingestion_pipeline(self) -> list[TextNode]:
        """加载 Markdown 知识库并切成带业务 metadata 的 TextNode。

        返回 TextNode 列表有两个用途：
        1. VectorStoreIndex 用它构建向量索引。
        2. BM25Retriever 用它构建关键词索引。
        """
        # 从仓库层拿知识文件列表，避免服务层关心具体目录结构。
        knowledge_paths = self.knowledge_repository.list_knowledge_paths()
        if not knowledge_paths:
            raise ValueError("知识目录为空，无法构建索引。")
        # 确保切片进入 VectorStoreIndex 时使用项目配置的 embedding 模型。
        Settings.embed_model = self._get_embed_model()
        # SimpleDirectoryReader 负责读取 Markdown 文件内容。
        documents = SimpleDirectoryReader(input_files=knowledge_paths).load_data()
        # IngestionPipeline 顺序执行：
        # 1. front matter 抽取；
        # 2. Markdown 章节切片；
        # 3. 业务 metadata 补齐。
        nodes = self._pipeline.run(documents=documents)
        # 过滤出 TextNode，保证后续 BM25/向量检索处理的是文本切片。
        return [n for n in nodes if isinstance(n, TextNode)]

    def _get_embed_model(self):
        """懒加载 Embedding 模型。

        ``create_embed_model`` 已经接入模型网关配置：
        可以根据企业配置选择云端 embedding 或本地开源 embedding。

        这里不在 __init__ 里加载，是为了避免服务启动时就触发模型下载、
        API 连通性检查或云端鉴权；真正需要 RAG 时再初始化。
        """
        if self._embed_model is None:
            self._embed_model = create_embed_model(self._settings)
        return self._embed_model

    def _get_vector_store(self):
        """懒创建向量库连接。

        当前项目按配置返回 PGVector vector store。
        PGVector 不可用时直接抛错，不再回退到本地索引。
        """
        if self._vector_store is None:
            self._vector_store = create_vector_store(self._settings)
        return self._vector_store

    def _configured_external_vector_store(self) -> bool:
        """当前配置是否期望使用外部向量库。

        这个判断和 ``_get_vector_store()`` 的返回值不是完全一回事：
        - 配置期望外部向量库时，factory 连接失败会直接抛错。
        - 这里仍保留独立判断，便于跳过历史本地缓存复用。
        """
        return str(getattr(self._settings, "vector_store_type", "local") or "local").strip().lower() in {
            "pgvector",
        }

    def _get_or_build_bm25_retriever(self) -> BM25Retriever:
        """懒加载 BM25Retriever。

        BM25 是关键词检索，适合 SKU、规则名称、仓库名这类精确词。
        向量检索适合语义相近表达，BM25 和向量刚好互补。
        """
        if self._bm25_retriever is None:
            if not self._nodes:
                # BM25 依赖 self._nodes；如果还没有节点，先触发向量索引构建流程。
                self._get_or_build_index()
            self._bm25_retriever = BM25Retriever.from_defaults(
                nodes=self._nodes,
                similarity_top_k=self._retrieval_top_k(),
            )
        return self._bm25_retriever

    def _retrieval_top_k(self) -> int:
        """计算本次召回 top_k。

        小知识库里 chunk 数可能少于 RETRIEVAL_TOP_K。
        BM25Retriever 在 k 大于语料数量时可能报错，所以这里做保护。
        """
        # len(self._nodes) 为 0 时用 RETRIEVAL_TOP_K 兜底，再用 max 保证至少为 1。
        return max(1, min(RETRIEVAL_TOP_K, len(self._nodes) or RETRIEVAL_TOP_K))

    def _build_document_fingerprint(self) -> str:
        """计算索引缓存指纹。

        指纹不仅包含文档文件，还包含 embedding 配置、向量后端、chunk 参数。
        原因是：同一份文档换了 embedding 模型后，旧向量索引就不应该复用。
        """
        # 先把影响向量结果的配置写入指纹。
        parts = [
            f"embed_provider:{getattr(self._settings, 'embed_provider', 'local')}",
            f"embed_model:{getattr(self._settings, 'embed_model_name', '')}",
            f"vector_store:{getattr(self._settings, 'vector_store_type', 'local')}",
            f"chunk:{CHUNK_SIZE}:{CHUNK_OVERLAP}",
        ]
        try:
            from app.infrastructure.llm.model_gateway import ModelGateway

            # 如果模型网关里配置了 embedding profile，也加入指纹。
            # 这样从本地 embedding 换到云端 embedding 时，会自动重建索引。
            embedding_profile = ModelGateway(self._settings).resolve_profile(
                use_case="embedding",
                model_type="embedding",
            )
            if embedding_profile is not None:
                parts.extend(
                    [
                        f"gateway_embedding:{embedding_profile.id}",
                        f"gateway_embedding_model:{embedding_profile.model}",
                        f"gateway_embedding_dim:{embedding_profile.dimension}",
                    ]
                )
        except Exception:
            # 指纹计算不能因为模型网关配置异常导致服务完全不可用。
            # 真正使用模型时仍会在 create_embed_model 阶段暴露错误。
            pass
        for path_text in self.knowledge_repository.list_knowledge_paths():
            path = Path(path_text)
            stat = path.stat()
            # 文件名 + 修改时间 + 大小足够判断知识文件是否变化。
            parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
        # 排序后拼接，保证不同遍历顺序生成同一个指纹。
        return "|".join(sorted(parts))

    def _build_hybrid_retriever(self, filter_categories: list[str]) -> QueryFusionRetriever:
        """构建向量 + BM25 的混合召回器。

        QueryFusionRetriever 会分别调用两个 retriever：
        - vector_retriever：语义相似度召回。
        - bm25_retriever：关键词召回。

        mode="reciprocal_rerank" 表示用 RRF 融合排序。
        RRF 不直接比较两路分数，而是融合各自排名，更适合不同检索器混用。

        这里传入 MockLLM，是因为本项目的 query rewrite 已经由 RAGQueryPlanner 负责；
        QueryFusionRetriever 不需要再额外调用 LLM 生成子查询，避免重复改写和额外耗时。
        """
        # 向量检索：擅长语义相似，比如“库存不够”和“缺货处理”。
        vector_retriever = self._get_or_build_index().as_retriever(
            similarity_top_k=self._retrieval_top_k(),
            filters=self._metadata_filters(filter_categories),
        )
        # BM25 检索：擅长关键词精确匹配，比如 SKU、仓库、规则名。
        bm25_retriever = self._get_or_build_bm25_retriever()

        # QueryFusionRetriever 负责把多路 retriever 的结果融合成一个候选列表。
        return QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            mode="reciprocal_rerank",
            similarity_top_k=RETRIEVAL_TOP_K,
            num_queries=1,
            llm=MockLLM(),
            use_async=True,
            verbose=False,
        )

    def _metadata_filters(self, categories: list[str]) -> MetadataFilters | None:
        """构建 LlamaIndex metadata 过滤器。

        这个过滤器会传给向量 retriever，让向量检索只查指定 category。
        注意 BM25 不一定支持同样的过滤，所以后面还会二次过滤。

        过滤器只在用户显式选择规则来源/业务类别时生效；
        无过滤条件时返回 None，表示全知识库检索。
        """
        if not categories:
            return None
        # OR 表示 category 命中任意一个即可。
        return MetadataFilters(
            filters=[
                ExactMatchFilter(key="category", value=normalize_knowledge_category(category))
                for category in categories
            ],
            condition=FilterCondition.OR,
        )

    def _retrieve_with_expanded_queries(
        self,
        hybrid_retriever: QueryFusionRetriever,
        expanded_queries: list[str],
        filter_categories: list[str] | None = None,
    ) -> list[NodeWithScore]:
        """同步执行多个扩展查询，并合并结果。

        QueryPlanner 可能会生成多条 query，例如：
        - 用户原问题。
        - 带订单上下文的改写问题。
        - 带业务关键词的扩展问题。
        每条 query 都跑一次混合召回，最后统一去重。
        """
        retrieved: list[NodeWithScore] = []
        for query in expanded_queries:
            # retrieve 返回这一条 query 的融合结果。
            retrieved.extend(hybrid_retriever.retrieve(query))
        return self._dedupe_and_filter_nodes(retrieved, filter_categories)

    def _dedupe_and_filter_nodes(
        self,
        nodes: list[NodeWithScore],
        filter_categories: list[str] | None,
    ) -> list[NodeWithScore]:
        """合并多路召回结果，并在融合后统一应用类别过滤。

        这里有一个容易忽略的点：
        MetadataFilters 只传给了向量 retriever，BM25 retriever 不一定支持同样的过滤。
        所以融合之后还要再过滤一次，保证最终 hits 一定符合 filter_categories。
        """
        allowed = {normalize_knowledge_category(category) for category in (filter_categories or [])}
        seen: dict[str, NodeWithScore] = {}
        for node_with_score in nodes:
            # BM25 可能没有吃到 metadata filter，所以这里再拦一次。
            if allowed and normalize_knowledge_category(node_with_score.node.metadata.get("category")) not in allowed:
                continue
            # 过期规则检测必须放在融合后的统一过滤里。
            # 原因：向量检索可以做 metadata filter，但 BM25 召回通常不支持同样的过滤表达式。
            # 如果只在向量侧过滤，过期规则仍可能通过 BM25 混进最终结果。
            if self._is_expired_knowledge(node_with_score.node.metadata.get("expires_at")):
                continue
            if self._is_inactive_knowledge(
                node_with_score.node.metadata.get("is_active"),
                node_with_score.node.metadata.get("version_status"),
            ):
                continue
            node_key = str(
                node_with_score.node.metadata.get("chunk_id") or node_with_score.node.node_id
            )
            existing = seen.get(node_key)
            # 同一个 chunk 可能被多条 query、多路 retriever 命中。
            # 保留分数最高的那一次，避免重复片段挤占最终 top5。
            if existing is None or (node_with_score.score or 0) > (existing.score or 0):
                seen[node_key] = node_with_score
        return list(seen.values())

    @staticmethod
    def _is_expired_knowledge(value: object) -> bool:
        """判断知识片段是否已经过期。

        支持 front matter 中常见的 ``YYYY-MM-DD`` 或 ISO datetime 字符串。
        解析失败时不把文档当成过期，因为错误 metadata 应该在入库质检报告中暴露，
        检索侧保持保守，避免误杀仍然有效的规则。
        """

        if not value:
            return False
        try:
            expires_at = datetime.fromisoformat(str(value)[:10]).date()
        except ValueError:
            return False
        return expires_at < datetime.now(timezone.utc).date()

    @staticmethod
    def _is_inactive_knowledge(is_active: object, version_status: object) -> bool:
        if is_active is not None and str(is_active).strip().lower() in {"false", "0", "no", "inactive"}:
            return True
        if version_status and str(version_status).strip().lower() not in {"active", "current", "published"}:
            return True
        return False

    @staticmethod
    def _keyword_overlap_score(text: str, query: str) -> float:
        """计算 query 与 chunk 的轻量关键词重合分，范围 0~1。

        这个分数不是 BM25 原始分，而是工程上可解释的关键词信号：
        - 优先统计供应链业务术语是否同时出现在 query 和 chunk。
        - 如果没有业务术语，再退回到英文/数字词的重合度，例如 SKU、仓库编号。
        """
        text_lower = text.lower()
        query_lower = query.lower()
        matched_terms = [
            term for term in RAG_KEYWORD_TERMS
            if term.lower() in text_lower and term.lower() in query_lower
        ]
        query_terms = [term for term in RAG_KEYWORD_TERMS if term.lower() in query_lower]
        if query_terms:
            return round(min(len(matched_terms) / max(len(query_terms), 1), 1.0), 4)

        query_words = set(re.findall(r"[a-zA-Z0-9_-]{2,}", query_lower))
        text_words = set(re.findall(r"[a-zA-Z0-9_-]{2,}", text_lower))
        if not query_words:
            return 0.0
        return round(len(query_words & text_words) / len(query_words), 4)

    def _node_to_hit(
        self,
        node_with_score: NodeWithScore,
        merged_query: str,
        intent: QueryIntent,
    ) -> KnowledgeHit:
        """把 LlamaIndex 的 NodeWithScore 转成项目自己的 KnowledgeHit。

        NodeWithScore 是框架内部对象；KnowledgeHit 是我们对外暴露的结构。
        转换时会补齐 source_file、category、score_detail、matched_terms 等字段。
        """
        # node 是真正的文本切片；node_with_score 外层只额外包了一层检索分数。
        node = node_with_score.node
        # get_content() 取出用于展示/给 LLM 的知识正文。
        text = node.get_content()
        # metadata 是前面 BusinessMetadataEnricher 补进去的业务字段。
        metadata = node.metadata
        # category 用于前端展示和后续业务分析；没有时兜底 general。
        category = metadata.get("category", "general")
        # source_file 优先从 file_path/source_file 推出来，最终只保留文件名。
        source_file = Path(
            str(metadata.get("file_path") or metadata.get("source_file", "unknown"))
        ).name

        # matched_terms 用于解释“为什么这段知识被命中”。
        matched_terms = self._answer_builder.extract_matched_terms(text, merged_query)
        # 这里把排序链路中逐步写入 metadata 的分数拆出来：
        # semantic_score：向量/RRF 基础分。
        # keyword_score：query 和 chunk 的关键词重合度。
        # business_rule_score：业务意图类别加权。
        # rerank_score：reranker 的归一化分。
        semantic_score = float(metadata.get("_rag_semantic_score", metadata.get("_rag_base_score", node_with_score.score or 0.0)))
        keyword_score = float(metadata.get("_rag_keyword_score", self._keyword_overlap_score(text, merged_query)))
        business_rule_score = float(metadata.get("_rag_business_rule_score", 0.0))
        rerank_score = float(metadata.get("_rag_rerank_normalized_score", 0.0))
        # final_score 是当前最终排序分数，保留四位方便前端展示。
        final_score = round(float(metadata.get("_rag_final_score", node_with_score.score or 0.0)), 4)
        score_detail = HybridScoreDetail(
            semantic_score=round(semantic_score, 4),
            keyword_score=round(keyword_score, 4),
            business_rule_score=round(business_rule_score, 4),
            rerank_score=round(rerank_score, 4),
            final_score=final_score,
        )
        retrieval_channels = self._metadata_list(metadata.get("_rag_retrieval_channels", ["semantic"]))
        if keyword_score > 0 and "keyword" not in retrieval_channels:
            retrieval_channels.append("keyword")
        if business_rule_score > 0:
            retrieval_channels.append("business_rule")
        if rerank_score > 0:
            retrieval_channels.append("reranker")

        # KnowledgeMetadata 是对外稳定结构，避免把 LlamaIndex 原始 metadata 直接暴露出去。
        km = KnowledgeMetadata(
            document_id=str(metadata.get("document_id", "unknown")),
            chunk_id=str(metadata.get("chunk_id", node.node_id)),
            parent_chunk_id=str(metadata.get("parent_chunk_id", "")),
            parent_section_path=self._metadata_list(metadata.get("parent_section_path", [])),
            parent_context_excerpt=str(metadata.get("parent_context_excerpt", "")),
            child_index_in_parent=int(metadata.get("child_index_in_parent", 0) or 0),
            child_count_in_parent=int(metadata.get("child_count_in_parent", 1) or 1),
            business_unit_type=str(metadata.get("business_unit_type", "general")),
            title=str(metadata.get("title", "unknown")),
            section_path=self._metadata_list(metadata.get("section_path", [])),
            tags=self._metadata_list(metadata.get("tags", [])),
            source_file=str(metadata.get("source_file", source_file)),
            source_path=str(metadata.get("source_path", source_file)),
            version=str(metadata.get("version", "")),
            owner=str(metadata.get("owner", "")),
            effective_date=str(metadata.get("effective_date", "")),
            expires_at=str(metadata.get("expires_at", "")),
            region=str(metadata.get("region", "")),
            business_scope=self._metadata_list(metadata.get("business_scope", [])),
            knowledge_source=str(metadata.get("knowledge_source", "")),
            collection_name=str(metadata.get("collection_name", "")),
            version_status=str(metadata.get("version_status", "")),
            is_active=str(metadata.get("is_active", "")),
            vector_backend=str(metadata.get("vector_backend", "")),
            source_case_id=str(metadata.get("source_case_id", "")),
        )

        # KnowledgeHit 是最终给 API/Agent 的命中文档片段。
        return KnowledgeHit(
            score=final_score,
            score_detail=score_detail,
            source_file=source_file,
            category=category,
            metadata=km,
            retrieval_channels=list(dict.fromkeys(retrieval_channels)),
            matched_terms=matched_terms,
            text=text,
        )

    @staticmethod
    def _metadata_list(value: object) -> list[str]:
        """把 metadata 中的列表字段统一还原成 ``list[str]``。

        为什么需要这个方法：
        - 本地 LlamaIndex 节点里，``tags``/``section_path`` 可能本来就是 list。
        - 写入 PGVector 等外部向量库前，复杂 metadata 会被转成 JSON 字符串。
        - 少数情况下字段可能只是普通字符串。

        对外返回前统一转成 list，可以让前端和 API schema 不用关心底层存储差异。
        """
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except Exception:
                    # 不是合法 JSON 时按普通字符串处理，避免 metadata 解析问题影响整次检索。
                    pass
            return [stripped]
        return [str(value)]
