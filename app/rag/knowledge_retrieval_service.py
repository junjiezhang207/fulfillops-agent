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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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
from app.domain.inventory.analysis import InventoryAnalysisService
from app.rag.query_rewriter import QueryRewriter
from app.rag.rag_answer_builder import RAGAnswerBuilder
from app.rag.rag_query_planner import RAGQueryPlanner, RetrievalInputs
from app.rag.reranker import ContextualCompressor, PassageReranker


logger = logging.getLogger(__name__)

# 每个知识片段的目标长度。太短会丢上下文，太长会让召回不够精准。
CHUNK_SIZE = 220
# 相邻切片保留一点重叠，避免规则条款刚好被切断后检索不到完整语义。
CHUNK_OVERLAP = 40
# 每一路检索最多召回多少个候选片段；后面还会去重、重排、截断。
RETRIEVAL_TOP_K = 8
# 最终返回给前端/Agent 的知识片段数量，控制回答上下文不要过长。
FINAL_HIT_COUNT = 5
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
)

# 文件名关键字 -> 业务知识分类。
# 这里不是为了“硬编码业务”，而是给 Markdown 知识文件建立最基础的可过滤标签。
CATEGORY_BY_FILE_KEYWORD = {
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
            source_file = Path(
                str(node.metadata.get("file_path") or node.metadata.get("filename") or "unknown")
            ).name
            # 通过文件名推断知识类别，例如 stockout_rules.md -> stockout_rule。
            category = self._infer_category(source_file)
            # document_id 用文件名去掉 .md，方便前端或日志展示。
            document_id = source_file.removesuffix(".md")

            # chunk_id 必须稳定：同一个文档同一个切片顺序生成相同 id。
            # 后面向量 + BM25 融合去重时，就靠这个字段判断是不是同一段知识。
            # 注意这里必须按 document_id 独立计数，不能用整批节点的 enumerate。
            # 否则 A 文档切片数量变化后，B 文档的 chunk 编号也会漂移，Milvus 旧数据就很难清理干净。
            order = chunk_counters.get(document_id, 0)
            chunk_counters[document_id] = order + 1
            chunk_id = f"{document_id}::chunk-{order:04d}"
            node.id_ = chunk_id

            # title/section_path/tags 都是为了“可解释召回”。
            # 面试时可以说：不是只返回一段文本，还能说清楚它来自哪个业务章节。
            title = self._extract_title(node.get_content(), source_file)
            section_path = self._extract_section_path(node.get_content(), title)
            tags = sorted(set([category, *section_path]))

            # metadata 会随 node 一起写入向量库。Milvus/本地索引都可以用这些字段过滤或展示。
            node.metadata.update(
                {
                    "category": category,
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "section_path": section_path,
                    "tags": tags,
                    "source_file": source_file,
                    "source_path": str(node.metadata.get("file_path", source_file)),
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


class KnowledgeRetrievalService:
    """面向履约场景的 RAG 检索服务。

    这个类是 RAG 的“编排层”，它不自己写 embedding 算法，也不自己写 BM25 算法，
    而是把 LlamaIndex、模型网关、知识仓库、业务意图识别组合起来。

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
        cross_encoder: PassageReranker | None = None,
        compressor: ContextualCompressor | None = None,
    ) -> None:
        # 知识文件访问层。这个服务不直接写死 app/data/knowledge，方便未来换存储。
        self.knowledge_repository = knowledge_repository

        # 向量索引懒加载缓存。第一次检索才构建，避免项目启动时加载模型很慢。
        self._index: VectorStoreIndex | None = None
        # 当前知识库切出来的 TextNode。BM25 必须基于内存节点构建。
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
        )
        # AnswerBuilder 负责“把命中的片段组装成结果/摘要”，和召回逻辑分离。
        self._answer_builder = RAGAnswerBuilder()

        self._settings = get_settings()
        # Embedding/RAG 向量库可能会加载本地模型或连接 Milvus，放到第一次检索时懒初始化，
        # 避免 FastAPI 启动阶段被模型权重加载卡住。
        self._embed_model = None
        self._vector_store = None
        # 最近一次重建时的文档变化摘要，用于 API 返回和排查“旧文档是否已清理”。
        self._last_registry_changes: list[dict[str, object]] = []
        # IngestionPipeline 是 LlamaIndex 的文档处理流水线：
        # 先切片，再给每个切片补业务 metadata。
        self._pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
                BusinessMetadataEnricher(),
            ]
        )

    def retrieve(
        self,
        order_id: str,
        question: str,
        filter_categories: list[str] | None = None,
    ) -> KnowledgeRetrieveResult:
        """同步检索入口。

        同步链路一般给普通 FastAPI service 或单元测试使用。
        流程固定为：
        1. prepare：识别意图、整理订单上下文、确定过滤类别。
        2. retrieve：混合召回候选知识片段。
        3. rank/build：业务重排、可选 rerank、转 KnowledgeHit。
        4. build_result：组装最终结果。
        """
        prepared = self._query_planner.prepare(order_id, question, filter_categories)
        retrieved = self._retrieve_nodes(question, prepared)
        hits = self._rank_and_build_hits(question, retrieved, prepared.intent)
        return self._build_result(order_id, question, prepared, retrieved.queries, hits)

    async def aretrieve(
        self,
        order_id: str,
        question: str,
        filter_categories: list[str] | None = None,
    ) -> "KnowledgeRetrieveResult":
        """异步检索入口。

        给异步 API / Agent 调用使用。它和同步版本返回结构一致，
        只是 query rewrite、检索、rerank、compress 尽量走异步方法。
        """
        prepared = self._query_planner.prepare(order_id, question, filter_categories)
        retrieved = await self._aretrieve_nodes(question, prepared)
        hits = await self._arank_and_build_hits(question, retrieved, prepared.intent)
        return self._build_result(order_id, question, prepared, retrieved.queries, hits)

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
        """
        keyword_scored = self._annotate_keyword_scores(nodes, expanded_queries)
        ranked = self._postprocess_by_business_rules(keyword_scored, expanded_queries, intent)
        if self._cross_encoder is None or not ranked:
            return ranked
        # reranker 通常输入纯文本列表，返回最相关的 passage 文本和相关性分数。
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
            semantic_score = float(node_with_score.score or 0.0)
            keyword_score = self._keyword_overlap_score(
                node_with_score.node.get_content(),
                merged_query,
            )
            keyword_boost = keyword_score * KEYWORD_SCORE_WEIGHT
            node_with_score.node.metadata.update(
                {
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
        # 重新构建向量索引；本地模式会写缓存，Milvus 模式会写外部向量库。
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
        """懒加载向量索引。"""
        # 第一次访问时加载/构建；后续复用，避免重复 embedding。
        if self._index is None:
            self._index = self._load_or_build_index()
        return self._index

    def _load_or_build_index(self) -> VectorStoreIndex:
        """优先加载缓存索引，缓存不可用时重新构建。"""
        # LlamaIndex 的索引构建会读取全局 Settings.embed_model。
        Settings.embed_model = self._get_embed_model()

        vector_store = self._get_vector_store()
        if vector_store is not None:
            # 外部向量库由后端自己管理持久化，不依赖本地 doc_fingerprint。
            return self._build_index(persist=False)

        if self._configured_external_vector_store():
            # 配置期望使用 Milvus/Zilliz，但连接失败后按开发策略回退到了本地存储。
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

    def _build_index(self, persist: bool) -> VectorStoreIndex:
        """构建向量索引，并按配置选择本地持久化或外部向量库。

        VectorStoreIndex 是 LlamaIndex 的核心索引对象。
        它负责把 TextNode 通过 embedding 模型转成向量，并写入对应的存储后端。
        """
        # 构建索引前先重新跑文档流水线，拿到最新 TextNode。
        self._nodes = self._run_ingestion_pipeline()
        current_documents = self._build_current_document_records(self._nodes)
        previous_registry = self._read_document_registry()
        changes = self._build_document_change_plan(
            current_documents=current_documents,
            previous_documents=previous_registry.get("documents", {}),
        )

        vector_store = self._get_vector_store()
        if vector_store is not None:
            # 外部向量库不会自动知道“某个文档删了/变短了”。
            # 所以重建前先按旧 registry 里的 chunk_id 删除过期向量，再写入新切片。
            self._delete_stale_vector_chunks(vector_store, changes)
            # 外部向量库模式：把 Milvus 等 vector_store 注入 LlamaIndex StorageContext。
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

    def list_document_statuses(self) -> list[dict[str, object]]:
        """返回知识文档与索引注册表的对比状态。

        这个方法给知识库管理 API 使用，用来判断：
        - 文件是否已进入索引。
        - 当前文件内容是否和上一次索引时一致。
        - 是否存在已经从目录删除、但注册表里仍有记录的文档。
        """

        registry = self._read_document_registry()
        indexed_docs = registry.get("documents", {})
        current_files = self._scan_knowledge_files()
        rows: list[dict[str, object]] = []

        for document_id, file_info in current_files.items():
            indexed = indexed_docs.get(document_id)
            if indexed is None:
                sync_status = "new"
                version = 1
                indexed_hash = ""
                chunk_count = 0
            elif indexed.get("content_hash") != file_info["content_hash"]:
                sync_status = "changed"
                version = int(indexed.get("version", 1))
                indexed_hash = str(indexed.get("content_hash", ""))
                chunk_count = len(indexed.get("chunk_ids", []))
            else:
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
        """返回知识文档注册表快照，供状态页展示和面试讲解。"""

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
        """读取知识文档注册表；不存在或损坏时返回空注册表。"""

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
        下次重建索引时，可以知道旧版本有哪些 chunk_id，从而在 Milvus 里先删旧 chunk。
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
        """扫描当前知识目录中的 Markdown 文档，并计算内容 hash。"""

        files: dict[str, dict[str, object]] = {}
        for path_text in self.knowledge_repository.list_knowledge_paths():
            path = Path(path_text)
            if not path.exists() or not path.is_file():
                continue
            content = path.read_bytes()
            document_id = path.stem
            files[document_id] = {
                "document_id": document_id,
                "filename": path.name,
                "source_path": str(path),
                "category": BusinessMetadataEnricher()._infer_category(path.name),
                "size_bytes": len(content),
                "last_modified": path.stat().st_mtime,
                "content_hash": hashlib.sha256(content).hexdigest(),
            }
        return files

    def _build_current_document_records(
        self,
        nodes: list[TextNode],
    ) -> dict[str, dict[str, object]]:
        """根据当前文件和切片结果生成新的文档注册表 documents 字段。"""

        old_documents = self._read_document_registry().get("documents", {})
        current_files = self._scan_knowledge_files()
        chunk_ids_by_doc: dict[str, list[str]] = {}
        for node in nodes:
            document_id = str(node.metadata.get("document_id", "unknown"))
            chunk_id = str(node.metadata.get("chunk_id", node.node_id))
            chunk_ids_by_doc.setdefault(document_id, []).append(chunk_id)

        documents: dict[str, dict[str, object]] = {}
        for document_id, file_info in current_files.items():
            old = old_documents.get(document_id, {})
            content_changed = old.get("content_hash") != file_info["content_hash"]
            old_version = int(old.get("version", 0) or 0)
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

        返回中的 stale_chunk_ids 会交给 Milvus delete_nodes。
        - created：新文档，没有旧 chunk 需要删。
        - updated：同 document_id 内容或切片发生变化，先删旧 chunk 再写新 chunk。
        - deleted：文件已不存在，删除旧 chunk。
        - unchanged：没有变化，不需要清理。
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

        LlamaIndex 的 MilvusVectorStore 支持 delete_nodes(node_ids=[...])。
        这里按 registry 里记录的旧 chunk_id 删除，可以覆盖三类企业常见场景：
        1. 同名文档替换：删旧 chunk，再写新 chunk。
        2. 文档变短：旧版本多出来的 chunk 不会残留。
        3. 文档删除：旧文档不会继续被 RAG 命中。
        """

        stale_chunk_ids = sorted({
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
        # IngestionPipeline 顺序执行 SentenceSplitter 和 BusinessMetadataEnricher。
        nodes = self._pipeline.run(documents=documents)
        # 过滤出 TextNode，保证后续 BM25/向量检索处理的是文本切片。
        return [n for n in nodes if isinstance(n, TextNode)]

    def _get_embed_model(self):
        """懒加载 Embedding 模型。

        ``create_embed_model`` 已经接入模型网关配置：
        可以根据企业配置选择云端 embedding 或本地开源 embedding。
        """
        if self._embed_model is None:
            self._embed_model = create_embed_model(self._settings)
        return self._embed_model

    def _get_vector_store(self):
        """懒创建向量库连接。

        当前项目可以按配置返回 Milvus vector store；
        如果返回 None，就代表使用 LlamaIndex 本地索引缓存。
        """
        if self._vector_store is None:
            self._vector_store = create_vector_store(self._settings)
        return self._vector_store

    def _configured_external_vector_store(self) -> bool:
        """当前配置是否期望使用外部向量库。"""
        return str(getattr(self._settings, "vector_store_type", "local") or "local").strip().lower() in {
            "milvus",
            "zilliz",
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
        """
        if not categories:
            return None
        # OR 表示 category 命中任意一个即可。
        return MetadataFilters(
            filters=[
                ExactMatchFilter(key="category", value=category)
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
        allowed = set(filter_categories or [])
        seen: dict[str, NodeWithScore] = {}
        for node_with_score in nodes:
            # BM25 可能没有吃到 metadata filter，所以这里再拦一次。
            if allowed and node_with_score.node.metadata.get("category") not in allowed:
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
        retrieval_channels = list(metadata.get("_rag_retrieval_channels", ["semantic"]))
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
            title=str(metadata.get("title", "unknown")),
            section_path=list(metadata.get("section_path", [])),
            tags=list(metadata.get("tags", [])),
            source_file=str(metadata.get("source_file", source_file)),
            source_path=str(metadata.get("source_path", source_file)),
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
