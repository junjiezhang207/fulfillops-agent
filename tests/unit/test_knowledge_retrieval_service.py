"""KnowledgeRetrievalService 的轻量回归测试。

这些测试不是为了跑完整 RAG 流程，而是锁住几个容易出问题的小逻辑：
1. 多路召回后的去重和过滤。
2. 小知识库下 top_k 不能超过语料大小。
3. 有 LLM Query Rewriter 时，也不能丢掉规则扩展 query。

学习时可以把这些测试当成 RAG 工程边界的例子。
"""

from types import SimpleNamespace

from llama_index.core.schema import NodeWithScore, TextNode

from app.schemas.knowledge import QueryIntent, QueryIntentType
from app.rag.knowledge_retrieval_service import BusinessMetadataEnricher, KnowledgeRetrievalService
from app.rag.rag_answer_builder import RAGAnswerBuilder
from app.rag.rag_query_planner import RAGQueryPlanner, RetrievalInputs
from app.rag.reranker import RankedPassage


def _node(chunk_id: str, category: str, score: float, text: str | None = None) -> NodeWithScore:
    """构造一个最小 NodeWithScore。

    真正的检索结果来自 LlamaIndex，这里只测我们自己的处理逻辑，
    所以手动造 TextNode 就够了。
    """
    text_node = TextNode(
        text=text or f"{category} content",
        metadata={"chunk_id": chunk_id, "category": category},
    )
    return NodeWithScore(node=text_node, score=score)


def test_dedupe_and_filter_nodes_applies_to_all_retrieval_channels():
    """融合后过滤必须覆盖 BM25 和向量两路结果。

    背景：
    - 向量检索可以接 MetadataFilters。
    - BM25 检索不一定支持同样的 metadata 过滤。
    - 所以最终合并后必须再过滤一次。

    这个测试还顺带验证：同一个 chunk 被多路召回时，保留最高分。
    """
    service = KnowledgeRetrievalService.__new__(KnowledgeRetrievalService)
    nodes = [
        _node("same", "stockout_rule", 0.3),
        _node("same", "stockout_rule", 0.8),
        _node("other", "regional_strategy", 0.9),
    ]

    result = service._dedupe_and_filter_nodes(nodes, ["stockout_rule"])

    assert len(result) == 1
    assert result[0].node.metadata["chunk_id"] == "same"
    assert result[0].score == 0.8
    assert result[0].node.metadata["category"] == "stockout_rule"


def test_retrieval_top_k_never_exceeds_corpus_size():
    """BM25 的 top_k 不能大于语料数量。

    小知识库或测试知识库里可能只有几个 chunk。
    如果仍然传 top_k=8，部分 BM25 实现会直接报错。
    """
    service = KnowledgeRetrievalService.__new__(KnowledgeRetrievalService)
    service._nodes = [TextNode(text="a"), TextNode(text="b")]

    assert service._retrieval_top_k() == 2


def test_retrieval_top_k_keeps_default_for_unbuilt_corpus():
    """索引还没构建时，先保留默认 top_k。

    _nodes 为空可能有两种情况：
    1. 真的没有文档。
    2. 索引还没有懒加载。

    这里选择返回默认值，让真正构建索引的地方决定后续行为。
    """
    service = KnowledgeRetrievalService.__new__(KnowledgeRetrievalService)
    service._nodes = []

    assert service._retrieval_top_k() == 8


def test_llm_rewrite_keeps_rule_based_business_queries():
    """LLM 改写不能替代业务规则扩展。

    LLM 可能改写出更自然的 query，但它也可能漏掉文档里的固定术语。
    所以最终 query 列表应该同时包含：
    - LLM 改写结果
    - 基于 QueryIntent 的规则 query
    - 基于库存状态的业务 query
    """
    planner = RAGQueryPlanner(
        inventory_analysis_service=SimpleNamespace(),
        query_rewriter=SimpleNamespace(rewrite=lambda question, n: [question, "LLM 改写查询"]),
    )
    inventory = SimpleNamespace(
        insufficient_skus=["SKU-A"],
        fulfillment_ready=False,
    )
    prepared = RetrievalInputs(
        active_filters=[],
        inventory_result=inventory,
        intent=QueryIntent(
            primary_intent=QueryIntentType.STOCKOUT_HANDLING,
            confidence=0.9,
            reasoning="测试",
            secondary_intents=[],
        ),
        retrieval_context="订单库存不足，需要查询缺货处理规则",
    )

    queries = planner.build_queries("库存不足怎么办", prepared)

    assert "LLM 改写查询" in queries
    assert "缺货履约规则" in queries
    assert "库存不足 SKU：SKU-A，优先检索缺货处理与跨仓规则" in queries


def test_business_metadata_enricher_uses_stable_chunk_id_as_node_id():
    """Milvus upsert 依赖稳定 node_id，避免重启后重复写入同一 chunk。"""
    node = TextNode(text="# 缺货规则\n\n库存不足时需要人工复核", metadata={"file_path": "stockout_rules.md"})

    [enriched] = BusinessMetadataEnricher()([node])

    assert enriched.metadata["chunk_id"] == "stockout_rules::chunk-0000"
    assert enriched.node_id == "stockout_rules::chunk-0000"


def test_business_metadata_enricher_numbers_chunks_per_document():
    """chunk_id must be stable per document, not based on the whole batch order."""
    nodes = [
        TextNode(text="A chunk 1", metadata={"file_path": "stockout_rules.md"}),
        TextNode(text="A chunk 2", metadata={"file_path": "stockout_rules.md"}),
        TextNode(text="B chunk 1", metadata={"file_path": "priority_orders.md"}),
    ]

    enriched = BusinessMetadataEnricher()(nodes)

    assert enriched[0].metadata["chunk_id"] == "stockout_rules::chunk-0000"
    assert enriched[1].metadata["chunk_id"] == "stockout_rules::chunk-0001"
    assert enriched[2].metadata["chunk_id"] == "priority_orders::chunk-0000"


def test_document_registry_change_plan_deletes_updated_and_removed_chunks():
    """Document replacement should delete old vector chunks before writing new ones."""
    service = KnowledgeRetrievalService.__new__(KnowledgeRetrievalService)
    previous = {
        "stockout_rules": {
            "content_hash": "old",
            "chunk_size": 220,
            "chunk_overlap": 40,
            "chunk_ids": ["stockout_rules::chunk-0000", "stockout_rules::chunk-0001"],
        },
        "deleted_rules": {
            "content_hash": "gone",
            "chunk_size": 220,
            "chunk_overlap": 40,
            "chunk_ids": ["deleted_rules::chunk-0000"],
        },
    }
    current = {
        "stockout_rules": {
            "content_hash": "new",
            "chunk_size": 220,
            "chunk_overlap": 40,
            "chunk_ids": ["stockout_rules::chunk-0000"],
        },
        "priority_orders": {
            "content_hash": "created",
            "chunk_size": 220,
            "chunk_overlap": 40,
            "chunk_ids": ["priority_orders::chunk-0000"],
        },
    }

    changes = service._build_document_change_plan(
        current_documents=current,
        previous_documents=previous,
    )
    by_id = {change["document_id"]: change for change in changes}

    assert by_id["stockout_rules"]["change_type"] == "updated"
    assert by_id["stockout_rules"]["stale_chunk_ids"] == [
        "stockout_rules::chunk-0000",
        "stockout_rules::chunk-0001",
    ]
    assert by_id["deleted_rules"]["change_type"] == "deleted"
    assert by_id["deleted_rules"]["stale_chunk_ids"] == ["deleted_rules::chunk-0000"]
    assert by_id["priority_orders"]["change_type"] == "created"
    assert by_id["priority_orders"]["stale_chunk_ids"] == []


def test_delete_stale_vector_chunks_calls_vector_store_delete_nodes():
    """Milvus/LlamaIndex vector stores should receive stale node ids for cleanup."""
    service = KnowledgeRetrievalService.__new__(KnowledgeRetrievalService)

    class FakeVectorStore:
        def __init__(self):
            self.deleted_node_ids = []

        def delete_nodes(self, node_ids):
            self.deleted_node_ids = node_ids

    store = FakeVectorStore()

    service._delete_stale_vector_chunks(
        store,
        [
            {"document_id": "a", "change_type": "updated", "stale_chunk_ids": ["a::chunk-0001"]},
            {"document_id": "b", "change_type": "deleted", "stale_chunk_ids": ["b::chunk-0000"]},
        ],
    )

    assert store.deleted_node_ids == ["a::chunk-0001", "b::chunk-0000"]


def test_rebuild_index_cleans_vectors_when_all_documents_deleted(tmp_path):
    """If the knowledge directory becomes empty, old Milvus chunks still need cleanup."""
    service = KnowledgeRetrievalService.__new__(KnowledgeRetrievalService)
    service.knowledge_repository = SimpleNamespace(
        rebuild_index_required=lambda: True,
        list_knowledge_paths=lambda: [],
    )
    service._settings = SimpleNamespace(knowledge_index_cache_dir=str(tmp_path))
    service._index = "old-index"
    service._nodes = [TextNode(text="old")]
    service._bm25_retriever = object()

    class FakeVectorStore:
        def __init__(self):
            self.deleted_node_ids = []

        def delete_nodes(self, node_ids):
            self.deleted_node_ids = node_ids

    service._vector_store = FakeVectorStore()
    registry_path = tmp_path / "knowledge_document_registry.json"
    registry_path.write_text(
        """
        {
          "schema_version": 1,
          "documents": {
            "old_rules": {
              "content_hash": "old",
              "chunk_ids": ["old_rules::chunk-0000"]
            }
          }
        }
        """,
        encoding="utf-8",
    )

    message = service.rebuild_index()

    assert "知识目录为空" in message
    assert service._vector_store.deleted_node_ids == ["old_rules::chunk-0000"]
    assert service._index is None
    assert service._nodes == []


def test_score_detail_explains_keyword_and_business_rule_signals():
    """RAG 命中结果要能解释关键词分和业务规则分。

    这个测试锁住企业级 RAG 面试里最容易被追问的点：
    不是只有一个黑盒 final_score，而是能看到每个排序信号。
    """
    service = KnowledgeRetrievalService.__new__(KnowledgeRetrievalService)
    service._cross_encoder = None
    service._answer_builder = RAGAnswerBuilder()
    intent = QueryIntent(
        primary_intent=QueryIntentType.STOCKOUT_HANDLING,
        confidence=0.9,
        reasoning="测试",
        secondary_intents=[],
    )
    nodes = [
        _node(
            "stockout",
            "stockout_rule",
            0.2,
            text="库存不足时进入缺货订单 SOP，并建议跨仓调拨和人工复核。",
        ),
        _node("general", "general", 0.1, text="通用履约说明。"),
    ]

    ranked = service._rank_nodes("库存不足怎么办", ["库存不足怎么办", "缺货履约规则"], nodes, intent)
    [hit] = service._hits_from_nodes(ranked[:1], ["库存不足怎么办"], intent)

    assert hit.category == "stockout_rule"
    assert hit.score_detail.semantic_score == 0.2
    assert hit.score_detail.keyword_score > 0
    assert hit.score_detail.business_rule_score == 0.35
    assert hit.score_detail.final_score == hit.score
    assert "keyword" in hit.retrieval_channels
    assert "business_rule" in hit.retrieval_channels


class _FakeReranker:
    """测试用 reranker：故意把第二段排到第一，模拟重排模型生效。"""

    def rerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        return [item.text for item in self.rerank_with_scores(query, passages, top_k)]

    async def arerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        return self.rerank(query, passages, top_k)

    def rerank_with_scores(
        self, query: str, passages: list[str], top_k: int | None = None
    ) -> list[RankedPassage]:
        preferred = passages[1]
        return [
            RankedPassage(text=preferred, original_score=0.0, rerank_score=2.0),
            RankedPassage(text=passages[0], original_score=0.0, rerank_score=0.5),
        ][: top_k or 2]


def test_reranker_score_is_written_to_hit_detail():
    """开启 reranker 后，命中结果要能看到 rerank_score 和 reranker 通道。"""
    service = KnowledgeRetrievalService.__new__(KnowledgeRetrievalService)
    service._cross_encoder = _FakeReranker()
    service._answer_builder = RAGAnswerBuilder()
    intent = QueryIntent(
        primary_intent=QueryIntentType.GENERAL,
        confidence=0.5,
        reasoning="测试",
        secondary_intents=[],
    )
    nodes = [
        _node("a", "general", 0.9, text="普通履约规则。"),
        _node("b", "general", 0.2, text="更相关的履约规则。"),
    ]

    ranked = service._rank_nodes("履约规则", ["履约规则"], nodes, intent)
    [hit] = service._hits_from_nodes(ranked[:1], ["履约规则"], intent)

    assert hit.metadata.chunk_id == "b"
    assert hit.score_detail.rerank_score == 1.0
    assert "reranker" in hit.retrieval_channels


def test_rag_golden_dataset_expected_category_ranks_first():
    """轻量 golden dataset：固定问题应该把对应知识类别排到第一。

    这里不启动真实向量库，而是固定一组候选 chunk，验证 RAG 排序层能根据
    intent + keyword signal 把正确类别顶上来。
    """
    service = KnowledgeRetrievalService.__new__(KnowledgeRetrievalService)
    service._cross_encoder = None
    cases = [
        ("库存不足应该怎么处理？", QueryIntentType.STOCKOUT_HANDLING, "stockout_rule"),
        ("高优先级订单要怎么履约？", QueryIntentType.PRIORITY_FULFILLMENT, "priority_rule"),
        ("售后补发要检查什么规则？", QueryIntentType.AFTER_SALES, "after_sales_rule"),
    ]
    candidate_nodes = [
        _node("stockout", "stockout_rule", 0.1, text="缺货和库存不足时，需要跨仓调拨并进入 SOP。"),
        _node("priority", "priority_rule", 0.1, text="高优先级订单需要优先履约并人工关注时效。"),
        _node("after_sales", "after_sales_rule", 0.1, text="售后补发、退货和质检需要检查库存更新规则。"),
    ]

    for question, primary_intent, expected_category in cases:
        intent = QueryIntent(
            primary_intent=primary_intent,
            confidence=0.9,
            reasoning="golden dataset",
            secondary_intents=[],
        )
        ranked = service._rank_nodes(question, [question], list(candidate_nodes), intent)

        assert ranked[0].node.metadata["category"] == expected_category
