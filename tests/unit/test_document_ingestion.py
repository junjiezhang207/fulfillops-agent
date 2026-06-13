"""RAG 文档入库流水线的轻量回归测试。

这些测试不跑真实 embedding，也不依赖向量库，只锁住入库前最关键的工程边界：
- 表格转 Markdown；
- 敏感信息脱敏和重复段落去重；
- pending -> confirm 人工确认流程；
- 版本报告落盘；
- 过期规则不会进入 RAG 最终候选。
"""

from llama_index.core.schema import NodeWithScore, TextNode

from app.rag.document_ingestion import DocumentIngestionService, MarkdownTableCleaner, TextCleaner
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService


def test_table_cleaner_turns_rows_into_markdown_table():
    """Word/PDF 表格进入 RAG 前应保持结构，而不是被压成难读的一行文本。"""

    table = [
        ["客户等级", "SLA", "处理动作"],
        ["VIP", "24小时", "优先发货"],
        ["普通", "48小时", "标准发货"],
    ]

    markdown = MarkdownTableCleaner.table_to_markdown(table)

    assert "| 客户等级 | SLA | 处理动作 |" in markdown
    assert "| VIP | 24小时 | 优先发货 |" in markdown


def test_text_cleaner_redacts_sensitive_and_dedupes_blocks():
    """敏感字段不应该进入向量库，重复段落也不应该反复污染召回。"""

    raw = """
    # 缺货规则

    联系运营 13812345678 或 ops@example.com。

    本文档仅供内部使用，请勿外传。

    本文档仅供内部使用，请勿外传。
    """

    cleaned = TextCleaner.clean(raw)

    assert "[PHONE]" in cleaned.markdown
    assert "[EMAIL]" in cleaned.markdown
    assert cleaned.removed_duplicate_blocks == 1
    assert cleaned.redacted_sensitive_items == 2


def test_ingestion_pending_confirm_writes_markdown_and_version_report(tmp_path):
    """PDF/Word 这类解析型文档默认应先 pending，人工确认后才进入正式知识目录。"""

    knowledge_dir = tmp_path / "knowledge"
    storage_dir = tmp_path / "ingestion"
    service = DocumentIngestionService(knowledge_dir=str(knowledge_dir), storage_dir=storage_dir)

    report = service.prepare_upload(
        filename="stockout.txt",
        raw="缺货时先检查同区域库存，再考虑跨仓调拨。".encode("utf-8"),
        document_id="stockout_rules",
        metadata={"category": "stockout_rule", "owner": "ops"},
        require_confirmation=True,
        replace_existing=False,
    )

    assert report.status == "pending_confirmation"
    detail = service.get_pending(report.ingestion_id)
    assert "markdown_preview" in detail

    committed = service.confirm_pending(report.ingestion_id, replace_existing=False)

    assert committed["document_id"] == "stockout_rules"
    assert (knowledge_dir / "stockout_rules.md").exists()
    versions = service.list_versions("stockout_rules")
    assert len(versions) == 1
    assert versions[0]["metadata"]["category"] == "stockout_rule"


def test_explicit_direct_commit_skips_pending_even_when_quality_warns(tmp_path):
    """前端直入库入口显式关闭人工确认时，不再把低质量文档强制放入 pending。"""

    knowledge_dir = tmp_path / "knowledge"
    storage_dir = tmp_path / "ingestion"
    service = DocumentIngestionService(knowledge_dir=str(knowledge_dir), storage_dir=storage_dir)

    report = service.prepare_upload(
        filename="short-rule.txt",
        raw="规则".encode("utf-8"),
        document_id="short_rule",
        metadata={"category": "general", "expires_at": "2000-01-01"},
        require_confirmation=False,
        replace_existing=False,
        allow_low_quality_direct_commit=True,
    )

    assert report.status == "indexed_source_ready"
    assert (knowledge_dir / "short_rule.md").exists()
    assert service.list_pending() == []
    assert any("跳过人工确认" in warning for warning in report.warnings)


def test_expired_knowledge_is_filtered_after_retrieval_fusion():
    """过期规则过滤必须覆盖 BM25 和向量两路召回后的合并结果。"""

    service = KnowledgeRetrievalService.__new__(KnowledgeRetrievalService)
    expired = NodeWithScore(
        node=TextNode(
            text="旧版缺货规则",
            metadata={
                "chunk_id": "old",
                "category": "stockout_rule",
                "expires_at": "2000-01-01",
            },
        ),
        score=0.9,
    )
    active = NodeWithScore(
        node=TextNode(
            text="新版缺货规则",
            metadata={
                "chunk_id": "new",
                "category": "stockout_rule",
                "expires_at": "2999-01-01",
            },
        ),
        score=0.8,
    )

    result = service._dedupe_and_filter_nodes([expired, active], ["stockout_rule"])

    assert [item.node.metadata["chunk_id"] for item in result] == ["new"]
