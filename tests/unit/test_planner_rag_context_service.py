"""Planner RAG Context 服务测试。"""

from types import SimpleNamespace

from app.rag.rag_context_service import CASE_FILTERS, SOP_FILTERS, PlannerRAGContextService


class FakeKnowledgeService:
    def __init__(self):
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        filters = kwargs["filter_categories"]
        if filters == CASE_FILTERS:
            hit = SimpleNamespace(
                source_file="excellent_case_case-1.md",
                category="excellent_case",
                score=0.89,
                text="相似订单通过跨仓调拨加客服确认完成履约。",
                metadata=SimpleNamespace(
                    title="跨仓调拨优秀案例",
                    chunk_id="case-1::chunk-0000",
                    source_case_id="case-1",
                ),
            )
            return SimpleNamespace(
                applied_filters=filters,
                answer_summary=SimpleNamespace(
                    key_rules=["历史案例显示先调拨再客服确认可降低延误投诉。"],
                    conclusion="可参考跨仓调拨优秀案例。",
                    coverage_note="命中优秀案例。",
                ),
                hits=[hit],
            )
        hit = SimpleNamespace(
            source_file="stockout_sop.md",
            category="stockout_rule",
            score=0.93,
            text="缺货订单先查同区域仓，再评估跨仓调拨。",
            metadata=SimpleNamespace(
                title="缺货 SOP",
                chunk_id="stockout::chunk-0000",
                source_case_id="",
            ),
        )
        return SimpleNamespace(
            applied_filters=filters,
            answer_summary=SimpleNamespace(
                key_rules=["缺货订单先查同区域仓，再评估跨仓调拨。"],
                conclusion="按缺货 SOP 处理。",
                coverage_note="命中 SOP。",
            ),
            hits=[hit],
        )


class ConflictingCaseKnowledgeService:
    def __init__(self):
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        filters = kwargs["filter_categories"]
        if filters == CASE_FILTERS:
            hit = SimpleNamespace(
                source_file="excellent_case_conflict.md",
                category="excellent_case",
                score=0.91,
                text="历史案例曾对 SKU-X 直接修改订单并拆单多包裹发出，且无需审批。",
                metadata=SimpleNamespace(
                    title="冲突历史案例",
                    chunk_id="case-conflict::chunk-0000",
                    source_case_id="case-conflict",
                ),
            )
            return SimpleNamespace(
                applied_filters=filters,
                answer_summary=SimpleNamespace(
                    key_rules=["SKU-X 可以拆单并直接执行，无需审批。"],
                    conclusion="可照历史案例执行。",
                    coverage_note="命中历史案例。",
                ),
                hits=[hit],
            )
        hit = SimpleNamespace(
            source_file="split_sop.md",
            category="split_merge_rule",
            score=0.95,
            text="拆单必须先检查 PIM 限制，并保留人工审批记录。",
            metadata=SimpleNamespace(
                title="拆单 SOP",
                chunk_id="split-sop::chunk-0000",
                source_case_id="",
            ),
        )
        return SimpleNamespace(
            applied_filters=filters,
            answer_summary=SimpleNamespace(
                key_rules=["拆单必须先检查 PIM 限制，并保留人工审批记录。"],
                conclusion="按当前 SOP 处理。",
                coverage_note="命中 SOP。",
            ),
            hits=[hit],
        )


def test_planner_rag_context_separates_sop_and_case_retrieval():
    knowledge = FakeKnowledgeService()
    service = PlannerRAGContextService(knowledge)

    context = service.build(
        order_id="SO-1",
        question="请生成履约执行提案",
        session_memory={
            "structured": {
                "current_topic": "库存调拨/跨仓调货",
                "user_preferences": {"priority": "speed"},
            }
        },
    )

    assert knowledge.calls[0]["filter_categories"] == SOP_FILTERS
    assert knowledge.calls[1]["filter_categories"] == CASE_FILTERS
    assert knowledge.calls[0]["question"] == "请生成履约执行提案"
    assert knowledge.calls[0]["session_memory"]["structured"]["user_preferences"] == {"priority": "speed"}
    assert "运营偏好" in context.sop_evidence.query
    assert context.sop_evidence.evidence[0].source_type == "sop"
    assert context.similar_cases.evidence[0].source_type == "excellent_case"
    assert context.source_collections == {
        "sop": "sop_collection",
        "excellent_case": "case_collection",
    }
    assert context.sop_evidence.collection_name == "sop_collection"
    assert context.similar_cases.collection_name == "case_collection"
    assert context.priority_rule == "current_business_state > current_sop > historical_case"


def test_planner_rag_context_ignores_cases_conflicting_with_current_data_or_sop():
    knowledge = ConflictingCaseKnowledgeService()
    service = PlannerRAGContextService(knowledge)

    context = service.build(
        order_id="SO-1",
        question="请生成履约执行提案",
        current_business_context={
            "product_restrictions": [
                {"sku_id": "SKU-X", "split_allowed": False},
            ]
        },
    )

    assert context.similar_cases.evidence == []
    assert context.similar_cases.key_points == []
    assert len(context.similar_cases.ignored_evidence) == 1
    ignored = context.similar_cases.ignored_evidence[0]
    assert ignored.conflict_status == "ignored_conflict"
    assert ignored.source_case_id == "case-conflict"
    assert "case_suggests_direct_business_mutation" in ignored.conflict_reasons
    assert "case_conflicts_with_active_sop" in ignored.conflict_reasons
    assert "case_conflicts_with_current_product_restriction" in ignored.conflict_reasons
    assert context.warnings == ["已忽略 1 条与当前业务数据或 SOP 冲突的历史案例。"]
