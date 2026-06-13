"""Hybrid 意图路由回归测试。"""

from types import SimpleNamespace

from app.application.routing.hybrid_service import HybridService
from app.application.routing.intent_classifier import IntentClassifier, IntentLevel


def test_intent_classifier_routes_common_frontend_questions():
    classifier = IntentClassifier()

    cases = [
        ("你好，你是谁？", IntentLevel.CASUAL),
        ("这个订单库存够不够，能不能发？", IntentLevel.SIMPLE),
        ("缺货订单 SOP 是什么？", IntentLevel.RAG),
        ("这个订单缺货了，给我处理建议", IntentLevel.COMPLEX),
        ("综合库存、风险、成本和时效给一个完整履约方案", IntentLevel.MULTI_DOMAIN),
    ]

    for question, expected in cases:
        result = classifier.classify(question)
        assert result.level == expected, f"{question} routed to {result.level}: {result.reasoning}"


class _FakeIntentModel:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, _prompt: str):
        return self.content


def _hybrid_with_model(content: str) -> HybridService:
    service = HybridService.__new__(HybridService)
    service.classifier = IntentClassifier()
    service.intent_model = _FakeIntentModel(content)
    return service


def test_hybrid_router_corrects_llm_workflow_for_rag_question():
    service = _hybrid_with_model(
        '{"route":"workflow","confidence":0.95,"reasoning":"订单相关","keywords":["订单"]}'
    )

    result = service._classify_intent("SO123", "缺货订单 SOP 是什么？")

    assert result.level == IntentLevel.RAG
    assert "纠偏" in result.reasoning


def test_hybrid_router_keeps_casual_question_out_of_workflow():
    service = _hybrid_with_model(
        '{"route":"workflow","confidence":0.9,"reasoning":"默认订单流程","keywords":["订单"]}'
    )

    result = service._classify_intent("SO123", "你好，你是谁？")

    assert result.level == IntentLevel.CASUAL


def test_hybrid_without_order_uses_rag_even_if_model_prefers_multi_agent():
    class FakeKnowledgeService:
        def __init__(self):
            self.calls = []

        def retrieve(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                order_id="ADHOC-KNOWLEDGE",
                summary="priority rules",
                answer_summary=SimpleNamespace(
                    conclusion="高优先级订单包括 VIP、加急和大促保障订单。",
                    key_rules=["VIP 订单", "加急订单", "大促保障订单"],
                    suggested_actions=[],
                ),
            )

    knowledge = FakeKnowledgeService()
    service = HybridService(
        workflow_service=SimpleNamespace(),
        agent_service=None,
        knowledge_service=knowledge,
        multi_agent_service=SimpleNamespace(run=lambda **_: (_ for _ in ()).throw(AssertionError("no multi-agent"))),
        intent_model=_FakeIntentModel(
            '{"route":"multi_agent","confidence":0.95,"reasoning":"misrouted","keywords":["order"]}'
        ),
    )

    result = service.process(order_id=None, question="高优先级订单包括哪些类型？")

    assert result.path_used == "rag"
    assert result.order_id == "ADHOC-KNOWLEDGE"
    assert "高优先级订单" in result.final_answer
    assert knowledge.calls[0]["order_id"] == ""
