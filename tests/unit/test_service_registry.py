from types import SimpleNamespace

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.core import service_registry
from app.rag.query_rewriter import QueryRewriter


def test_knowledge_retrieval_service_wires_rag_models_through_model_gateway(monkeypatch, tmp_path):
    calls: list[str] = []
    captured: dict[str, object] = {}
    rewrite_model = FakeListChatModel(responses=["缺货订单处理规则"])
    intent_model = FakeListChatModel(responses=['{"primary_intent":"stockout_handling","confidence":0.8}'])

    def fake_create_chat_model(_settings, *, use_case: str, model_id=None):
        calls.append(use_case)
        return rewrite_model if use_case == "query_rewrite" else intent_model

    class FakeKnowledgeRetrievalService:
        def __init__(
            self,
            *,
            knowledge_repository,
            inventory_analysis_service,
            query_rewriter,
            intent_classifier_model,
            cross_encoder,
            compressor,
        ) -> None:
            captured["knowledge_repository"] = knowledge_repository
            captured["inventory_analysis_service"] = inventory_analysis_service
            captured["query_rewriter"] = query_rewriter
            captured["intent_classifier_model"] = intent_classifier_model
            captured["cross_encoder"] = cross_encoder
            captured["compressor"] = compressor

    monkeypatch.setattr(
        service_registry,
        "get_settings",
        lambda: SimpleNamespace(
            knowledge_dir=str(tmp_path),
            knowledge_extra_dirs="",
            knowledge_recursive=True,
        ),
    )
    monkeypatch.setattr(service_registry.LLMFactory, "create_chat_model", fake_create_chat_model)
    monkeypatch.setattr(service_registry, "KnowledgeRetrievalService", FakeKnowledgeRetrievalService)
    monkeypatch.setattr(service_registry, "get_inventory_analysis_service", lambda: "inventory-service")
    service_registry.get_knowledge_retrieval_service.cache_clear()

    try:
        service_registry.get_knowledge_retrieval_service()
    finally:
        service_registry.get_knowledge_retrieval_service.cache_clear()

    assert calls == ["query_rewrite", "intent_classification"]
    assert isinstance(captured["query_rewriter"], QueryRewriter)
    assert captured["query_rewriter"]._chat_model is rewrite_model
    assert captured["intent_classifier_model"] is intent_model
    assert captured["inventory_analysis_service"] == "inventory-service"
    assert captured["cross_encoder"] is None
    assert captured["compressor"] is None
