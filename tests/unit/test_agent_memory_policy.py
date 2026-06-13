from app.agents.runtime.agent_service import AgentService, ExtractedMemory, ExtractedMemoryBatch
from app.schemas.agent import ToolCallDetail


def _service_for_memory_extractor() -> AgentService:
    service = object.__new__(AgentService)
    service._memory_extractor_model_id = "deepseek-v4-flash"
    return service


def test_memory_candidates_ignore_low_value_chitchat():
    candidates = AgentService._build_memory_candidates(
        session_id="s1",
        message="谢谢",
        reply="不客气",
        tools_called=[],
    )

    assert candidates == []


def test_memory_candidates_write_tool_grounded_order_decision():
    candidates = AgentService._build_memory_candidates(
        session_id="s1",
        message="SO202502140001 缺货了，需要人工复核吗？",
        reply="订单 SO202502140001 库存不足，建议进入人工复核并通知客户延期。",
        tools_called=["analyze_order", "check_inventory"],
    )

    order_candidates = [item for item in candidates if item.value["memory_type"] == "order_decision"]
    assert order_candidates
    assert order_candidates[0].namespace == ("orders", "SO202502140001")
    assert order_candidates[0].key.startswith("decision-")
    assert order_candidates[0].value["write_reason"] == "tool_grounded_order_decision"
    assert order_candidates[0].value["business_score"] >= 0.65


def test_memory_candidates_redact_sensitive_text_before_persisting():
    candidates = AgentService._build_memory_candidates(
        session_id="s1",
        message="SO202502140001 缺货，客户手机号 13812345678，邮箱 ops@example.com，需要延期。",
        reply="建议延期并给 110101199001011234 这个联系人发送通知。",
        tools_called=["analyze_order"],
    )

    joined = "\n".join(str(item.value) for item in candidates)
    assert "13812345678" not in joined
    assert "ops@example.com" not in joined
    assert "110101199001011234" not in joined
    assert "[PHONE]" in joined
    assert "[EMAIL]" in joined
    assert "[ID_CARD]" in joined


def test_memory_candidates_promote_preference_to_session_and_customer_scope():
    candidates = AgentService._build_memory_candidates(
        session_id="s1",
        message="客户 C-VIP-001 不接受替代 SKU，以后缺货先人工确认。",
        reply="已记录：缺货时优先人工确认，不推荐替代 SKU。",
        tools_called=[],
    )

    preference_candidates = [item for item in candidates if item.value["memory_type"] == "user_preference"]
    namespaces = {item.namespace for item in preference_candidates}
    assert ("sessions", "s1", "preferences") in namespaces
    assert ("customers", "C-VIP-001", "preferences") in namespaces
    assert all(item.key.startswith("pref-") for item in preference_candidates)


def test_llm_memory_candidate_maps_customer_preference():
    service = _service_for_memory_extractor()
    candidate = service._memory_candidate_from_extracted(
        session_id="s1",
        message="客户 C-VIP-001 以后优先走 DHL。",
        reply="已记录偏好。",
        tools_called=[],
        has_tool_observation=False,
        memory=ExtractedMemory(
            memory_type="user_preference",
            scope="customer",
            preference="客户偏好优先使用 DHL 承运商",
            customer_id="C-VIP-001",
            memory_facts={"preferred_carrier": "DHL"},
            evidence="客户 C-VIP-001 以后优先走 DHL。",
            confidence=0.86,
            importance=0.8,
        ),
    )

    assert candidate is not None
    assert candidate.namespace == ("customers", "C-VIP-001", "preferences")
    assert candidate.value["source"] == "llm_memory_extractor"
    assert candidate.value["extractor_model"] == "deepseek-v4-flash"
    assert candidate.value["memory_facts"] == {"preferred_carrier": "DHL"}


def test_llm_order_decision_requires_tool_observation():
    service = _service_for_memory_extractor()
    memory = ExtractedMemory(
        memory_type="order_decision",
        scope="order",
        order_id="SO202502140001",
        summary="订单库存不足，建议人工复核并通知客户延期。",
        evidence="工具返回库存不足，助手建议人工复核。",
        confidence=0.82,
        importance=0.85,
    )

    without_observation = service._memory_candidate_from_extracted(
        session_id="s1",
        message="SO202502140001 库存不足怎么办？",
        reply="建议人工复核并通知客户延期。",
        tools_called=["check_inventory"],
        has_tool_observation=False,
        memory=memory,
    )
    with_observation = service._memory_candidate_from_extracted(
        session_id="s1",
        message="SO202502140001 库存不足怎么办？",
        reply="建议人工复核并通知客户延期。",
        tools_called=["check_inventory"],
        has_tool_observation=True,
        memory=memory,
    )

    assert without_observation is None
    assert with_observation is not None
    assert with_observation.namespace == ("orders", "SO202502140001")
    assert with_observation.value["write_reason"] == "llm_tool_grounded_order_decision"


async def test_llm_memory_extractor_output_becomes_candidates():
    class FakeMemoryExtractor:
        async def ainvoke(self, payload):
            self.payload = payload
            return ExtractedMemoryBatch(
                memories=[
                    ExtractedMemory(
                        memory_type="conversation_summary",
                        scope="session",
                        summary="用户正在处理缺货订单，关注人工复核和延期通知。",
                        evidence="用户询问缺货处理，助手建议人工复核。",
                        confidence=0.72,
                        importance=0.7,
                    )
                ]
            )

    service = _service_for_memory_extractor()
    service._memory_extractor = FakeMemoryExtractor()

    candidates = await service._build_llm_memory_candidates(
        session_id="s1",
        message="SO202502140001 缺货怎么办？",
        reply="建议人工复核并通知客户延期。",
        tools_called=["check_inventory"],
        tool_call_details=[
            ToolCallDetail(
                tool_name="check_inventory",
                input_args={"order_id": "SO202502140001"},
                output='{"status":"ok","summary":"库存不足"}',
                order=1,
            )
        ],
    )

    assert len(candidates) == 1
    assert candidates[0].namespace == ("sessions", "s1")
    assert candidates[0].value["source"] == "llm_memory_extractor"
    assert candidates[0].value["extractor_model"] == "deepseek-v4-flash"
