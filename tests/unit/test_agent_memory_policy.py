from app.services.agent_service import AgentService


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
