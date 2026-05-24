import json

from app.agents.orchestration.react_agent import ToolObservation, _reflect_on_answer


def _obs(tool_name: str, data: dict, summary: str) -> ToolObservation:
    return ToolObservation(
        tool_name=tool_name,
        content=json.dumps(
            {"status": "ok", "data": data, "summary": summary},
            ensure_ascii=False,
        ),
    )


def test_reflection_retries_when_inventory_question_has_no_tool_grounding():
    result = _reflect_on_answer(
        question="分析订单 SO202502140001 的履约风险，看看能不能发货",
        answer="整体看风险较低，建议正常发货。",
        tools_called=[],
    )

    assert not result.passed
    assert "未调用问题所需工具" in result.reason


def test_reflection_passes_when_answer_is_grounded_in_relevant_tool_output():
    order_obs = _obs(
        "analyze_order",
        {"order_id": "SO202502140001", "total_quantity": 12, "region": "华东"},
        "订单 SO202502140001 共 12 件，区域华东。",
    )
    inventory_obs = _obs(
        "check_inventory",
        {
            "order_id": "SO202502140001",
            "fulfillment_ready": False,
            "insufficient_skus": ["SKU-IPHONE-CASE-001"],
        },
        "订单 SO202502140001 无法全量履约，SKU-IPHONE-CASE-001 缺货。",
    )

    result = _reflect_on_answer(
        question="分析订单 SO202502140001 的履约风险，看看能不能发货",
        answer=(
            "结论：订单 SO202502140001 目前不能全量发货，主要风险是 "
            "SKU-IPHONE-CASE-001 缺货。建议先锁定可发库存，再处理缺货 SKU。"
        ),
        tools_called=["analyze_order", "check_inventory"],
        tool_observations=[order_obs, inventory_obs],
    )

    assert result.passed
    assert result.reason == "质量达标"


def test_reflection_retries_when_required_tool_is_missing():
    result = _reflect_on_answer(
        question="订单 SO202502140001 能不能发货，有没有缺货 SKU？",
        answer="根据规则，缺货时应优先查询替代品并通知客户。",
        tools_called=["retrieve_knowledge"],
        tool_observations=[
            _obs("retrieve_knowledge", {"hit_count": 1}, "缺货时应优先查询替代品并通知客户。")
        ],
    )

    assert not result.passed
    assert "check_inventory" in result.reason


def test_reflection_retries_when_answer_invents_business_entity():
    inventory_obs = _obs(
        "check_inventory",
        {
            "order_id": "SO202502140001",
            "fulfillment_ready": False,
            "insufficient_skus": ["SKU-IPHONE-CASE-001"],
        },
        "订单 SO202502140001 无法全量履约，SKU-IPHONE-CASE-001 缺货。",
    )

    result = _reflect_on_answer(
        question="订单 SO202502140001 缺哪个 SKU？",
        answer="订单 SO202502140001 缺货 SKU 是 SKU-FAKE-999，建议改发替代品。",
        tools_called=["check_inventory"],
        tool_observations=[inventory_obs],
    )

    assert not result.passed
    assert "未支撑的关键实体" in result.reason
