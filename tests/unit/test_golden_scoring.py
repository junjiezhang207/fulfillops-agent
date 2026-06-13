import json

from app.agents.quality.evaluation.golden_dataset import GOLDEN_DATASET
from app.agents.quality.evaluation.scoring import evaluate_agent_result
from app.schemas.agent import AgentExecutionTrace, ToolCallDetail


def _case(case_id: str):
    return next(case for case in GOLDEN_DATASET if case.id == case_id)


def _tool_output(data: dict, summary: str) -> str:
    return json.dumps(
        {"schema_version": "1.0", "status": "ok", "data": data, "summary": summary},
        ensure_ascii=False,
    )


def _result(reply: str, tool_details: list[ToolCallDetail]) -> dict:
    return {
        "reply": reply,
        "tools_called": [detail.tool_name for detail in tool_details],
        "trace": AgentExecutionTrace(
            session_id="unit-eval",
            user_message="",
            tools_called=tool_details,
            final_reply=reply,
            execution_steps=len(tool_details),
        ),
    }


def test_score_passes_when_stockout_answer_is_grounded():
    case = _case("gc-inv-001")
    inventory_output = _tool_output(
        {
            "order_id": "SO202502140001",
            "fulfillment_ready": False,
            "insufficient_skus": ["SKU-CHARGER-020W-002"],
        },
        "订单 SO202502140001 无法全量履约，SKU-CHARGER-020W-002 库存不足。",
    )
    order_output = _tool_output(
        {"order_id": "SO202502140001", "priority": "高"},
        "订单 SO202502140001 为高优先级待履约订单。",
    )
    score = evaluate_agent_result(
        case,
        _result(
            "结论：SO202502140001 目前不能全量履约，SKU-CHARGER-020W-002 库存不足。"
            "建议先查仓库库存分布，必要时拆单或调拨处理。",
            [
                ToolCallDetail(tool_name="analyze_order", input_args={}, output=order_output, order=1),
                ToolCallDetail(tool_name="check_inventory", input_args={}, output=inventory_output, order=2),
            ],
        ),
    )

    assert score.passed, score.assert_message()
    assert score.breakdown.factual_grounding > 20


def test_score_hard_fails_when_stockout_is_claimed_fulfillable():
    case = _case("gc-inv-001")
    inventory_output = _tool_output(
        {
            "order_id": "SO202502140001",
            "fulfillment_ready": False,
            "insufficient_skus": ["SKU-CHARGER-020W-002"],
        },
        "订单 SO202502140001 无法全量履约，SKU-CHARGER-020W-002 库存不足。",
    )
    score = evaluate_agent_result(
        case,
        _result(
            "SO202502140001 库存充足，可以全量正常发货，无需处理。",
            [ToolCallDetail(tool_name="check_inventory", input_args={}, output=inventory_output, order=1)],
        ),
    )

    assert not score.passed
    assert "claims_full_fulfillment_ready" in score.hard_failures


def test_score_hard_fails_when_answer_fabricates_sku():
    case = _case("gc-inv-001")
    inventory_output = _tool_output(
        {
            "order_id": "SO202502140001",
            "fulfillment_ready": False,
            "insufficient_skus": ["SKU-CHARGER-020W-002"],
        },
        "订单 SO202502140001 无法全量履约，SKU-CHARGER-020W-002 库存不足。",
    )
    score = evaluate_agent_result(
        case,
        _result(
            "SO202502140001 不能全量履约，缺货 SKU 是 SKU-FAKE-999，建议调拨。",
            [ToolCallDetail(tool_name="check_inventory", input_args={}, output=inventory_output, order=1)],
        ),
    )

    assert not score.passed
    assert "unsupported_entity" in score.hard_failures


def test_score_hard_fails_when_rag_evidence_is_missing():
    case = _case("gc-rag-001")
    score = evaluate_agent_result(
        case,
        _result(
            "高优先级客户缺货时建议优先处理，并通知客户。",
            [],
        ),
    )

    assert not score.passed
    assert "missing_required_tool" in score.hard_failures
    assert "missing_rag_evidence" in score.hard_failures


def test_score_handles_not_found_case_without_hallucinating():
    case = _case("gc-edge-001")
    order_output = _tool_output(
        {},
        "订单查询失败：订单不存在：SO999999",
    )
    score = evaluate_agent_result(
        case,
        _result(
            "未找到订单 SO999999，不能确认其发货或库存状态，建议核对订单号后重试。",
            [ToolCallDetail(tool_name="analyze_order", input_args={}, output=order_output, order=1)],
        ),
    )

    assert score.passed, score.assert_message()
