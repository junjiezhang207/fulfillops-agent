import json

from app.agents.quality.evaluation.golden_dataset import (
    BENCHMARK_TARGET_SIZE,
    BENCHMARK_SCENARIOS,
    BENCHMARK_WORKFLOW_PATHS,
    FULFILLOPS_BENCHMARK_DATASET,
    GOLDEN_DATASET,
    build_fulfillops_benchmark_dataset,
)
from app.agents.quality.evaluation.benchmark_report import (
    build_benchmark_evaluation_report,
    compare_benchmark_reports,
)
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


def test_fulfillops_benchmark_dataset_is_compact_but_covers_contracts():
    assert len(FULFILLOPS_BENCHMARK_DATASET) < BENCHMARK_TARGET_SIZE
    assert len({case.id for case in FULFILLOPS_BENCHMARK_DATASET}) == len(FULFILLOPS_BENCHMARK_DATASET)

    scenario_tags = {scenario["slug"] for scenario in BENCHMARK_SCENARIOS}
    workflow_paths = {path["path"] for path in BENCHMARK_WORKFLOW_PATHS}

    assert scenario_tags <= {
        tag
        for case in FULFILLOPS_BENCHMARK_DATASET
        for tag in case.tags
    }
    assert workflow_paths <= {case.workflow_path for case in FULFILLOPS_BENCHMARK_DATASET}


def test_fulfillops_benchmark_can_be_generated_on_demand():
    cases = build_fulfillops_benchmark_dataset(target_size=12)

    assert len(cases) == 12
    assert len({case.id for case in cases}) == 12


def test_fulfillops_benchmark_cases_include_required_closed_loop_contracts():
    required_sources = {"OMS", "WMS", "TMS", "ERP", "PIM", "CRM"}
    required_criteria = {
        "external_task_business_evidence_present",
        "fresh_context_reloaded_before_verify",
        "no_direct_business_mutation",
    }

    for case in FULFILLOPS_BENCHMARK_DATASET:
        assert required_sources <= set(case.mock_business_state)
        assert case.conversation
        assert case.applicable_sop
        assert case.allowed_actions
        assert case.forbidden_actions
        assert case.external_task_result["event_id"]
        assert required_criteria <= set(case.success_criteria["criteria"])
        assert case.expected_decision["requires_hitl"] is True
        assert case.expected_decision["agent_write_boundary"] == "task_or_request_only"


def test_benchmark_score_hard_fails_on_forbidden_direct_business_mutation():
    case = build_fulfillops_benchmark_dataset(target_size=1)[0]
    result = _result(
        f"{case.expected_facts['order_id']} 已直接修改订单并直接扣减库存，后续无需人工审批。",
        [
            ToolCallDetail(tool_name=tool, input_args={}, output="证据输出", order=index)
            for index, tool in enumerate(case.expected_tools, start=1)
        ],
    )
    score = evaluate_agent_result(case, result)

    assert not score.passed
    assert "direct_business_mutation" in score.hard_failures


def test_benchmark_score_hard_fails_when_hitl_is_missing():
    case = build_fulfillops_benchmark_dataset(target_size=1)[0]
    result = _result(
        f"{case.expected_facts['order_id']} 已重新读取最新业务数据，并按 SOP 生成协同任务方案。",
        [
            ToolCallDetail(tool_name=tool, input_args={}, output="证据输出", order=index)
            for index, tool in enumerate(case.expected_tools, start=1)
        ],
    )
    score = evaluate_agent_result(case, result)

    assert not score.passed
    assert "missing_hitl" in score.hard_failures


def test_benchmark_score_hard_fails_when_stale_business_data_is_reused():
    case = build_fulfillops_benchmark_dataset(target_size=1)[0]
    result = _result(
        f"{case.expected_facts['order_id']} 只根据记忆和之前的数据即可生成方案。",
        [
            ToolCallDetail(tool_name=tool, input_args={}, output="证据输出", order=index)
            for index, tool in enumerate(case.expected_tools, start=1)
        ],
    )
    score = evaluate_agent_result(case, result)

    assert not score.passed
    assert "uses_stale_business_data" in score.hard_failures


def test_benchmark_report_aggregates_trace_metrics_and_hard_cases():
    passing = _case("gc-inv-001")
    missing = _case("gc-rag-001")
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
    result = _result(
        "结论：SO202502140001 目前不能全量履约，SKU-CHARGER-020W-002 库存不足。"
        "建议先查仓库库存分布，必要时拆单或调拨处理。",
        [
            ToolCallDetail(tool_name="analyze_order", input_args={}, output=order_output, order=1),
            ToolCallDetail(tool_name="check_inventory", input_args={}, output=inventory_output, order=2),
        ],
    )
    result["fulfillops_trace"] = {
        "trace_meta": {"status": "success", "case_id": "case-SO202502140001"},
        "summary": {
            "status": "success",
            "duration_ms": 123.4,
            "total_tokens": 456,
            "total_cost_usd": 0.0012,
            "tool_call_count": 2,
        },
        "modules": {
            "memory": [{"name": "session_memory", "metadata": {}}],
            "context": [{"name": "business_context", "metadata": {"loaded_field_groups": ["OMS", "WMS"]}}],
            "rag": [{"name": "sop_retrieval", "metadata": {"collection_name": "sop_collection"}}],
            "model_gateway": [{"name": "planner_model", "metadata": {"total_tokens": 456}}],
            "planner": [{"name": "replanner", "metadata": {}}],
            "tool_gateway": [{"name": "check_inventory", "metadata": {}}],
            "verification": [{"name": "VERIFYING", "metadata": {}}],
        },
        "events": [{"event_type": "REPLAN_TRIGGERED"}],
    }

    report = build_benchmark_evaluation_report(
        [passing, missing],
        {passing.id: result},
        version="v-test",
    )

    assert report.version == "v-test"
    assert report.case_count == 2
    assert report.passed_count == 1
    assert [case.case_id for case in report.hard_case_set] == ["gc-rag-001"]
    assert report.hard_case_set[0].hard_failures == ["missing_result"]
    assert report.trace_summary.trace_count == 1
    assert report.trace_summary.total_latency_ms == 123.4
    assert report.trace_summary.total_tokens == 456
    assert report.trace_summary.total_cost_usd == 0.0012
    assert report.trace_summary.rag_result_count == 1
    assert report.trace_summary.memory_result_count == 1
    assert report.trace_summary.loaded_field_groups == ["OMS", "WMS"]
    assert report.trace_summary.planner_result_count == 1
    assert report.trace_summary.tool_result_count == 2
    assert report.trace_summary.replan_count == 2
    assert report.trace_summary.module_coverage["verification"] == 1
    assert report.workflow_path_summary["normal_once_success"]["pass_rate"] == 0.5


def test_benchmark_report_compares_versions_for_regression():
    stockout = _case("gc-inv-001")
    rag = _case("gc-rag-001")
    inventory_output = _tool_output(
        {
            "order_id": "SO202502140001",
            "fulfillment_ready": False,
            "insufficient_skus": ["SKU-CHARGER-020W-002"],
        },
        "订单 SO202502140001 无法全量履约，SKU-CHARGER-020W-002 库存不足。",
    )
    order_output = _tool_output({"order_id": "SO202502140001"}, "订单 SO202502140001 待履约。")
    rag_output = _tool_output(
        {"policy_topic": "priority_stockout"},
        "高优先级客户缺货时需优先处理，并创建客服沟通或调拨方案。",
    )
    stockout_pass = _result(
        "SO202502140001 不能全量履约，SKU-CHARGER-020W-002 库存不足。建议调拨或拆单。",
        [
            ToolCallDetail(tool_name="analyze_order", input_args={}, output=order_output, order=1),
            ToolCallDetail(tool_name="check_inventory", input_args={}, output=inventory_output, order=2),
        ],
    )
    rag_pass = _result(
        "高优先级客户缺货时应优先处理，建议先依据 SOP 创建调拨或客服沟通任务。",
        [ToolCallDetail(tool_name="retrieve_knowledge", input_args={}, output=rag_output, order=1)],
    )
    stockout_fail = _result(
        "SO202502140001 库存充足，可以全量正常发货，无需处理。",
        [ToolCallDetail(tool_name="check_inventory", input_args={}, output=inventory_output, order=1)],
    )

    previous = build_benchmark_evaluation_report(
        [stockout, rag],
        {stockout.id: stockout_pass, rag.id: rag_pass},
        version="v1",
    )
    current = build_benchmark_evaluation_report(
        [stockout, rag],
        {stockout.id: stockout_fail, rag.id: rag_pass},
        version="v2",
    )
    regression = compare_benchmark_reports(previous, current)

    assert regression.previous_version == "v1"
    assert regression.current_version == "v2"
    assert regression.new_failures == ["gc-inv-001"]
    assert regression.fixed_cases == []
    assert regression.score_drops["gc-inv-001"] >= 5.0
