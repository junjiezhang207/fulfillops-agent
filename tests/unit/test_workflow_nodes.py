"""Workflow 节点单元测试 — 验证每个节点的 state 增量格式。

测试策略：
  - mock 所有 service 依赖，专注于节点自身逻辑
  - 验证成功路径产出正确的 state 增量
  - 验证错误路径写入 errors 或返回 Command(goto=...)
  - 验证 trace 事件始终被写入

全部无 LLM 依赖，CI 毫秒级完成。
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from app.workflows.fulfillment.nodes import WorkflowNodes
from app.workflows.fulfillment.state import GraphState
from app.workflows.fulfillment.trace import TraceEvent, ErrorEvent
from app.domain.orders.analysis import OrderNotFoundError
from app.schemas.workflow import ExecutionProposal, ProposalAction
from langgraph.types import Command


@pytest.fixture
def mock_services():
    order_svc = MagicMock()
    inv_svc = MagicMock()
    know_svc = MagicMock()
    return order_svc, inv_svc, know_svc


@pytest.fixture
def nodes(mock_services):
    order_svc, inv_svc, know_svc = mock_services
    return WorkflowNodes(
        order_service=order_svc,
        inventory_service=inv_svc,
        knowledge_service=know_svc,
        chat_model=None,
    )


def _base_state(**kwargs) -> GraphState:
    return {
        "order_id": kwargs.get("order_id", "SO-TEST"),
        "question": kwargs.get("question", "能否发货"),
        "filter_categories": kwargs.get("filter_categories", []),
        "trace": [],
        "errors": [],
    }


# ── dispatch 节点 ─────────────────────────────────────────────────────────────

class TestDispatchNode:
    def test_valid_order_id_writes_trace(self, nodes):
        state = _base_state()
        result = nodes.dispatch(state)
        assert isinstance(result, dict)
        assert "trace" in result
        assert len(result["trace"]) == 1
        assert result["trace"][0].node == "dispatch"
        assert result["trace"][0].status == "ok"

    def test_missing_order_id_raises(self, nodes):
        state = _base_state(order_id="")
        with pytest.raises(ValueError, match="order_id"):
            nodes.dispatch(state)


# ── order_analysis 节点 ───────────────────────────────────────────────────────

class TestOrderAnalysisNode:
    def test_success_path(self, nodes, mock_services):
        order_svc, _, _ = mock_services
        order_result = MagicMock()
        order_result.item_count = 3
        order_result.total_quantity = 100
        order_result.summary = "订单包含 3 个 SKU"
        order_svc.analyze_order.return_value = order_result

        state = _base_state()
        result = nodes.order_analysis(state)

        assert isinstance(result, dict)
        assert result["order_result"] is order_result
        assert len(result["trace"]) == 1
        assert result["trace"][0].status == "ok"

    def test_order_not_found_returns_command(self, nodes, mock_services):
        order_svc, _, _ = mock_services
        order_svc.analyze_order.side_effect = OrderNotFoundError("订单不存在")

        state = _base_state()
        result = nodes.order_analysis(state)

        # OrderNotFoundError 应返回 Command(goto="finalize")，不是普通 dict
        assert isinstance(result, Command)
        assert result.goto == "finalize"
        assert len(result.update["errors"]) == 1
        assert result.update["errors"][0].exception_type == "OrderNotFoundError"

    def test_generic_exception_writes_errors(self, nodes, mock_services):
        order_svc, _, _ = mock_services
        order_svc.analyze_order.side_effect = RuntimeError("数据库连接失败")

        state = _base_state()
        result = nodes.order_analysis(state)

        # 普通异常写 errors，继续流程（不用 Command 短路）
        assert isinstance(result, dict)
        assert len(result["errors"]) == 1
        assert result["errors"][0].node == "order_analysis"
        assert result["trace"][0].status == "error"


# ── inventory_analysis 节点 ───────────────────────────────────────────────────

class TestInventoryAnalysisNode:
    def _make_inv_result(self, ready: bool, insufficient: list[str] | None = None):
        m = MagicMock()
        m.fulfillment_ready = ready
        m.insufficient_skus = insufficient or []
        m.summary = "库存充足" if ready else f"缺货 SKU: {insufficient}"
        m.item_count = 2
        return m

    def test_fulfillable_branch(self, nodes, mock_services):
        order_svc, inv_svc, _ = mock_services
        order_result = MagicMock()
        order_result.total_amount = 5000
        order_result.customer_level = "normal"
        order_result.hours_to_deadline = 48
        order_result.sku_types = []
        order_result.cross_region = False

        inv_svc.analyze_inventory.return_value = self._make_inv_result(ready=True)

        state = _base_state()
        state["order_result"] = order_result
        result = nodes.inventory_analysis(state)

        assert isinstance(result, dict)
        assert result.get("fulfillment_branch") == "fulfillable"
        assert result["inventory_result"].fulfillment_ready is True

    def test_stockout_branch(self, nodes, mock_services):
        _, inv_svc, _ = mock_services
        inv_result = self._make_inv_result(ready=False, insufficient=["SKU-A", "SKU-B"])
        inv_svc.analyze_inventory.return_value = inv_result

        state = _base_state()
        state["order_result"] = MagicMock(
            total_amount=5000, customer_level="normal",
            hours_to_deadline=48, sku_types=[], cross_region=False,
        )
        result = nodes.inventory_analysis(state)

        assert isinstance(result, dict)
        assert result.get("fulfillment_branch") == "stockout"

    def test_skipped_when_no_order_result(self, nodes):
        state = _base_state()
        # order_result 为 None 时跳过本节点
        result = nodes.inventory_analysis(state)
        assert isinstance(result, dict)
        assert result["trace"][0].status == "skipped"
        assert result.get("fulfillment_branch") == "stockout"  # 兜底


# ── knowledge_retrieval 节点 ──────────────────────────────────────────────────

class TestKnowledgeRetrievalNode:
    def test_success_path(self, nodes, mock_services):
        _, _, know_svc = mock_services
        know_result = MagicMock()
        know_result.hits = [MagicMock(), MagicMock()]
        know_result.matched_categories = ["stockout_rules"]
        know_svc.retrieve.return_value = know_result

        state = _base_state()
        result = nodes.knowledge_retrieval(state)

        assert result["knowledge_result"] is know_result
        assert result["trace"][0].status == "ok"

    def test_default_question_when_none(self, nodes, mock_services):
        _, _, know_svc = mock_services
        know_result = MagicMock(hits=[], matched_categories=[])
        know_svc.retrieve.return_value = know_result

        state = _base_state(question=None)
        nodes.knowledge_retrieval(state)

        # 验证使用了默认问题（不因 question=None 崩溃）
        call_kwargs = know_svc.retrieve.call_args
        assert call_kwargs is not None


# ── finalize 节点 ─────────────────────────────────────────────────────────────

class TestFinalizeNode:
    def test_fast_path_produces_answer(self, nodes):
        inv = MagicMock()
        inv.fulfillment_ready = True
        inv.order_id = "SO-TEST"
        inv.summary = "库存充足"

        state = _base_state()
        state["inventory_result"] = inv
        state["order_result"] = MagicMock(summary="订单正常")

        result = nodes.finalize(state)
        assert result["final_answer"] is not None
        assert result["final_answer"].conclusion != ""
        assert result["trace"][0].status == "ok"

    def test_knowledge_path_includes_rules(self, nodes):
        inv = MagicMock(fulfillment_ready=False, order_id="SO-TEST", summary="缺货")
        know = MagicMock()
        know.answer_summary = MagicMock(
            key_rules=["规则1", "规则2"],
            suggested_actions=["动作1"],
        )

        state = _base_state()
        state["inventory_result"] = inv
        state["knowledge_result"] = know

        result = nodes.finalize(state)
        evidences = result["final_answer"].key_evidences
        assert any("规则依据" in e for e in evidences)

    def test_no_inventory_result_safe(self, nodes):
        state = _base_state()
        # 没有任何中间结果时不应崩溃
        result = nodes.finalize(state)
        assert "final_answer" in result


# ── execution proposal policy validator ─────────────────────────────────────

def _policy_ready_action(**overrides) -> ProposalAction:
    action = ProposalAction(
        action_id="a1",
        action_type="split_order",
        sku_id="SKU-A",
        quantity=2,
        from_warehouse="WH-A",
        carrier="FAST",
        reason="拆单先发",
        responsibility_domain="WMS",
        business_evidence=["OMS订单待履约", "WMS库存可覆盖"],
    )
    return action.model_copy(update=overrides)


def _policy_ready_proposal(**overrides) -> ExecutionProposal:
    action = overrides.pop("action", _policy_ready_action())
    actions = overrides.pop("actions", [action])
    goal_type = overrides.pop("goal_type", "split_fulfillment")
    action_dag = overrides.pop("action_dag", WorkflowNodes._build_action_dag(actions))
    success_criteria = overrides.pop(
        "success_criteria",
        WorkflowNodes._success_criteria_template(goal_type, actions),
    )
    proposal = ExecutionProposal(
        proposal_id="prop-policy",
        order_id="SO-POLICY",
        title="策略校验",
        summary="策略校验",
        actions=actions,
        data_fingerprint="fp-policy",
        goal_type=goal_type,
        action_dag=action_dag,
        success_criteria=success_criteria,
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    )
    return proposal.model_copy(update=overrides)


class TestExecutionProposalPolicyValidator:
    def test_valid_proposal_passes_policy_validator(self):
        proposal = _policy_ready_proposal()

        failures = WorkflowNodes._validate_execution_proposal_policy(proposal)

        assert failures == []

    def test_rejects_direct_business_mutation_action(self):
        proposal = _policy_ready_proposal(
            action=_policy_ready_action(action_type="direct_inventory_mutation")
        )

        failures = WorkflowNodes._validate_execution_proposal_policy(proposal)

        assert any(item["name"].startswith("hard_constraint.no_direct_business_mutation") for item in failures)

    def test_rejects_dangling_and_cyclic_action_dag(self):
        first = _policy_ready_action(action_id="a1", depends_on=["missing"])
        second = _policy_ready_action(action_id="a2", depends_on=["a1"])
        proposal = _policy_ready_proposal(
            actions=[first, second],
            action_dag={
                "nodes": [
                    {"action_id": "a1", "action_type": "split_order"},
                    {"action_id": "a2", "action_type": "split_order"},
                ],
                "edges": [
                    {"from": "missing", "to": "a1"},
                    {"from": "a1", "to": "a2"},
                    {"from": "a2", "to": "a1"},
                ],
            },
        )

        failures = WorkflowNodes._validate_execution_proposal_policy(proposal)

        failure_names = {item["name"] for item in failures}
        assert "action_dag.dependencies:a1" in failure_names
        assert "action_dag.edge_reference" in failure_names
        assert "action_dag.acyclic" in failure_names

    def test_rejects_incomplete_success_criteria(self):
        proposal = _policy_ready_proposal(
            success_criteria={
                "goal_type": "split_fulfillment",
                "criteria": [{"name": "fresh_business_state_reloaded"}],
                "max_replan_count": 2,
            }
        )

        failures = WorkflowNodes._validate_execution_proposal_policy(proposal)

        failure_names = {item["name"] for item in failures}
        assert "success_criteria.max_replan_count" in failure_names
        assert "success_criteria.required_names" in failure_names
        assert "success_criteria.evidence_scope" in failure_names

    def test_rejects_expired_proposal(self):
        proposal = _policy_ready_proposal(
            expires_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        )

        failures = WorkflowNodes._validate_execution_proposal_policy(proposal)

        assert any(item["name"] == "plan_schema.expires_at" for item in failures)

    def test_preflight_short_circuits_invalidated_policy_proposal(self, nodes, mock_services):
        order_svc, inv_svc, _ = mock_services
        proposal = _policy_ready_proposal(status="invalidated", invalidation_reason="Action DAG 不能包含循环依赖。")

        validation = nodes._preflight_validate(proposal)

        assert validation.status == "invalidated"
        assert validation.checks[0]["name"] == "policy_schema_validator"
        order_svc.analyze_order.assert_not_called()
        inv_svc.analyze_inventory.assert_not_called()

    def test_preflight_rejects_expired_proposal_before_runtime_lookup(self, nodes, mock_services):
        order_svc, inv_svc, _ = mock_services
        proposal = _policy_ready_proposal(
            expires_at=(datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        )

        validation = nodes._preflight_validate(proposal)

        assert validation.status == "invalidated"
        assert validation.checks[0]["name"] == "proposal_expiry"
        order_svc.analyze_order.assert_not_called()
        inv_svc.analyze_inventory.assert_not_called()
