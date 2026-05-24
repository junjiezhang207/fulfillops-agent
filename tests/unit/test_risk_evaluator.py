"""RiskEvaluator 单元测试 — 验证所有风险等级和规则触发路径。

大厂标准：每条规则至少有一个正向（触发）和一个负向（不触发）测试。
全部无 LLM 依赖，纯计算，CI 毫秒级完成。
"""

import pytest

from app.workflows.fulfillment.risk_evaluator import RiskAssessment, RiskEvaluator, RiskLevel


@pytest.fixture
def evaluator():
    return RiskEvaluator()


def _ctx(**kwargs):
    """构造风险评估上下文，提供合理默认值。"""
    return {
        "order_id": "SO-TEST",
        "order_value": kwargs.get("order_value", 0),
        "customer_level": kwargs.get("customer_level", "normal"),
        "hours_to_deadline": kwargs.get("hours_to_deadline", 999),
        "stock_gap_ratio": kwargs.get("stock_gap_ratio", 0.0),
        "split_order_required": kwargs.get("split_order_required", False),
        "sku_types": kwargs.get("sku_types", []),
        "cross_region": kwargs.get("cross_region", False),
    }


class TestRiskLevels:
    """验证不同场景下的风险等级评估。"""

    def test_normal_order_is_low(self, evaluator):
        ctx = _ctx(order_value=5000, customer_level="normal", hours_to_deadline=48)
        assessment = evaluator.evaluate(ctx)
        assert assessment.max_level == RiskLevel.LOW
        assert not assessment.needs_human_review

    def test_high_value_order_triggers_high(self, evaluator):
        ctx = _ctx(order_value=150_000)
        assessment = evaluator.evaluate(ctx)
        assert assessment.max_level >= RiskLevel.HIGH
        assert assessment.needs_human_review
        assert "high_value_order" in assessment.signal_names

    def test_vip_critical_urgent_triggers_critical(self, evaluator):
        ctx = _ctx(
            order_value=600_000,
            customer_level="vip",
            hours_to_deadline=1,
        )
        assessment = evaluator.evaluate(ctx)
        assert assessment.max_level == RiskLevel.CRITICAL
        assert assessment.interrupt_type == "critical_approval"
        assert "vip_critical_urgent" in assessment.signal_names

    def test_severe_stockout_triggers_high(self, evaluator):
        ctx = _ctx(stock_gap_ratio=0.8)
        assessment = evaluator.evaluate(ctx)
        assert assessment.max_level >= RiskLevel.HIGH
        assert "severe_stockout" in assessment.signal_names

    def test_vip_near_deadline_triggers_high(self, evaluator):
        ctx = _ctx(customer_level="vip", hours_to_deadline=3)
        assessment = evaluator.evaluate(ctx)
        assert assessment.max_level >= RiskLevel.HIGH
        assert "vip_deadline_urgent" in assessment.signal_names

    def test_new_customer_large_order_triggers_high(self, evaluator):
        ctx = _ctx(customer_level="new_customer", order_value=60_000)
        assessment = evaluator.evaluate(ctx)
        assert assessment.max_level >= RiskLevel.HIGH
        assert "new_customer_large_order" in assessment.signal_names

    def test_split_order_triggers_medium(self, evaluator):
        ctx = _ctx(split_order_required=True)
        assessment = evaluator.evaluate(ctx)
        assert RiskLevel.MEDIUM in [s.level for s in assessment.signals]
        assert "split_order_required" in assessment.signal_names

    def test_cold_chain_triggers_medium(self, evaluator):
        ctx = _ctx(sku_types=["cold_chain"])
        assessment = evaluator.evaluate(ctx)
        assert "cold_chain_product" in assessment.signal_names

    def test_cross_region_triggers_medium(self, evaluator):
        ctx = _ctx(cross_region=True)
        assessment = evaluator.evaluate(ctx)
        assert "cross_region_fulfillment" in assessment.signal_names


class TestRiskAssessmentProperties:
    """验证 RiskAssessment 的属性计算逻辑。"""

    def test_empty_signals_is_low(self):
        a = RiskAssessment(signals=[])
        assert a.max_level == RiskLevel.LOW
        assert not a.needs_human_review
        assert a.interrupt_type == "auto_pass"

    def test_needs_human_review_above_high(self, evaluator):
        ctx = _ctx(order_value=200_000)
        a = evaluator.evaluate(ctx)
        assert a.needs_human_review

    def test_timeout_seconds_by_level(self, evaluator):
        critical_ctx = _ctx(order_value=600_000, customer_level="vip", hours_to_deadline=1)
        critical_a = evaluator.evaluate(critical_ctx)
        assert critical_a.timeout_seconds() == 300

        high_ctx = _ctx(order_value=200_000)
        high_a = evaluator.evaluate(high_ctx)
        assert high_a.timeout_seconds() == 1800

    def test_interrupt_context_structure(self, evaluator):
        ctx = _ctx(order_value=150_000)
        a = evaluator.evaluate(ctx)
        interrupt_ctx = evaluator.build_interrupt_context(
            order_id="SO-TEST", assessment=a,
            inventory_summary="库存不足", order_summary="高价值订单",
        )
        assert "order_id" in interrupt_ctx
        assert "risk_level" in interrupt_ctx
        assert "risk_signals" in interrupt_ctx
        assert "suggested_actions" in interrupt_ctx
        assert "timeout_seconds" in interrupt_ctx

    def test_multiple_rules_max_level(self, evaluator):
        # 同时触发 HIGH 和 MEDIUM 规则，max_level 应该是 HIGH
        ctx = _ctx(order_value=150_000, split_order_required=True, cross_region=True)
        a = evaluator.evaluate(ctx)
        assert a.max_level == RiskLevel.HIGH
