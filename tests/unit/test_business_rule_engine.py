from datetime import datetime, timedelta

from app.schemas.advanced_order import (
    AdvancedOrderDetails,
    CustomerLevel,
    CustomerSegment,
    Location,
    OrderLineItem,
    OrderType,
    QualityLevel,
    ShippingMethod,
    WarehouseInfo,
)
from app.application.routing.advanced_hybrid_service import AdvancedHybridService
from app.domain.rules.business_rule_engine import BusinessRuleEngine, RuleImpactAnalysis


def _order(**overrides) -> AdvancedOrderDetails:
    data = {
        "order_id": "SO-TEST-001",
        "customer_id": "C-VIP-001",
        "created_at": datetime.now(),
        "customer_level": CustomerLevel.VIP,
        "customer_name": "Test Customer",
        "customer_contact": "ops@example.com",
        "order_type": OrderType.URGENT,
        "required_delivery_date": datetime.now() + timedelta(hours=12),
        "preferred_delivery_date": datetime.now() + timedelta(days=1),
        "destination": Location(province="Shanghai", city="Shanghai", district="Pudong"),
        "priority": 4,
        "shipping_method": ShippingMethod.STANDARD,
        "line_items": [OrderLineItem(sku="SKU-1", quantity=2, unit_price=30000)],
        "budget": 65000,
        "cost_sensitive": True,
        "quality_requirement": QualityLevel.STANDARD,
        "special_handling": ["fragile"],
    }
    data.update(overrides)
    return AdvancedOrderDetails(**data)


def _vip_segment(order: AdvancedOrderDetails) -> CustomerSegment:
    return CustomerSegment(
        customer_id=order.customer_id,
        level=CustomerLevel.VIP,
        default_priority=8,
        target_delivery_days=1,
        max_acceptable_delay=0,
        preferred_shipping_method=ShippingMethod.OVERNIGHT,
        max_shipping_cost_ratio=0.1,
        required_quality_level=QualityLevel.SUPER,
        inspection_required=True,
        consolidation_allowed=False,
        max_wait_for_consolidation_hours=0,
        substitution_allowed=False,
        substitution_approval_required=True,
        return_period_days=30,
        free_return=True,
        discount_ratio=0.95,
    )


def _warehouse() -> WarehouseInfo:
    return WarehouseInfo(
        warehouse_id="WH-TEST",
        name="Test Warehouse",
        location=Location(province="Shanghai", city="Shanghai", district="Pudong"),
        total_capacity=10000,
        used_capacity=5000,
        storage_cost_per_unit_day=0.5,
        picking_cost_per_unit=2,
        packing_cost_per_shipment=30,
        handling_cost_per_unit=1,
        quality_rating=98,
    )


def test_business_rules_do_not_mutate_input_and_keep_audit_trail():
    order = _order()
    engine = BusinessRuleEngine(customer_segment=_vip_segment(order))

    result = engine.execute(order)

    assert order.priority == 4
    assert order.shipping_method == ShippingMethod.STANDARD
    assert order.risk_flags == []
    assert result.original_order.priority == 4
    assert result.final_order.priority == 10
    assert result.final_order.shipping_method == ShippingMethod.OVERNIGHT
    assert "urgent_high_value" in result.final_order.risk_flags

    vip_rule = next(rule for rule in result.applied_rules if rule.rule_id == "vip_priority")
    assert vip_rule.before["priority"] == 4
    assert vip_rule.after["priority"] == 10


def test_lower_priority_rule_cannot_silently_override_higher_priority_rule():
    order = _order()
    engine = BusinessRuleEngine(customer_segment=_vip_segment(order))

    result = engine.execute(order)

    cost_rule = next(rule for rule in result.applied_rules if rule.rule_id == "cost_optimization")
    assert cost_rule.applied is False
    assert "优先级低于" in cost_rule.conflict_note
    assert result.final_order.shipping_method == ShippingMethod.OVERNIGHT
    assert result.conflict_notes


def test_rule_impact_analysis_reports_real_original_priority():
    order = _order(priority=3)
    engine = BusinessRuleEngine(customer_segment=_vip_segment(order))

    result = engine.execute(order)
    impact = RuleImpactAnalysis.analyze(
        order=result.final_order,
        applied_rules=result.applied_rules,
        original_order=result.original_order,
        conflict_notes=result.conflict_notes,
    )

    assert impact["original_priority"] == 3
    assert impact["final_priority"] == 10
    assert impact["conflict_notes"] == result.conflict_notes
    assert impact["applied_rules"][0]["before"]


def test_advanced_hybrid_service_exposes_rule_details_and_pricing():
    service = AdvancedHybridService(warehouses=[_warehouse()])

    result = service.process(order=_order(), inventory_snapshot={"inventory_level": "low"})

    assert result.rule_details
    assert result.rule_conflicts
    assert result.pricing_adjustment["base_price"] == 60000
    assert result.pricing_adjustment["final_price"] > 0
    assert result.pricing_adjustment["factors"]
    assert result.fulfillment_options
