#!/usr/bin/env python3
"""高级业务系统演示 — 展示复杂化后的供应链决策系统。

演示流程：
  1. 高级订单输入（包含真实业务要素）
  2. 应用业务规则（根据客户分级、订单类型等）
  3. 生成多个履约方案（快速、平衡、经济等）
  4. 对比方案评分（成本、时间、质量维度）
  5. 最优方案选择（基于权重）

运行方式：
  python -m scripts.advanced_business_demo
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas.advanced_order import (
    AdvancedOrderDetails,
    OrderLineItem,
    Location,
    CustomerLevel,
    OrderType,
    ShippingMethod,
    QualityLevel,
    InventoryLevel,
    CustomerSegment,
    WarehouseInfo,
)
from app.services.business_rule_engine import (
    BusinessRuleEngine,
    DynamicPricingEngine,
    RuleImpactAnalysis,
)
from app.services.fulfillment_options_generator import (
    FulfillmentOptionsGenerator,
)


def create_demo_order_1() -> AdvancedOrderDetails:
    """创建演示订单 1：VIP 急单"""
    return AdvancedOrderDetails(
        order_id="SO-2026-001",
        customer_id="CUST-VIP-001",
        customer_name="阿里巴巴采购部",
        customer_contact="张三",
        created_at=datetime.now(),
        customer_level=CustomerLevel.VIP,
        order_type=OrderType.URGENT,
        priority=5,
        required_delivery_date=datetime.now() + timedelta(hours=24),
        preferred_delivery_date=datetime.now() + timedelta(hours=12),
        allowed_delay_days=0,
        destination=Location(
            province="浙江省",
            city="杭州市",
            district="滨江区",
            latitude=30.2741,
            longitude=120.1551,
        ),
        shipping_method=ShippingMethod.OVERNIGHT,
        consolidation_allowed=False,
        line_items=[
            OrderLineItem(sku="SKU-A001", quantity=500, unit_price=100),
            OrderLineItem(sku="SKU-B002", quantity=200, unit_price=50),
        ],
        budget=60000,
        cost_sensitive=False,
        quality_requirement=QualityLevel.SUPER,
        special_handling=["易碎", "防潮"],
        require_inspection=True,
        risk_flags=["high_value"],
        notes="重要客户，必须满足交期",
    )


def create_demo_order_2() -> AdvancedOrderDetails:
    """创建演示订单 2：新客户可延期订单"""
    return AdvancedOrderDetails(
        order_id="SO-2026-002",
        customer_id="CUST-NEW-001",
        customer_name="小微企业 ABC",
        customer_contact="李四",
        created_at=datetime.now(),
        customer_level=CustomerLevel.NEW_CUSTOMER,
        order_type=OrderType.FLEXIBLE,
        priority=3,
        required_delivery_date=datetime.now() + timedelta(days=7),
        preferred_delivery_date=datetime.now() + timedelta(days=5),
        allowed_delay_days=3,
        destination=Location(
            province="山东省",
            city="青岛市",
            district="市北区",
        ),
        shipping_method=ShippingMethod.STANDARD,
        consolidation_allowed=True,
        line_items=[
            OrderLineItem(sku="SKU-C003", quantity=100, unit_price=25),
            OrderLineItem(sku="SKU-D004", quantity=50, unit_price=30),
        ],
        budget=5000,
        cost_sensitive=True,
        quality_requirement=QualityLevel.STANDARD,
        special_handling=[],
        require_inspection=False,
        notes="成本敏感，可以延迟以降低成本",
    )


def create_demo_warehouses() -> list:
    """创建演示仓库"""
    return [
        WarehouseInfo(
            warehouse_id="WH-SH",
            name="上海仓库",
            location=Location(province="上海市", city="上海市", district="浦东新区"),
            total_capacity=50000,
            used_capacity=30000,
            max_daily_shipments=500,
            processing_time_hours=2,
            quality_rating=98.5,
            storage_cost_per_unit_day=0.5,
            picking_cost_per_unit=2.0,
            packing_cost_per_shipment=50,
            handling_cost_per_unit=1.0,
        ),
        WarehouseInfo(
            warehouse_id="WH-GZ",
            name="广州仓库",
            location=Location(province="广东省", city="广州市", district="番禺区"),
            total_capacity=40000,
            used_capacity=25000,
            max_daily_shipments=400,
            processing_time_hours=4,
            quality_rating=96.0,
            storage_cost_per_unit_day=0.4,
            picking_cost_per_unit=1.8,
            packing_cost_per_shipment=40,
            handling_cost_per_unit=0.9,
        ),
        WarehouseInfo(
            warehouse_id="WH-CS",
            name="长沙仓库",
            location=Location(province="湖南省", city="长沙市", district="天心区"),
            total_capacity=30000,
            used_capacity=18000,
            max_daily_shipments=300,
            processing_time_hours=6,
            quality_rating=92.0,
            storage_cost_per_unit_day=0.3,
            picking_cost_per_unit=1.5,
            packing_cost_per_shipment=30,
            handling_cost_per_unit=0.8,
        ),
    ]


def print_order_summary(order: AdvancedOrderDetails):
    """打印订单摘要"""
    print()
    print("[订单信息]")
    print(f"  订单 ID: {order.order_id}")
    print(f"  客户: {order.customer_name} ({order.customer_level.value})")
    print(f"  订单类型: {order.order_type.value}")
    print(f"  优先级: {order.priority}/10")
    print(f"  订单总额: ${order.total_amount:,.2f}")
    print(f"  总数量: {order.total_quantity} 件")
    print()
    print("[时间约束]")
    print(f"  必须交期: {order.required_delivery_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"  距离截止: {order.days_to_deadline} 天")
    print(f"  是否紧急: {'是' if order.is_urgent else '否'}")
    print()
    print("[特殊要求]")
    print(f"  质量等级: {order.quality_requirement.value}")
    print(f"  特殊处理: {', '.join(order.special_handling) if order.special_handling else '无'}")
    print(f"  风险标签: {', '.join(order.risk_flags) if order.risk_flags else '无'}")


def main():
    """运行演示"""
    print("=" * 80)
    print("高级业务系统演示 — 真实供应链决策系统")
    print("=" * 80)
    print()

    # ========== 演示 1：VIP 急单 ==========
    print("演示 1：VIP 急单处理")
    print("-" * 80)

    order1 = create_demo_order_1()
    warehouses = create_demo_warehouses()

    print_order_summary(order1)

    # 应用业务规则
    print("[应用业务规则]")
    customer_segment = CustomerSegment(
        customer_id=order1.customer_id,
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

    engine = BusinessRuleEngine(customer_segment)
    order1_after_rules = engine.apply_rules(order1)

    rule_analysis = RuleImpactAnalysis.analyze(order1_after_rules, engine.applied_rules)
    print(f"  应用规则数: {len(engine.applied_rules)}")
    for rule in engine.applied_rules:
        print(f"    - {rule.rule_name}")

    print()
    print(f"  优先级调整: 5 → {order1_after_rules.priority}")
    print(f"  质量等级: {order1_after_rules.quality_requirement.value}")
    print(f"  物流方式: {order1_after_rules.shipping_method.value}")
    print(f"  是否允许合并: {order1_after_rules.consolidation_allowed}")
    print(f"  是否需要检查: {order1_after_rules.require_inspection}")

    # 生成履约方案
    print()
    print("[生成履约方案]")
    options_generator = FulfillmentOptionsGenerator(warehouses)
    options = options_generator.generate_options(order1_after_rules, {})

    print(f"  生成方案数: {len(options)}")
    print()

    for i, option in enumerate(options, 1):
        print(f"  [方案 {i}] {option.option_name}")
        print(f"    策略: {option.strategy.value}")
        print(f"    仓库: {option.primary_warehouse_id}")
        print(f"    物流方式: {option.shipping_method.value}")
        print(f"    处理时间: {option.processing_time_hours}h + 物流 {option.shipping_time_hours}h")
        print(f"    成本: ${option.total_cost:.2f}")
        print(f"      - 仓库处理: ${option.warehouse_handling_cost:.2f}")
        print(f"      - 物流费: ${option.shipping_cost:.2f}")
        print(f"      - 打包费: ${option.packaging_cost:.2f}")
        print(f"    质量风险: {option.quality_risk}")
        print(f"    评分:")
        print(f"      - 成本评分: {option.cost_score:.1f}")
        print(f"      - 时间评分: {option.time_score:.1f}")
        print(f"      - 质量评分: {option.quality_score:.1f}")
        print(f"      - 综合评分: {option.overall_score:.1f} [SCORE]")
        if option.special_notes:
            print(f"    特殊说明: {option.special_notes}")
        print()

    # 推荐方案
    if options:
        best_option = options[0]
        print(f"[推荐方案] {best_option.option_name}")
        print(f"  综合评分最高: {best_option.overall_score:.1f}")
        print(f"  理由: 满足 VIP 客户的快速交期要求，质量等级超级，风险最低")
        print()

    # ========== 演示 2：新客户可延期订单 ==========
    print()
    print("=" * 80)
    print("演示 2：新客户成本优化")
    print("-" * 80)

    order2 = create_demo_order_2()
    print_order_summary(order2)

    # 应用业务规则
    print("[应用业务规则]")
    customer_segment2 = CustomerSegment(
        customer_id=order2.customer_id,
        level=CustomerLevel.NEW_CUSTOMER,
        default_priority=3,
        target_delivery_days=7,
        max_acceptable_delay=3,
        preferred_shipping_method=ShippingMethod.ECONOMY,
        max_shipping_cost_ratio=0.15,
        required_quality_level=QualityLevel.STANDARD,
        inspection_required=False,
        consolidation_allowed=True,
        max_wait_for_consolidation_hours=48,
        substitution_allowed=True,
        substitution_approval_required=False,
        return_period_days=7,
        free_return=False,
        discount_ratio=1.05,  # 新客户可能加价
    )

    engine2 = BusinessRuleEngine(customer_segment2)
    order2_after_rules = engine2.apply_rules(order2)

    print(f"  应用规则数: {len(engine2.applied_rules)}")
    for rule in engine2.applied_rules:
        print(f"    - {rule.rule_name}")

    print()
    print(f"  优先级调整: 3 → {order2_after_rules.priority}")
    print(f"  质量等级: {order2_after_rules.quality_requirement.value}")
    print(f"  物流方式: {order2_after_rules.shipping_method.value}")
    print(f"  是否允许合并: {order2_after_rules.consolidation_allowed}")

    # 生成履约方案
    print()
    print("[生成履约方案]")
    options2 = options_generator.generate_options(order2_after_rules, {})

    print(f"  生成方案数: {len(options2)}")
    print()

    for i, option in enumerate(options2, 1):
        print(f"  [方案 {i}] {option.option_name}")
        print(f"    成本: ${option.total_cost:.2f}")
        print(f"    综合评分: {option.overall_score:.1f} [SCORE]")
        print()

    if options2:
        best_option2 = options2[0]
        print(f"[推荐方案] {best_option2.option_name}")
        print(f"  综合评分最高: {best_option2.overall_score:.1f}")
        print(f"  理由: 新客户成本敏感，推荐最经济方案，允许延迟交期以降低成本")
        if best_option2.consolidation_opportunity:
            print(f"  优化机会: 支持订单合并，进一步节省成本")
        print()

    # ========== 总结 ==========
    print()
    print("=" * 80)
    print("总结")
    print("=" * 80)
    print()
    print("✓ 系统成功演示了真实供应链决策的复杂性：")
    print()
    print("1. [高级订单模型]")
    print("   - 客户分级（VIP / 一级 / 二级 / 新客）")
    print("   - 多维度时间约束（必须交期、优先交期、允许延迟）")
    print("   - 成本预算和敏感度")
    print("   - 质量和特殊需求")
    print()
    print("2. [业务规则引擎]")
    print("   - 根据客户等级自动调整优先级")
    print("   - 根据订单类型（急单 / 可延期）调整策略")
    print("   - 根据时间约束和风险自动应用规则")
    print()
    print("3. [多方案生成]")
    print("   - 快速方案：满足紧急交期，成本最高")
    print("   - 平衡方案：性价比最高（推荐）")
    print("   - 经济方案：最低成本，适合成本敏感客户")
    print("   - 库存优化方案：兼顾成本和库存周转")
    print()
    print("4. [智能评分]")
    print("   - 多维度评分：成本、时间、质量")
    print("   - 加权计算：根据订单特征动态调整权重")
    print("   - VIP 订单优先时间，新客户优先成本")
    print()
    print("Next Steps:")
    print("  1. 集成到 HybridService（作为新的决策引擎）")
    print("  2. 扩展 API 支持高级订单模型")
    print("  3. 添加数据库存储和查询")
    print("  4. 实现 Agent 进阶决策（选择最优方案）")


if __name__ == "__main__":
    main()
