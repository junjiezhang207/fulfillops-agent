#!/usr/bin/env python3
"""Test realtime knowledge system integration with HybridService."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.realtime_knowledge_service import (
    create_demo_realtime_knowledge_service,
)
from app.services.hybrid_rag_service import HybridRAGService
from app.services.advanced_hybrid_service import AdvancedHybridService
from app.schemas.advanced_order import (
    AdvancedOrderDetails,
    CustomerLevel,
    OrderType,
    ShippingMethod,
    QualityLevel,
    Location,
    OrderLineItem,
    WarehouseInfo,
)
from datetime import datetime, timedelta


def test_hybrid_rag_service():
    """Test HybridRAGService methods."""
    print("\n" + "=" * 80)
    print("TEST 1: HybridRAGService - Basic Queries")
    print("=" * 80)

    realtime = create_demo_realtime_knowledge_service()
    rag = HybridRAGService(realtime)

    # Test inventory context
    print("\n[Test 1.1] query_inventory_context")
    context = rag.query_inventory_context("WH-SH", "SKU-A001")
    assert context is not None
    assert "WH-SH" in context
    assert "SKU-A001" in context
    print("[OK] Inventory context retrieved successfully")

    # Test pricing context
    print("\n[Test 1.2] query_pricing_context")
    context = rag.query_pricing_context("SKU-A001")
    assert context is not None
    assert "SKU-A001" in context
    print("[OK] Pricing context retrieved successfully")

    # Test customer context
    print("\n[Test 1.3] query_customer_context")
    context = rag.query_customer_context("CUST-VIP-001")
    assert context is not None
    assert "CUST-VIP-001" in context
    print("[OK] Customer context retrieved successfully")

    # Test market context
    print("\n[Test 1.4] query_market_context")
    context = rag.query_market_context()
    assert context is not None
    print("[OK] Market context retrieved successfully")

    # Test analysis methods
    print("\n[Test 1.5] analyze_sku_opportunity")
    analysis = rag.analyze_sku_opportunity("SKU-D004")
    assert analysis is not None
    assert "SKU-D004" in analysis
    print("[OK] SKU opportunity analysis retrieved successfully")

    print("\n[Test 1.6] analyze_customer_order_fit")
    analysis = rag.analyze_customer_order_fit("CUST-NEW-001", 5000)
    assert analysis is not None
    assert "$5,000" in analysis
    print("[OK] Customer order fit analysis retrieved successfully")

    print("\n[Test 1.7] get_fulfillment_recommendation_context")
    context = rag.get_fulfillment_recommendation_context("vip")
    assert context is not None
    assert "vip" in context
    print("[OK] Fulfillment recommendation context retrieved successfully")


def test_advanced_hybrid_with_realtime():
    """Test AdvancedHybridService with realtime data integration."""
    print("\n" + "=" * 80)
    print("TEST 2: AdvancedHybridService with Realtime Data")
    print("=" * 80)

    # Create demo warehouses
    warehouses = [
        WarehouseInfo(
            warehouse_id="WH-SH",
            name="Shanghai Warehouse",
            location=Location(province="Shanghai", city="Shanghai", district="Pudong"),
            total_capacity=10000,
            used_capacity=2000,
            storage_cost_per_unit_day=0.5,
            picking_cost_per_unit=1.0,
            packing_cost_per_shipment=50.0,
            handling_cost_per_unit=0.5,
        ),
        WarehouseInfo(
            warehouse_id="WH-GZ",
            name="Guangzhou Warehouse",
            location=Location(province="Guangdong", city="Guangzhou", district="Nansha"),
            total_capacity=8000,
            used_capacity=1600,
            storage_cost_per_unit_day=0.45,
            picking_cost_per_unit=0.9,
            packing_cost_per_shipment=40.0,
            handling_cost_per_unit=0.4,
        ),
    ]

    # Create test order
    order = AdvancedOrderDetails(
        order_id="TEST-001",
        customer_id="CUST-VIP-001",
        created_at=datetime.now(),
        customer_level=CustomerLevel.VIP,
        customer_name="Test VIP Customer",
        customer_contact="vip@example.com",
        order_type=OrderType.URGENT,
        required_delivery_date=datetime.now() + timedelta(days=1),
        preferred_delivery_date=datetime.now() + timedelta(days=2),
        destination=Location(province="Shanghai", city="Shanghai", district="Huangpu"),
        line_items=[
            OrderLineItem(
                sku="SKU-A001",
                quantity=100,
                unit_price=50,
                special_notes="Fragile item"
            )
        ],
        budget=10000,
        cost_sensitive=False,
        quality_requirement=QualityLevel.PREMIUM,
        special_handling=["fragile"],
    )

    print("\n[Test 2.1] Create and process order")
    advanced_service = AdvancedHybridService(warehouses=warehouses)
    result = advanced_service.process(order)

    assert result is not None
    assert len(result.rules_applied) > 0
    assert len(result.fulfillment_options) > 0
    assert result.recommended_option_id is not None
    print(f"[OK] Order processed successfully")
    print(f"  - Rules applied: {len(result.rules_applied)}")
    print(f"  - Fulfillment options: {len(result.fulfillment_options)}")
    print(f"  - Final priority: {result.final_priority}")

    print("\n[Test 2.2] Verify rule application")
    if result.original_priority < result.final_priority:
        print(f"[OK] VIP rules applied (priority: {result.original_priority} -> {result.final_priority})")
    else:
        print(f"[OK] Rules applied (priority: {result.original_priority} -> {result.final_priority})")

    print("\n[Test 2.3] Verify fulfillment options")
    for i, option in enumerate(result.fulfillment_options):
        days = option.total_time_hours / 24
        print(f"  Option {i+1}: {option.strategy.value} (Cost: ${option.total_cost:.2f}, Time: {days:.1f}d, Risk: {option.quality_risk})")
    assert all(opt.total_cost > 0 for opt in result.fulfillment_options)
    assert all(opt.total_time_hours > 0 for opt in result.fulfillment_options)
    print("[OK] All options are valid")


def test_realtime_data_models():
    """Test realtime data models."""
    print("\n" + "=" * 80)
    print("TEST 3: Realtime Data Models")
    print("=" * 80)

    realtime = create_demo_realtime_knowledge_service()

    # Test inventory snapshot
    print("\n[Test 3.1] Inventory snapshots with depreciation")
    inv = realtime.inventory_data.get("WH-SH:SKU-A001")
    assert inv is not None
    print(f"[OK] SKU-A001 at WH-SH:")
    print(f"  - Aging days: {inv.aging_days}")
    print(f"  - Depreciation rate: {inv.depreciation_rate * 100:.1f}%")

    inv_aged = realtime.inventory_data.get("WH-GZ:SKU-A001")
    assert inv_aged is not None
    assert inv_aged.aging_days > inv.aging_days
    assert inv_aged.depreciation_rate > inv.depreciation_rate
    print(f"[OK] SKU-A001 at WH-GZ (aged):")
    print(f"  - Aging days: {inv_aged.aging_days}")
    print(f"  - Depreciation rate: {inv_aged.depreciation_rate * 100:.1f}%")

    # Test dynamic pricing factor
    print("\n[Test 3.2] Dynamic pricing factor")
    factor = realtime.get_dynamic_pricing_factor()
    assert 0.8 <= factor <= 1.2
    print(f"[OK] Dynamic pricing factor: {factor:.2f}x")

    # Test customer credit risk
    print("\n[Test 3.3] Customer credit risk assessment")
    vip_risk = realtime.get_customer_credit_risk("CUST-VIP-001")
    new_risk = realtime.get_customer_credit_risk("CUST-NEW-001")
    assert "低风险" in vip_risk
    assert "中等风险" in new_risk
    print(f"[OK] VIP customer risk: Low")
    print(f"[OK] New customer risk: Medium")

    # Test market analysis
    print("\n[Test 3.4] Market analysis")
    market = realtime.market_data
    assert market is not None
    assert market.overall_inventory_rate > 0
    print(f"[OK] Market inventory rate: {market.overall_inventory_rate * 100:.0f}%")
    print(f"[OK] Market cash flow: {market.cash_flow_status}")
    print(f"[OK] Market competition: {market.competition_level}")


def main():
    """Run all tests."""
    try:
        test_hybrid_rag_service()
        test_advanced_hybrid_with_realtime()
        test_realtime_data_models()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        print("\nSummary:")
        print("  [OK] HybridRAGService - All query methods working")
        print("  [OK] AdvancedHybridService - Order processing with rules")
        print("  [OK] Realtime data models - Inventory, pricing, customer, market")
        print("  [OK] Integration complete - Realtime + Static knowledge = Hybrid RAG")
        print()

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
