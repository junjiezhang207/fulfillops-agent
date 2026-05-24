from datetime import datetime

from app.repositories.composite_repositories import (
    CompositeInventoryRepository,
    CompositeOrderRepository,
)
from app.repositories.enterprise_data_repository import EnterpriseDataRepository
from app.repositories.in_memory_inventory_repository import InMemoryInventoryRepository
from app.repositories.in_memory_order_repository import InMemoryOrderRepository
from app.schemas.enterprise_data import EnterpriseDataSourceCreate
from app.schemas.inventory import InventoryRecord
from app.schemas.orders import OrderItem, OrderRecord
from app.services.inventory_analysis_service import InventoryAnalysisService
from app.services.order_analysis_service import OrderAnalysisService


def _order(order_id: str = "SO-ENT-001", sku_id: str = "SKU-ENT-001") -> OrderRecord:
    return OrderRecord(
        order_id=order_id,
        platform="ERP",
        order_time=datetime(2026, 5, 11, 9, 30, 0),
        order_status="待履约",
        region="华东-上海",
        priority="高",
        items=[
            OrderItem(
                sku_id=sku_id,
                product_name="企业导入商品",
                quantity=3,
                unit_price=99.0,
            )
        ],
    )


def _inventory(
    sku_id: str = "SKU-ENT-001",
    warehouse_id: str = "WH-ENT-001",
    available_stock: int = 8,
) -> InventoryRecord:
    return InventoryRecord(
        warehouse_id=warehouse_id,
        warehouse_name="企业上海仓",
        region="华东-上海",
        sku_id=sku_id,
        available_stock=available_stock,
        locked_stock=1,
        updated_at=datetime(2026, 5, 11, 10, 0, 0),
    )


def test_enterprise_orders_are_persisted(tmp_path):
    repo = EnterpriseDataRepository(tmp_path)

    repo.import_orders("erp-main", [_order()])

    reloaded = EnterpriseDataRepository(tmp_path)
    saved = reloaded.get_order_by_id("SO-ENT-001")

    assert saved is not None
    assert saved.platform == "ERP"
    assert reloaded.stats().order_count == 1


def test_source_id_can_be_generated_from_source_name(tmp_path):
    repo = EnterpriseDataRepository(tmp_path)

    source = repo.create_source(EnterpriseDataSourceCreate(name="ERP Main"))

    assert source.source_id == "erp-main"


def test_inventory_replace_only_affects_the_selected_source(tmp_path):
    repo = EnterpriseDataRepository(tmp_path)
    repo.import_inventory("erp-a", [_inventory(warehouse_id="WH-A")])
    repo.import_inventory("erp-b", [_inventory(warehouse_id="WH-B", available_stock=2)])

    repo.import_inventory(
        "erp-a",
        [_inventory(warehouse_id="WH-A-NEW", available_stock=6)],
        replace_source=True,
    )

    records = repo.list_inventory_by_sku("SKU-ENT-001")
    warehouse_ids = {record.warehouse_id for record in records}

    assert warehouse_ids == {"WH-A-NEW", "WH-B"}
    assert sum(record.available_stock for record in records) == 8


def test_composite_repositories_prefer_enterprise_data(tmp_path):
    enterprise = EnterpriseDataRepository(tmp_path)
    enterprise.import_orders("erp-main", [_order(order_id="SO202502140001")])
    enterprise.import_inventory("erp-main", [_inventory(sku_id="SKU-ENT-001")])

    order_repo = CompositeOrderRepository(
        primary=enterprise,
        fallback=InMemoryOrderRepository(),
    )
    inventory_repo = CompositeInventoryRepository(
        primary=enterprise,
        fallback=InMemoryInventoryRepository(),
    )

    order = order_repo.get_order_by_id("SO202502140001")
    inventory = inventory_repo.list_inventory_by_sku("SKU-ENT-001")

    assert order is not None
    assert order.platform == "ERP"
    assert inventory[0].warehouse_id == "WH-ENT-001"


def test_inventory_analysis_uses_enterprise_order_and_inventory(tmp_path):
    enterprise = EnterpriseDataRepository(tmp_path)
    enterprise.import_orders("erp-main", [_order()])
    enterprise.import_inventory("erp-main", [_inventory(available_stock=3)])

    order_service = OrderAnalysisService(
        CompositeOrderRepository(
            primary=enterprise,
            fallback=InMemoryOrderRepository(),
        )
    )
    inventory_service = InventoryAnalysisService(
        inventory_repository=CompositeInventoryRepository(
            primary=enterprise,
            fallback=InMemoryInventoryRepository(),
        ),
        order_analysis_service=order_service,
    )

    result = inventory_service.analyze_inventory("SO-ENT-001")

    assert result.fulfillment_ready is True
    assert result.sku_checks[0].total_available_stock == 3
