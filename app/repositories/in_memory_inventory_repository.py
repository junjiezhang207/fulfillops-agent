"""In-memory inventory repository.

Learning notes:
- Demo fallback data source for local development.
- Real enterprise inventory should come from WMS, databases or sync jobs.
"""

from datetime import datetime

from app.repositories.inventory_repository import InventoryRepository
from app.schemas.inventory import InventoryRecord


class InMemoryInventoryRepository(InventoryRepository):
    """基于内存模拟数据的库存仓储实现。"""

    def __init__(self) -> None:
        self._inventory_by_sku: dict[str, list[InventoryRecord]] = {
            "SKU-IPHONE-CASE-001": [
                InventoryRecord(
                    warehouse_id="WH-SH-001",
                    warehouse_name="上海一仓",
                    region="华东-上海",
                    sku_id="SKU-IPHONE-CASE-001",
                    available_stock=5,
                    locked_stock=1,
                    updated_at=datetime(2025, 2, 14, 9, 30, 0),
                ),
                InventoryRecord(
                    warehouse_id="WH-HZ-002",
                    warehouse_name="杭州二仓",
                    region="华东-杭州",
                    sku_id="SKU-IPHONE-CASE-001",
                    available_stock=8,
                    locked_stock=2,
                    updated_at=datetime(2025, 2, 14, 9, 40, 0),
                ),
            ],
            "SKU-CHARGER-020W-002": [
                InventoryRecord(
                    warehouse_id="WH-SH-001",
                    warehouse_name="上海一仓",
                    region="华东-上海",
                    sku_id="SKU-CHARGER-020W-002",
                    available_stock=0,
                    locked_stock=3,
                    updated_at=datetime(2025, 2, 14, 9, 35, 0),
                ),
                InventoryRecord(
                    warehouse_id="WH-SZ-003",
                    warehouse_name="深圳三仓",
                    region="华南-深圳",
                    sku_id="SKU-CHARGER-020W-002",
                    available_stock=2,
                    locked_stock=0,
                    updated_at=datetime(2025, 2, 14, 9, 50, 0),
                ),
            ],
            "SKU-BOTTLE-INS-001": [
                InventoryRecord(
                    warehouse_id="WH-GZ-004",
                    warehouse_name="广州四仓",
                    region="华南-广州",
                    sku_id="SKU-BOTTLE-INS-001",
                    available_stock=12,
                    locked_stock=1,
                    updated_at=datetime(2025, 2, 14, 10, 0, 0),
                )
            ],
            "SKU-ROUTER-WIFI7-001": [
                InventoryRecord(
                    warehouse_id="WH-BJ-005",
                    warehouse_name="北京五仓",
                    region="华北-北京",
                    sku_id="SKU-ROUTER-WIFI7-001",
                    available_stock=1,
                    locked_stock=0,
                    updated_at=datetime(2025, 2, 14, 10, 10, 0),
                )
            ],
            "SKU-CABLE-TYPEC-003": [
                InventoryRecord(
                    warehouse_id="WH-BJ-005",
                    warehouse_name="北京五仓",
                    region="华北-北京",
                    sku_id="SKU-CABLE-TYPEC-003",
                    available_stock=2,
                    locked_stock=0,
                    updated_at=datetime(2025, 2, 14, 10, 15, 0),
                ),
                InventoryRecord(
                    warehouse_id="WH-TJ-006",
                    warehouse_name="天津六仓",
                    region="华北-天津",
                    sku_id="SKU-CABLE-TYPEC-003",
                    available_stock=5,
                    locked_stock=1,
                    updated_at=datetime(2025, 2, 14, 10, 20, 0),
                ),
            ],
        }

    def list_inventory_by_sku(self, sku_id: str) -> list[InventoryRecord]:
        """根据 SKU 查询所有相关库存记录。"""

        return self._inventory_by_sku.get(sku_id, [])
