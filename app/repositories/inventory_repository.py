"""Inventory repository interface.

Learning notes:
- Defines the minimum ability needed by services: list inventory records by SKU.
- Production can add WMS/API/SQL implementations without changing service logic.
"""

from typing import Protocol

from app.schemas.inventory import InventoryRecord


class InventoryRepository(Protocol):
    """库存仓储接口协议。"""

    def list_inventory_by_sku(self, sku_id: str) -> list[InventoryRecord]:
        """根据 SKU 查询所有相关库存记录。"""
