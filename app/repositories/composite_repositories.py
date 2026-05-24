"""Composite repositories.

Learning notes:
- Encapsulates the strategy: enterprise data first, demo data as fallback.
- Service code depends on one repository interface and does not know how many sources are checked.
"""

from app.repositories.inventory_repository import InventoryRepository
from app.repositories.order_repository import OrderRepository
from app.schemas.inventory import InventoryRecord
from app.schemas.orders import OrderRecord


class CompositeOrderRepository(OrderRepository):
    """先查主数据源，查不到再回退到兜底数据源。

    在这个项目里，主数据源就是企业后台导入的数据，兜底数据源是原来的
    InMemoryOrderRepository。这样做的好处是：你可以继续用 demo 订单学习和演示，
    一旦企业导入真实订单，同一个 order_id 会优先使用真实数据。
    """

    def __init__(self, primary: OrderRepository, fallback: OrderRepository) -> None:
        self.primary = primary
        self.fallback = fallback

    def get_order_by_id(self, order_id: str) -> OrderRecord | None:
        return self.primary.get_order_by_id(order_id) or self.fallback.get_order_by_id(order_id)


class CompositeInventoryRepository(InventoryRepository):
    """先查企业库存，企业库存没有命中时再回退到 demo 库存。"""

    def __init__(
        self,
        primary: InventoryRepository,
        fallback: InventoryRepository,
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    def list_inventory_by_sku(self, sku_id: str) -> list[InventoryRecord]:
        records = self.primary.list_inventory_by_sku(sku_id)
        return records if records else self.fallback.list_inventory_by_sku(sku_id)
