"""In-memory order repository.

Learning notes:
- Demo fallback data source for local development.
- Enterprise imported orders take priority through CompositeOrderRepository.
"""

from datetime import datetime

from app.repositories.order_repository import OrderRepository
from app.schemas.orders import OrderItem, OrderRecord


class InMemoryOrderRepository(OrderRepository):
    """基于内存模拟数据的订单仓储实现。

    设计说明：
    - 当前阶段故意使用内存模拟数据，而不是数据库。
    - 这样可以让你先专注理解“仓储层接口”和“服务层分析逻辑”的关系。
    - 后续替换成 MySQL 实现时，服务层可以保持基本不变。
    """

    def __init__(self) -> None:
        self._orders: dict[str, OrderRecord] = {
            "SO202502140001": OrderRecord(
                order_id="SO202502140001",
                platform="抖音商城",
                order_time=datetime(2025, 2, 14, 10, 30, 0),
                order_status="待履约",
                region="华东-上海",
                priority="高",
                items=[
                    OrderItem(
                        sku_id="SKU-IPHONE-CASE-001",
                        product_name="磁吸手机壳",
                        quantity=2,
                        unit_price=89.0,
                    ),
                    OrderItem(
                        sku_id="SKU-CHARGER-020W-002",
                        product_name="20W 快充头",
                        quantity=1,
                        unit_price=129.0,
                    ),
                ],
            ),
            "SO202502140002": OrderRecord(
                order_id="SO202502140002",
                platform="天猫旗舰店",
                order_time=datetime(2025, 2, 14, 14, 5, 0),
                order_status="待履约",
                region="华南-广州",
                priority="中",
                items=[
                    OrderItem(
                        sku_id="SKU-BOTTLE-INS-001",
                        product_name="316 不锈钢保温杯",
                        quantity=1,
                        unit_price=199.0,
                    )
                ],
            ),
            "SO202502140003": OrderRecord(
                order_id="SO202502140003",
                platform="京东自营",
                order_time=datetime(2025, 2, 14, 18, 20, 0),
                order_status="已支付待审核",
                region="华北-北京",
                priority="高",
                items=[
                    OrderItem(
                        sku_id="SKU-ROUTER-WIFI7-001",
                        product_name="WiFi 7 路由器",
                        quantity=1,
                        unit_price=899.0,
                    ),
                    OrderItem(
                        sku_id="SKU-CABLE-TYPEC-003",
                        product_name="Type-C 数据线",
                        quantity=3,
                        unit_price=39.9,
                    ),
                ],
            ),
        }

    def get_order_by_id(self, order_id: str) -> OrderRecord | None:
        """根据订单编号获取订单。

        这里直接从内存字典中读取，模拟真实仓储查询行为。
        当前阶段的重点不是存储技术，而是先把上层业务流程跑通。
        """

        return self._orders.get(order_id)
