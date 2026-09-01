"""Order repository interface.

Learning notes:
- Defines the ability to fetch one order by order_id.
- Services depend on this Protocol so storage implementations can be replaced.
"""

from typing import Protocol

from app.schemas.orders import OrderRecord


class OrderRepository(Protocol):
    """订单仓储接口协议。

    设计说明：
    - 当前阶段先用 Protocol 定义最小仓储能力。
    - 这样做的好处是，服务层依赖的是“能力接口”，而不是某个具体实现。
    - 后续可以很自然地扩展出 PostgreSQL 实现、外部 API 实现、测试替身实现。
    """

    def get_order_by_id(self, order_id: str) -> OrderRecord | None:
        """根据订单编号获取订单。"""
