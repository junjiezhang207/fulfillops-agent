"""订单分析服务（学习版注释）。

这个文件负责把“原始订单记录”转换成“业务可读的订单分析结果”。
它处在 Repository 和 API/Agent 中间：
- Repository 只负责取数据。
- Service 负责业务理解和结构化分析。
- API/Agent 只消费分析结果，不关心订单数据怎么拼出来。
"""

from app.repositories.order_repository import OrderRepository
from app.schemas.orders import OrderAnalysisResult, OrderRecord


class OrderNotFoundError(Exception):
    """订单不存在异常。

    单独定义异常的好处是：API 层可以精确捕获它并返回 404，
    而不是把所有错误都当成 500。
    """


class OrderAnalysisService:
    """订单分析服务。

    设计说明：
    - 这个服务代表“领域服务层”，负责真正的订单分析业务逻辑。
    - API 层不应该直接操作模拟数据，更不应该自己写分析逻辑。
    - 后续如果订单数据来源变化，或者分析规则变复杂，优先改这里。
    """

    def __init__(self, repository: OrderRepository) -> None:
        # 注入订单仓库，方便未来把模拟数据换成数据库、ERP、OMS 接口。
        self.repository = repository

    def analyze_order(self, order_id: str) -> OrderAnalysisResult:
        """分析指定订单并输出结构化结果。

        这个方法是外部调用入口，只做三件事：
        1. 按 order_id 读取订单。
        2. 处理订单不存在的情况。
        3. 把真实分析交给 _build_analysis_result。
        """

        # Repository 返回的是原始订单记录，可能存在也可能不存在。
        order = self.repository.get_order_by_id(order_id)
        if order is None:
            raise OrderNotFoundError(f"订单不存在：{order_id}")

        # 找到订单后，进入结构化分析逻辑。
        return self._build_analysis_result(order)

    def _build_analysis_result(self, order: OrderRecord) -> OrderAnalysisResult:
        """将原始订单数据转换为结构化分析结果。

        这里故意拆成独立私有方法，原因是：
        1. 让 `analyze_order` 保持清晰，只负责流程控制。
        2. 让真正的分析逻辑集中在一个地方，便于后续扩展。
        3. 后续增加订单风险标签、平台规则提示时，也更容易继续往这里追加。
        """

        # 提取订单包含的 SKU 列表，后续库存分析会用到。
        sku_list = [item.sku_id for item in order.items]
        # 统计订单总件数，不是订单行数；一行商品可能购买多个。
        total_quantity = sum(item.quantity for item in order.items)
        # 订单行数，表示订单里有几种商品。
        item_count = len(order.items)

        # summary 是给人读的一句话业务摘要，Agent 回答时也可以直接引用。
        summary = (
            f"订单 {order.order_id} 来自 {order.platform}，"
            f"当前状态为 {order.order_status}，"
            f"收货区域为 {order.region}，"
            f"优先级为 {order.priority}。"
            f"本单共包含 {item_count} 行商品，"
            f"总购买件数为 {total_quantity}。"
        )

        # 返回 Pydantic schema，而不是直接返回 ORM/字典。
        # 这样 API、Workflow、Agent 都拿到稳定的字段结构。
        return OrderAnalysisResult(
            order_id=order.order_id,
            platform=order.platform,
            order_status=order.order_status,
            region=order.region,
            priority=order.priority,
            item_count=item_count,
            total_quantity=total_quantity,
            sku_list=sku_list,
            items=order.items,
            summary=summary,
        )
