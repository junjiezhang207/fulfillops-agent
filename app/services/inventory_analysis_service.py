"""库存分析服务（学习版注释）。

这个文件回答一个核心业务问题：
“某个订单里每个 SKU 的全国可售库存，是否足够覆盖订单需求？”

它不负责找替代 SKU，也不负责生成履约方案，只做库存是否充足的判断。
这样职责边界比较清楚：库存分析是后续缺货处理、RAG 查询、Workflow 决策的输入。
"""

from app.repositories.inventory_repository import InventoryRepository
from app.schemas.inventory import InventoryAnalysisResult, SkuInventoryCheckResult
from app.services.order_analysis_service import OrderAnalysisService


class InventoryAnalysisService:
    """库存判断服务。

    设计说明：
    - 这个服务是订单分析模块的自然下一步。
    - 它不会自己解析订单，而是复用已经完成的订单分析服务。
    - 这样可以避免重复写订单读取和订单结构解析逻辑。
    """

    def __init__(
        self,
        inventory_repository: InventoryRepository,
        order_analysis_service: OrderAnalysisService,
    ) -> None:
        # 库存仓库负责按 SKU 查询各仓库存量。
        self.inventory_repository = inventory_repository
        # 订单分析服务负责把 order_id 变成订单结构，避免这里重复读订单。
        self.order_analysis_service = order_analysis_service

    def analyze_inventory(self, order_id: str) -> InventoryAnalysisResult:
        """根据订单编号判断当前库存是否满足履约。

        主流程：
        1. 先分析订单，拿到订单商品行。
        2. 对每个 SKU 汇总全国仓库可售库存。
        3. 判断每个 SKU 是否满足订单需求。
        4. 汇总成订单级别的 fulfillment_ready。
        """

        # 复用订单分析结果，拿到 items、订单摘要等信息。
        order_analysis = self.order_analysis_service.analyze_order(order_id)

        # sku_checks 保存每个 SKU 的库存检查结果。
        sku_checks: list[SkuInventoryCheckResult] = []
        # insufficient_skus 只记录缺货 SKU，方便上层快速判断风险点。
        insufficient_skus: list[str] = []

        for item in order_analysis.items:
            # 查询这个 SKU 在所有仓库里的库存记录。
            warehouse_records = self.inventory_repository.list_inventory_by_sku(
                item.sku_id
            )
            # 这里用“可用库存”汇总，而不是 total_stock。
            # 预留库存通常已经被其他订单占用，不能拿来履约本单。
            total_available_stock = sum(
                record.available_stock for record in warehouse_records
            )
            # 只要全国可用库存总和覆盖订单需求，就认为这个 SKU 具备履约条件。
            fulfillment_ready = total_available_stock >= item.quantity

            if not fulfillment_ready:
                insufficient_skus.append(item.sku_id)

            # 保存 SKU 级别明细，前端和 Agent 都可以展开说明。
            sku_checks.append(
                SkuInventoryCheckResult(
                    sku_id=item.sku_id,
                    required_quantity=item.quantity,
                    total_available_stock=total_available_stock,
                    fulfillment_ready=fulfillment_ready,
                    warehouse_records=warehouse_records,
                )
            )

        # 订单整体是否可履约：所有 SKU 都充足才算 true。
        overall_ready = len(insufficient_skus) == 0
        summary = self._build_summary(
            order_id=order_analysis.order_id,
            fulfillment_ready=overall_ready,
            insufficient_skus=insufficient_skus,
        )

        # 返回订单级别 + SKU 级别的完整库存判断。
        return InventoryAnalysisResult(
            order_id=order_analysis.order_id,
            fulfillment_ready=overall_ready,
            insufficient_skus=insufficient_skus,
            sku_checks=sku_checks,
            order_summary=order_analysis.summary,
            summary=summary,
        )

    def _build_summary(
        self,
        order_id: str,
        fulfillment_ready: bool,
        insufficient_skus: list[str],
    ) -> str:
        """生成库存分析摘要。

        这个摘要不是最终决策，只是把结构化结果转成人能快速理解的句子。
        """

        if fulfillment_ready:
            return (
                f"订单 {order_id} 当前库存满足履约条件，"
                f"所有 SKU 的汇总可售库存均覆盖订单需求。"
            )

        insufficient_text = "、".join(insufficient_skus)
        return (
            f"订单 {order_id} 当前库存不足，"
            f"以下 SKU 无法满足订单需求：{insufficient_text}。"
        )
