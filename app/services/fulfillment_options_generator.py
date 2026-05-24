"""多方案生成器（学习版注释）— 为同一订单生成多个备选履约方案。

每个方案都有不同的成本、时间、质量权衡。
Agent 可以根据客户偏好选择最优方案。

这个文件偏“策略模拟/评分模型”：
- FulfillmentPlanService 输出一个可执行方案。
- FulfillmentOptionsGenerator 输出多个可比较方案，并给每个方案打分。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from datetime import datetime, timedelta

from app.schemas.advanced_order import (
    AdvancedOrderDetails,
    InventoryLevel,
    WarehouseInfo,
    ShippingMethod,
)


class FulfillmentStrategy(Enum):
    """履约策略枚举。

    用枚举而不是字符串，可以减少拼写错误，并让 IDE 有自动补全。
    """
    FAST_TRACK = "fast_track"  # 快速（最快交期，最高成本）
    MIXED = "mixed"  # 混合（平衡成本和时间）
    ECONOMY = "economy"  # 经济（最低成本，较长交期）
    INVENTORY_OPTIMIZATION = "inventory_optimization"  # 库存清理（优先消耗积压）


@dataclass
class FulfillmentOption:
    """履约方案。

    一个 FulfillmentOption 是“候选方案”，不是最终决策。
    方案包含成本、时间、质量等维度，最后通过 calculate_scores 得到综合评分。
    """

    # 方案唯一 ID，便于前端选择或日志追踪。
    option_id: str
    # 策略类型，例如快速、经济、库存优化。
    strategy: FulfillmentStrategy
    # 给人看的方案名称。
    option_name: str
    # 给人看的方案描述。
    description: str

    # ========== 物流信息 ==========
    # 主发货仓。
    primary_warehouse_id: str
    # 备用仓，拆单/调拨时可使用。
    backup_warehouse_id: Optional[str] = None
    # 运输方式，影响时间和成本。
    shipping_method: ShippingMethod = ShippingMethod.STANDARD

    # ========== 成本分解 ==========
    warehouse_handling_cost: float = 0.0  # 仓库处理成本
    shipping_cost: float = 0.0  # 运费
    packaging_cost: float = 0.0  # 打包成本
    storage_cost: float = 0.0  # 额外仓储成本（如延迟）
    insurance_cost: float = 0.0  # 保险成本

    @property
    def total_cost(self) -> float:
        """总成本。

        把各类成本加总，后续评分只读这个属性。
        """
        return (
            self.warehouse_handling_cost +
            self.shipping_cost +
            self.packaging_cost +
            self.storage_cost +
            self.insurance_cost
        )

    # ========== 时间指标 ==========
    processing_time_hours: int = 0  # 处理时间
    shipping_time_hours: int = 0  # 物流时间
    estimated_delivery_date: Optional[datetime] = None  # 预计送达时间

    @property
    def total_time_hours(self) -> int:
        """总耗时 = 仓库处理时间 + 物流运输时间。"""
        return self.processing_time_hours + self.shipping_time_hours

    # ========== 质量指标 ==========
    quality_risk: str = "low"  # low / medium / high
    defect_probability: float = 0.0  # 0-1，破损概率
    warehouse_quality_score: float = 100.0
    shipping_quality_score: float = 100.0

    # ========== 其他特征 ==========
    consolidation_opportunity: bool = False  # 是否有合并机会
    inventory_optimization_benefit: str = ""  # 库存优化收益
    special_notes: str = ""

    # ========== 评分 ==========
    cost_score: float = 0.0  # 0-100，分数越高越便宜
    time_score: float = 0.0  # 0-100，分数越高越快
    quality_score: float = 0.0  # 0-100，分数越高质量越好
    overall_score: float = 0.0  # 综合评分

    def calculate_scores(
        self,
        order: AdvancedOrderDetails,
        max_cost: float = 1000,
        budget_constraint: Optional[float] = None,
    ):
        """计算各维度评分。

        评分越高越好：
        - cost_score：越便宜越高。
        - time_score：越快越高。
        - quality_score：质量越稳定越高。
        - overall_score：根据订单特点加权。
        """

        # 成本评分（越便宜越高）。
        # max_cost 可以理解为本次订单可接受的成本上限。
        if self.total_cost <= max_cost / 4:
            self.cost_score = 100
        else:
            self.cost_score = max(0, 100 - (self.total_cost / max_cost * 100))

        # 时间评分（越快越高）。
        # 先计算距离客户要求送达时间还有多少小时。
        hours_to_deadline = (
            order.required_delivery_date - datetime.now()
        ).total_seconds() / 3600
        if self.total_time_hours <= hours_to_deadline * 0.5:
            self.time_score = 100
        else:
            # time_buffer 越大，说明离截止时间越安全。
            time_buffer = hours_to_deadline - self.total_time_hours
            self.time_score = max(0, 50 + (time_buffer / hours_to_deadline * 50))

        # 质量评分：仓库质量和物流质量取平均，再乘以“不破损概率”。
        self.quality_score = (
            (self.warehouse_quality_score + self.shipping_quality_score) / 2
        ) * (1 - self.defect_probability)

        # 综合评分（根据订单优先级权衡）。
        if order.is_urgent:
            # 紧急订单优先时间
            self.overall_score = (
                self.time_score * 0.5 +
                self.quality_score * 0.3 +
                self.cost_score * 0.2
            )
        elif order.cost_sensitive:
            # 成本敏感优先成本
            self.overall_score = (
                self.cost_score * 0.5 +
                self.quality_score * 0.3 +
                self.time_score * 0.2
            )
        else:
            # 平衡权重
            self.overall_score = (
                self.cost_score * 0.3 +
                self.time_score * 0.3 +
                self.quality_score * 0.4
            )


class FulfillmentOptionsGenerator:
    """生成多个履约方案。

    这个类的输入是订单和库存快照，输出是一组已经排序的候选方案。
    """

    def __init__(self, warehouses: List[WarehouseInfo]):
        # 可用仓库列表。这里用构造函数注入，便于测试时传不同仓库集合。
        self.warehouses = warehouses

    def generate_options(
        self,
        order: AdvancedOrderDetails,
        inventory_snapshot: dict,  # sku -> [warehouse_inventories]
    ) -> List[FulfillmentOption]:
        """为订单生成多个备选方案。

        这里不是只生成一个“最优”，而是把不同偏好的方案都生成出来，
        再通过评分排序，让 Agent/前端可以解释取舍。
        """

        options = []

        # 方案 1：快速方案（最快交期，最高成本）
        fast_option = self._generate_fast_track_option(
            order, inventory_snapshot
        )
        if fast_option:
            options.append(fast_option)

        # 方案 2：平衡方案（成本和时间平衡）
        mixed_option = self._generate_mixed_option(order, inventory_snapshot)
        if mixed_option:
            options.append(mixed_option)

        # 方案 3：经济方案（最低成本）
        economy_option = self._generate_economy_option(order, inventory_snapshot)
        if economy_option:
            options.append(economy_option)

        # 方案 4：库存优化方案（如果有积压库存）
        inventory_opt_option = self._generate_inventory_optimization_option(
            order, inventory_snapshot
        )
        if inventory_opt_option:
            options.append(inventory_opt_option)

        # 计算所有方案的评分。
        for option in options:
            option.calculate_scores(order)

        # 按综合评分排序，最高分排第一。
        options.sort(key=lambda x: x.overall_score, reverse=True)

        return options

    def _generate_fast_track_option(
        self, order: AdvancedOrderDetails, inventory: dict
    ) -> Optional[FulfillmentOption]:
        """生成快速方案。

        目标是最快送达，所以倾向最近仓和最快物流，成本较高。
        """

        # 选择距离最近、质量最好的仓库。
        best_warehouse = self._find_nearest_warehouse(order.destination)

        if not best_warehouse:
            return None

        # 构造快速方案的成本/时间/质量参数。
        option = FulfillmentOption(
            option_id="opt_001_fast_track",
            strategy=FulfillmentStrategy.FAST_TRACK,
            option_name="快速方案",
            description="优先满足交期，从距离最近的仓库快速发货",
            primary_warehouse_id=best_warehouse.warehouse_id,
            shipping_method=ShippingMethod.OVERNIGHT,
            processing_time_hours=2,
            shipping_time_hours=24,
            warehouse_handling_cost=order.total_quantity * 5,
            shipping_cost=100,
            packaging_cost=50,
            quality_risk="low",
            warehouse_quality_score=best_warehouse.quality_rating,
            special_notes=f"从 {best_warehouse.name} 快速发货，保证 24 小时内发出",
        )

        option.estimated_delivery_date = datetime.now() + timedelta(
            hours=option.total_time_hours
        )

        return option

    def _generate_mixed_option(
        self, order: AdvancedOrderDetails, inventory: dict
    ) -> Optional[FulfillmentOption]:
        """生成平衡方案（推荐）。

        在成本、时间、质量之间取折中，通常适合作为默认推荐。
        """

        # 选择仓库（库存充足 + 成本合理）。
        warehouse = self._find_optimal_warehouse(order)

        if not warehouse:
            return None

        option = FulfillmentOption(
            option_id="opt_002_mixed",
            strategy=FulfillmentStrategy.MIXED,
            option_name="平衡方案（推荐）",
            description="在成本和时间之间取得平衡，性价比最高",
            primary_warehouse_id=warehouse.warehouse_id,
            shipping_method=ShippingMethod.STANDARD,
            processing_time_hours=4,
            shipping_time_hours=72,
            warehouse_handling_cost=order.total_quantity * 3,
            shipping_cost=50,
            packaging_cost=40,
            quality_risk="low",
            warehouse_quality_score=warehouse.quality_rating,
            special_notes=f"从 {warehouse.name} 标准物流发货，3-5 天送达",
        )

        option.estimated_delivery_date = datetime.now() + timedelta(
            hours=option.total_time_hours
        )

        return option

    def _generate_economy_option(
        self, order: AdvancedOrderDetails, inventory: dict
    ) -> Optional[FulfillmentOption]:
        """生成经济方案。

        目标是成本最低，所以运输更慢、质量风险略高。
        """

        # 选择成本最低的仓库。
        warehouse = self._find_cheapest_warehouse(order)

        if not warehouse:
            return None

        option = FulfillmentOption(
            option_id="opt_003_economy",
            strategy=FulfillmentStrategy.ECONOMY,
            option_name="经济方案",
            description="最低成本，适合对价格敏感的订单",
            primary_warehouse_id=warehouse.warehouse_id,
            shipping_method=ShippingMethod.ECONOMY,
            processing_time_hours=6,
            shipping_time_hours=168,  # 7 天
            warehouse_handling_cost=order.total_quantity * 2,
            shipping_cost=20,
            packaging_cost=25,
            quality_risk="medium",
            warehouse_quality_score=warehouse.quality_rating * 0.9,
            special_notes=f"从 {warehouse.name} 经济物流发货，7-14 天送达",
        )

        option.estimated_delivery_date = datetime.now() + timedelta(
            hours=option.total_time_hours
        )

        return option

    def _generate_inventory_optimization_option(
        self, order: AdvancedOrderDetails, inventory: dict
    ) -> Optional[FulfillmentOption]:
        """生成库存优化方案（针对积压库存）。

        如果某仓有老库存，可以优先从该仓发货，提高库存周转。
        """

        # 查找有积压库存的仓库。
        warehouse_with_aged_inventory = (
            self._find_warehouse_with_aged_inventory(order)
        )

        if not warehouse_with_aged_inventory:
            return None

        warehouse_id, inventory_age_days = warehouse_with_aged_inventory

        warehouse = next(
            (w for w in self.warehouses if w.warehouse_id == warehouse_id), None
        )
        if not warehouse:
            return None

        option = FulfillmentOption(
            option_id="opt_004_inventory_optimization",
            strategy=FulfillmentStrategy.INVENTORY_OPTIMIZATION,
            option_name=f"库存清理方案",
            description=f"此仓库此 SKU 库存积压 {inventory_age_days} 天，优先清理，价格更优",
            primary_warehouse_id=warehouse_id,
            shipping_method=ShippingMethod.STANDARD,
            processing_time_hours=4,
            shipping_time_hours=72,
            warehouse_handling_cost=order.total_quantity * 2.5,
            shipping_cost=40,
            packaging_cost=35,
            quality_risk="medium",
            warehouse_quality_score=max(80, warehouse.quality_rating - 10),
            consolidation_opportunity=True,
            inventory_optimization_benefit=f"清理 {inventory_age_days} 天的积压库存，节省日均仓储成本 ${order.total_quantity * warehouse.storage_cost_per_unit_day * inventory_age_days:.2f}",
            special_notes=f"优先选择此方案可优化库存周转，建议给客户 5-10% 折扣",
        )

        option.estimated_delivery_date = datetime.now() + timedelta(
            hours=option.total_time_hours
        )

        return option

    def _find_nearest_warehouse(self, location) -> Optional[WarehouseInfo]:
        """找距离最近的仓库。

        当前是演示版：直接返回第一个仓库。
        生产版应根据目的地经纬度、仓库地址、物流时效计算。
        """
        if not self.warehouses:
            return None
        # 简化：直接返回第一个（实际应该计算距离）
        return self.warehouses[0]

    def _find_optimal_warehouse(self, order) -> Optional[WarehouseInfo]:
        """找最优仓库（库存充足 + 质量好 + 成本合理）。

        当前简化为质量评分最高的仓。
        """
        if not self.warehouses:
            return None
        # 返回质量评分最高的仓库
        return max(self.warehouses, key=lambda w: w.quality_rating)

    def _find_cheapest_warehouse(self, order) -> Optional[WarehouseInfo]:
        """找成本最低的仓库。

        当前按拣货、处理、打包摊销成本估算。
        """
        if not self.warehouses:
            return None
        # 返回处理成本最低的仓库
        return min(
            self.warehouses,
            key=lambda w: (
                w.picking_cost_per_unit +
                w.handling_cost_per_unit +
                w.packing_cost_per_shipment / order.total_quantity
            ),
        )

    def _find_warehouse_with_aged_inventory(
        self, order
    ) -> Optional[tuple[str, int]]:
        """找有积压库存的仓库。

        当前返回 None，表示演示数据里没有接入真实库存年龄。
        生产版应查询库存批次、入库日期、库龄等字段。
        """
        # 返回 None 示例（实际应该查询库存）。
        return None
