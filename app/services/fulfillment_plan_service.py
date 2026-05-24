"""履约方案服务（学习版注释）。

这个文件负责把多个基础能力组合成“订单履约方案”：
- 库存服务：判断每个 SKU 够不够。
- 仓库服务：找到有货且适合发货的仓库。
- 替代 SKU 服务：缺货时给出可替代商品。

它输出的是可执行的动作列表，例如“从某仓发货 / 用某替代品 / 预订补货”。
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.services.inventory_analysis_service import InventoryAnalysisService
from app.services.substitute_sku_service import SubstituteSkuService
from app.services.warehouse_service import WarehouseService


@dataclass
class FulfillmentAction:
    """单个履约动作。

    一个订单可能有多行商品，每行商品可能对应不同动作。
    例如 A 商品直接发货，B 商品用替代品，C 商品预订。
    """

    # 原订单中的 SKU。
    sku_id: str
    # 原商品名称。
    product_name: str
    # 本动作处理的数量，当前用字符串是为了直接展示“2件”。
    quantity: str
    # 动作类型：发货、替代、预订。
    action_type: str  # "ship_from_warehouse", "substitute", "backorder"
    # 发货仓信息，仅 ship_from_warehouse 时有值。
    warehouse_id: str | None = None
    warehouse_name: str | None = None
    # 替代商品信息，仅 substitute 时有值。
    substitute_sku: str | None = None
    substitute_product: str | None = None
    # 预计完成天数，用于方案整体 ETA。
    estimated_days: int | None = None
    # 给运营/客服看的补充说明。
    note: str = ""


@dataclass
class FulfillmentPlan:
    """完整的履约方案。

    这是服务最终返回的结构，包含策略名、动作列表、成本影响和摘要。
    """

    order_id: str
    plan_strategy: str  # "fast_track", "mixed", "backorder"
    actions: list[FulfillmentAction]
    total_cost_impact: float
    estimated_completion_days: int
    summary: str


class FulfillmentPlanService:
    """履约方案生成服务。

    这里是典型的“应用服务”：不直接保存数据，而是协调多个领域服务完成业务流程。
    """

    def __init__(
        self,
        inventory_service: InventoryAnalysisService,
        warehouse_service: WarehouseService,
        substitute_service: SubstituteSkuService,
    ) -> None:
        # 库存分析：判断订单整体和每个 SKU 是否充足。
        self.inventory_service = inventory_service
        # 仓库查询：库存充足时选择发货仓。
        self.warehouse_service = warehouse_service
        # 替代 SKU：库存不足时尝试找替代方案。
        self.substitute_service = substitute_service

    def generate_plan(self, order_id: str) -> FulfillmentPlan:
        """为订单生成履约方案。

        策略：
          1. 库存充足 → fast_track（直接从最近仓发货，3-5天）
          2. 部分缺货 → mixed（充足的正常发，缺货的用替代或预订，5-10天）
          3. 全部缺货 → backorder（全部预订，10-15天或建议取消）

        Args:
            order_id: 订单ID

        Returns:
            完整的履约方案
        """
        # 获取订单和库存信息。这里捕获异常是为了让前端拿到结构化失败结果，
        # 而不是整个请求直接崩掉。
        try:
            order_result = self.inventory_service.order_analysis_service.analyze_order(
                order_id
            )
            inventory_result = self.inventory_service.analyze_inventory(order_id)
        except Exception as exc:
            return FulfillmentPlan(
                order_id=order_id,
                plan_strategy="error",
                actions=[],
                total_cost_impact=0,
                estimated_completion_days=0,
                summary=f"方案生成失败：{exc}",
            )

        # actions 是最终方案里的“动作清单”。
        actions: list[FulfillmentAction] = []
        # 替代品可能比原商品贵或便宜，这里累计成本影响。
        total_cost_impact = 0.0
        # fulfilled_count 记录能被直接/替代履约的 SKU 行数。
        fulfilled_count = 0

        # 遍历订单中的每个SKU，生成对应的履约动作
        if order_result and inventory_result:
            for item in order_result.items:
                sku_id = item.sku_id
                qty = item.quantity

                if inventory_result.fulfillment_ready:
                    # 库存充足：从库存最多的仓库直接发货。
                    warehouse_result = self.warehouse_service.search_sku_inventory(sku_id)
                    if warehouse_result.warehouse_list:
                        best_wh = warehouse_result.warehouse_list[0]
                        actions.append(
                            FulfillmentAction(
                                sku_id=sku_id,
                                product_name=item.product_name,
                                quantity=f"{qty}件",
                                action_type="ship_from_warehouse",
                                warehouse_id=best_wh.warehouse_id,
                                warehouse_name=best_wh.warehouse_name,
                                estimated_days=3,
                                note=f"从{best_wh.warehouse_name}直接发货",
                            )
                        )
                        fulfilled_count += 1
                else:
                    # 库存不足：优先尝试替代，其次预订。
                    substitute_result = self.substitute_service.search_substitutes(sku_id)
                    if substitute_result.substitutes:
                        # 用兼容度最高的替代品；替代是否需要客户确认，由上层流程决定。
                        best_sub = substitute_result.substitutes[0]
                        # 单件价差 * 数量 = 本 SKU 的成本影响。
                        price_diff = best_sub.unit_price - item.unit_price
                        total_cost_impact += price_diff * qty

                        actions.append(
                            FulfillmentAction(
                                sku_id=sku_id,
                                product_name=item.product_name,
                                quantity=f"{qty}件",
                                action_type="substitute",
                                substitute_sku=best_sub.sku_id,
                                substitute_product=best_sub.product_name,
                                estimated_days=3,
                                note=f"用{best_sub.product_name}替代，兼容度{best_sub.compatibility_score}%，"
                                f"单价差异 ￥{price_diff:.1f}/件",
                            )
                        )
                        fulfilled_count += 1
                    else:
                        # 无替代方案：只能预订补货。
                        actions.append(
                            FulfillmentAction(
                                sku_id=sku_id,
                                product_name=item.product_name,
                                quantity=f"{qty}件",
                                action_type="backorder",
                                estimated_days=10,
                                note="库存预订，预计10天后补货",
                            )
                        )

        # 根据已履约 SKU 数量决定整体策略。
        if fulfilled_count == len(order_result.items) if order_result else False:
            plan_strategy = "fast_track"
            estimated_days = 3
        elif fulfilled_count > 0:
            plan_strategy = "mixed"
            estimated_days = 7
        else:
            plan_strategy = "backorder"
            estimated_days = 10

        # 摘要面向人读，结构化 actions 面向系统读。
        summary = self._build_summary(
            order_id, plan_strategy, fulfilled_count, len(actions), total_cost_impact
        )

        return FulfillmentPlan(
            order_id=order_id,
            plan_strategy=plan_strategy,
            actions=actions,
            total_cost_impact=total_cost_impact,
            estimated_completion_days=estimated_days,
            summary=summary,
        )

    def _build_summary(
        self, order_id: str, strategy: str, fulfilled: int, total: int, cost_impact: float
    ) -> str:
        """生成方案摘要。

        这里把内部策略代码翻译成中文，方便前端和 Agent 直接展示。
        """
        strategy_cn = {
            "fast_track": "快速履约（3-5天）",
            "mixed": "混合履约（部分替代，5-7天）",
            "backorder": "预订策略（10-15天）",
        }.get(strategy, "未知策略")

        lines = [
            f"订单 {order_id} 的履约方案：",
            f"  整体策略：{strategy_cn}",
            f"  可直接履约：{fulfilled}/{total} 个SKU",
        ]

        if cost_impact > 0:
            lines.append(f"  成本增加：￥{cost_impact:.2f}（由于替代或升级）")
        elif cost_impact < 0:
            lines.append(f"  成本节省：￥{-cost_impact:.2f}（通过替代或降级）")

        return "\n".join(lines)
