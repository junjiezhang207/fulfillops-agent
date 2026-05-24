"""高级混合服务（学习版注释）— 支持真实业务复杂性的订单处理。

与基础 HybridService 的区别：
  • 输入：AdvancedOrderDetails（包含客户分级、时间约束、特殊需求）
  • 处理：应用业务规则 → 生成多个履约方案 → 智能评分
  • 输出：多个 FulfillmentOption（不是单一答案）
"""

from dataclasses import dataclass, field
from typing import Optional
import time

from app.schemas.advanced_order import AdvancedOrderDetails, WarehouseInfo
from app.domain.rules.business_rule_engine import (
    BusinessRuleEngine,
    DynamicPricingEngine,
    RuleImpactAnalysis,
)
from app.domain.fulfillment.options_generator import (
    FulfillmentOption,
    FulfillmentOptionsGenerator,
)
from app.application.memory.session_memory_service import SessionMemoryService


@dataclass
class AdvancedHybridResult:
    """高级混合服务的统一返回格式。

    这个结果不是自然语言回答，而是“规则 + 方案 + 评分”的结构化结果。
    """

    order_id: str
    customer_id: str
    customer_level: str
    question: str

    # 规则应用结果
    rules_applied: list[str] = field(default_factory=list)
    rule_details: list[dict] = field(default_factory=list)
    rule_conflicts: list[str] = field(default_factory=list)
    original_priority: int = 5
    final_priority: int = 5
    risk_level: str = "low"
    risk_flags: list[str] = field(default_factory=list)
    pricing_adjustment: dict | None = None

    # 履约方案
    fulfillment_options: list[FulfillmentOption] = field(default_factory=list)
    recommended_option_id: Optional[str] = None

    # 统计信息
    execution_time_ms: float = 0.0
    from_cache: bool = False
    thread_id: Optional[str] = None


class AdvancedHybridService:
    """高级混合服务 — 处理真实业务复杂的订单。

    和普通 HybridService 不同，它不根据用户问题路由 Agent/Workflow，
    而是针对 AdvancedOrderDetails 直接跑规则引擎和方案生成器。
    """

    def __init__(
        self,
        warehouses: list[WarehouseInfo],
        session_cache: Optional[SessionMemoryService] = None,
    ):
        """初始化高级混合服务。

        Args:
            warehouses: 仓库列表（用于履约方案生成）
            session_cache: 会话缓存（用于短期记忆）
        """
        # 仓库列表用于生成候选履约方案。
        self.warehouses = warehouses
        # 预留会话缓存字段，方便未来把高级订单处理也接入短期上下文。
        self.session_cache = session_cache
        # 多方案生成器负责快速/平衡/经济/库存优化等候选方案。
        self.options_generator = FulfillmentOptionsGenerator(warehouses)

    def process(
        self,
        order: AdvancedOrderDetails,
        inventory_snapshot: Optional[dict] = None,
        thread_id: Optional[str] = None,
    ) -> AdvancedHybridResult:
        """处理高级订单，生成多个履约方案。

        流程：
          1. 应用业务规则（根据客户分级和订单特征）
          2. 生成多个履约方案（快速、平衡、经济等）
          3. 对方案进行智能评分（成本、时间、质量）
          4. 推荐最优方案

        Args:
            order: 高级订单详情
            inventory_snapshot: 库存快照（SKU -> [warehouse_inventories]）
            thread_id: 会话 ID（用于短期记忆）

        Returns:
            AdvancedHybridResult，包含规则应用结果和履约方案
        """
        start_time = time.time()

        if inventory_snapshot is None:
            # 没传库存快照时使用空 dict，保证流程可运行。
            inventory_snapshot = {}

        # 第一步：应用业务规则
        rule_engine = BusinessRuleEngine(
            customer_segment=self._create_customer_segment(order)
        )
        # execute 返回原订单、副本处理后的订单、规则审计记录。
        rule_result = rule_engine.execute(order)
        order_after_rules = rule_result.final_order
        # rules_applied 只保留实际生效的规则名，适合概览展示。
        rules_applied = [r.rule_name for r in rule_result.applied_rules if r.applied]
        # RuleImpactAnalysis 把规则变化整理成前端更容易展示的 dict。
        rule_impact = RuleImpactAnalysis.analyze(
            order=rule_result.final_order,
            applied_rules=rule_result.applied_rules,
            original_order=rule_result.original_order,
            conflict_notes=rule_result.conflict_notes,
        )
        rule_details = rule_impact["applied_rules"]

        # 规则处理后计算一次可解释动态定价，用于展示业务因子如何影响成本。
        # 这里不一定真的改订单价格，更多是展示“如果按规则定价会怎样”。
        pricing_adjustment = DynamicPricingEngine.calculate_adjustment(
            base_price=order_after_rules.total_amount,
            order=order_after_rules,
            inventory=inventory_snapshot,
        )

        # 第二步：生成多个履约方案
        # 方案生成使用“规则处理后的订单”，因为优先级/物流/质量要求可能已经变了。
        fulfillment_options = self.options_generator.generate_options(
            order_after_rules, inventory_snapshot
        )

        # 第三步：推荐最优方案
        recommended_option = None
        if fulfillment_options:
            # FulfillmentOptionsGenerator 已经按 overall_score 降序排序。
            recommended_option = fulfillment_options[0]

        execution_time_ms = (time.time() - start_time) * 1000

        # 汇总结构化处理结果。
        result = AdvancedHybridResult(
            order_id=order.order_id,
            customer_id=order.customer_id,
            customer_level=order.customer_level.value,
            question="",  # 高级服务不处理问题，处理订单
            rules_applied=rules_applied,
            rule_details=rule_details,
            rule_conflicts=rule_result.conflict_notes,
            original_priority=rule_result.original_order.priority,
            final_priority=order_after_rules.priority,
            risk_level=order_after_rules.risk_level,
            risk_flags=order_after_rules.risk_flags,
            pricing_adjustment=pricing_adjustment,
            fulfillment_options=fulfillment_options,
            recommended_option_id=(
                recommended_option.option_id if recommended_option else None
            ),
            execution_time_ms=execution_time_ms,
            thread_id=thread_id,
        )

        return result

    def _create_customer_segment(self, order: AdvancedOrderDetails):
        """从订单信息创建客户分级规则。

        CustomerSegment 可以理解为“客户等级对应的默认服务承诺”。
        规则引擎会用它来决定优先级、物流方式、退货权益、是否允许替代等。
        """
        from app.schemas.advanced_order import (
            CustomerSegment,
            ShippingMethod,
            QualityLevel,
        )

        # 根据客户等级设置默认规则。
        customer_level = order.customer_level

        # VIP：时效、质量、权益都最高，但限制也最强。
        if customer_level.value == "vip":
            return CustomerSegment(
                customer_id=order.customer_id,
                level=customer_level,
                default_priority=8,
                target_delivery_days=1,
                max_acceptable_delay=0,
                preferred_shipping_method=ShippingMethod.OVERNIGHT,
                max_shipping_cost_ratio=0.1,
                required_quality_level=QualityLevel.SUPER,
                inspection_required=True,
                consolidation_allowed=False,
                max_wait_for_consolidation_hours=0,
                substitution_allowed=False,
                substitution_approval_required=True,
                return_period_days=30,
                free_return=True,
                discount_ratio=0.95,
            )
        elif customer_level.value == "first_tier":
            # 一级客户：高质量和较快物流，但比 VIP 稍宽松。
            return CustomerSegment(
                customer_id=order.customer_id,
                level=customer_level,
                default_priority=7,
                target_delivery_days=2,
                max_acceptable_delay=1,
                preferred_shipping_method=ShippingMethod.EXPRESS,
                max_shipping_cost_ratio=0.12,
                required_quality_level=QualityLevel.PREMIUM,
                inspection_required=False,
                consolidation_allowed=False,
                max_wait_for_consolidation_hours=12,
                substitution_allowed=True,
                substitution_approval_required=True,
                return_period_days=14,
                free_return=True,
                discount_ratio=0.98,
            )
        elif customer_level.value == "second_tier":
            # 二级客户：标准服务，允许一定合并和替代。
            return CustomerSegment(
                customer_id=order.customer_id,
                level=customer_level,
                default_priority=5,
                target_delivery_days=5,
                max_acceptable_delay=2,
                preferred_shipping_method=ShippingMethod.STANDARD,
                max_shipping_cost_ratio=0.15,
                required_quality_level=QualityLevel.STANDARD,
                inspection_required=False,
                consolidation_allowed=True,
                max_wait_for_consolidation_hours=48,
                substitution_allowed=True,
                substitution_approval_required=False,
                return_period_days=7,
                free_return=False,
                discount_ratio=1.0,
            )
        else:  # NEW_CUSTOMER
            # 新客户：成本控制更强，允许更长等待和经济物流。
            return CustomerSegment(
                customer_id=order.customer_id,
                level=customer_level,
                default_priority=3,
                target_delivery_days=7,
                max_acceptable_delay=3,
                preferred_shipping_method=ShippingMethod.ECONOMY,
                max_shipping_cost_ratio=0.15,
                required_quality_level=QualityLevel.STANDARD,
                inspection_required=False,
                consolidation_allowed=True,
                max_wait_for_consolidation_hours=72,
                substitution_allowed=True,
                substitution_approval_required=False,
                return_period_days=7,
                free_return=False,
                discount_ratio=1.05,
            )
