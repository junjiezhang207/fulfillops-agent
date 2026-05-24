"""业务规则引擎（学习版注释）— 高确定性业务约束的前置处理。

它不是替代 Agent 的推理层，而是把 VIP、急单、交期、预算、质量等确定性规则
先落到订单约束上，再交给后续履约方案生成器评分。

设计重点：
  - 不原地修改传入订单，避免调用方拿不到原始状态。
  - 每条规则记录 before/after，方便审计和面试解释。
  - 当多个规则覆盖同一字段时，记录冲突说明，而不是悄悄覆盖。
"""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List

from app.schemas.advanced_order import (
    AdvancedOrderDetails,
    CustomerSegment,
    CustomerLevel,
    OrderType,
    ShippingMethod,
    QualityLevel,
)


@dataclass
class RuleApplication:
    """单条规则应用结果。

    这相当于一条规则审计日志：谁改了什么，改之前/之后是什么。
    """

    # 规则唯一 ID，适合日志和测试断言。
    rule_id: str
    # 人类可读规则名。
    rule_name: str
    # 是否真的造成字段变化。
    applied: bool
    # 本规则希望写入的字段和值。
    changes: dict
    # 规则优先级，数字越小越优先。
    priority: int = 100
    # 修改前字段快照。
    before: dict = field(default_factory=dict)
    # 修改后字段快照。
    after: dict = field(default_factory=dict)
    # 字段冲突说明。
    conflict_note: str = ""


@dataclass
class RuleExecutionResult:
    """一次规则执行报告。"""

    # 原始订单副本。
    original_order: AdvancedOrderDetails
    # 应用规则后的订单副本。
    final_order: AdvancedOrderDetails
    # 所有尝试应用的规则记录。
    applied_rules: list[RuleApplication]
    # 字段覆盖冲突说明。
    conflict_notes: list[str]


class BusinessRuleEngine:
    """业务规则引擎。

    规则优先级数字越小越先执行。当前实现仍是代码规则，但执行结果已经具备
    审计字段；后续可以把规则条件和 changes 移到数据库或配置中心。
    """

    def __init__(self, customer_segment: CustomerSegment):
        # 客户分层规则，是很多业务规则的基础参数。
        self.customer_segment = customer_segment
        # 本次执行应用过的规则。
        self.applied_rules: List[RuleApplication] = []
        # 多规则覆盖同一字段时记录冲突。
        self.conflict_notes: list[str] = []
        # 记录字段最后由哪个规则写入，用于冲突判断。
        self._field_owner: dict[str, str] = {}
        # 记录字段写入规则的优先级。
        self._field_priority: dict[str, int] = {}

    def apply_rules(self, order: AdvancedOrderDetails) -> AdvancedOrderDetails:
        """兼容旧调用：返回应用规则后的订单。"""
        return self.execute(order).final_order

    def execute(self, order: AdvancedOrderDetails) -> RuleExecutionResult:
        """应用所有规则，并返回完整执行报告。

        传入的 order 不会被原地修改；所有规则都作用在副本上。
        """
        # 保存原始副本用于审计对比。
        original_order = deepcopy(order)
        # 所有规则都作用在 working_order 上，保护调用方传入对象。
        working_order = deepcopy(order)
        # 每次 execute 前清空上一次执行状态。
        self.applied_rules = []
        self.conflict_notes = []
        self._field_owner = {}
        self._field_priority = {}

        # 按业务维度应用规则。内部每条规则还会带 priority 处理字段冲突。
        self._apply_customer_level_rules(working_order)
        self._apply_order_type_rules(working_order)
        self._apply_time_constraint_rules(working_order)
        self._apply_cost_constraint_rules(working_order)
        self._apply_quality_rules(working_order)
        self._apply_risk_rules(working_order)

        return RuleExecutionResult(
            original_order=original_order,
            final_order=working_order,
            applied_rules=list(self.applied_rules),
            conflict_notes=list(self.conflict_notes),
        )

    def _apply_customer_level_rules(self, order: AdvancedOrderDetails):
        """根据客户等级应用规则。

        客户等级是最基础的服务承诺来源，例如 VIP 更高优先级、更高质检要求。
        """

        if order.customer_level == CustomerLevel.VIP:
            # VIP 客户规则
            self._apply_rule(
                "vip_priority",
                "VIP 客户优先级提升",
                {"priority": min(max(order.priority, self.customer_segment.default_priority) + 3, 10)},
                order,
                priority=10,
            )

            self._apply_rule(
                "vip_quality",
                "VIP 要求超级质量",
                {"quality_requirement": self.customer_segment.required_quality_level},
                order,
                priority=11,
            )

            self._apply_rule(
                "vip_no_consolidation",
                "VIP 订单不允许合并",
                {
                    "consolidation_allowed": self.customer_segment.consolidation_allowed,
                    "allowed_delay_days": self.customer_segment.max_acceptable_delay,
                },
                order,
                priority=12,
            )

            self._apply_rule(
                "vip_fast_shipping",
                "VIP 默认快速物流",
                {"shipping_method": self.customer_segment.preferred_shipping_method},
                order,
                priority=13,
            )

        elif order.customer_level == CustomerLevel.FIRST_TIER:
            # 一级客户规则
            self._apply_rule(
                "first_tier_priority",
                "一级客户优先级不低于客户分层默认值",
                {"priority": max(order.priority, self.customer_segment.default_priority)},
                order,
                priority=20,
            )

            self._apply_rule(
                "first_tier_quality",
                "一级客户要求高级质量",
                {"quality_requirement": self.customer_segment.required_quality_level},
                order,
                priority=21,
            )

        elif order.customer_level == CustomerLevel.NEW_CUSTOMER:
            # 新客户规则
            self._apply_rule(
                "new_customer_priority",
                "新客户优先级降低",
                {"priority": max(order.priority - 2, 1)},
                order,
                priority=30,
            )

            self._apply_rule(
                "new_customer_allow_consolidation",
                "新客户允许合并以降低成本",
                {
                    "consolidation_allowed": self.customer_segment.consolidation_allowed,
                    "allowed_delay_days": self.customer_segment.max_acceptable_delay,
                },
                order,
                priority=31,
            )

            self._apply_rule(
                "new_customer_economy_shipping",
                "新客户默认经济物流",
                {"shipping_method": self.customer_segment.preferred_shipping_method},
                order,
                priority=32,
            )

    def _apply_order_type_rules(self, order: AdvancedOrderDetails):
        """根据订单类型应用规则。

        急单和可延期订单的目标完全不同：
        - 急单追求时效。
        - 可延期订单可以牺牲时间换成本。
        """

        if order.order_type == OrderType.URGENT:
            # 急单规则
            self._apply_rule(
                "urgent_priority",
                "急单提升优先级",
                {"priority": min(order.priority + 5, 10)},
                order,
                priority=40,
            )

            self._apply_rule(
                "urgent_fast_shipping",
                "急单强制快速物流",
                {"shipping_method": ShippingMethod.OVERNIGHT},
                order,
                priority=41,
            )

            self._apply_rule(
                "urgent_no_consolidation",
                "急单不允许合并",
                {"consolidation_allowed": False},
                order,
                priority=42,
            )

            # 如果订单金额大，添加风险标签
            if order.total_amount > 50000:
                self._add_risk_flag(order, "urgent_high_value", "urgent_high_value", "急单高金额风险", priority=43)

        elif order.order_type == OrderType.FLEXIBLE:
            # 可延期订单 - 可以优化成本
            self._apply_rule(
                "flexible_consolidation",
                "可延期订单优先合并",
                {"consolidation_allowed": True, "consolidation_window_hours": 72},
                order,
                priority=50,
            )

            self._apply_rule(
                "flexible_economy_shipping",
                "可延期订单优先经济物流",
                {"shipping_method": ShippingMethod.ECONOMY},
                order,
                priority=51,
            )

    def _apply_time_constraint_rules(self, order: AdvancedOrderDetails):
        """根据时间约束应用规则。

        越接近交付截止时间，越应该提升优先级并限制合并等待。
        """

        if order.days_to_deadline < 1:
            # 紧急交期（< 1 天）
            self._apply_rule(
                "critical_deadline",
                "关键交期：必须快速发货",
                {
                    "priority": min(order.priority + 5, 10),
                    "shipping_method": ShippingMethod.OVERNIGHT,
                    "consolidation_allowed": False,
                },
                order,
                priority=60,
            )
            self._add_risk_flag(order, "critical_deadline", "critical_deadline_risk", "关键交期风险", priority=61)

        elif order.days_to_deadline < 3:
            # 紧张交期（< 3 天）
            self._apply_rule(
                "tight_deadline",
                "紧张交期：推荐快速方案",
                {"priority": min(order.priority + 3, 10)},
                order,
                priority=62,
            )

    def _apply_cost_constraint_rules(self, order: AdvancedOrderDetails):
        """根据成本约束应用规则。

        成本敏感订单会倾向经济物流；预算接近上限时要打风险标记。
        """

        if order.cost_sensitive and order.budget:
            # 成本敏感的订单 - 优化成本
            self._apply_rule(
                "cost_optimization",
                "成本优化：推荐经济方案",
                {"shipping_method": ShippingMethod.ECONOMY},
                order,
                priority=70,
            )

            # 如果订单已接近预算，标记为风险
            if order.budget_ratio > 0.9:
                self._add_risk_flag(order, "approaching_budget", "approaching_budget", "预算接近上限", priority=71)

    def _apply_quality_rules(self, order: AdvancedOrderDetails):
        """根据质量需求应用规则。

        高质量/易碎品要求通常会触发质检、特殊包装或人工关注。
        """

        if order.quality_requirement == QualityLevel.SUPER:
            # 超级质量要求
            self._apply_rule(
                "super_quality",
                "超级质量：检查和特殊处理",
                {"require_inspection": True},
                order,
                priority=80,
            )

        if "易碎" in order.special_handling or "fragile" in order.special_handling:
            self._add_risk_flag(
                order,
                "fragile_handling_required",
                "fragile_handling_required",
                "易碎品特殊处理",
                priority=81,
            )

    def _apply_risk_rules(self, order: AdvancedOrderDetails):
        """根据风险标签应用规则。

        这里处理高价值、缺货等横跨客户/时间/成本之外的风险。
        """

        # 高价值订单
        if order.total_amount > 100000:
            self._add_risk_flag(order, "high_value", "high_value_risk", "高价值订单风险", priority=90)
            self._apply_rule(
                "high_value_approval",
                "高价值订单需要批准",
                {"require_inspection": True},
                order,
                priority=91,
            )

        # 缺货 SKU
        for line_item in order.line_items:
            if line_item.special_notes and "缺货" in line_item.special_notes:
                self._add_risk_flag(
                    order,
                    f"out_of_stock_{line_item.sku}",
                    f"out_of_stock_{line_item.sku}",
                    f"SKU {line_item.sku} 缺货风险",
                    priority=92,
                )

    def _apply_rule(
        self,
        rule_id: str,
        rule_name: str,
        changes: dict,
        order: AdvancedOrderDetails,
        *,
        priority: int = 100,
    ):
        """应用单个规则，并记录 before/after 和字段覆盖冲突。"""
        before: dict[str, Any] = {}
        after: dict[str, Any] = {}
        conflict_messages: list[str] = []

        # 更新订单。changes 里每个 key 都表示希望修改的订单字段。
        for key, value in changes.items():
            if hasattr(order, key):
                before_value = getattr(order, key)
                before[key] = self._jsonable(before_value)
                if key in self._field_owner and before_value != value:
                    # 如果同一字段之前已经被其他规则改过，就判断优先级。
                    previous_rule = self._field_owner[key]
                    previous_priority = self._field_priority[key]
                    if priority > previous_priority:
                        # 当前规则优先级更低，不能覆盖之前的值。
                        conflict_messages.append(
                            f"{rule_id} 想覆盖字段 {key}，但优先级低于 {previous_rule}，已保留原值"
                        )
                        after[key] = self._jsonable(before_value)
                        continue
                    # 当前规则优先级更高或相同，允许覆盖，但记录冲突。
                    conflict_messages.append(
                        f"{rule_id} 覆盖字段 {key}，前一规则：{previous_rule}"
                    )
                setattr(order, key, value)
                after[key] = self._jsonable(getattr(order, key))
                # 记录字段由当前规则接管。
                self._field_owner[key] = rule_id
                self._field_priority[key] = priority

        if not after:
            # changes 里的字段在订单对象上不存在，或者都被跳过。
            return

        conflict_note = "；".join(conflict_messages)
        if conflict_note:
            self.conflict_notes.append(conflict_note)

        # applied 表示字段值是否真的发生变化。
        applied = any(before.get(key) != after.get(key) for key in after)

        # 记录规则应用
        self.applied_rules.append(
            RuleApplication(
                rule_id=rule_id,
                rule_name=rule_name,
                applied=applied,
                changes={key: self._jsonable(value) for key, value in changes.items()},
                priority=priority,
                before=before,
                after=after,
                conflict_note=conflict_note,
            )
        )

    def _add_risk_flag(
        self,
        order: AdvancedOrderDetails,
        flag: str,
        rule_id: str,
        rule_name: str,
        *,
        priority: int,
    ) -> None:
        """添加风险标签并记录审计信息，避免重复标签。"""
        if flag in order.risk_flags:
            return
        self._apply_rule(
            rule_id,
            rule_name,
            {"risk_flags": [*order.risk_flags, flag]},
            order,
            priority=priority,
        )

    @staticmethod
    def _jsonable(value: Any) -> Any:
        """把枚举/列表/dict 转成容易 JSON 序列化的值。"""
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, list):
            return [BusinessRuleEngine._jsonable(item) for item in value]
        if isinstance(value, dict):
            return {key: BusinessRuleEngine._jsonable(item) for key, item in value.items()}
        return value


class DynamicPricingEngine:
    """动态定价引擎。

    当前用于展示规则因子如何影响价格，不直接写回订单。
    """

    # 定价因子及其影响范围
    PRICING_FACTORS = {
        "customer_level": {
            "vip": 0.95,  # VIP 享受折扣
            "first_tier": 0.98,
            "second_tier": 1.0,
            "new_customer": 1.05,  # 新客户可能加价
        },
        "order_size": {
            "small": 1.1,  # < $1000
            "medium": 1.0,  # $1000-$10000
            "large": 0.95,  # $10000-$50000
            "xlarge": 0.90,  # > $50000
        },
        "urgency": {
            "standard": 1.0,  # 标准
            "urgent": 1.3,  # 急单加价 30%
            "critical": 1.5,  # 关键加价 50%
        },
        "inventory_level": {
            "abundant": 0.95,  # 库存充足
            "normal": 1.0,
            "low": 1.1,  # 库存低加价
            "critical": 1.2,  # 关键缺货加价
        },
        "quality_requirement": {
            "standard": 1.0,
            "premium": 1.1,
            "super": 1.2,
        },
    }

    @staticmethod
    def calculate_adjustment(
        base_price: float,
        order: AdvancedOrderDetails,
        inventory: dict,
    ) -> dict:
        """计算动态价格，并返回可解释的因子明细。"""
        factors: list[dict[str, float | str]] = []
        multiplier = 1.0

        def apply_factor(name: str, value: float) -> None:
            # 每个因子都乘到总 multiplier 上，并记录明细用于解释。
            nonlocal multiplier
            multiplier *= value
            factors.append({"name": name, "multiplier": round(value, 4)})

        # 客户等级因子：VIP 折扣，新客户可能加价。
        customer_level = order.customer_level.value.lower()
        if customer_level in DynamicPricingEngine.PRICING_FACTORS["customer_level"]:
            apply_factor(
                f"customer_level:{customer_level}",
                DynamicPricingEngine.PRICING_FACTORS["customer_level"][customer_level],
            )

        # 订单金额因子：订单越大，通常单价或服务费率越低。
        if order.total_amount < 1000:
            apply_factor("order_size:small", 1.1)
        elif 1000 <= order.total_amount < 10000:
            apply_factor("order_size:medium", 1.0)
        elif 10000 <= order.total_amount < 50000:
            apply_factor("order_size:large", 0.95)
        else:
            apply_factor("order_size:xlarge", 0.90)

        # 紧急程度因子：越急成本越高。
        if order.is_urgent and order.days_to_deadline < 1:
            apply_factor("urgency:critical", 1.5)
        elif order.order_type == OrderType.URGENT:
            apply_factor("urgency:urgent", 1.3)

        # 质量要求因子：更高质检/包装要求会增加成本。
        if order.quality_requirement == QualityLevel.PREMIUM:
            apply_factor("quality:premium", 1.1)
        elif order.quality_requirement == QualityLevel.SUPER:
            apply_factor("quality:super", 1.2)

        # 库存紧张程度因子：低库存/关键缺货会提高履约成本。
        inventory_level = str(inventory.get("inventory_level", "")).lower() if inventory else ""
        if inventory_level in DynamicPricingEngine.PRICING_FACTORS["inventory_level"]:
            apply_factor(
                f"inventory:{inventory_level}",
                DynamicPricingEngine.PRICING_FACTORS["inventory_level"][inventory_level],
            )

        # 最终价格 = 基础价格 * 所有因子乘积。
        final_price = base_price * multiplier
        return {
            "base_price": round(base_price, 2),
            "final_price": round(final_price, 2),
            "multiplier": round(multiplier, 4),
            "factors": factors,
        }

    @staticmethod
    def calculate_price(
        base_price: float,
        order: AdvancedOrderDetails,
        inventory: dict,
    ) -> float:
        """计算最终价格"""
        return DynamicPricingEngine.calculate_adjustment(base_price, order, inventory)["final_price"]


class RuleImpactAnalysis:
    """规则影响分析。

    把规则执行结果整理成前端/面试展示更友好的 dict。
    """

    @staticmethod
    def analyze(
        order: AdvancedOrderDetails,
        applied_rules: List[RuleApplication],
        original_order: AdvancedOrderDetails | None = None,
        conflict_notes: list[str] | None = None,
    ) -> dict:
        """分析规则对订单的影响。"""

        # 这里不再重新计算规则，只做格式整理。
        return {
            "order_id": order.order_id,
            "original_priority": original_order.priority if original_order else None,
            "final_priority": order.priority,
            "applied_rules": [
                {
                    "rule_id": rule.rule_id,
                    "rule_name": rule.rule_name,
                    "priority": rule.priority,
                    "applied": rule.applied,
                    "changes": rule.changes,
                    "before": rule.before,
                    "after": rule.after,
                    "conflict_note": rule.conflict_note,
                }
                for rule in applied_rules
            ],
            "conflict_notes": conflict_notes or [],
            "risk_level": order.risk_level,
            "risk_flags": order.risk_flags,
            "quality_requirement": order.quality_requirement.value,
            "shipping_method": order.shipping_method.value,
            "consolidation_allowed": order.consolidation_allowed,
        }
