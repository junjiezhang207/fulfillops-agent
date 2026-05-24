"""多规则风险评估引擎 — Human-in-the-Loop 的触发决策核心。

解决的问题：
  旧版：只有一条规则（库存充足就中断），所有订单一刀切审批
  新版：可配置的多规则引擎，按风险等级决定是否需要人工、需要谁审批

风险等级体系：
  LOW      — 自动放行，不打断流程，仅记录
  MEDIUM   — 打标记，但默认自动通过（可配置要求确认）
  HIGH     — 必须人工审批才能继续
  CRITICAL — 需要高级别审批，同时触发告警通知

典型触发场景（按业务优先级排序）：
  CRITICAL: 金额 > 50万 的 VIP 急单
  HIGH:     金额 > 10万 / 缺货率 > 50% / VIP 2小时内截止
  MEDIUM:   新客户大单 / 需要拆单 / 冷链商品 / 跨境订单
  LOW:      常规订单（不打断）
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class RiskLevel(Enum):
    """风险等级。值越大风险越高，便于数值比较。"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __lt__(self, other: "RiskLevel") -> bool:
        return self.value < other.value

    def __ge__(self, other: "RiskLevel") -> bool:
        return self.value >= other.value


@dataclass
class RiskSignal:
    """单条风险规则触发产生的信号。"""
    rule_name: str
    level: RiskLevel
    reason: str                          # 给审批员看的说明
    suggested_action: str = ""           # 建议的处理方式
    auto_approve_fallback: str = ""      # 超时时自动执行的动作


@dataclass
class RiskAssessment:
    """对一个订单的完整风险评估结果。"""
    signals: list[RiskSignal] = field(default_factory=list)

    @property
    def max_level(self) -> RiskLevel:
        if not self.signals:
            return RiskLevel.LOW
        return max(s.level for s in self.signals)

    @property
    def needs_human_review(self) -> bool:
        """HIGH 及以上必须人工审批。"""
        return self.max_level >= RiskLevel.HIGH

    @property
    def interrupt_type(self) -> str:
        """根据最高风险信号确定中断类型（用于 InterruptEvent.type）。"""
        if not self.needs_human_review:
            return "auto_pass"
        level = self.max_level
        if level == RiskLevel.CRITICAL:
            return "critical_approval"
        return "standard_approval"

    @property
    def summary(self) -> str:
        if not self.signals:
            return "无风险信号，自动放行"
        parts = [f"[{s.level.name}] {s.reason}" for s in self.signals]
        return "；".join(parts)

    @property
    def signal_names(self) -> list[str]:
        return [s.rule_name for s in self.signals]

    def timeout_seconds(self) -> int:
        """根据风险等级返回超时秒数（超时后自动降级处理）。"""
        if self.max_level == RiskLevel.CRITICAL:
            return 300    # 5 分钟，紧急告警
        if self.max_level == RiskLevel.HIGH:
            return 1800   # 30 分钟
        return 86400      # 24 小时（MEDIUM 级别）


# ============================================================================
# 规则定义
# ============================================================================

@dataclass
class RiskRule:
    """单条风险规则。

    这里把规则拆成数据对象，而不是在 evaluate 里写一长串 if/else。
    好处是：
      - 新增规则只需要追加 RiskRule，不破坏评估主流程。
      - 每条规则都带 level / reason / suggested_action，审批台可以直接展示。
      - 单元测试可以单独构造规则列表，覆盖边界条件更容易。
    """
    name: str
    level: RiskLevel
    reason: str
    check: Callable[[dict], bool]        # 接受 context dict，返回 bool
    suggested_action: str = ""
    auto_approve_fallback: str = "rejected"  # 超时时的默认动作


# 内置规则集（按风险等级从高到低排列）
_DEFAULT_RULES: list[RiskRule] = [

    # ── CRITICAL ──────────────────────────────────────────────────────────
    RiskRule(
        name="vip_critical_urgent",
        level=RiskLevel.CRITICAL,
        reason="VIP 客户 + 金额超 50 万 + 2 小时内截止",
        check=lambda c: (
            c.get("customer_level") == "vip"
            and c.get("order_value", 0) > 500_000
            and c.get("hours_to_deadline", 999) < 2
        ),
        suggested_action="立即联系 VIP 客户经理，优先协调库存",
        auto_approve_fallback="escalate",
    ),

    # ── HIGH ──────────────────────────────────────────────────────────────
    RiskRule(
        name="high_value_order",
        level=RiskLevel.HIGH,
        reason="订单金额超 10 万元",
        check=lambda c: c.get("order_value", 0) > 100_000,
        suggested_action="财务复核后确认发货",
        auto_approve_fallback="rejected",
    ),
    RiskRule(
        name="vip_deadline_urgent",
        level=RiskLevel.HIGH,
        reason="VIP 客户且距截止不足 4 小时",
        check=lambda c: (
            c.get("customer_level") == "vip"
            and c.get("hours_to_deadline", 999) < 4
        ),
        suggested_action="优先处理，确认是否走加急通道",
        auto_approve_fallback="approved",  # 超时默认批准（不能让 VIP 等）
    ),
    RiskRule(
        name="severe_stockout",
        level=RiskLevel.HIGH,
        reason="缺货率超过 50%",
        check=lambda c: c.get("stock_gap_ratio", 0) > 0.5,
        suggested_action="告知客户预计延迟，确认是否接受替代品",
        auto_approve_fallback="rejected",
    ),
    RiskRule(
        name="new_customer_large_order",
        level=RiskLevel.HIGH,
        reason="新客户订单金额超 5 万（信用风险）",
        check=lambda c: (
            c.get("customer_level") == "new_customer"
            and c.get("order_value", 0) > 50_000
        ),
        suggested_action="核实客户资质，要求预付款",
        auto_approve_fallback="rejected",
    ),

    # ── MEDIUM ────────────────────────────────────────────────────────────
    RiskRule(
        name="split_order_required",
        level=RiskLevel.MEDIUM,
        reason="需要拆单发货，可能影响时效",
        check=lambda c: c.get("split_order_required", False),
        suggested_action="确认客户接受分批发货",
        auto_approve_fallback="approved",
    ),
    RiskRule(
        name="cold_chain_product",
        level=RiskLevel.MEDIUM,
        reason="含冷链商品，需确认配送条件",
        check=lambda c: "cold_chain" in c.get("sku_types", []),
        suggested_action="确认目的地冷链配送覆盖",
        auto_approve_fallback="approved",
    ),
    RiskRule(
        name="cross_region_fulfillment",
        level=RiskLevel.MEDIUM,
        reason="跨大区调配，运费显著增加",
        check=lambda c: c.get("cross_region", False),
        suggested_action="确认客户承担额外运费",
        auto_approve_fallback="approved",
    ),
]


# ============================================================================
# 评估器
# ============================================================================

class RiskEvaluator:
    """多规则风险评估引擎。

    使用方式：
        evaluator = RiskEvaluator()
        context = {
            "order_value": 150_000,
            "customer_level": "vip",
            "hours_to_deadline": 3,
            "stock_gap_ratio": 0.1,
            "split_order_required": False,
            "sku_types": [],
            "cross_region": False,
        }
        assessment = evaluator.evaluate(context)
        if assessment.needs_human_review:
            # 触发 interrupt()
            ...

    设计取舍：
      规则输入统一使用 context dict，而不是直接依赖 OrderAnalysisResult /
      InventoryAnalysisResult。这样风险引擎不认识 LangGraph，也不认识具体 service，
      只是一个可复用的纯规则模块。
    """

    def __init__(self, rules: list[RiskRule] | None = None):
        self._rules = rules if rules is not None else _DEFAULT_RULES

    def evaluate(self, context: dict) -> RiskAssessment:
        """对上下文 dict 运行所有规则，返回完整评估结果。

        单条规则异常会被吞掉，这是故意的“局部失败不拖垮整体”策略。
        风险评估宁可少一个信号，也不应该因为一条可配置规则写错而让整个
        workflow 无法给出履约结论。
        """
        signals: list[RiskSignal] = []
        for rule in self._rules:
            try:
                if rule.check(context):
                    signals.append(RiskSignal(
                        rule_name=rule.name,
                        level=rule.level,
                        reason=rule.reason,
                        suggested_action=rule.suggested_action,
                        auto_approve_fallback=rule.auto_approve_fallback,
                    ))
            except Exception:
                pass  # 单条规则异常不影响整体评估
        return RiskAssessment(signals=signals)

    def build_interrupt_context(
        self,
        order_id: str,
        assessment: RiskAssessment,
        inventory_summary: str,
        order_summary: str,
    ) -> dict:
        """构造传给 interrupt() 的 context dict。"""
        signals_detail = [
            {
                "rule": s.rule_name,
                "level": s.level.name,
                "reason": s.reason,
                "suggested_action": s.suggested_action,
            }
            for s in assessment.signals
        ]
        # 整合所有触发规则的建议动作（去重）
        suggested_actions = list(dict.fromkeys(
            s.suggested_action for s in assessment.signals if s.suggested_action
        ))
        return {
            "order_id": order_id,
            "risk_level": assessment.max_level.name,
            "risk_signals": signals_detail,
            "suggested_actions": suggested_actions,
            "inventory_summary": inventory_summary,
            "order_summary": order_summary,
            "timeout_seconds": assessment.timeout_seconds(),
        }
