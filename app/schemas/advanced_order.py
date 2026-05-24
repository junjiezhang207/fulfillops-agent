"""高级订单模型 — 支持真实业务复杂性。

包含客户分级、时间约束、特殊需求等真实供应链要素。
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional


class CustomerLevel(Enum):
    """客户分级"""
    VIP = "vip"
    FIRST_TIER = "first_tier"
    SECOND_TIER = "second_tier"
    NEW_CUSTOMER = "new_customer"


class OrderType(Enum):
    """订单类型"""
    STANDARD = "standard"  # 常规
    URGENT = "urgent"  # 急单
    FLEXIBLE = "flexible"  # 可延期
    SAMPLE = "sample"  # 样品


class ShippingMethod(Enum):
    """物流方式"""
    OVERNIGHT = "overnight"  # 次日达
    EXPRESS = "express"  # 快递 3-5 天
    STANDARD = "standard"  # 标准 7-10 天
    ECONOMY = "economy"  # 经济 14-21 天
    BULK = "bulk"  # 整车 21+ 天


class QualityLevel(Enum):
    """质量等级"""
    STANDARD = "standard"  # 标准
    PREMIUM = "premium"  # 高级
    SUPER = "super"  # 超级


@dataclass
class Location:
    """地理位置"""
    province: str
    city: str
    district: str
    latitude: float = 0.0
    longitude: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.province}-{self.city}-{self.district}"


@dataclass
class OrderLineItem:
    """订单行项目"""
    sku: str
    quantity: int
    unit_price: float

    # 行项目特殊需求
    special_notes: Optional[str] = None
    substitution_allowed: bool = True
    max_acceptable_substitution: Optional[str] = None  # 替代 SKU
    batch_size: Optional[int] = None  # 如果订单必须是某个倍数


@dataclass
class AdvancedOrderDetails:
    """高级订单详情 — 包含真实业务复杂性"""

    # ========== 基础信息 ==========
    order_id: str
    customer_id: str
    created_at: datetime

    # ========== 客户信息 ==========
    customer_level: CustomerLevel  # VIP / 一级 / 二级 / 新客
    customer_name: str
    customer_contact: str

    # ========== 订单类型和优先级 ==========
    order_type: OrderType  # 常规 / 急单 / 可延期

    # ========== 时间约束 ==========
    required_delivery_date: datetime  # 必须交期（硬约束）
    preferred_delivery_date: datetime  # 优先交期（软约束）
    destination: Location  # 目的地

    # ========== 订单优先级和延期 ==========
    priority: int = 5  # 1-10，默认 5，越高越重要
    allowed_delay_days: int = 0  # VIP 0, 新客 7

    # 计算属性
    @property
    def days_to_deadline(self) -> int:
        """距离截止日期的天数"""
        return (self.required_delivery_date - datetime.now()).days

    @property
    def is_urgent(self) -> bool:
        """是否紧急（< 2 天）"""
        return self.days_to_deadline < 2

    # ========== 物流要求 ==========
    shipping_method: ShippingMethod = ShippingMethod.STANDARD
    consolidation_allowed: bool = True  # VIP 不允许，新客允许

    # ========== 商品明细 ==========
    line_items: List[OrderLineItem] = field(default_factory=list)

    @property
    def total_amount(self) -> float:
        """订单总金额"""
        return sum(item.quantity * item.unit_price for item in self.line_items)

    @property
    def total_quantity(self) -> int:
        """订单总数量"""
        return sum(item.quantity for item in self.line_items)

    # ========== 成本和预算 ==========
    budget: Optional[float] = None  # 客户预算上限
    cost_sensitive: bool = False  # 是否对成本敏感

    @property
    def budget_ratio(self) -> float:
        """已用预算占比"""
        if self.budget is None:
            return 0.0
        return self.total_amount / self.budget

    # ========== 质量要求 ==========
    quality_requirement: QualityLevel = QualityLevel.STANDARD
    special_handling: List[str] = field(default_factory=list)  # 易碎、防潮、冷链等
    require_inspection: bool = False

    # ========== 关联订单 ==========
    related_order_ids: List[str] = field(default_factory=list)  # 同客户相关订单
    consolidation_window_hours: int = 24  # 等待合并的时间窗口

    # ========== 风险标签 ==========
    risk_flags: List[str] = field(default_factory=list)  # 高价值、敏感品、缺货SKU等

    @property
    def risk_level(self) -> str:
        """风险等级"""
        if not self.risk_flags:
            return "low"
        if len(self.risk_flags) >= 3 or "critical" in self.risk_flags:
            return "critical"
        if len(self.risk_flags) >= 2:
            return "high"
        return "medium"

    # ========== 备注 ==========
    notes: str = ""
    internal_notes: str = ""


@dataclass
class InventoryLevel:
    """库存分层模型 — 真实库存管理"""

    # ========== 基础信息 ==========
    sku: str
    warehouse_id: str

    # ========== 库存分层 ==========
    available_qty: int = 0  # 可用库存（可以立即发货）
    locked_qty: int = 0  # 已锁定（其他订单预留）
    reserved_qty: int = 0  # 预留（确认但未出库）
    defective_qty: int = 0  # 不良品
    expired_qty: int = 0  # 过期品
    damaged_qty: int = 0  # 损坏品

    # ========== 在途库存 ==========
    transit_qty: int = 0  # 在途（采购单）
    expected_arrival: Optional[datetime] = None

    # ========== 库存质量指标 ==========
    quality_score: float = 100.0  # 0-100，基于历史缺陷率
    turnover_rate: float = 1.0  # 周转率（月）
    holding_cost_per_day: float = 1.0  # 日仓储成本
    days_in_storage: int = 0  # 在库天数

    # ========== 库存约束 ==========
    min_stock_level: int = 0  # 安全库存
    max_stock_level: int = 999999  # 最大库存
    batch_size: int = 1  # 订单倍数
    shelf_life_days: Optional[int] = None  # 保质期

    @property
    def days_until_expiry(self) -> Optional[int]:
        """距离过期的天数"""
        if self.shelf_life_days is None:
            return None
        return self.shelf_life_days - self.days_in_storage

    @property
    def is_expiring_soon(self) -> bool:
        """是否即将过期"""
        if self.days_until_expiry is None:
            return False
        return self.days_until_expiry <= 30

    # ========== 总库存计算 ==========
    @property
    def total_stock(self) -> int:
        """总库存"""
        return self.available_qty + self.locked_qty + self.reserved_qty

    @property
    def usable_stock(self) -> int:
        """可用库存（不含预留和锁定）"""
        return self.available_qty

    @property
    def can_fulfill(self, qty: int) -> bool:
        """是否可以满足订单"""
        return self.available_qty >= qty and qty % self.batch_size == 0

    # ========== 价格 ==========
    unit_price: float = 0.0

    @property
    def inventory_value(self) -> float:
        """库存价值"""
        return self.total_stock * self.unit_price


@dataclass
class CustomerSegment:
    """客户分级规则"""

    customer_id: str
    level: CustomerLevel

    # ========== 优先级规则 ==========
    default_priority: int

    # ========== 交期规则 ==========
    target_delivery_days: int  # VIP 1, 新客 7
    max_acceptable_delay: int  # VIP 0, 新客 5

    # ========== 物流规则 ==========
    preferred_shipping_method: ShippingMethod
    max_shipping_cost_ratio: float  # 运费/订单金额上限

    # ========== 质量规则 ==========
    required_quality_level: QualityLevel
    inspection_required: bool

    # ========== 订单合并规则 ==========
    consolidation_allowed: bool  # VIP 否，新客 是
    max_wait_for_consolidation_hours: int

    # ========== 替代和返回规则 ==========
    substitution_allowed: bool
    substitution_approval_required: bool
    return_period_days: int  # VIP 30, 新客 7
    free_return: bool

    # ========== 成本规则 ==========
    discount_ratio: float = 1.0  # 1.0 无折扣，0.9 九折


@dataclass
class WarehouseInfo:
    """仓库信息"""

    warehouse_id: str
    name: str
    location: Location

    # ========== 仓库容量 ==========
    total_capacity: float  # 立方米
    used_capacity: float

    # ========== 成本 ==========
    storage_cost_per_unit_day: float
    picking_cost_per_unit: float
    packing_cost_per_shipment: float
    handling_cost_per_unit: float

    @property
    def available_capacity(self) -> float:
        return self.total_capacity - self.used_capacity

    @property
    def capacity_utilization(self) -> float:
        return self.used_capacity / self.total_capacity

    # ========== 作业能力 ==========
    max_daily_shipments: int = 1000
    processing_time_hours: int = 4  # 订单处理时间
    quality_rating: float = 95.0  # 0-100

    def calculate_handling_cost(self, qty: int) -> float:
        """计算订单处理成本"""
        return (
            qty * self.picking_cost_per_unit +
            self.packing_cost_per_shipment +
            qty * self.handling_cost_per_unit
        )
