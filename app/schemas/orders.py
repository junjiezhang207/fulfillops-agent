"""Order Pydantic schemas.

Learning notes:
- Defines raw order data, analysis requests and analysis results.
- Repositories return OrderRecord; services transform it into analysis output.
"""

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class OrderItem(BaseModel):
    """订单商品明细模型。"""

    sku_id: str = Field(..., description="SKU 编码。")
    product_name: str = Field(..., description="商品名称。")
    quantity: int = Field(..., ge=1, description="购买数量。")
    unit_price: float = Field(..., ge=0, description="商品单价。")
    allocated_quantity: int = Field(default=0, ge=0, description="已分配数量。")
    shipped_quantity: int = Field(default=0, ge=0, description="已发货数量。")
    sku_status: str = Field(default="pending", description="商品履约状态。")
    split_allowed: bool = Field(default=True, description="是否允许拆单。")
    special_storage: str | None = Field(default=None, description="冷链、危险品等特殊仓储要求。")
    sku_type: str | None = Field(default=None, description="商品类型，用于匹配候选仓支持能力。")


class OrderRecord(BaseModel):
    """订单原始数据模型。

    说明：
    - 这个模型代表仓储层返回的订单原始结构。
    - 后续如果把模拟数据替换成 PostgreSQL 或 ERP 接口，
      也应尽量返回与这个模型相近的结构，降低上层改动成本。
    """

    order_id: str = Field(..., description="订单编号。")
    platform: str = Field(..., description="订单来源平台。")
    order_time: datetime | None = Field(default=None, description="下单时间。")
    order_status: str = Field(..., description="订单状态。")
    region: str = Field(..., description="用户收货区域。")
    priority: str = Field(..., description="订单优先级。")
    items: list[OrderItem] = Field(..., description="订单商品明细列表。")
    created_at: datetime | None = Field(default=None, description="下单时间，兼容 OMS 字段名。")
    promise_delivery_time: datetime | None = Field(default=None, description="承诺送达时间。")
    current_warehouse_id: str | None = Field(default=None, description="当前履约仓。")
    shipping_region: str | None = Field(default=None, description="收货区域，省/市/区。")
    fulfillment_type: str = Field(default="普通", description="履约类型：普通/预售/同城等。")
    already_split: bool = Field(default=False, description="是否已经分仓。")
    inventory_reserved: bool = Field(default=False, description="是否已经占库存。")
    package_created: bool = Field(default=False, description="是否已经生成包裹。")
    waybill_created: bool = Field(default=False, description="是否已经生成运单。")
    outbound_completed: bool = Field(default=False, description="是否已经出库。")
    active_fulfillment_tasks: list[str] = Field(
        default_factory=list,
        description="正在执行的调拨、拆单或其他履约任务。",
    )

    @model_validator(mode="after")
    def normalize_operational_fields(self) -> "OrderRecord":
        if self.order_time is None and self.created_at is not None:
            self.order_time = self.created_at
        if self.created_at is None and self.order_time is not None:
            self.created_at = self.order_time
        if not self.shipping_region:
            self.shipping_region = self.region
        return self


class OrderAnalysisRequest(BaseModel):
    """订单分析请求模型。"""

    order_id: str = Field(..., description="待分析订单编号。")


class OrderAnalysisResult(BaseModel):
    """订单分析结果模型。

    说明：
    - 这是订单模块输出给上层接口或后续库存模块的结构化结果。
    - 后续进入 LangGraph 状态建模时，这个结果也可以直接作为状态中的一部分。
    """

    order_id: str = Field(..., description="订单编号。")
    platform: str = Field(..., description="订单来源平台。")
    order_status: str = Field(..., description="订单状态。")
    region: str = Field(..., description="用户收货区域。")
    priority: str = Field(..., description="订单优先级。")
    created_at: datetime | None = Field(default=None, description="下单时间。")
    promise_delivery_time: datetime | None = Field(default=None, description="承诺送达时间。")
    current_warehouse_id: str | None = Field(default=None, description="当前履约仓。")
    shipping_region: str | None = Field(default=None, description="收货区域。")
    fulfillment_type: str = Field(default="普通", description="履约类型。")
    fulfillment_status_flags: dict = Field(default_factory=dict, description="已发生履约状态。")
    item_count: int = Field(..., description="商品明细行数。")
    total_quantity: int = Field(..., description="订单总购买件数。")
    sku_list: list[str] = Field(..., description="订单中包含的 SKU 列表。")
    items: list[OrderItem] = Field(..., description="订单商品明细。")
    summary: str = Field(..., description="订单分析摘要。")
