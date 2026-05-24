"""Order Pydantic schemas.

Learning notes:
- Defines raw order data, analysis requests and analysis results.
- Repositories return OrderRecord; services transform it into analysis output.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class OrderItem(BaseModel):
    """订单商品明细模型。"""

    sku_id: str = Field(..., description="SKU 编码。")
    product_name: str = Field(..., description="商品名称。")
    quantity: int = Field(..., ge=1, description="购买数量。")
    unit_price: float = Field(..., ge=0, description="商品单价。")


class OrderRecord(BaseModel):
    """订单原始数据模型。

    说明：
    - 这个模型代表仓储层返回的订单原始结构。
    - 后续如果把模拟数据替换成 MySQL 或 ERP 接口，
      也应尽量返回与这个模型相近的结构，降低上层改动成本。
    """

    order_id: str = Field(..., description="订单编号。")
    platform: str = Field(..., description="订单来源平台。")
    order_time: datetime = Field(..., description="下单时间。")
    order_status: str = Field(..., description="订单状态。")
    region: str = Field(..., description="用户收货区域。")
    priority: str = Field(..., description="订单优先级。")
    items: list[OrderItem] = Field(..., description="订单商品明细列表。")


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
    item_count: int = Field(..., description="商品明细行数。")
    total_quantity: int = Field(..., description="订单总购买件数。")
    sku_list: list[str] = Field(..., description="订单中包含的 SKU 列表。")
    items: list[OrderItem] = Field(..., description="订单商品明细。")
    summary: str = Field(..., description="订单分析摘要。")
