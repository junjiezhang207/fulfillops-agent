"""Inventory Pydantic schemas.

Learning notes:
- InventoryRecord represents one warehouse/SKU inventory snapshot.
- Inventory analysis outputs are reused by Workflow, Agent and the frontend.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class InventoryRecord(BaseModel):
    """库存原始记录模型。"""

    warehouse_id: str = Field(..., description="仓库编号。")
    warehouse_name: str = Field(..., description="仓库名称。")
    region: str = Field(..., description="仓库所属区域。")
    sku_id: str = Field(..., description="SKU 编码。")
    available_stock: int = Field(..., ge=0, description="可售库存。")
    locked_stock: int = Field(..., ge=0, description="锁定库存。")
    updated_at: datetime = Field(..., description="库存更新时间。")


class InventoryAnalysisRequest(BaseModel):
    """库存判断请求模型。"""

    order_id: str = Field(..., description="待做库存判断的订单编号。")


class SkuInventoryCheckResult(BaseModel):
    """单个 SKU 的库存判断结果。"""

    sku_id: str = Field(..., description="SKU 编码。")
    required_quantity: int = Field(..., description="订单所需数量。")
    total_available_stock: int = Field(..., description="汇总可售库存。")
    fulfillment_ready: bool = Field(..., description="该 SKU 是否满足履约。")
    warehouse_records: list[InventoryRecord] = Field(..., description="该 SKU 关联的仓库库存记录。")


class InventoryAnalysisResult(BaseModel):
    """库存判断结果模型。"""

    order_id: str = Field(..., description="订单编号。")
    fulfillment_ready: bool = Field(..., description="订单整体是否可履约。")
    insufficient_skus: list[str] = Field(..., description="库存不足的 SKU 列表。")
    sku_checks: list[SkuInventoryCheckResult] = Field(..., description="逐个 SKU 的库存判断结果。")
    order_summary: str = Field(..., description="关联订单摘要。")
    summary: str = Field(..., description="库存分析摘要。")
