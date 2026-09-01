"""Inventory Pydantic schemas.

Learning notes:
- InventoryRecord represents one warehouse/SKU inventory snapshot.
- Inventory analysis outputs are reused by Workflow, Agent and the frontend.
"""

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class InventoryRecord(BaseModel):
    """库存原始记录模型。"""

    warehouse_id: str = Field(..., description="仓库编号。")
    warehouse_name: str = Field(..., description="仓库名称。")
    region: str = Field(..., description="仓库所属区域。")
    sku_id: str = Field(..., description="SKU 编码。")
    available_stock: int = Field(..., ge=0, description="可售库存。")
    locked_stock: int = Field(..., ge=0, description="锁定库存。")
    updated_at: datetime = Field(..., description="库存更新时间。")
    on_hand_stock: int | None = Field(default=None, ge=0, description="实物库存。")
    reserved_stock: int = Field(default=0, ge=0, description="已预占库存。")
    inbound_stock: int = Field(default=0, ge=0, description="在途/即将入库库存。")
    expected_inbound_time: datetime | None = Field(default=None, description="预计到仓时间。")
    inventory_version: str | None = Field(default=None, description="库存版本。")
    warehouse_status: str = Field(default="normal", description="仓库是否正常可履约。")
    service_region: str | None = Field(default=None, description="可配送区域。")
    supported_sku_types: list[str] = Field(default_factory=list, description="仓库支持的商品类型。")
    cutoff_time: str | None = Field(default=None, description="当天仓库截单时间。")
    capacity_status: str = Field(default="normal", description="当前仓库履约能力。")

    @model_validator(mode="after")
    def normalize_inventory_fields(self) -> "InventoryRecord":
        if self.on_hand_stock is None:
            self.on_hand_stock = self.available_stock + self.reserved_stock + self.locked_stock
        if not self.service_region:
            self.service_region = self.region
        if not self.inventory_version:
            self.inventory_version = self.updated_at.isoformat()
        return self


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
