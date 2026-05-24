"""Enterprise data ingestion schemas.

Learning notes:
- Data source models describe ERP/OMS/WMS/SFTP-style sources.
- Import requests convert enterprise data into the unified order/inventory models used by services.
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.schemas.inventory import InventoryRecord
from app.schemas.orders import OrderRecord


def normalize_source_id_value(value: str | None) -> str | None:
    """把来源 ID 规整成适合作为文件内主键的形式。"""

    if value is None:
        return None
    normalized = value.strip().lower().replace(" ", "-")
    if not normalized:
        return None
    return normalized


class EnterpriseDataSourceType(StrEnum):
    """企业数据源类型。

    这里先把类型建出来，是为了让后台接口具备真实项目的扩展点：
    当前版本支持手动 JSON 导入，后面要接 ERP API、数据库、SFTP 文件时，
    不需要重新设计接口，只需要增加对应的同步任务实现。
    """

    MANUAL = "manual"
    JSON = "json"
    API = "api"
    DATABASE = "database"
    SFTP = "sftp"


class EnterpriseDataSourceCreate(BaseModel):
    """创建或登记一个企业数据源。

    config 用 dict 而不是写死字段，是因为不同数据源的连接参数差异很大：
    API 可能需要 base_url/token，数据库需要 dsn/table，SFTP 需要 host/path。
    对外暴露时仍然通过 Pydantic 校验基础结构，避免完全散乱。
    """

    source_id: str | None = Field(default=None, description="数据源 ID；为空时由系统生成。")
    name: str = Field(..., min_length=1, description="数据源名称。")
    source_type: EnterpriseDataSourceType = Field(
        default=EnterpriseDataSourceType.MANUAL,
        description="数据源类型。",
    )
    description: str = Field(default="", description="数据源说明。")
    config: dict[str, Any] = Field(default_factory=dict, description="连接配置。")
    enabled: bool = Field(default=True, description="是否启用。")

    @field_validator("source_id")
    @classmethod
    def normalize_source_id(cls, value: str | None) -> str | None:
        return normalize_source_id_value(value)


class EnterpriseDataSourceInfo(EnterpriseDataSourceCreate):
    """已登记的数据源信息。"""

    source_id: str = Field(..., description="数据源 ID。")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    order_count: int = Field(default=0, description="该数据源导入的订单数。")
    inventory_record_count: int = Field(default=0, description="该数据源导入的库存记录数。")


class EnterpriseOrderImportRequest(BaseModel):
    """企业订单导入请求。

    orders 直接复用项目已有 OrderRecord，核心好处是：
    导入层只负责把企业数据转换成统一业务模型，后面的订单分析、库存分析、
    Agent 工具和 Workflow 都不需要知道数据到底来自 demo、ERP 还是数据库。
    """

    source_id: str = Field(default="manual", description="数据归属的数据源 ID。")
    orders: list[OrderRecord] = Field(..., min_length=1, description="订单列表。")
    replace_source: bool = Field(
        default=False,
        description="为 true 时先清空该数据源已有订单，再写入本次数据。",
    )

    @field_validator("source_id")
    @classmethod
    def normalize_source_id(cls, value: str) -> str:
        return normalize_source_id_value(value) or "manual"


class EnterpriseInventoryImportRequest(BaseModel):
    """企业库存导入请求。"""

    source_id: str = Field(default="manual", description="数据归属的数据源 ID。")
    records: list[InventoryRecord] = Field(..., min_length=1, description="库存记录列表。")
    replace_source: bool = Field(
        default=False,
        description="为 true 时先清空该数据源已有库存，再写入本次数据。",
    )

    @field_validator("source_id")
    @classmethod
    def normalize_source_id(cls, value: str) -> str:
        return normalize_source_id_value(value) or "manual"


class EnterpriseImportResult(BaseModel):
    """导入后的统计结果。"""

    source_id: str
    imported_count: int
    total_orders: int
    total_inventory_records: int


class EnterpriseDataStats(BaseModel):
    """企业数据接入层的整体统计。"""

    source_count: int
    order_count: int
    inventory_record_count: int
    inventory_sku_count: int
