"""企业数据接入 API。

本模块模拟 ERP/OMS/WMS 数据接入层，负责登记数据源、导入订单、
导入库存，以及查询企业数据接入后的结果。

``sources`` 表示数据来源，例如 ERP、OMS、WMS、Excel 导入或第三方平台。
``orders/import`` 和 ``inventory/import`` 是批量导入入口，``replace_source``
用于控制是否替换同一来源的旧数据。
"""

from fastapi import APIRouter, HTTPException, Query, status

from app.core.service_registry import get_enterprise_data_repository
from app.schemas.common import ApiResponse
from app.schemas.enterprise_data import (
    EnterpriseDataSourceCreate,
    EnterpriseInventoryImportRequest,
    EnterpriseImportResult,
    EnterpriseOrderImportRequest,
)

router = APIRouter(prefix="/enterprise-data")


@router.post("/sources", response_model=ApiResponse)
def upsert_source(request: EnterpriseDataSourceCreate) -> ApiResponse:
    """登记企业数据源。

    这个接口只负责保存“数据从哪里来”的后台配置；真正业务查询仍然走订单、
    库存仓库接口。这样边界会更像企业项目里的数据接入层。
    """

    repository = get_enterprise_data_repository()
    source = repository.create_source(request, allow_update=True)
    return ApiResponse(
        success=True,
        message="企业数据源已保存。",
        data={"source": source.model_dump(mode="json")},
    )


@router.get("/sources", response_model=ApiResponse)
def list_sources() -> ApiResponse:
    """列出所有已登记的数据源。"""
    repository = get_enterprise_data_repository()
    sources = [source.model_dump(mode="json") for source in repository.list_sources()]
    return ApiResponse(
        success=True,
        message=f"共 {len(sources)} 个企业数据源。",
        data={"sources": sources},
    )


@router.delete("/sources/{source_id}", response_model=ApiResponse)
def delete_source(source_id: str) -> ApiResponse:
    """删除数据源配置。

    注意：这里删除的是数据源及其关联导入数据的入口语义，具体清理逻辑由 repository 决定。
    """
    repository = get_enterprise_data_repository()
    deleted = repository.delete_source(source_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"数据源不存在：{source_id}",
        )
    return ApiResponse(success=True, message="企业数据源已删除。", data={})


@router.post("/orders/import", response_model=ApiResponse)
def import_orders(request: EnterpriseOrderImportRequest) -> ApiResponse:
    """批量导入企业订单。

    ``normalize_source_id`` 会把用户传入的数据源标识规整成稳定 ID，避免同一个来源因为
    大小写、空格等问题被当成多个来源。
    """
    repository = get_enterprise_data_repository()
    source_id = repository.normalize_source_id(request.source_id)
    imported_count = repository.import_orders(
        source_id=source_id,
        orders=request.orders,
        replace_source=request.replace_source,
    )
    stats = repository.stats()
    result = EnterpriseImportResult(
        source_id=source_id,
        imported_count=imported_count,
        total_orders=stats.order_count,
        total_inventory_records=stats.inventory_record_count,
    )
    return ApiResponse(
        success=True,
        message=f"已导入 {imported_count} 条企业订单。",
        data=result.model_dump(mode="json"),
    )


@router.post("/inventory/import", response_model=ApiResponse)
def import_inventory(request: EnterpriseInventoryImportRequest) -> ApiResponse:
    """批量导入企业库存。

    库存数据通常来自 WMS 或 ERP。导入后，库存分析服务可以优先读取企业数据，
    这比只依赖项目内置模拟数据更接近真实业务。
    """
    repository = get_enterprise_data_repository()
    source_id = repository.normalize_source_id(request.source_id)
    imported_count = repository.import_inventory(
        source_id=source_id,
        records=request.records,
        replace_source=request.replace_source,
    )
    stats = repository.stats()
    result = EnterpriseImportResult(
        source_id=source_id,
        imported_count=imported_count,
        total_orders=stats.order_count,
        total_inventory_records=stats.inventory_record_count,
    )
    return ApiResponse(
        success=True,
        message=f"已导入 {imported_count} 条企业库存记录。",
        data=result.model_dump(mode="json"),
    )


@router.get("/orders/{order_id}", response_model=ApiResponse)
def get_enterprise_order(order_id: str) -> ApiResponse:
    """按订单号查询企业导入订单。"""
    repository = get_enterprise_data_repository()
    order = repository.get_order_by_id(order_id)
    if order is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"企业订单不存在：{order_id}",
        )
    return ApiResponse(
        success=True,
        message="企业订单查询完成。",
        data={"order": order.model_dump(mode="json")},
    )


@router.get("/orders", response_model=ApiResponse)
def list_enterprise_orders(limit: int = Query(default=100, ge=1, le=500)) -> ApiResponse:
    """查询企业订单列表。

    前端运营台使用这个接口展示真实订单；没有数据时返回空列表，不再提供 demo 订单。
    """
    repository = get_enterprise_data_repository()
    orders = repository.list_orders(limit=limit)
    return ApiResponse(
        success=True,
        message=f"共 {len(orders)} 条企业订单。",
        data={"orders": [order.model_dump(mode="json") for order in orders]},
    )


@router.get("/inventory/{sku_id}", response_model=ApiResponse)
def get_enterprise_inventory(sku_id: str) -> ApiResponse:
    """按 SKU 查询企业导入库存。

    这里返回列表，是因为同一个 SKU 可能分布在多个仓、多个批次或多个数据源里。
    """
    repository = get_enterprise_data_repository()
    records = repository.list_inventory_by_sku(sku_id)
    return ApiResponse(
        success=True,
        message=f"共 {len(records)} 条企业库存记录。",
        data={"records": [record.model_dump(mode="json") for record in records]},
    )


@router.get("/stats", response_model=ApiResponse)
def get_enterprise_data_stats() -> ApiResponse:
    """返回企业数据接入统计，用于前端看板和调试导入结果。"""
    repository = get_enterprise_data_repository()
    return ApiResponse(
        success=True,
        message="企业数据统计完成。",
        data=repository.stats().model_dump(mode="json"),
    )
