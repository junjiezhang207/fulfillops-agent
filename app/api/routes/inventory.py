"""库存分析 API。

本模块暴露 ``POST /inventory/analyze``，根据订单号判断库存是否足够、哪些 SKU 缺货、
履约是否有风险。它是订单分析之后的第二个基础业务接口。

路由层不做库存计算，只负责把请求交给 ``InventoryAnalysisService``，
并把业务异常转换成 HTTP 状态码。库存规则保留在 service 层，供 API、Agent 工具、
Workflow 节点和测试复用。
"""

from fastapi import APIRouter, HTTPException, status

from app.core.service_registry import get_inventory_analysis_service
from app.schemas.common import ApiResponse
from app.schemas.inventory import InventoryAnalysisRequest
from app.domain.orders.analysis import (
    OrderNotFoundError,
)

router = APIRouter(prefix="/inventory")

inventory_analysis_service = get_inventory_analysis_service()


@router.post("/analyze", response_model=ApiResponse)
def analyze_inventory(request: InventoryAnalysisRequest) -> ApiResponse:
    """库存判断接口。

    这个函数故意很薄：输入校验由 Pydantic 完成，业务判断由 service 完成，
    HTTP 状态码转换由 except 分支完成。
    """

    try:
        result = inventory_analysis_service.analyze_inventory(request.order_id)
    except OrderNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return ApiResponse(
        success=True,
        message="库存判断完成。",
        data=result.model_dump(mode="json"),
    )
