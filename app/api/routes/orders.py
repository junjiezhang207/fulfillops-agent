"""订单分析 API。

本模块暴露 ``POST /orders/analyze``，负责把 HTTP 请求转换成订单分析服务调用。
路由层接收 Pydantic 请求模型，真正业务逻辑在 ``OrderAnalysisService``；
业务异常 ``OrderNotFoundError`` 在这里映射成 HTTP 404。
"""

from fastapi import APIRouter, HTTPException, status

from app.core.service_registry import get_order_analysis_service
from app.schemas.common import ApiResponse
from app.schemas.orders import OrderAnalysisRequest
from app.domain.orders.analysis import (
    OrderNotFoundError,
)

router = APIRouter(prefix="/orders")

# 当前项目在模块内装配依赖；如果后续服务规模变大，可切换为 Depends 或容器注入。
order_analysis_service = get_order_analysis_service()


@router.post("/analyze", response_model=ApiResponse)
def analyze_order(request: OrderAnalysisRequest) -> ApiResponse:
    """订单分析接口。

    当前接口只做三件事：
    1. 接收请求参数。
    2. 调用服务层完成订单分析。
    3. 把结果包装成统一响应结构返回。

    边界：
    - 路由层不直接写分析逻辑。
    - 路由层也不直接操作模拟数据。
    """

    # API 层负责把业务异常翻译成 HTTP 语义；订单不存在对应 404。
    try:
        result = order_analysis_service.analyze_order(request.order_id)
    except OrderNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return ApiResponse(
        success=True,
        message="订单分析完成。",
        data=result.model_dump(mode="json"),
    )
