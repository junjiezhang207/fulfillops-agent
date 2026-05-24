"""订单分析 API。

文件作用摘要：
这个文件暴露 ``POST /orders/analyze``，负责把 HTTP 请求转换成一次订单分析服务调用。
它是最适合学习“路由层 → 服务层 → 仓储层”调用链的入口。

学习重点：
1. 路由层接收的是 Pydantic 请求模型 ``OrderAnalysisRequest``。
2. 真正业务逻辑在 ``OrderAnalysisService``，路由层不直接分析订单字段。
3. 业务异常 ``OrderNotFoundError`` 在这里映射成 HTTP 404。
4. 正常结果统一用 ``ApiResponse`` 包装，前端不用关心每个业务接口的返回形状差异。
"""

from fastapi import APIRouter, HTTPException, status

from app.core.service_registry import get_order_analysis_service
from app.schemas.common import ApiResponse
from app.schemas.orders import OrderAnalysisRequest
from app.services.order_analysis_service import (
    OrderNotFoundError,
)

router = APIRouter(prefix="/orders")

# 当前阶段先直接在模块内组装依赖，目的是让你更容易看懂：
# 路由 -> 服务 -> 仓储 这条最小调用链是如何跑起来的。
# 后续如果项目变大，再把这部分切换成依赖注入模式。
order_analysis_service = get_order_analysis_service()


@router.post("/analyze", response_model=ApiResponse)
def analyze_order(request: OrderAnalysisRequest) -> ApiResponse:
    """订单分析接口。

    当前接口只做三件事：
    1. 接收请求参数。
    2. 调用服务层完成订单分析。
    3. 把结果包装成统一响应结构返回。

    注意：
    - 路由层不直接写分析逻辑。
    - 路由层也不直接操作模拟数据。
    - 这样你后面学习分层时，边界会更清晰。
    """

    # 面试官可能问：为什么这里 catch OrderNotFoundError，而不是让异常直接抛出？
    # 回答：service 层抛的是业务异常，API 层负责把业务异常翻译成 HTTP 语义；
    # “订单不存在”对 HTTP 客户端来说就是 404，而不是 500。
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
