"""健康检查 API。

``GET /health`` 是整个后端最轻量的探活接口。React 前端、部署平台、
反向代理或监控脚本都可以用它判断 FastAPI 进程是否启动成功。

健康检查不初始化 LLM、Milvus、Redis、MySQL 等重依赖。它只证明应用进程
和路由系统可用，不代表所有下游服务健康。
"""

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.common import ApiResponse, HealthResponse

router = APIRouter()


@router.get("/health", response_model=ApiResponse)
def health_check() -> ApiResponse:
    """基础健康检查接口。

    只验证 FastAPI 应用和路由聚合是否可用，并返回统一响应结构。
    """

    # 这里只读取配置对象里的静态字段，不触发模型或数据库初始化。
    # 这样即使 LLM/Milvus/MySQL 没配好，健康检查也能告诉你“后端程序本身是活的”。
    settings = get_settings()
    payload = HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
    )

    return ApiResponse(
        success=True,
        message="服务运行正常。",
        data=payload.model_dump(),
    )
