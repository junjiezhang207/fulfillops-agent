"""健康检查 API。

文件作用摘要：
``GET /health`` 是整个后端最轻量的探活接口。React 前端、部署平台、
反向代理或监控脚本都可以用它判断 FastAPI 进程是否启动成功。

学习重点：
1. 健康检查要非常轻，不应该初始化 LLM、Milvus、Redis、MySQL 这类重依赖。
2. 它只证明“应用进程和路由系统可用”，不证明所有下游服务都健康。
3. 如果未来要做企业级探活，可以拆成：
   - liveness：进程还活着即可。
   - readiness：依赖都准备好，流量才可以打进来。
"""

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.common import ApiResponse, HealthResponse

router = APIRouter()


@router.get("/health", response_model=ApiResponse)
def health_check() -> ApiResponse:
    """基础健康检查接口。

    这是整个后端项目的第一个接口，虽然简单，但非常重要：
    1. 它可以帮助我们验证 FastAPI 应用是否已经正确启动。
    2. 它可以帮助我们验证路由聚合是否生效。
    3. 它也为后续统一响应结构提供一个最小示例。
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
