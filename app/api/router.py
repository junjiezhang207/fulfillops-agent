"""顶层 API 路由注册中心。

文件作用摘要：
这个文件是所有 API 子路由的“总装配点”。每个功能模块都在 ``app/api/routes/`` 下维护
自己的 ``APIRouter``，这里统一 ``include_router``，最后由 ``app/main.py`` 把总路由挂到
``/api/v1``。

为什么不把所有接口都写在一个文件？
1. 订单、库存、知识库、Agent、Workflow 的依赖和错误处理都不同，拆开后更容易维护。
2. FastAPI 的 ``APIRouter`` 天然支持按模块拆分，路由 prefix、tags、依赖都可以局部管理。
3. 面试或排查时可以按 URL 快速定位文件：例如 ``/agent/chat`` 对应 ``routes/agent.py``。

新增 API 模块的步骤：
1. 在 ``app/api/routes`` 下创建新文件，并定义 ``router = APIRouter(prefix="...")``。
2. 在本文件 import 这个 router。
3. 调用 ``router.include_router`` 注册，并给一个 OpenAPI tag。
"""

from fastapi import APIRouter

from app.api.routes.agent import router as agent_router
from app.api.routes.enterprise_data import router as enterprise_data_router
from app.api.routes.health import router as health_router
from app.api.routes.hybrid import router as hybrid_router
from app.api.routes.inventory import router as inventory_router
from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.knowledge_mgmt import router as knowledge_mgmt_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.models import router as models_router
from app.api.routes.orders import router as order_router
from app.api.routes.workflow import router as workflow_router

# 这里的 router 是“总路由”，不是某个具体业务模块的路由。
# main.py 通常只需要 include 这一个总 router，避免主程序知道太多业务文件。
router = APIRouter()

# tags 会显示在 FastAPI 自动生成的 OpenAPI/Swagger 文档里。
# 注意：每个子 router 自己已经带了 prefix，这里只负责分组，不再重复写 URL 前缀。
router.include_router(health_router, tags=["health"])
router.include_router(enterprise_data_router, tags=["enterprise-data"])
router.include_router(order_router, tags=["orders"])
router.include_router(inventory_router, tags=["inventory"])
router.include_router(knowledge_router, tags=["knowledge"])
router.include_router(knowledge_mgmt_router, tags=["knowledge-mgmt"])
router.include_router(workflow_router, tags=["workflow"])
router.include_router(agent_router, tags=["agent"])
router.include_router(hybrid_router, tags=["hybrid"])
router.include_router(models_router, tags=["models"])
router.include_router(metrics_router, tags=["observability"])  # 获取 /metrics 指标
