"""顶层 API 路由注册中心。

每个功能模块都在 ``app/api/routes/`` 下维护自己的 ``APIRouter``，
本模块统一注册子路由，最后由 ``app/main.py`` 挂载到 ``/api/v1``。

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
from app.api.routes.observability import router as observability_router
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
router.include_router(observability_router, tags=["observability"])
router.include_router(metrics_router, tags=["observability"])  # 获取 /metrics 指标
