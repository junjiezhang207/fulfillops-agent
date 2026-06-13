"""FastAPI 应用入口。

学习笔记：
- create_app() 读取配置、初始化日志，并注册所有路由。
- TraceIdMiddleware 会给请求日志附加 trace_id。
- Uvicorn 会从这个文件加载全局 app 对象。
"""

from pathlib import Path
import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.router import router as api_router
from app.core.config import get_settings
from app.core.infrastructure_checks import verify_required_infrastructure
from app.core.service_registry import get_knowledge_retrieval_service
from app.core.logging import TraceIdMiddleware, setup_logging
from app.observability.trace_context import BusinessTraceMiddleware

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。

    设计说明：
    1. 使用工厂函数而不是直接在全局创建应用，便于后续测试和扩展。
    2. 应用初始化时统一完成配置读取、日志初始化、路由注册。
    3. TraceIdMiddleware 注入全链路 trace_id，让日志自动携带请求标识。
    """

    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )
    cors_origins = [
        origin.strip()
        for origin in settings.frontend_cors_origins.split(",")
        if origin.strip()
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    # 全链路 trace_id：从 X-Trace-Id 请求头读取或自动生成，并注入所有日志。
    # BusinessTraceMiddleware 负责业务决策链路 trace 的创建和持久化。
    # TraceIdMiddleware 继续服务旧的 JSON 日志关联，但只注册一次，避免重复中间件开销。
    app.add_middleware(BusinessTraceMiddleware)
    app.add_middleware(TraceIdMiddleware)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    frontend_dist = Path("frontend") / "dist"
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    @app.on_event("startup")
    async def verify_infrastructure() -> None:
        """启动时检查生产必需基础设施，避免运行中才发现降级或状态丢失。"""
        results = await asyncio.to_thread(verify_required_infrastructure, settings)
        logger.info("基础设施检查通过：%s", ", ".join(f"{item.name}={item.detail}" for item in results))

    @app.on_event("startup")
    async def warmup_knowledge_index() -> None:
        """Warm RAG index before the first user question reaches retrieval."""
        if not bool(getattr(settings, "knowledge_warmup_on_startup", True)):
            return
        try:
            result = await asyncio.to_thread(get_knowledge_retrieval_service().warmup_index)
            logger.info("RAG 索引启动预热完成：%s", result)
        except Exception as exc:
            logger.warning("RAG 索引启动预热失败，首次检索可能变慢：%s", exc)

    return app


app = create_app()
