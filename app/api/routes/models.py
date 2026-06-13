"""模型网关查询接口。

本模块暴露模型配置的只读查询能力。前端可以通过这些接口展示“当前有哪些模型可选”
以及“某个用途正在使用哪个模型”，但不能看到 API Key、环境变量或其它敏感配置。

``use_case`` 表示业务用途，例如 agent、workflow、reranker；
``model_type`` 用来区分 chat、embedding、reranker。这里不创建模型实例，
真正的模型构造由 ``LLMFactory`` 完成。
"""

from fastapi import APIRouter, Query

from app.schemas.common import ApiResponse
from app.infrastructure.llm.model_gateway import get_model_gateway

router = APIRouter(prefix="/models")


@router.get("", response_model=ApiResponse)
def list_models(
    use_case: str | None = Query(
        default=None,
        description="按用途过滤：agent / workflow / plan_execute / supervisor / workflow_finalize / rag_rewrite / rag_compress / structured_extract / judge",
    ),
    model_type: str | None = Query(default=None, description="模型类型：chat / embedding / reranker"),
    include_disabled: bool = Query(default=False, description="是否包含未启用模型"),
) -> ApiResponse:
    """返回模型列表。

    include_disabled 默认 False，避免前端把未配置或禁用模型展示成可用项。
    这里只返回模型元数据，模型密钥等敏感信息必须留在服务端。
    """
    gateway = get_model_gateway()
    return ApiResponse(
        success=True,
        message="模型列表已返回。",
        data={
            "models": gateway.list_models(
                use_case=use_case,
                include_disabled=include_disabled,
                model_type=model_type,
            ),
            "model_type": model_type,
        },
    )


@router.get("/active", response_model=ApiResponse)
def active_model(
    use_case: str = Query(
        default="agent",
        description="用途：agent / workflow / plan_execute / supervisor / workflow_finalize / rag_rewrite / rag_compress / structured_extract / judge",
    ),
    model_id: str | None = Query(default=None, description="可选：指定模型 ID"),
    model_type: str | None = Query(default=None, description="模型类型：chat / embedding / reranker"),
) -> ApiResponse:
    """返回某个用途当前实际会使用的模型。

    如果传入 model_id，则查询指定模型；否则按 use_case 的默认策略解析。
    这个接口适合前端状态栏展示，也适合调试“为什么请求走了某个模型”。
    """
    gateway = get_model_gateway()
    model = gateway.active_model(use_case=use_case, model_id=model_id, model_type=model_type)
    return ApiResponse(
        success=bool(model),
        message="当前模型已返回。" if model else "没有可用模型配置。",
        data={"model": model or {}},
    )
