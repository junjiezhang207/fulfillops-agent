"""RAG 知识检索 API。

文件作用摘要：
这个文件暴露知识库检索能力，主要用于前端直接调试 RAG，也可以帮助你理解 Agent 工具
``retrieve_knowledge`` 背后实际调用的服务是什么。

学习重点：
1. 路由层只接收 ``order_id``、``question`` 和类别过滤，不关心向量检索细节。
2. 检索、改写、BM25/向量混合召回、重排、摘要都在 ``KnowledgeRetrievalService``。
3. ``/reindex`` 是手动重建索引入口，适合本地开发或知识文档变更后触发。

面试官可能问：为什么 RAG 也要提供 HTTP 接口，而不是只给 Agent 用？
回答：RAG 是一个可独立测试的能力。提供 HTTP 入口后，前端、测试、运维脚本和 Agent
工具都能复用同一套检索服务，也方便排查“是检索没命中，还是 Agent 没用好检索结果”。
"""

from fastapi import APIRouter, HTTPException, status

from app.core.service_registry import get_knowledge_retrieval_service
from app.schemas.common import ApiResponse
from app.schemas.knowledge import KnowledgeRetrieveRequest
from app.services.order_analysis_service import (
    OrderNotFoundError,
)

router = APIRouter(prefix="/knowledge")

knowledge_retrieval_service = get_knowledge_retrieval_service()


@router.post("/retrieve", response_model=ApiResponse)
def retrieve_knowledge(request: KnowledgeRetrieveRequest) -> ApiResponse:
    """知识检索接口。

    这里把业务问题交给 RAG 服务，不直接构造 Prompt，也不直接访问向量库。
    如果你学习 RAG 链路，可以从这个接口打断点，然后继续追到 service 层。
    """

    try:
        result = knowledge_retrieval_service.retrieve(
            order_id=request.order_id,
            question=request.question,
            filter_categories=request.filter_categories,
        )
    except OrderNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        # ValueError 通常代表知识库配置或索引状态异常。
        # 这里映射成 500，是因为客户端参数未必有错，更多是服务端检索环境不可用。
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return ApiResponse(
        success=True,
        message="知识检索完成。",
        data=result.model_dump(mode="json"),
    )


@router.post("/reindex", response_model=ApiResponse)
def rebuild_knowledge_index() -> ApiResponse:
    """手动重建知识索引接口。

    注意：这个是基础版同步重建入口；更完整的知识库管理在 ``knowledge_mgmt.py``，
    那里支持上传/删除文档和后台重建。
    """

    message = knowledge_retrieval_service.rebuild_index()
    return ApiResponse(
        success=True,
        message=message,
        data={},
    )
