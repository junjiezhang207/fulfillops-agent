"""知识库管理 API。

这个文件负责“知识文档运维入口”，和 ``knowledge.py`` 的检索入口分工不同：

``knowledge.py``
    面向业务调用方，提供知识检索和基础索引重建。

``knowledge_mgmt.py``
    面向知识库运营/后台管理，负责上传、解析、清洗、人工确认、版本管理、删除和重建。

本次增强后的入库流程：

1. 上传 .md/.txt/.pdf/.docx。
2. PDF/Word 自动解析成 Markdown；TXT 会补一个标题。
3. 统一执行页眉页脚删除、乱码修复、重复段落去重、敏感信息脱敏、表格转 Markdown。
4. 输出解析报告和质量校验结果。
5. PDF/Word 默认进入 pending 区，等待人工确认；确认后才写入正式知识目录。
6. 确认入库时保存版本快照，并按需触发后台索引重建。

人工确认入库的风险控制：
- 企业知识库不是普通文件上传。错误解析、旧规则、隐私字段一旦进入向量库，就会影响 Agent 决策。
- pending -> confirm 的设计让运营人员能先看 Markdown 预览和报告，再决定是否真的入库。
"""

import asyncio
import re
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile, status

from app.core.config import get_settings
from app.core.service_registry import get_knowledge_retrieval_service
from app.rag.document_ingestion import DocumentIngestionService, SUPPORTED_SOURCE_SUFFIXES
from app.schemas.common import ApiResponse


router = APIRouter(prefix="/knowledge-mgmt")

# ---------------------------------------------------------------------------
# 依赖装配
# ---------------------------------------------------------------------------

_settings = get_settings()
_knowledge_svc = get_knowledge_retrieval_service()
_ingestion_svc = DocumentIngestionService(_settings.knowledge_dir)

# 重建状态目前保存在进程内存中。
# 学习项目这样最直观；真实多实例部署时应换成 Redis / 数据库 / 任务队列状态表。
_rebuild_status: dict = {"running": False, "last_result": None, "last_error": None}


# ---------------------------------------------------------------------------
# 知识文档管理
# ---------------------------------------------------------------------------


@router.get("/documents", response_model=ApiResponse)
async def list_documents() -> ApiResponse:
    """列出正式知识目录中已经入库的文档。

    注意：pending 区的待确认文档不会出现在这里，因为它们还没有进入 RAG 正式索引。
    """

    docs = await asyncio.to_thread(_knowledge_svc.list_document_statuses)
    return ApiResponse(
        success=True,
        message=f"共有 {len(docs)} 个正式知识文档。",
        data={"documents": docs, "total": len(docs)},
    )


@router.post("/documents", response_model=ApiResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: str | None = Form(default=None),
    replace_existing: bool = Form(default=False),
    rebuild: bool = Form(default=True),
    confirm_before_index: bool | None = Form(default=None),
    category: str | None = Form(default=None),
    title: str | None = Form(default=None),
    owner: str | None = Form(default=None),
    version: str | None = Form(default=None),
    effective_date: str | None = Form(default=None),
    expires_at: str | None = Form(default=None),
    region: str | None = Form(default=None),
    business_scope: str | None = Form(default=None),
) -> ApiResponse:
    """上传知识文档。

    新旧行为兼容说明：
    - .md/.txt 默认直接写入正式知识库，并按需触发索引重建。
    - .pdf/.docx 默认只解析到 pending 区，等待人工确认后再入库。
    - 调用方可以显式传 ``confirm_before_index=true``，让任意文档都先进入人工确认。

    表单 metadata 会写入 Markdown front matter，被后续 RAG 切片流程读取：
    - category：知识类别，比如 stockout_rule、priority_rule。
    - version：业务版本号；不传时系统自动生成时间版本。
    - expires_at：过期时间；过期文档会在质检中报错，检索侧也会过滤过期 chunk。
    """

    filename = file.filename or "knowledge.md"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SOURCE_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仅支持上传 .md、.txt、.pdf 或 .docx 知识文档。",
        )

    raw = await file.read()
    doc_id = _safe_document_id(document_id or Path(filename).stem)
    # PDF/Word 默认需要人工确认，因为解析质量和表格结构都需要人看一眼。
    require_confirmation = confirm_before_index if confirm_before_index is not None else suffix in {".pdf", ".docx"}
    metadata = {
        "category": category,
        "title": title,
        "owner": owner,
        "version": version,
        "effective_date": effective_date,
        "expires_at": expires_at,
        "region": region,
        "business_scope": _split_csv(business_scope),
    }

    try:
        report = await asyncio.to_thread(
            _ingestion_svc.prepare_upload,
            filename=filename,
            raw=raw,
            document_id=doc_id,
            metadata=metadata,
            require_confirmation=require_confirmation,
            replace_existing=replace_existing,
            allow_low_quality_direct_commit=confirm_before_index is False,
        )
    except FileExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "同名知识文档已存在，请确认是否覆盖旧版本。",
                "document_id": doc_id,
                "suggestion": "重新提交时设置 replace_existing=true。",
            },
        ) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    rebuild_scheduled = False
    if report.status == "indexed_source_ready":
        rebuild_scheduled = _schedule_rebuild_if_needed(background_tasks, rebuild)

    return ApiResponse(
        success=True,
        message="文档已解析，等待人工确认后入库。" if report.status == "pending_confirmation" else "知识文档已入库。",
        data={
            "document_id": report.document_id,
            "filename": f"{report.document_id}.md",
            "ingestion_id": report.ingestion_id,
            "status": report.status,
            "version": report.version,
            "quality_score": report.quality_score,
            "quality_passed": report.quality_passed,
            "expired": report.expired,
            "warnings": report.warnings,
            "errors": report.errors,
            "cleaning": report.cleaning,
            "rebuild_scheduled": rebuild_scheduled,
        },
    )


@router.get("/documents/{doc_id}", response_model=ApiResponse)
async def get_document(doc_id: str) -> ApiResponse:
    """查看正式知识文档内容预览。

    这里只返回前 2000 字符，避免后台页面一次性拉取超大文档。
    """

    safe_doc_id = _safe_document_id(doc_id)
    path = Path(_settings.knowledge_dir) / f"{safe_doc_id}.md"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档 '{safe_doc_id}' 不存在。",
        )

    content = await asyncio.to_thread(path.read_text, encoding="utf-8")
    truncated = len(content) > 2000
    return ApiResponse(
        success=True,
        message="文档内容已返回。",
        data={
            "id": safe_doc_id,
            "content": content[:2000],
            "truncated": truncated,
            "total_chars": len(content),
        },
    )


@router.delete("/documents/{doc_id}", response_model=ApiResponse)
async def delete_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    rebuild: bool = Query(default=True),
) -> ApiResponse:
    """删除正式知识文档。

    删除源文件后需要重建索引；``KnowledgeRetrievalService`` 会根据注册表删除旧向量 chunk。
    """

    safe_doc_id = _safe_document_id(doc_id)
    path = Path(_settings.knowledge_dir) / f"{safe_doc_id}.md"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档 '{safe_doc_id}' 不存在。",
        )

    await asyncio.to_thread(path.unlink)
    rebuild_scheduled = _schedule_rebuild_if_needed(background_tasks, rebuild)
    return ApiResponse(
        success=True,
        message="知识文档已删除，旧向量会在索引重建时清理。",
        data={"document_id": safe_doc_id, "rebuild_scheduled": rebuild_scheduled},
    )


@router.get("/documents/{doc_id}/versions", response_model=ApiResponse)
async def list_document_versions(doc_id: str) -> ApiResponse:
    """查看某个知识文档的版本历史。"""

    safe_doc_id = _safe_document_id(doc_id)
    versions = await asyncio.to_thread(_ingestion_svc.list_versions, safe_doc_id)
    return ApiResponse(
        success=True,
        message=f"文档 {safe_doc_id} 共有 {len(versions)} 个版本。",
        data={"document_id": safe_doc_id, "versions": versions},
    )


# ---------------------------------------------------------------------------
# 人工确认入库
# ---------------------------------------------------------------------------


@router.get("/ingestions", response_model=ApiResponse)
async def list_pending_ingestions() -> ApiResponse:
    """列出等待人工确认的解析任务。"""

    items = await asyncio.to_thread(_ingestion_svc.list_pending)
    return ApiResponse(
        success=True,
        message=f"共有 {len(items)} 个待确认解析任务。",
        data={"items": items, "total": len(items)},
    )


@router.get("/ingestions/{ingestion_id}", response_model=ApiResponse)
async def get_pending_ingestion(ingestion_id: str) -> ApiResponse:
    """查看解析报告和 Markdown 预览。"""

    try:
        item = await asyncio.to_thread(_ingestion_svc.get_pending, ingestion_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ApiResponse(success=True, message="解析任务详情已返回。", data=item)


@router.post("/ingestions/{ingestion_id}/confirm", response_model=ApiResponse)
async def confirm_pending_ingestion(
    ingestion_id: str,
    background_tasks: BackgroundTasks,
    replace_existing: bool = Form(default=False),
    rebuild: bool = Form(default=True),
) -> ApiResponse:
    """确认解析结果并写入正式知识库。"""

    try:
        committed = await asyncio.to_thread(
            _ingestion_svc.confirm_pending,
            ingestion_id,
            replace_existing=replace_existing,
        )
    except FileExistsError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "同名知识文档已存在，请确认是否覆盖旧版本。",
                "suggestion": "重新确认时设置 replace_existing=true。",
            },
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    rebuild_scheduled = _schedule_rebuild_if_needed(background_tasks, rebuild)
    return ApiResponse(
        success=True,
        message="解析结果已确认入库。",
        data={**committed, "rebuild_scheduled": rebuild_scheduled},
    )


@router.delete("/ingestions/{ingestion_id}", response_model=ApiResponse)
async def reject_pending_ingestion(ingestion_id: str) -> ApiResponse:
    """拒绝解析结果并删除 pending 暂存文件。"""

    try:
        await asyncio.to_thread(_ingestion_svc.reject_pending, ingestion_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return ApiResponse(
        success=True,
        message="待确认解析任务已拒绝并删除。",
        data={"ingestion_id": ingestion_id},
    )


# ---------------------------------------------------------------------------
# 索引重建和状态
# ---------------------------------------------------------------------------


@router.post("/rebuild", response_model=ApiResponse)
async def rebuild_index(background_tasks: BackgroundTasks) -> ApiResponse:
    """触发知识库索引后台重建。"""

    if _rebuild_status["running"]:
        return ApiResponse(
            success=False,
            message="索引重建正在进行中，请稍后查询 /status。",
            data={"running": True},
        )

    background_tasks.add_task(_do_rebuild_background)
    return ApiResponse(
        success=True,
        message="索引重建已在后台启动，通过 GET /knowledge-mgmt/status 查看进度。",
        data={"running": True, "status": "started"},
    )


@router.get("/status", response_model=ApiResponse)
async def index_status() -> ApiResponse:
    """查看知识库索引、待确认任务和后台重建状态。"""

    def _get_stats():
        doc_count = len(_knowledge_svc.knowledge_repository.list_knowledge_paths())
        cache_dir = Path(_settings.knowledge_index_cache_dir)
        index_exists = (cache_dir / "doc_fingerprint.txt").exists()
        registry = _knowledge_svc.document_registry_snapshot()
        pending = _ingestion_svc.list_pending()
        registry_docs = registry.get("documents", {}) if isinstance(registry, dict) else {}
        total_chunk_count = sum(
            len(document.get("chunk_ids") or [])
            for document in registry_docs.values()
            if isinstance(document, dict)
        )
        indexed_document_count = sum(
            1
            for document in registry_docs.values()
            if isinstance(document, dict) and document.get("chunk_ids")
        )
        return doc_count, index_exists, registry, pending, total_chunk_count, indexed_document_count

    doc_count, index_exists, registry, pending, total_chunk_count, indexed_document_count = await asyncio.to_thread(_get_stats)

    return ApiResponse(
        success=True,
        message="知识库状态已返回。",
        data={
            "document_count": doc_count,
            "indexed_document_count": indexed_document_count,
            "unindexed_document_count": max(doc_count - indexed_document_count, 0),
            "total_chunk_count": total_chunk_count,
            "embedding_ready": index_exists and total_chunk_count > 0 and not _rebuild_status["last_error"],
            "vector_store_type": getattr(_settings, "vector_store_type", "local"),
            "index_built": index_exists,
            "rebuild_running": _rebuild_status["running"],
            "last_rebuild_result": _rebuild_status["last_result"],
            "last_rebuild_error": _rebuild_status["last_error"],
            "pending_ingestion_count": len(pending),
            "knowledge_dir": str(Path(_settings.knowledge_dir)),
            "knowledge_extra_dirs": getattr(_settings, "knowledge_extra_dirs", ""),
            "index_cache_dir": str(Path(_settings.knowledge_index_cache_dir)),
            "document_registry": registry,
        },
    )


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


async def _do_rebuild_background() -> None:
    """后台执行索引重建。

    索引重建会读文件、切片、生成 embedding、写向量库，可能耗时较长；
    放到后台任务可以避免上传接口阻塞。
    """

    _rebuild_status["running"] = True
    _rebuild_status["last_error"] = None
    try:
        result = await asyncio.to_thread(_knowledge_svc.rebuild_index)
        _rebuild_status["last_result"] = result
    except Exception as exc:
        _rebuild_status["last_error"] = str(exc)
    finally:
        _rebuild_status["running"] = False


def _safe_document_id(value: str) -> str:
    """把用户传入的 document_id 规范成安全文件名。

    核心目的不是美化名字，而是防止路径穿越，例如 ``../../app/.env``。
    """

    normalized = re.sub(r"[^\w.-]+", "-", value.strip(), flags=re.UNICODE).strip(".-_")
    if not normalized or "/" in normalized or "\\" in normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="document_id 不合法。",
        )
    return normalized


def _split_csv(value: str | None) -> list[str]:
    """把表单里的逗号分隔字段转成列表。"""

    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _schedule_rebuild_if_needed(background_tasks: BackgroundTasks, rebuild: bool) -> bool:
    """按需调度后台重建，避免上传/确认接口被 embedding 和向量库写入阻塞。"""

    if not rebuild or _rebuild_status["running"]:
        return False
    background_tasks.add_task(_do_rebuild_background)
    return True
