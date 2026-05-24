"""知识库管理 API — 文档查询、上传、删除、索引重建、版本信息。

文件作用摘要：
``knowledge.py`` 更像“业务检索入口”，而本文件是“知识库运维入口”。它负责管理原始
知识文档，并触发 RAG 索引重建，让企业规则可以持续更新。

异步改造说明：
  list_documents / get_document — 文件 I/O 改为 asyncio.to_thread()，
                                   不阻塞 FastAPI 事件循环
  rebuild_index                 — 索引重建是耗时操作（秒级到分钟级），
                                   改为 BackgroundTasks 立即返回，后台执行，
                                   轮询 /status 端点查看完成状态

学习重点：
1. 上传/删除文档只是改源文件，真正让 RAG 生效还要重建索引。
2. 上传同名文档时返回 409，让前端显式确认是否覆盖，避免误删知识。
3. 文档 ID 必须安全化，防止路径穿越，例如 ``../../secret``。
4. 后台任务状态目前存在内存里，生产环境可以换成 Redis/数据库。

面试官可能问：为什么重建索引要后台执行？
回答：重建会涉及文件读取、切分、embedding、写入向量库，耗时不可控。如果同步阻塞
HTTP 请求，前端容易超时，也会占住 worker。后台执行 + 状态轮询更适合运维型任务。
"""

import asyncio
import re
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile, status

from app.core.config import get_settings
from app.core.service_registry import get_knowledge_retrieval_service
from app.schemas.common import ApiResponse

router = APIRouter(prefix="/knowledge-mgmt")

# ── 依赖装配 ──────────────────────────────────────────────────────────────────
_settings = get_settings()
_knowledge_svc = get_knowledge_retrieval_service()

# 重建状态追踪（简单内存标志，生产可改为 Redis）。
# 为什么这里先用内存？
# - 学习项目更直观，少一个外部依赖。
# - 单进程本地开发足够。
# - 真正多实例部署时，内存状态无法共享，所以要迁移到 Redis/数据库。
_rebuild_status: dict = {"running": False, "last_result": None, "last_error": None}


# ── 端点 ──────────────────────────────────────────────────────────────────────

@router.get("/documents", response_model=ApiResponse)
async def list_documents() -> ApiResponse:
    """列出知识库中所有文档的基本信息。

    文件 I/O 通过 asyncio.to_thread() 执行，不阻塞事件循环。
    """
    docs = await asyncio.to_thread(_knowledge_svc.list_document_statuses)
    return ApiResponse(
        success=True,
        message=f"共 {len(docs)} 个知识文档。",
        data={"documents": docs, "total": len(docs)},
    )


@router.post("/documents", response_model=ApiResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: str | None = Form(default=None),
    replace_existing: bool = Form(default=False),
    rebuild: bool = Form(default=True),
) -> ApiResponse:
    """上传或替换知识库文档。

    真实企业知识库接入不能只“追加文件”，必须明确处理冲突：
    - 同 document_id 已存在且 replace_existing=false：返回 409，提醒前端让用户确认。
    - replace_existing=true：覆盖旧文件，并通过后台重建清理旧向量 chunk。
    """

    filename = file.filename or "knowledge.md"
    # 只允许 .md/.txt，是为了让知识库内容保持可读、可 diff、可审查。
    # 如果未来支持 PDF/Word，应该先做解析和清洗，再进入同一套文档注册/重建流程。
    if not filename.lower().endswith((".md", ".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仅支持上传 .md 或 .txt 知识文档。",
        )

    raw = await file.read()
    try:
        # 统一 UTF-8 可以避免 Windows/服务器之间编码不一致导致分词和检索异常。
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="知识文档必须使用 UTF-8 编码。",
        ) from exc

    doc_id = _safe_document_id(document_id or Path(filename).stem)
    knowledge_dir = Path(_settings.knowledge_dir)
    target = knowledge_dir / f"{doc_id}.md"
    existed = target.exists()
    if existed and not replace_existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "同名知识文档已存在，请确认是否覆盖旧版本。",
                "document_id": doc_id,
                "filename": target.name,
                "suggestion": "重新提交时设置 replace_existing=true，系统会替换文件并在重建索引时清理旧向量。",
            },
        )

    def _write_file() -> None:
        # 先写临时文件再 replace，是一个轻量的原子写思路：
        # 避免写到一半进程异常时留下半截知识文档。
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".md.tmp")
        temp.write_text(content, encoding="utf-8")
        temp.replace(target)

    await asyncio.to_thread(_write_file)
    rebuild_scheduled = _schedule_rebuild_if_needed(background_tasks, rebuild)
    return ApiResponse(
        success=True,
        message="知识文档已上传。" if not existed else "知识文档已替换，旧版本会在索引重建时清理。",
        data={
            "document_id": doc_id,
            "filename": target.name,
            "replaced": existed,
            "rebuild_scheduled": rebuild_scheduled,
        },
    )


@router.get("/documents/{doc_id}", response_model=ApiResponse)
async def get_document(doc_id: str) -> ApiResponse:
    """查看指定文档的内容（前 2000 字符）。

    文件读取通过 asyncio.to_thread() 执行，不阻塞事件循环。
    """
    knowledge_dir = Path(_settings.knowledge_dir)
    path = knowledge_dir / f"{doc_id}.md"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档 '{doc_id}' 不存在。",
        )

    content = await asyncio.to_thread(path.read_text, encoding="utf-8")
    truncated = len(content) > 2000
    return ApiResponse(
        success=True,
        message="文档内容已返回。",
        data={
            "id": doc_id,
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
    """删除知识库文档。

    删除文件后需要重建索引；重建时 KnowledgeRetrievalService 会根据注册表删除 Milvus 旧 chunk。
    """

    safe_doc_id = _safe_document_id(doc_id)
    knowledge_dir = Path(_settings.knowledge_dir)
    path = knowledge_dir / f"{safe_doc_id}.md"
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
        data={
            "document_id": safe_doc_id,
            "rebuild_scheduled": rebuild_scheduled,
        },
    )


@router.post("/rebuild", response_model=ApiResponse)
async def rebuild_index(background_tasks: BackgroundTasks) -> ApiResponse:
    """触发知识库索引重建（后台执行，立即返回）。

    改造前：同步阻塞，大型知识库可能让请求卡住数分钟
    改造后：BackgroundTasks 立即返回 202，重建在后台进行
            调用 GET /status 轮询重建进度

    适用场景：
      - 添加了新的规则文档
      - 修改了现有规则内容
      - 删除了过时的规则
    """
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
    """查看知识库索引当前状态（含后台重建进度）。"""
    knowledge_dir = Path(_settings.knowledge_dir)

    def _get_stats():
        doc_count = len(list(knowledge_dir.glob("*.md"))) if knowledge_dir.exists() else 0
        cache_dir = Path(_settings.knowledge_index_cache_dir)
        index_exists = (cache_dir / "doc_fingerprint.txt").exists()
        registry = _knowledge_svc.document_registry_snapshot()
        return doc_count, index_exists, registry

    doc_count, index_exists, registry = await asyncio.to_thread(_get_stats)

    return ApiResponse(
        success=True,
        message="知识库状态已返回。",
        data={
            "document_count": doc_count,
            "index_built": index_exists,
            "rebuild_running": _rebuild_status["running"],
            "last_rebuild_result": _rebuild_status["last_result"],
            "last_rebuild_error": _rebuild_status["last_error"],
            "knowledge_dir": str(knowledge_dir),
            "index_cache_dir": str(Path(_settings.knowledge_index_cache_dir)),
            "document_registry": registry,
        },
    )


# ── 内部辅助 ──────────────────────────────────────────────────────────────────

async def _do_rebuild_background() -> None:
    """后台重建任务：在线程池里执行耗时的同步重建逻辑。"""
    _rebuild_status["running"] = True
    _rebuild_status["last_error"] = None
    try:
        result = await asyncio.to_thread(_knowledge_svc.rebuild_index)
        _rebuild_status["last_result"] = result
    except Exception as exc:
        _rebuild_status["last_error"] = str(exc)
    finally:
        _rebuild_status["running"] = False


def _infer_category(stem: str) -> str:
    """根据文件名推断知识类别。

    当前函数是一个轻量 fallback，主要用于本地学习和 demo。企业项目里更常见的做法是：
    - 上传时让运营人员显式选择 category；
    - 或在文档 front matter 里写 metadata；
    - 或从知识管理系统同步结构化标签。
    """
    _MAP = {
        "stockout": "缺货处理", "priority": "优先级规则",
        "regional": "区域履约", "split": "拆单策略",
        "after_sales": "售后规则",
    }
    stem_lower = stem.lower()
    for key, val in _MAP.items():
        if key in stem_lower:
            return val
    return "通用规则"


def _safe_document_id(value: str) -> str:
    """把用户传入的文档 ID 规范成安全文件名。"""

    # 这里的目标不是“美化名字”，而是防止路径穿越和奇怪字符污染知识库目录。
    # 例如用户传 ../../app/.env，规范化后如果仍包含 / 或 \ 就直接拒绝。
    normalized = re.sub(r"[^\w.-]+", "-", value.strip(), flags=re.UNICODE).strip(".-_")
    if not normalized or "/" in normalized or "\\" in normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="document_id 不合法。",
        )
    return normalized


def _schedule_rebuild_if_needed(background_tasks: BackgroundTasks, rebuild: bool) -> bool:
    """按需调度后台重建，避免上传接口被 embedding/Milvus 阻塞。"""

    if not rebuild or _rebuild_status["running"]:
        return False
    background_tasks.add_task(_do_rebuild_background)
    return True
