"""工作流外观服务（Facade，学习版注释）。

可以把这个类理解为“LangGraph 编译图的使用说明书”：
  - ``run``：普通同步调用，一次性拿最终结果。
  - ``run_stream_async``：异步流式调用，节点完成就推事件。
  - ``run_with_timeout``：带超时保护的调用。
  - ``run_stream`` / ``resume``：支持 Human-in-the-Loop 中断和恢复。

图本身在 ``app.graph.workflow.build_workflow`` 中定义；这里负责给 API 层提供
稳定、好用的运行接口。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from typing import AsyncIterator

from langchain_core.language_models import BaseChatModel
from langgraph.types import Command

from app.graph.llm_adapter import LLMFactory
from app.graph.nodes import WorkflowNodes
from app.graph.workflow import build_workflow
from app.schemas.workflow import (
    ApprovalAuditEntry,
    ApprovalRequest,
    InterruptEvent,
    WorkflowRunRequest,
    WorkflowRunResult,
)
from app.services.inventory_analysis_service import InventoryAnalysisService
from app.services.knowledge_retrieval_service import KnowledgeRetrievalService
from app.services.order_analysis_service import OrderAnalysisService


class WorkflowTimeoutError(Exception):
    """工作流执行超时专用异常，路由层捕获后返回 408。"""


def _safe_serialize(obj: object) -> object:
    """将 state delta 中不可直接 JSON 序列化的对象转为可序列化格式。

    LangGraph state 里可能包含 Pydantic 对象、datetime、ErrorEvent、TraceEvent 等。
    SSE 或 JSON API 只能发送可 JSON 序列化的结构，所以所有事件出门前都走这里。
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(v) for v in obj]
    # Pydantic v2 model
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    # Pydantic v1 model
    if hasattr(obj, "dict"):
        return obj.dict()
    # datetime / date
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    # 其余类型尝试 JSON dumps，失败则 str()
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ── 幂等性存储（内存 TTL 缓存）────────────────────────────────────────────────

class _IdempotencyStore:
    """相同请求在 TTL 内直接返回缓存结果，避免重复执行。

    key 由 (order_id + question + sorted(filter_categories)) 的 SHA-256 决定。
    生产环境替换为 Redis SET + EXPIRE 即可，接口不变。
    """

    def __init__(self, ttl: float = 300.0) -> None:
        self._store: dict[str, tuple[WorkflowRunResult, float]] = {}
        self._ttl = ttl

    @staticmethod
    def make_key(request: WorkflowRunRequest) -> str:
        """为一次业务请求生成幂等 key。

        注意 categories 排序后再参与 hash，这样 ["a", "b"] 和 ["b", "a"]
        会命中同一个缓存，避免无意义重复执行。
        """
        content = (
            f"{request.order_id}:{request.question or ''}:"
            f"{sorted(request.filter_categories or [])}"
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, key: str) -> WorkflowRunResult | None:
        """按 key 读取幂等缓存；过期则删除。"""
        entry = self._store.get(key)
        if entry and time.monotonic() - entry[1] < self._ttl:
            return entry[0]
        if entry:
            del self._store[key]
        return None

    def set(self, key: str, result: WorkflowRunResult) -> None:
        """写入幂等缓存。"""
        self._store[key] = (result, time.monotonic())


class WorkflowService:
    """固定 workflow 外观服务。

    它持有两样东西：
      - ``self._nodes``：带业务 service 依赖的节点集合。
      - ``self._graph``：LangGraph 编译后的可运行图。
    """

    _idempotency = _IdempotencyStore(ttl=300.0)

    @staticmethod
    def _extract_pending_interrupt(state_snapshot) -> dict:
        """从 get_state() 的快照中提取中断信息。

        LangGraph 在 interrupt 后会把中断对象放到 state snapshot 上。
        API 层需要把它转成 InterruptEvent 返回给前端审批台。
        """
        interrupts = getattr(state_snapshot, "interrupts", None) or ()
        for interrupt_obj in interrupts:
            if hasattr(interrupt_obj, "value"):
                return interrupt_obj.value
        return {}

    @staticmethod
    def _build_completed_result(
        request: WorkflowRunRequest | None,
        final_state: dict,
    ) -> dict:
        """从最终 state 构建 WorkflowRunResult。

        这个方法被 run_stream 和 resume 共用。
        resume 时没有原始 request，所以会优先从 final_state 里取 order_id/question。
        """
        order_id = final_state.get("order_id", "")
        question = final_state.get("question")
        filter_categories = final_state.get("filter_categories", [])
        if request is not None:
            order_id = request.order_id
            question = request.question
            filter_categories = request.filter_categories

        return {
            "status": "completed",
            "result": WorkflowRunResult(
                order_id=order_id,
                question=question,
                filter_categories=filter_categories,
                order_result=final_state.get("order_result"),
                inventory_result=final_state.get("inventory_result"),
                knowledge_result=final_state.get("knowledge_result"),
                final_answer=final_state.get("final_answer"),
                trace=final_state.get("trace") or [],
                errors=final_state.get("errors") or [],
            ),
            "interrupt": None,
        }

    def __init__(
        self,
        order_service: OrderAnalysisService,
        inventory_service: InventoryAnalysisService,
        knowledge_service: KnowledgeRetrievalService,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        """构造节点集合并编译 workflow 图。

        如果调用方没有传 chat_model，会尝试从配置创建一个。
        没有 LLM 也没关系，finalize 节点会走规则模板 fallback。
        """
        from app.core.config import get_settings
        from app.agent.checkpointer import create_checkpointer
        from app.memory import create_long_term_memory_store

        settings = get_settings()
        # 如果没有显式传 chat_model，就从模型网关/配置创建 workflow_finalize 用模型。
        effective_chat_model = (
            chat_model if chat_model is not None
            else LLMFactory.create_chat_model(settings, use_case="workflow_finalize")
        )

        # WorkflowNodes 是图节点集合，节点内部会调用订单/库存/RAG service。
        self._nodes = WorkflowNodes(
            order_service=order_service,
            inventory_service=inventory_service,
            knowledge_service=knowledge_service,
            chat_model=effective_chat_model,
        )
        memory_embed_model = None
        # 长期记忆使用向量后端时，需要 embedding 模型；这里懒加载，减少启动压力。
        long_term_backend = settings.long_term_memory_backend.strip().lower()
        if long_term_backend in {"mysql_milvus", "mysql+milvus", "mysql", "milvus"}:
            from app.graph.embed_adapter import create_lazy_embed_model

            memory_embed_model = create_lazy_embed_model(settings)
        # build_workflow 会把节点、短期 checkpointer、长期 store 编译成 LangGraph。
        self._graph = build_workflow(
            self._nodes,
            checkpointer=create_checkpointer(
                settings.redis_url,
                ttl_seconds=settings.short_term_memory_ttl_seconds,
            ),
            store=create_long_term_memory_store(
                db_path=settings.long_term_memory_db_path,
                backend=settings.long_term_memory_backend,
                mysql_url=settings.long_term_memory_mysql_url or settings.mysql_url,
                milvus_uri=settings.milvus_uri or f"http://{settings.milvus_host}:{settings.milvus_port}",
                milvus_token=settings.milvus_token,
                milvus_database=settings.milvus_database,
                milvus_collection=settings.long_term_memory_milvus_collection,
                milvus_alias=settings.long_term_memory_milvus_alias,
                milvus_timeout_seconds=settings.milvus_timeout_seconds,
                milvus_similarity_metric=settings.milvus_similarity_metric,
                embedding_model=memory_embed_model,
                vector_dimension=settings.long_term_memory_vector_dimension,
                default_ttl_days=settings.long_term_memory_ttl_days,
            ),
        )

    # =========================================================================
    # 同步执行（向后兼容，自动通过所有 HITL 检查点）
    # =========================================================================

    def run(self, request: WorkflowRunRequest) -> WorkflowRunResult:
        """驱动一次完整工作流执行（含幂等性检查，自动批准 HITL 检查点）。

        幂等性：相同 (order_id + question + categories) 在 5 分钟内直接返回缓存，
                不重复触发 LLM 调用，防止网络重试、重复点击导致重复执行。

        同步模式的设计取舍：
          为了兼容普通 HTTP 请求，它不会把 HITL 中断暴露给用户，
          而是遇到中断后一律用 approved 自动恢复。
          需要真实审批体验时，应使用 run_stream/resume。
        """
        idem_key = self._idempotency.make_key(request)
        cached = self._idempotency.get(idem_key)
        if cached is not None:
            # 同样请求短时间重复提交时直接返回，避免重复执行图。
            return cached

        thread_id = f"sync-{request.order_id}-{uuid.uuid4().hex[:8]}"
        # thread_id 是 LangGraph checkpointer 的会话 key。
        # 即使是同步调用也要传，因为 interrupt/resume 需要依赖它保存状态。
        initial_state: dict = {
            "order_id": request.order_id,
            "question": request.question,
            "filter_categories": request.filter_categories,
            "trace": [],
            "errors": [],
        }
        config = {"configurable": {"thread_id": thread_id}}

        # graph.invoke() 一次性跑完整张图，返回最终 state。
        final_state: dict = self._graph.invoke(initial_state, config)

        # 自动处理 HITL 中断（同步模式一律自动批准）。
        # get_state(config).next 非空，说明图停在某个 interrupt 之后，还有后续节点没跑。
        state_snapshot = self._graph.get_state(config)
        while state_snapshot and state_snapshot.next:
            final_state = self._graph.invoke(Command(resume="approved"), config)
            state_snapshot = self._graph.get_state(config)

        result = WorkflowRunResult(
            order_id=request.order_id,
            question=request.question,
            filter_categories=request.filter_categories,
            order_result=final_state.get("order_result"),
            inventory_result=final_state.get("inventory_result"),
            knowledge_result=final_state.get("knowledge_result"),
            final_answer=final_state.get("final_answer"),
            trace=final_state.get("trace") or [],
            errors=final_state.get("errors") or [],
        )
        # 完整执行成功后写入幂等缓存。
        self._idempotency.set(idem_key, result)
        return result

    # =========================================================================
    # 真正的异步 SSE 流式执行（每完成一个节点推一次事件）
    # =========================================================================

    async def run_stream_async(
        self,
        request: WorkflowRunRequest,
        thread_id: str,
        timeout: float = 60.0,
    ) -> AsyncIterator[dict]:
        """异步流式执行工作流，每完成一个节点 yield 一个事件。

        事件格式：
          {"type": "node_complete", "node": "order_analysis", "data": {...}}
          {"type": "interrupted",   "interrupt": InterruptEvent}
          {"type": "completed",     "final_state": {...}}
          {"type": "error",         "message": "..."}

        LangGraph API：
          graph.astream(input, config, stream_mode="updates") 每完成一个节点
          yield 一个 {node_name: state_delta} dict，
          是 stream() 的 async 版本，配合 FastAPI StreamingResponse 或 SSE 使用。

        超时：asyncio.wait_for 包裹整个 astream 生成器，超过 timeout 秒抛出 TimeoutError。

        注意这里 yield 的不是最终 API schema，而是“事件”。
        路由层会把这些 dict 包装成 SSE data 行发给前端。
        """
        initial_state: dict = {
            "order_id": request.order_id,
            "question": request.question,
            "filter_categories": request.filter_categories or [],
            "trace": [],
            "errors": [],
        }
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # asyncio.timeout()（Python 3.11+）真正对 async generator 生效，
            # 旧版 asyncio.wait_for 只能包裹 coroutine，无法包裹 async generator。
            async with asyncio.timeout(timeout):
                async for event in self._graph.astream(
                    initial_state, config, stream_mode="updates"
                ):
                    # stream_mode="updates" 每次返回 {node_name: state_delta}。
                    # state_delta 只包含该节点本次写入的字段，不是完整 state。
                    node_name = next(iter(event), "unknown")
                    state_delta = event.get(node_name, {})
                    # 节点完成后立刻把 delta 序列化成前端可消费事件。
                    yield {
                        "type": "node_complete",
                        "node": node_name,
                        "data": _safe_serialize(state_delta),
                    }

        except asyncio.TimeoutError:
            yield {"type": "error", "message": f"工作流超时（{timeout:.0f}s）"}
            return
        except Exception as exc:
            yield {"type": "error", "message": str(exc)}
            return

        state_snapshot = self._graph.get_state(config)
        if state_snapshot and state_snapshot.next:
            # 如果图停在 interrupt，这里不会返回 completed，而是把审批信息交给前端。
            # 前端拿到 thread_id 后，后续调用 resume(thread_id, decision) 继续执行。
            interrupt_info = self._extract_pending_interrupt(state_snapshot)
            interrupt_event = InterruptEvent(
                type=interrupt_info.get("type", "unknown"),
                node=interrupt_info.get("node", "unknown"),
                prompt=interrupt_info.get("prompt", ""),
                context=interrupt_info.get("context", {}),
                options=interrupt_info.get("options", ["approved", "rejected", "escalate"]),
                thread_id=thread_id,
                risk_level=interrupt_info.get("risk_level", "HIGH"),
                risk_signals=interrupt_info.get("risk_signals", []),
                timeout_seconds=interrupt_info.get("timeout_seconds", 1800),
            )
            yield {"type": "interrupted", "interrupt": interrupt_event.model_dump()}
            return

        final_state = self._graph.get_state(config).values
        # 没有中断，说明图已经走到 END；此时 snapshot.values 就是完整最终 state。
        yield {
            "type": "completed",
            "final_state": _safe_serialize(final_state),
        }

    # =========================================================================
    # 带超时的同步执行（asyncio.wait_for 包裹）
    # =========================================================================

    async def run_with_timeout(
        self,
        request: WorkflowRunRequest,
        timeout: float = 30.0,
    ) -> WorkflowRunResult:
        """带 asyncio 超时保护的异步执行版本。

        超过 timeout 秒抛出 WorkflowTimeoutError，路由层捕获后返回 408。

        它适合“调用方想等最终结果，但又不希望请求无限挂住”的场景。
        """
        thread_id = f"timeout-{request.order_id}-{uuid.uuid4().hex[:8]}"
        initial_state: dict = {
            "order_id": request.order_id,
            "question": request.question,
            "filter_categories": request.filter_categories or [],
            "trace": [],
            "errors": [],
        }
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # ainvoke 是 LangGraph 的异步完整执行接口。
            final_state = await asyncio.wait_for(
                self._graph.ainvoke(initial_state, config),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise WorkflowTimeoutError(
                f"工作流执行超时（{timeout:.0f}s），订单 {request.order_id}"
            )

        return WorkflowRunResult(
            order_id=request.order_id,
            question=request.question,
            filter_categories=request.filter_categories or [],
            order_result=final_state.get("order_result"),
            inventory_result=final_state.get("inventory_result"),
            knowledge_result=final_state.get("knowledge_result"),
            final_answer=final_state.get("final_answer"),
            trace=final_state.get("trace") or [],
            errors=final_state.get("errors") or [],
        )

    # =========================================================================
    # 流式执行（支持 HITL 中断）
    # =========================================================================

    def run_stream(self, request: WorkflowRunRequest, thread_id: str) -> dict:
        """驱动工作流执行，遇到中断时返回中断信息。

        Returns:
            {"status": "completed"|"interrupted"|"error",
             "result": WorkflowRunResult | None,
             "interrupt": InterruptEvent | None}

        和 run_stream_async 的区别：
          - run_stream_async 是真正流式，适合 SSE。
          - run_stream 是“执行到完成或中断就返回一个 dict”，适合普通接口。
        """
        initial_state: dict = {
            "order_id": request.order_id,
            "question": request.question,
            "filter_categories": request.filter_categories,
            "trace": [],
            "errors": [],
        }
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # invoke 遇到 interrupt 时会停下，状态保存在 checkpointer 里。
            final_state: dict = self._graph.invoke(initial_state, config)
        except Exception as exc:
            return {"status": "error", "result": None, "interrupt": None, "error": str(exc)}

        state_snapshot = self._graph.get_state(config)
        if state_snapshot and state_snapshot.next:
            # next 非空表示图没有走完，通常是 interrupt 等待人工决策。
            interrupt_info = self._extract_pending_interrupt(state_snapshot)
            event = InterruptEvent(
                type=interrupt_info.get("type", "unknown"),
                node=interrupt_info.get("node", "unknown"),
                prompt=interrupt_info.get("prompt", ""),
                context=interrupt_info.get("context", {}),
                options=interrupt_info.get("options", ["approved", "rejected", "escalate"]),
                thread_id=thread_id,
                risk_level=interrupt_info.get("risk_level", "HIGH"),
                risk_signals=interrupt_info.get("risk_signals", []),
                timeout_seconds=interrupt_info.get("timeout_seconds", 1800),
            )
            return {"status": "interrupted", "result": None, "interrupt": event}

        return self._build_completed_result(request, final_state)

    # =========================================================================
    # 恢复执行（HITL resume）
    # =========================================================================

    def resume(self, thread_id: str, approval: ApprovalRequest) -> dict:
        """恢复被中断的工作流。

        LangGraph API：
          graph.invoke(Command(resume=decision), config) 恢复中断的图执行，
          替代旧版 for event in graph.stream(Command(...)):。

        关键点：
          resume 必须使用和中断时相同的 thread_id。
          因为 LangGraph 要靠 thread_id 找回之前停住的 state。
        """
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # Command(resume=...) 是 LangGraph HITL 标准 API。
            # approval.decision 会作为 interrupt(...) 的返回值回到节点内部。
            final_state: dict = self._graph.invoke(
                Command(resume=approval.decision), config
            )
        except Exception as exc:
            return {"status": "error", "result": None, "interrupt": None, "error": str(exc)}

        state_snapshot = self._graph.get_state(config)
        if state_snapshot and state_snapshot.next:
            # resume 后仍可能再次中断，例如一个流程里有多个审批点。
            interrupt_info = self._extract_pending_interrupt(state_snapshot)
            event = InterruptEvent(
                type=interrupt_info.get("type", "unknown"),
                node=interrupt_info.get("node", "unknown"),
                prompt=interrupt_info.get("prompt", ""),
                context=interrupt_info.get("context", {}),
                options=interrupt_info.get("options", ["approved", "rejected", "escalate"]),
                thread_id=thread_id,
                risk_level=interrupt_info.get("risk_level", "HIGH"),
                risk_signals=interrupt_info.get("risk_signals", []),
                timeout_seconds=interrupt_info.get("timeout_seconds", 1800),
            )
            return {"status": "interrupted", "result": None, "interrupt": event}

        return self._build_completed_result(None, final_state)

    # =========================================================================
    # 审批查询接口（预留，接入 MySQL 后实现）
    # =========================================================================

    def list_pending_approvals(self, risk_level: str | None = None) -> list[ApprovalAuditEntry]:
        """查询当前待审批工单列表（TODO: 接入 MySQL）。"""
        return []

    def get_approval_history(self, order_id: str) -> list[ApprovalAuditEntry]:
        """查询订单审批历史（TODO: 接入 MySQL）。"""
        return []

    def get_hitl_stats(self) -> dict:
        """HITL 统计（TODO: 接入 MySQL）。"""
        return {"pending_count": 0, "total_decisions": 0, "approval_rate": 0.0}
