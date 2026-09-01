"""混合服务（学习版注释）— 根据意图动态选择 Workflow、Agent、RAG 或 Multi-Agent。

混合策略：
  - SIMPLE       → Workflow（快速、确定性、支持 HITL 中断）
  - RAG          → KnowledgeRetrievalService（规则/SOP/政策问答）
  - MEDIUM       → Agent（单 ReAct Agent，自主选择工具）
  - COMPLEX      → Agent（单域深度分析）
  - MULTI_DOMAIN → Multi-Agent（跨域协同，Supervisor 调度）

统一的响应格式，上层 API 无需关心底层用的哪条路径。
"""

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from app.schemas.workflow import WorkflowRunRequest
from app.agents.runtime.agent_service import AgentService
from app.application.cache.client_cache_service import ResponseCacheService
from app.application.routing.intent_classifier import (
    IntentClassificationResult,
    IntentLevel,
    get_classifier,
)
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
from app.agents.runtime.multi_agent_service import MultiAgentService
from app.application.memory.session_memory_service import (
    SessionMemoryService,
)
from app.application.routing.ops_case_service import OpsCaseAnalysisService
from app.application.workflow.workflow_service import WorkflowService
from app.infrastructure.llm.model_gateway import get_model_gateway
from app.observability.business_trace import add_trace_step


@dataclass
class HybridResult:
    """混合服务的统一返回格式。

    无论底层走 Workflow、Agent 还是 Multi-Agent，API 都返回这个结构。
    """

    # 当前处理的订单。
    order_id: str
    # 用户原始问题。
    question: str
    # 实际使用的路径。
    path_used: str  # "workflow" | "agent" | "rag" | "multi_agent" | "chat"
    # 意图分类等级。
    intent_level: str  # "simple" | "rag" | "medium" | "complex" | "multi_domain" | "casual"
    # 分类置信度。
    confidence: float
    # 最终给用户看的回答。
    final_answer: str
    # 底层路径调用过的工具/Agent。
    tools_called: list[str]
    execution_time_ms: float = 0.0
    thread_id: Optional[str] = None
    conversation_turns: int = 0
    cache_hit: bool = False
    from_cache: bool = False
    # HITL: 当 workflow 路径被中断时，以下字段有值
    status: str = "completed"  # "completed" | "interrupted" | "error"
    interrupt: Optional[dict] = None
    # 前端 Action Card：可审批履约提案和执行前二次校验。
    action_card: Optional[dict[str, Any]] = None
    preflight_validation: Optional[dict[str, Any]] = None
    # 路由解释，给前端展示“为什么进这条链路”。
    intent_reasoning: str = ""
    intent_keywords: list[str] | None = None
    session_memory: Optional[dict[str, Any]] = None
    rag_context: Optional[dict[str, Any]] = None

    def __post_init__(self):
        # 保证 tools_called 始终是 list，避免前端处理 None。
        if self.tools_called is None:
            self.tools_called = []
        if self.intent_keywords is None:
            self.intent_keywords = []


class HybridService:
    """混合服务 — 协调 Agent、Workflow、Multi-Agent 的动态路由。

    这是项目的“总入口型服务”之一：
    它不亲自分析订单，也不亲自调用工具，而是决定应该让哪条链路处理。
    """

    def __init__(
        self,
        workflow_service: WorkflowService,
        agent_service: AgentService | None,
        knowledge_service: KnowledgeRetrievalService,
        multi_agent_service: Optional[MultiAgentService] = None,
        ops_case_service: Optional[OpsCaseAnalysisService] = None,
        intent_model: object | None = None,
        session_cache: Optional[SessionMemoryService] = None,
        response_cache: Optional[ResponseCacheService] = None,
        client_cache: Optional[ResponseCacheService] = None,
        simple_threshold: float = 0.5,
        complex_threshold: float = 0.7,
    ):
        # Workflow 适合固定、可解释、带 HITL 的流程。
        self.workflow_service = workflow_service
        # Agent 适合中等/复杂但仍属于单域的问题；模型未配置时允许为空。
        self.agent_service = agent_service
        # RAG 适合回答企业规则、SOP、政策依据类问题。
        self.knowledge_service = knowledge_service
        # Multi-Agent 适合跨领域复杂问题，可选依赖。
        self.multi_agent_service = multi_agent_service
        # 运营异常案件路径：整理案件材料，不执行系统动作。
        self.ops_case_service = ops_case_service
        # 大模型意图识别模型，通常用便宜的小模型或 workflow_finalize 模型。
        self.intent_model = intent_model
        # Hybrid 演示路由自己的轻量会话缓存。
        self.session_cache = session_cache
        # client_cache 是旧参数名，实际语义是接口层响应缓存，不是 Prompt Caching。
        self.response_cache = response_cache or client_cache
        # 轻量规则分类器，用来决定走哪条路径。
        self.classifier = get_classifier()

        self.simple_threshold = simple_threshold
        self.complex_threshold = complex_threshold

    # =========================================================================
    # 主入口
    # =========================================================================

    def process(
        self,
        order_id: str | None,
        question: str,
        filter_categories: list = None,
        thread_id: Optional[str] = None,
    ) -> HybridResult:
        """处理请求，根据意图复杂度动态选择路径。

        路由规则：
          CASUAL       → 普通聊天（不调用业务工具）
          SIMPLE       → Workflow（支持 HITL 中断）
          RAG          → RAG 知识检索
          MEDIUM       → Agent
          COMPLEX      → Agent
          MULTI_DOMAIN → Multi-Agent（fallback: Agent → Workflow）
        """
        import time

        start_time = time.time()

        if filter_categories is None:
            filter_categories = []
        normalized_order_id = (order_id or "").strip()

        # 第一步：短期结构化记忆。只记录运营偏好/约束/反馈/指代，不保存实时业务事实。
        conversation_turns = 0
        session_memory_payload: dict[str, Any] | None = None
        if thread_id and self.session_cache:
            session = self.session_cache.get_or_create_session(thread_id, normalized_order_id)
            conversation_turns = session.conversation_turns + 1
            session.conversation_turns = conversation_turns
            self.session_cache.set_session(session)
            snapshot = self.session_cache.update_from_user_message(
                thread_id=thread_id,
                order_id=normalized_order_id,
                message=question or "",
            )
            session_memory_payload = snapshot.model_dump(mode="json")

        # 第零步：检查响应缓存。这里只缓存低风险 completed 结果，不缓存 HITL/错误/高风险决策。
        cache_hit = False
        from_cache = False
        cache_context = self._cache_context(filter_categories, session_memory_payload)
        if self.response_cache:
            cached_entry = self.response_cache.get(normalized_order_id, question, cache_context=cache_context)
            if cached_entry:
                # 缓存命中时直接返回，不再走分类/Agent/Workflow。
                cache_hit = True
                from_cache = True
                execution_time_ms = (time.time() - start_time) * 1000
                add_trace_step(
                    step_type="router",
                    name="hybrid_response_cache",
                    status="success",
                    duration_ms=execution_time_ms,
                    summary="Hybrid 响应缓存命中",
                    metadata={"path_used": cached_entry.path_used, "cache_hit": True},
                )
                return HybridResult(
                    order_id=normalized_order_id,
                    question=question,
                    path_used=cached_entry.path_used,
                    intent_level="cached",
                    confidence=1.0,
                    final_answer=cached_entry.answer,
                    tools_called=cached_entry.tools_called,
                    execution_time_ms=execution_time_ms,
                    thread_id=thread_id,
                    conversation_turns=conversation_turns,
                    cache_hit=cache_hit,
                    from_cache=from_cache,
                    session_memory=session_memory_payload,
                )

        # 第二步：意图分类。优先大模型 JSON 路由，失败时回退规则分类器。
        classification = self._classify_intent(normalized_order_id, question or "")
        if not normalized_order_id and classification.level != IntentLevel.CASUAL:
            classification = IntentClassificationResult(
                level=IntentLevel.RAG,
                score=max(classification.score, 0.86),
                primary_keywords=classification.primary_keywords,
                reasoning=f"无订单号请求，跳过订单链路，按知识库 RAG 检索处理；原路由：{classification.level.value}",
            )
        add_trace_step(
            step_type="router",
            name="hybrid_router",
            status="success",
            summary=f"识别意图并路由为 {classification.level.value}",
            input_summary={"order_id": normalized_order_id or None, "question": question},
            metadata={
                "intent_level": classification.level.value,
                "confidence": classification.score,
                "reasoning": classification.reasoning,
                "keywords": classification.primary_keywords,
            },
        )

        # 第三步：根据分类选择路径
        if self._is_fulfillment_action_question(question or "") and normalized_order_id:
            path_result = self._run_workflow_with_hitl(
                normalized_order_id, question, filter_categories, thread_id, session_memory_payload
            )
        elif self._is_ops_case_question(question or "") and normalized_order_id and self.ops_case_service:
            path_result = self._run_ops_case_path(normalized_order_id, question)
        elif classification.level == IntentLevel.CASUAL:
            path_result = self._run_casual_path(question)
        elif classification.level == IntentLevel.MULTI_DOMAIN:
            # 跨领域复杂问题优先走多 Agent。
            path_result = self._run_multi_agent_path(normalized_order_id, question, thread_id)
        elif classification.level == IntentLevel.RAG:
            # 规则、SOP、政策依据类问题走 RAG。
            path_result = self._run_rag_path(normalized_order_id, question, filter_categories)
        elif classification.level == IntentLevel.COMPLEX:
            # 单域复杂问题走 ReAct Agent。
            path_result = self._run_agent_path(normalized_order_id, question)
        elif classification.level == IntentLevel.MEDIUM:
            path_result = self._run_agent_path(normalized_order_id, question)
        else:
            # SIMPLE → Workflow（带 HITL 支持）
            path_result = self._run_workflow_with_hitl(
                normalized_order_id, question, filter_categories, thread_id, session_memory_payload
            )
        add_trace_step(
            step_type="hybrid_path",
            name=path_result.get("path", "unknown"),
            status=path_result.get("status", "completed"),
            summary=f"Hybrid 选择链路：{path_result.get('path', 'unknown')}",
            metadata={
                "tools_called": path_result.get("tools_called", []),
                "has_interrupt": bool(path_result.get("interrupt")),
            },
        )

        execution_time_ms = (time.time() - start_time) * 1000
        action_card = path_result.get("action_card")
        rag_context = path_result.get("rag_context")
        if rag_context is None and isinstance(action_card, dict):
            decision_context = action_card.get("decision_context") or {}
            if isinstance(decision_context, dict):
                rag_context = decision_context.get("rag_context")

        # 把底层路径返回的 dict 包成统一 HybridResult。
        result = HybridResult(
            order_id=path_result.get("order_id") or normalized_order_id,
            question=question,
            path_used=path_result["path"],
            intent_level=classification.level.value,
            confidence=classification.score,
            final_answer=path_result.get("reply", ""),
            tools_called=path_result.get("tools_called", []),
            execution_time_ms=execution_time_ms,
            thread_id=thread_id,
            conversation_turns=conversation_turns,
            cache_hit=cache_hit,
            from_cache=from_cache,
            status=path_result.get("status", "completed"),
            interrupt=path_result.get("interrupt"),
            action_card=action_card,
            preflight_validation=path_result.get("preflight_validation"),
            intent_reasoning=classification.reasoning,
            intent_keywords=classification.primary_keywords,
            session_memory=session_memory_payload,
            rag_context=rag_context if isinstance(rag_context, dict) else None,
        )

        # 第四步：写入响应缓存。缓存服务内部会继续判断风险和状态。
        if self.response_cache:
            # set 返回值这里不强依赖，因为是否缓存由策略决定。
            self.response_cache.set(
                order_id=normalized_order_id,
                question=question,
                answer=path_result["reply"],
                path_used=path_result["path"],
                tools_called=path_result["tools_called"],
                status=result.status,
                execution_time_ms=execution_time_ms,
                cache_context=cache_context,
            )

        # 第五步：更新会话记忆
        if thread_id and self.session_cache:
            session = self.session_cache.get_session(thread_id)
            if session:
                # 只记录轻量路由信息，不保存完整回答，避免内存缓存膨胀。
                session.conversation_turns = conversation_turns
                session.order_analysis_cache = {
                    "path_used": path_result["path"],
                    "intent_level": classification.level.value,
                    "confidence": classification.score,
                    "intent_reasoning": classification.reasoning,
                }
                self.session_cache.set_session(session)
                self.session_cache.record_assistant_summary(thread_id, path_result.get("reply") or result.status)
                result.session_memory = self.session_cache.snapshot(thread_id).model_dump(mode="json")

        add_trace_step(
            step_type="evaluation",
            name="hybrid_quality_snapshot",
            status="success" if result.status != "error" else "error",
            summary="Hybrid 响应质量闭环快照",
            metadata={
                "path_used": result.path_used,
                "intent_level": result.intent_level,
                "confidence": result.confidence,
                "has_final_answer": bool(result.final_answer),
                "final_answer_length": len(result.final_answer or ""),
                "tools_called_count": len(result.tools_called),
                "status": result.status,
                "hitl_required": result.status == "interrupted",
                "from_cache": result.from_cache,
                "execution_time_ms": result.execution_time_ms,
                "session_memory_updated": bool(result.session_memory),
                "rag_context_ready": bool(result.rag_context),
            },
        )

        return result

    # =========================================================================
    # Resume（HITL 恢复）
    # =========================================================================

    def resume_workflow(
        self,
        thread_id: str,
        decision: str,
        notes: str = "",
    ) -> HybridResult:
        """恢复被中断的工作流。

        Args:
            thread_id: 中断时返回的线程 ID。
            decision: 人工决策（"approved" 或 "rejected"）。
            notes: 可选的备注信息。
        """
        import time

        start_time = time.time()

        # resume 只恢复 Workflow，因为 HITL 中断来自 Workflow LangGraph。
        from app.schemas.workflow import ApprovalRequest

        stream_result = self.workflow_service.resume(
            thread_id,
            ApprovalRequest(
                decision=decision,
                reason=notes or "人工审核未填写备注。",
                approver_id="frontend-user",
            ),
        )

        execution_time_ms = (time.time() - start_time) * 1000

        if stream_result["status"] == "interrupted":
            add_trace_step(
                step_type="workflow",
                name="fulfillment_workflow",
                status="interrupted",
                summary="Workflow 暂停，等待人工审核",
                metadata={
                    "thread_id": thread_id,
                    "interrupt": stream_result["interrupt"].model_dump()
                    if stream_result["interrupt"]
                    else None,
                },
            )
            # 如果恢复后又遇到新的中断，继续把 interrupt 透传给前端。
            return HybridResult(
                order_id="",
                question="",
                path_used="workflow",
                intent_level="simple",
                confidence=1.0,
                final_answer="",
                tools_called=[],
                execution_time_ms=execution_time_ms,
                thread_id=thread_id,
                status="interrupted",
                interrupt=stream_result["interrupt"].model_dump()
                if stream_result["interrupt"]
                else None,
                action_card=self._action_card_from_interrupt(stream_result["interrupt"]),
            )

        result = stream_result.get("result")
        if result:
            add_trace_step(
                step_type="workflow",
                name="fulfillment_workflow",
                status="success",
                summary="Workflow 执行完成",
                metadata={
                    "thread_id": thread_id,
                    "node_count": len(result.trace or []),
                    "errors": [str(item) for item in (result.errors or [])[:5]],
                },
            )
            # WorkflowRunResult 中 final_answer 是结构化对象，取 conclusion 给前端。
            final_answer = result.final_answer.conclusion if result.final_answer else ""
            action_card = self._action_card_from_result(result)
            preflight_validation = self._preflight_from_result(result)
        else:
            final_answer = stream_result.get("error", "Unknown error")
            action_card = None
            preflight_validation = None

        return HybridResult(
            order_id=result.order_id if result else "",
            question=result.question if result else "",
            path_used="workflow",
            intent_level="simple",
            confidence=1.0,
            final_answer=final_answer,
            tools_called=["dispatch", "order_analysis", "inventory_analysis", "finalize"],
            execution_time_ms=execution_time_ms,
            thread_id=thread_id,
            status=stream_result["status"],
            action_card=action_card,
            preflight_validation=preflight_validation,
        )

    # =========================================================================
    # 路径实现
    # =========================================================================

    def _classify_intent(self, order_id: str | None, question: str) -> IntentClassificationResult:
        """优先用大模型识别路由目标，失败后回退规则分类器。"""
        fallback = self.classifier.classify(question or "")
        if self.intent_model is None:
            return fallback

        prompt = (
            f"{get_model_gateway().prompt_system(use_case='hybrid_router')}\n\n"
            f"订单: {order_id}\n"
            f"用户问题: {question or '按默认履约问题分析'}\n"
        )
        try:
            response = self.intent_model.invoke(prompt)
            content = getattr(response, "content", response)
            payload = self._parse_intent_json(str(content))
            route = str(payload.get("route", "")).strip().lower()
            level_map = {
                "casual": IntentLevel.CASUAL,
                "chat": IntentLevel.CASUAL,
                "smalltalk": IntentLevel.CASUAL,
                "workflow": IntentLevel.SIMPLE,
                "rag": IntentLevel.RAG,
                "agent": IntentLevel.COMPLEX,
                "multi_agent": IntentLevel.MULTI_DOMAIN,
                "multi-agent": IntentLevel.MULTI_DOMAIN,
            }
            level = level_map.get(route)
            if level is None:
                return fallback
            confidence = float(payload.get("confidence", fallback.score))
            keywords = payload.get("keywords") or fallback.primary_keywords
            if not isinstance(keywords, list):
                keywords = fallback.primary_keywords
            reasoning = str(payload.get("reasoning") or "LLM intent routing")
            reconciled = self._reconcile_intent(
                llm_level=level,
                llm_confidence=confidence,
                llm_keywords=[str(item) for item in keywords[:5]],
                llm_reasoning=reasoning,
                fallback=fallback,
                question=question,
            )
            if reconciled is not None:
                return reconciled
            return IntentClassificationResult(
                level=level,
                score=max(0.0, min(confidence, 1.0)),
                primary_keywords=[str(item) for item in keywords[:5]],
                reasoning=f"LLM: {reasoning}",
            )
        except Exception:
            return fallback

    def _reconcile_intent(
        self,
        *,
        llm_level: IntentLevel,
        llm_confidence: float,
        llm_keywords: list[str],
        llm_reasoning: str,
        fallback: IntentClassificationResult,
        question: str,
    ) -> IntentClassificationResult | None:
        """融合 LLM 路由和规则强信号，避免单边误判。"""
        safe_confidence = max(0.0, min(float(llm_confidence), 1.0))
        text = (question or "").lower()

        # 日常问题必须保持轻量，不要因为前端带了 order_id 就进入履约 workflow。
        if fallback.level == IntentLevel.CASUAL and fallback.score >= 0.85:
            return fallback

        # 硬业务状态判断：模型误判为 agent/rag 时，仍优先 workflow。
        if self.classifier._is_fixed_workflow_question(text) and llm_level != IntentLevel.RAG:
            return IntentClassificationResult(
                level=IntentLevel.SIMPLE,
                score=max(0.9, safe_confidence),
                primary_keywords=fallback.primary_keywords or llm_keywords,
                reasoning=f"规则强纠偏：固定履约状态判断优先 Workflow；LLM: {llm_reasoning}",
            )

        # 明确规则/SOP/政策问题：模型误判为 workflow/agent 时，优先 RAG。
        if (
            self.classifier._is_explicit_rag_question(text)
            and not self.classifier._is_plan_or_advice_question(text)
            and llm_level != IntentLevel.MULTI_DOMAIN
        ):
            return IntentClassificationResult(
                level=IntentLevel.RAG,
                score=max(0.88, safe_confidence),
                primary_keywords=fallback.primary_keywords or llm_keywords,
                reasoning=f"规则强纠偏：明确询问规则/SOP/政策依据，优先 RAG；LLM: {llm_reasoning}",
            )

        # 多域强信号优先 Multi-Agent，尤其是模型保守判成 workflow 的时候。
        if fallback.level == IntentLevel.MULTI_DOMAIN and fallback.score >= 0.75:
            return IntentClassificationResult(
                level=IntentLevel.MULTI_DOMAIN,
                score=max(fallback.score, safe_confidence),
                primary_keywords=fallback.primary_keywords or llm_keywords,
                reasoning=f"规则强纠偏：命中多领域综合问题；{fallback.reasoning}",
            )

        # 低置信 LLM 不覆盖较强规则结论。
        if safe_confidence < 0.62 and fallback.score >= 0.75:
            return IntentClassificationResult(
                level=fallback.level,
                score=fallback.score,
                primary_keywords=fallback.primary_keywords,
                reasoning=f"低置信 LLM 回退规则：{fallback.reasoning}",
            )

        # LLM 把明显方案/建议类问题判成 workflow 时，使用规则侧的 Agent/Multi-Agent 信号。
        if (
            llm_level == IntentLevel.SIMPLE
            and fallback.level in {IntentLevel.COMPLEX, IntentLevel.MULTI_DOMAIN, IntentLevel.RAG}
            and fallback.score >= 0.7
        ):
            return IntentClassificationResult(
                level=fallback.level,
                score=fallback.score,
                primary_keywords=fallback.primary_keywords,
                reasoning=f"规则纠偏：LLM 判为 workflow，但问题命中 {fallback.level.value} 强信号；{fallback.reasoning}",
            )

        return None

    @staticmethod
    def _parse_intent_json(content: str) -> dict:
        """兼容模型返回 ```json ...``` 或前后带解释文本的情况。"""
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.S)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _run_async(coro):
        """在同步 FastAPI 路由里执行 Agent 的 async chat。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        if loop.is_running():
            raise RuntimeError("当前同步路由不应在已运行事件循环中调用 async Agent。")
        return loop.run_until_complete(coro)

    def _run_workflow_with_hitl(
        self,
        order_id: str,
        question: str,
        filter_categories: list,
        thread_id: Optional[str],
        session_memory: dict[str, Any] | None = None,
    ) -> dict:
        """运行 Workflow 路径（带 HITL 支持）。

        使用 run_stream() 捕获中断事件，在库存充足时暂停等待人工确认。
        """
        effective_thread_id = thread_id or f"wf-{order_id}"
        # WorkflowRunRequest 是 WorkflowService 的标准入参。
        request = WorkflowRunRequest(
            order_id=order_id,
            question=question,
            filter_categories=filter_categories,
            session_memory=session_memory or {},
        )

        try:
            # run_stream 会返回 completed/interrupted/error 三种状态。
            stream_result = self.workflow_service.run_stream(request, effective_thread_id)
            workflow_step = add_trace_step(
                step_type="workflow",
                name="fulfillment_workflow",
                status=stream_result.get("status", "unknown"),
                summary=f"Workflow run_stream 返回状态：{stream_result.get('status', 'unknown')}",
                metadata={
                    "thread_id": effective_thread_id,
                    "has_interrupt": bool(stream_result.get("interrupt")),
                    "has_result": bool(stream_result.get("result")),
                },
            )
        except Exception as exc:
            add_trace_step(
                step_type="workflow",
                name="fulfillment_workflow",
                status="error",
                summary="Workflow 执行抛出异常",
                error_code=exc.__class__.__name__,
                error_message=str(exc),
                metadata={"thread_id": effective_thread_id},
            )
            return {
                "path": "workflow",
                "reply": f"Workflow error: {exc}",
                "tools_called": [],
                "status": "error",
                "interrupt": None,
            }

        if stream_result["status"] == "interrupted":
            self._record_workflow_node_trace(
                stream_result.get("trace") or [],
                parent_id=workflow_step.id if workflow_step else None,
            )
            # HITL 中断时没有最终答案，前端需要展示 interrupt 等待人工处理。
            action_card = self._action_card_from_interrupt(stream_result["interrupt"])
            return {
                "path": "workflow",
                "reply": "",
                "tools_called": ["dispatch", "order_analysis", "inventory_analysis", "proposal_generation", "human_approval"],
                "status": "interrupted",
                "interrupt": stream_result["interrupt"].model_dump()
                if stream_result["interrupt"]
                else None,
                "action_card": action_card,
                "rag_context": self._rag_context_from_action_card(action_card),
            }

        result = stream_result.get("result")
        if result:
            self._record_workflow_node_trace(
                result.trace,
                parent_id=workflow_step.id if workflow_step else None,
            )
            # completed 时提取最终结论和实际工具链路。
            final_answer = result.final_answer.conclusion if result.final_answer else ""
            tools_called = [
                "dispatch",
                "order_analysis",
                "inventory_analysis",
                "proposal_generation",
                "human_approval",
            ]
            if result.final_answer and result.final_answer.routing_path == "knowledge_path":
                tools_called.append("knowledge_retrieval")
            tools_called.append("finalize")
            action_card = self._action_card_from_result(result)
            return {
                "path": "workflow",
                "reply": final_answer,
                "tools_called": tools_called,
                "status": "completed",
                "interrupt": None,
                "action_card": action_card,
                "preflight_validation": self._preflight_from_result(result),
                "rag_context": self._rag_context_from_action_card(action_card),
            }

        return {
            "path": "workflow",
            "reply": stream_result.get("error", "Unknown error"),
            "tools_called": [],
            "status": "error",
            "interrupt": None,
            }

    @staticmethod
    def _action_card_from_interrupt(interrupt) -> dict[str, Any] | None:
        if not interrupt:
            return None
        context = getattr(interrupt, "context", {}) or {}
        proposal = context.get("proposal") if isinstance(context, dict) else None
        return proposal if isinstance(proposal, dict) else None

    @staticmethod
    def _action_card_from_result(result) -> dict[str, Any] | None:
        proposal = getattr(result, "execution_proposal", None)
        if proposal is None and getattr(result, "final_answer", None) is not None:
            proposal = getattr(result.final_answer, "execution_proposal", None)
        if proposal is None:
            return None
        return proposal.model_dump(mode="json") if hasattr(proposal, "model_dump") else proposal

    @staticmethod
    def _preflight_from_result(result) -> dict[str, Any] | None:
        validation = getattr(result, "preflight_validation", None)
        if validation is None and getattr(result, "final_answer", None) is not None:
            validation = getattr(result.final_answer, "preflight_validation", None)
        if validation is None:
            return None
        return validation.model_dump(mode="json") if hasattr(validation, "model_dump") else validation

    @staticmethod
    def _rag_context_from_action_card(action_card: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(action_card, dict):
            return None
        decision_context = action_card.get("decision_context") or {}
        if not isinstance(decision_context, dict):
            return None
        rag_context = decision_context.get("rag_context")
        return rag_context if isinstance(rag_context, dict) else None

    @staticmethod
    def _record_workflow_node_trace(trace_events: list, parent_id: str | None = None) -> None:
        """把 Workflow 内部节点 trace 展开写入业务 Trace。

        WorkflowRunResult.trace 是 LangGraph state 内的节点级轨迹；BusinessTrace
        是前端 Trace Center 使用的统一轨迹。这里做一次桥接，避免 Hybrid 页面
        只能看到一个折叠的 fulfillment_workflow 步骤。
        """
        status_map = {"ok": "success", "error": "error", "skipped": "skipped"}
        for event in trace_events or []:
            status = status_map.get(getattr(event, "status", ""), "success")
            add_trace_step(
                step_type="workflow_node",
                name=getattr(event, "node", "workflow_node"),
                parent_id=parent_id,
                status=status,
                duration_ms=float(getattr(event, "elapsed_ms", 0) or 0),
                summary=getattr(event, "note", "") or "",
                started_at=getattr(event, "start_ts", None).isoformat()
                if getattr(event, "start_ts", None)
                else None,
                ended_at=getattr(event, "end_ts", None).isoformat()
                if getattr(event, "end_ts", None)
                else None,
                metadata={"source": "workflow_state_trace"},
            )

    def _run_rag_path(self, order_id: str | None, question: str, filter_categories: list) -> dict:
        """运行 RAG 路径，专门回答规则、SOP、政策依据类问题。"""
        try:
            result = self.knowledge_service.retrieve(
                order_id=order_id,
                question=question or "请检索当前订单相关的履约规则。",
                filter_categories=filter_categories or [],
            )
            summary = result.answer_summary
            parts = [summary.conclusion or result.summary]
            if summary.key_rules:
                parts.append("关键规则：\n" + "\n".join(f"- {item}" for item in summary.key_rules[:5]))
            if summary.suggested_actions:
                parts.append("建议动作：\n" + "\n".join(f"- {item}" for item in summary.suggested_actions[:5]))
            return {
                "path": "rag",
                "order_id": result.order_id,
                "reply": "\n\n".join(parts),
                "tools_called": ["knowledge_retrieval", "hybrid_rag"],
                "status": "completed",
                "interrupt": None,
            }
        except Exception as exc:
            return {
                "path": "rag",
                "reply": f"RAG error: {exc}",
                "tools_called": ["knowledge_retrieval"],
                "status": "error",
                "interrupt": None,
            }

    def _run_casual_path(self, question: str) -> dict:
        """运行普通闲聊路径，不调用订单、库存、RAG 或 Agent 工具。"""
        if self.intent_model is None:
            reply = "你好，我是电商履约运营智能协同 Agent。你可以问我订单库存、缺货规则、履约方案或风险评估。"
        else:
            prompt = (
                f"{get_model_gateway().prompt_system(use_case='casual_chat')}\n\n"
                f"用户：{question or '你好'}"
            )
            try:
                response = self.intent_model.invoke(prompt)
                reply = str(getattr(response, "content", response)).strip()
            except Exception:
                reply = "你好，我是电商履约运营智能协同 Agent。你可以问我订单库存、缺货规则、履约方案或风险评估。"
        return {
            "path": "chat",
            "reply": reply,
            "tools_called": [],
            "status": "completed",
            "interrupt": None,
        }

    @staticmethod
    def _is_fulfillment_action_question(question: str) -> bool:
        """识别需要生成可审批执行提案的问题。"""
        text = question.lower()
        action_markers = [
            "执行提案",
            "action card",
            "操作卡片",
            "批准",
            "换仓",
            "拆单",
            "合单",
            "物流渠道",
            "渠道变更",
            "库存调拨",
            "跨仓调货",
            "缺货处置",
            "二次校验",
            "preflight",
        ]
        return any(marker in text for marker in action_markers)

    @staticmethod
    def _is_ops_case_question(question: str) -> bool:
        """识别前端运营案件 Copilot 问题。"""
        text = question.lower()
        markers = [
            "运营异常案件",
            "案件摘要",
            "异常案件",
            "分流",
            "沟通草稿",
            "客服沟通",
            "复盘",
            "商业影响",
            "sop 适配",
            "sop适配",
            "不要替 oms",
            "不要替 oms/wms/tms/erp",
        ]
        return any(marker in text for marker in markers)

    def _run_ops_case_path(self, order_id: str, question: str) -> dict:
        """运行运营异常案件路径。

        这条路径只组织事实、SOP 和运营建议，不执行订单/库存/物流动作。
        """
        if self.ops_case_service is None:
            return self._run_agent_path(order_id, question)
        try:
            result = self.ops_case_service.analyze(order_id=order_id, question=question)
            return {
                "path": "ops_case",
                "order_id": result.order_id,
                "reply": result.to_markdown(),
                "tools_called": [
                    "order_analysis",
                    "inventory_analysis",
                    "knowledge_retrieval",
                    "ops_case_analysis",
                ],
                "status": "completed",
                "interrupt": None,
            }
        except Exception as exc:
            return {
                "path": "ops_case",
                "reply": f"Ops case error: {exc}",
                "tools_called": ["ops_case_analysis"],
                "status": "error",
                "interrupt": None,
            }

    def _run_agent_path(self, order_id: str, question: str) -> dict:
        """运行 Agent 路径。"""
        if not order_id:
            return self._run_rag_path(order_id, question, [])
        if self.agent_service is None:
            return self._run_workflow_fallback(order_id, question, [])
        try:
            # Hybrid 内部给 Agent 一个稳定 session_id，方便短期记忆按订单隔离。
            agent_message = f"当前订单 ID：{order_id}\n运营问题：{question}"
            result = self._run_async(self.agent_service.chat(
                session_id=f"hybrid-{order_id}",
                message=agent_message,
                include_trace=False,
            ))
            return {
                "path": "agent",
                "reply": result["reply"],
                "tools_called": result["tools_called"],
                "status": "completed",
                "interrupt": None,
            }
        except Exception:
            # Agent 失败降级到 Workflow（无 HITL）
            return self._run_workflow_fallback(order_id, question, [])

    def _run_multi_agent_path(
        self,
        order_id: str,
        question: str,
        thread_id: Optional[str],
    ) -> dict:
        """运行 Multi-Agent 路径（Supervisor + 专业 Agent 协同）。

        不可用时依次降级：Multi-Agent → Agent → Workflow。
        """
        if not order_id:
            return self._run_rag_path(order_id, question, [])
        if self.multi_agent_service is None:
            # 没注入 MultiAgentService 时自动降级到单 Agent。
            return self._run_agent_path(order_id, question)

        try:
            # MultiAgentService 返回 dataclass，这里转成 Hybrid path_result dict。
            result = self.multi_agent_service.run(order_id=order_id, question=question)
            return {
                "path": "multi_agent",
                "reply": result.final_answer,
                "tools_called": result.agents_called,
                "status": "completed",
                "interrupt": None,
            }
        except Exception:
            # Multi-Agent 失败继续降级到单 Agent。
            return self._run_agent_path(order_id, question)

    def _run_workflow_fallback(
        self, order_id: str, question: str, filter_categories: list
    ) -> dict:
        """Agent 降级时的 Workflow 兜底（同步模式，无 HITL）。"""
        try:
            # fallback 使用同步 run，不处理 HITL 中断，避免 Agent fallback 链路过复杂。
            request = WorkflowRunRequest(
                order_id=order_id,
                question=question,
                filter_categories=filter_categories,
            )
            result = self.workflow_service.run(request)
            final_answer = (
                result.final_answer.conclusion if result.final_answer else ""
            )
            tools_called = [
                "dispatch",
                "order_analysis",
                "inventory_analysis",
                "proposal_generation",
                "human_approval",
            ]
            if result.final_answer and result.final_answer.routing_path == "knowledge_path":
                tools_called.append("knowledge_retrieval")
            tools_called.append("finalize")
            action_card = self._action_card_from_result(result)
            return {
                "path": "workflow",
                "reply": final_answer,
                "tools_called": tools_called,
                "status": "completed",
                "interrupt": None,
                "action_card": action_card,
                "preflight_validation": self._preflight_from_result(result),
                "rag_context": self._rag_context_from_action_card(action_card),
            }
        except Exception as exc:
            return {
                "path": "workflow",
                "reply": f"Error: {exc}",
                "tools_called": [],
                "status": "error",
                "interrupt": None,
            }

    # =========================================================================
    # 统计
    # =========================================================================

    def get_routing_stats(self) -> dict:
        """获取路由统计。"""
        return {
            "simple_threshold": self.simple_threshold,
            "complex_threshold": self.complex_threshold,
            "available_paths": [
                "chat",
                "workflow",
                "rag",
                "ops_case",
                "agent",
                *(["multi_agent"] if self.multi_agent_service else []),
            ],
            "description": "5-way routing: Chat + Workflow + RAG + Agent + Multi-Agent (with HITL)",
        }

    @staticmethod
    def _cache_context(filter_categories: list | None, session_memory: dict[str, Any] | None = None) -> str:
        """构造响应缓存上下文。

        真实生产里这里应接订单更新时间、库存快照版本、知识库版本和模型 ID。
        当前项目先把知识类别过滤和短期记忆摘要纳入 key，避免不同上下文复用同一答案。
        """
        parts: list[str] = []
        if filter_categories:
            parts.append("filters:" + ",".join(sorted(str(item) for item in filter_categories)))
        if session_memory:
            structured = session_memory.get("structured", {})
            digest = hashlib.sha256(
                json.dumps(structured, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            parts.append(f"memory:{digest}")
        return "|".join(parts)
