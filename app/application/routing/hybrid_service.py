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
import json
import re
from dataclasses import dataclass
from typing import Optional

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
    SessionContext,
    SessionMemoryService,
)
from app.application.workflow.workflow_service import WorkflowService


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
    path_used: str  # "workflow" | "agent" | "rag" | "multi_agent"
    # 意图分类等级。
    intent_level: str  # "simple" | "rag" | "medium" | "complex" | "multi_domain"
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
    # 路由解释，给前端展示“为什么进这条链路”。
    intent_reasoning: str = ""
    intent_keywords: list[str] | None = None

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
        order_id: str,
        question: str,
        filter_categories: list = None,
        thread_id: Optional[str] = None,
    ) -> HybridResult:
        """处理请求，根据意图复杂度动态选择路径。

        路由规则：
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

        # 第零步：检查响应缓存。这里只缓存低风险 completed 结果，不缓存 HITL/错误/高风险决策。
        cache_hit = False
        from_cache = False
        cache_context = self._cache_context(filter_categories)
        if self.response_cache:
            cached_entry = self.response_cache.get(order_id, question, cache_context=cache_context)
            if cached_entry:
                # 缓存命中时直接返回，不再走分类/Agent/Workflow。
                cache_hit = True
                from_cache = True
                execution_time_ms = (time.time() - start_time) * 1000
                return HybridResult(
                    order_id=order_id,
                    question=question,
                    path_used=cached_entry.path_used,
                    intent_level="cached",
                    confidence=1.0,
                    final_answer=cached_entry.answer,
                    tools_called=cached_entry.tools_called,
                    execution_time_ms=execution_time_ms,
                    thread_id=thread_id,
                    cache_hit=cache_hit,
                    from_cache=from_cache,
                )

        # 第一步：会话上下文
        conversation_turns = 0
        if thread_id and self.session_cache:
            session = self.session_cache.get_session(thread_id)
            if session:
                # 已有会话：轮次 +1。
                conversation_turns = session.conversation_turns + 1
            else:
                # 新会话：创建 SessionContext。
                session = SessionContext(thread_id=thread_id, order_id=order_id)
                self.session_cache.set_session(session)
                conversation_turns = 1

        # 第二步：意图分类。优先大模型 JSON 路由，失败时回退规则分类器。
        classification = self._classify_intent(order_id, question or "")

        # 第三步：根据分类选择路径
        if classification.level == IntentLevel.MULTI_DOMAIN:
            # 跨领域复杂问题优先走多 Agent。
            path_result = self._run_multi_agent_path(order_id, question, thread_id)
        elif classification.level == IntentLevel.RAG:
            # 规则、SOP、政策依据类问题走 RAG。
            path_result = self._run_rag_path(order_id, question, filter_categories)
        elif classification.level == IntentLevel.COMPLEX:
            # 单域复杂问题走 ReAct Agent。
            path_result = self._run_agent_path(order_id, question)
        elif classification.level == IntentLevel.MEDIUM:
            path_result = self._run_agent_path(order_id, question)
        else:
            # SIMPLE → Workflow（带 HITL 支持）
            path_result = self._run_workflow_with_hitl(
                order_id, question, filter_categories, thread_id
            )

        execution_time_ms = (time.time() - start_time) * 1000

        # 把底层路径返回的 dict 包成统一 HybridResult。
        result = HybridResult(
            order_id=order_id,
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
            intent_reasoning=classification.reasoning,
            intent_keywords=classification.primary_keywords,
        )

        # 第四步：写入响应缓存。缓存服务内部会继续判断风险和状态。
        if self.response_cache:
            # set 返回值这里不强依赖，因为是否缓存由策略决定。
            self.response_cache.set(
                order_id=order_id,
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
        stream_result = self.workflow_service.resume(thread_id, decision)

        execution_time_ms = (time.time() - start_time) * 1000

        if stream_result["status"] == "interrupted":
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
            )

        result = stream_result.get("result")
        if result:
            # WorkflowRunResult 中 final_answer 是结构化对象，取 conclusion 给前端。
            final_answer = result.final_answer.conclusion if result.final_answer else ""
        else:
            final_answer = stream_result.get("error", "Unknown error")

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
        )

    # =========================================================================
    # 路径实现
    # =========================================================================

    def _classify_intent(self, order_id: str, question: str) -> IntentClassificationResult:
        """优先用大模型识别路由目标，失败后回退规则分类器。"""
        fallback = self.classifier.classify(question or "")
        if self.intent_model is None:
            return fallback

        prompt = f"""你是电商供应链系统的意图路由器。请只输出 JSON，不要解释。

可选 route:
- workflow: 固定履约判断、库存是否可发、订单是否满足履约条件、需要可审计流程/HITL。
- rag: 规则、SOP、政策、知识库依据、售后制度、缺货/跨仓/优先级规则问答。
- agent: 开放式分析、需要自主调用工具、替代品、履约建议、单域多步骤推理。
- multi_agent: 同时涉及库存、履约、风险、成本、时效、仓库协同等多个领域的复杂综合问题。

订单: {order_id}
用户问题: {question or "按默认履约问题分析"}

输出格式:
{{"route":"workflow|rag|agent|multi_agent","confidence":0.0到1.0,"reasoning":"一句话理由","keywords":["关键词1","关键词2"]}}
"""
        try:
            response = self.intent_model.invoke(prompt)
            content = getattr(response, "content", response)
            payload = self._parse_intent_json(str(content))
            route = str(payload.get("route", "")).strip().lower()
            level_map = {
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
            # 大模型有时会把“生成方案/综合建议”过度保守地判成 workflow。
            # 如果规则分类器对 RAG / Agent / Multi-Agent 有较强信号，就用规则结果纠偏。
            if (
                level == IntentLevel.SIMPLE
                and fallback.level != IntentLevel.SIMPLE
                and fallback.score >= 0.7
            ):
                return IntentClassificationResult(
                    level=fallback.level,
                    score=fallback.score,
                    primary_keywords=fallback.primary_keywords,
                    reasoning=f"规则纠偏：LLM 判为 workflow，但问题命中 {fallback.level.value} 强信号；{fallback.reasoning}",
                )
            return IntentClassificationResult(
                level=level,
                score=max(0.0, min(confidence, 1.0)),
                primary_keywords=[str(item) for item in keywords[:5]],
                reasoning=f"LLM: {reasoning}",
            )
        except Exception:
            return fallback

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
        )

        try:
            # run_stream 会返回 completed/interrupted/error 三种状态。
            stream_result = self.workflow_service.run_stream(request, effective_thread_id)
        except Exception as exc:
            return {
                "path": "workflow",
                "reply": f"Workflow error: {exc}",
                "tools_called": [],
                "status": "error",
                "interrupt": None,
            }

        if stream_result["status"] == "interrupted":
            # HITL 中断时没有最终答案，前端需要展示 interrupt 等待人工处理。
            return {
                "path": "workflow",
                "reply": "",
                "tools_called": ["dispatch", "order_analysis", "inventory_analysis"],
                "status": "interrupted",
                "interrupt": stream_result["interrupt"].model_dump()
                if stream_result["interrupt"]
                else None,
            }

        result = stream_result.get("result")
        if result:
            # completed 时提取最终结论和实际工具链路。
            final_answer = result.final_answer.conclusion if result.final_answer else ""
            tools_called = ["dispatch", "order_analysis", "inventory_analysis"]
            if result.final_answer and result.final_answer.routing_path == "knowledge_path":
                tools_called.append("knowledge_retrieval")
            tools_called.append("finalize")
            return {
                "path": "workflow",
                "reply": final_answer,
                "tools_called": tools_called,
                "status": "completed",
                "interrupt": None,
            }

        return {
            "path": "workflow",
            "reply": stream_result.get("error", "Unknown error"),
            "tools_called": [],
            "status": "error",
            "interrupt": None,
            }

    def _run_rag_path(self, order_id: str, question: str, filter_categories: list) -> dict:
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

    def _run_agent_path(self, order_id: str, question: str) -> dict:
        """运行 Agent 路径。"""
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
            tools_called = ["dispatch", "order_analysis", "inventory_analysis"]
            if result.final_answer and result.final_answer.routing_path == "knowledge_path":
                tools_called.append("knowledge_retrieval")
            tools_called.append("finalize")
            return {
                "path": "workflow",
                "reply": final_answer,
                "tools_called": tools_called,
                "status": "completed",
                "interrupt": None,
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
                "workflow",
                "rag",
                "agent",
                *(["multi_agent"] if self.multi_agent_service else []),
            ],
            "description": "4-way routing: Workflow + RAG + Agent + Multi-Agent (with HITL)",
        }

    @staticmethod
    def _cache_context(filter_categories: list | None) -> str:
        """构造响应缓存上下文。

        真实生产里这里应接订单更新时间、库存快照版本、知识库版本和模型 ID。
        当前项目先把知识类别过滤纳入 key，避免不同过滤条件复用同一答案。
        """
        if not filter_categories:
            return ""
        return "filters:" + ",".join(sorted(str(item) for item in filter_categories))
