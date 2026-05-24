"""Agent 外观服务（学习版注释）。

这个文件可以理解成“业务系统和 LangChain Agent 之间的适配层”。

它不负责自己实现 ReAct 循环。模型什么时候调用工具、工具结果如何回到模型、
模型什么时候生成最终回答，这些都交给 ``langchain.agents.create_agent`` 生成的
Agent 图来完成。

这一层负责的是工程化编排：
  1. 把订单、库存、RAG 等业务服务包装成 LangChain Tool。
  2. 给工具统一加超时、重试、缓存和异常兜底。
  3. 接入 LangGraph checkpointer，保存会话内短期记忆。
  4. 接入长期记忆 store，只沉淀可复用的业务偏好和订单决策。
  5. 接入 Langfuse，把关键调用链路上报到观测平台。
  6. 把 LangChain 返回的消息历史整理成 API 需要的 reply、trace、decision。

学习时建议按这个顺序读：
  - ``__init__``：看 Agent 需要哪些依赖，以及短期/长期记忆怎么接进去。
  - ``chat``：普通非流式接口的一轮完整主线。
  - ``stream_chat``：流式接口如何把 LangChain chunk 转成前端事件。
  - ``_remember_turn`` / ``_search_memories``：长期记忆的写入和召回策略。
  - ``_extract_reply_and_tools``：如何从 LangChain 消息历史还原工程 trace。
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage, filter_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from app.agent.agent import ReflectiveAgentRunner, build_agent
from app.agent.checkpointer import create_checkpointer
from app.agent.context_manager import ContextWindowConfig, ContextWindowManager
from app.agent.evaluation.langfuse_tracer import LangfuseTracer, score_after_run
from app.agent.tool_wrapper import wrap_all_tools
from app.agent.tools import make_inventory_tool, make_knowledge_tool, make_order_tool
from app.memory import MemoryGovernanceService
from app.schemas.agent import (
    AgentExecutionTrace,
    FulfillmentDecision,
    ReActCycle,
    ReflectionInfo,
    ToolCallDetail,
)
from app.services.inventory_analysis_service import InventoryAnalysisService
from app.services.knowledge_retrieval_service import KnowledgeRetrievalService
from app.services.order_analysis_service import OrderAnalysisService


CATALOG_TOOL_NAMES = {"find_substitute_sku"}

# 只有命中这些业务信号时，普通对话才值得写入长期记忆。
# 目的不是“什么都记”，而是沉淀后续真的可能复用的履约偏好、风险结论和订单决策。
MEMORY_SIGNAL_KEYWORDS = (
    "缺货",
    "替代",
    "人工复核",
    "优先级",
    "跨仓",
    "调拨",
    "履约方案",
    "风险",
    "售后",
    "拆单",
    "合单",
    "延期",
    "承诺",
)

# 这些短句通常只是寒暄或确认，不应该污染长期记忆。
LOW_VALUE_MESSAGE_PATTERNS = (
    "你好",
    "您好",
    "谢谢",
    "感谢",
    "好的",
    "收到",
)

SENSITIVE_TEXT_PATTERNS = (
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"\b1[3-9]\d{9}\b"), "[PHONE]"),
    (re.compile(r"\b\d{17}[\dXx]\b"), "[ID_CARD]"),
)


@dataclass(frozen=True)
class MemoryWriteCandidate:
    """一次 Agent 轮次里准备写入长期记忆的候选项。

    企业级记忆写入不应该是“回答完就存”。先生成候选项，再按类型、业务信号、
    工具证据和隐私规则决定是否入库，能避免长期记忆被寒暄、重复结论和敏感信息污染。
    """

    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any]
    score: float
    reason: str

# 这里没有让主 Agent 一开始就强制输出结构化对象。
# 原因：
# 1. ReAct Agent 的自然语言回复对用户更友好，也方便展示工具分析结果。
# 2. 结构化对象主要给前端/系统做后续处理，不一定适合作为用户看到的第一响应。
# 所以流程是：Agent 先正常回答，再用一个小的 LCEL chain 把回答抽取成 Pydantic 模型。
_STRUCT_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是供应链履约决策分析器。根据以下 Agent 分析结果，提取结构化决策信息。"
        "只基于提供的文本内容，不要添加未提到的信息。",
    ),
    ("human", "问题：{question}\n\nAgent 分析结果：{agent_reply}"),
])


class AgentNotAvailableError(Exception):
    """未配置真实 LLM 时，Agent 无法初始化。

    Agent 主链路必须有 chat_model；如果没模型，应该在服务创建阶段就失败，
    而不是等用户请求进来后才报错。
    """


def _json_line(payload: dict) -> str:
    """流式接口每次 yield 一行 JSON，前端可以逐行解析。"""
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _tool_call_name(tool_call: Any) -> str:
    """兼容 LangChain 返回 dict 或对象两种 tool_call 表示。"""
    if isinstance(tool_call, dict):
        return str(tool_call.get("name") or "")
    return str(getattr(tool_call, "name", "") or "")


def _tool_call_args(tool_call: Any) -> dict:
    """从 tool_call 中取参数，trace 里会展示这些参数。"""
    if isinstance(tool_call, dict):
        return tool_call.get("args") or {}
    return getattr(tool_call, "args", {}) or {}


def _split_by_cache_policy(tools: list[BaseTool]) -> tuple[list[BaseTool], list[BaseTool]]:
    """按数据新鲜度拆工具。

    面试里可以这样解释：缓存不是越多越好。
    - 库存、订单、履约方案依赖实时状态，不能缓存。
    - 替代 SKU、知识规则更像目录数据，短时间缓存能减少重复调用。
    """
    realtime, catalog = [], []
    for tool in tools:
        (catalog if tool.name in CATALOG_TOOL_NAMES else realtime).append(tool)
    return realtime, catalog


class AgentService:
    """生产级 ReAct Agent 服务。

    LangChain 负责 Agent 循环、工具调用、结构化输出和 callbacks；本类负责业务侧
    的依赖装配、缓存策略、上下文裁剪和响应格式。
    """

    def __init__(
        self,
        order_service: OrderAnalysisService,
        inventory_service: InventoryAnalysisService,
        knowledge_service: KnowledgeRetrievalService,
        chat_model: object,
        extra_tools: list[BaseTool] | None = None,
        context_config: ContextWindowConfig | None = None,
        tool_timeout_seconds: float = 10.0,
        tool_max_retries: int = 2,
        enable_reflection: bool = False,
        reflection_threshold: float = 0.6,
        max_reflection_retries: int = 2,
        enable_structured_output: bool = True,
        structured_output_model: object | None = None,
        langfuse_tracer: Optional[LangfuseTracer] = None,
    ) -> None:
        """完成 Agent 的一次性装配。

        可以把这里拆成六步理解：
        1. 校验模型：没有真实 chat_model 时，Agent 主链路不能工作。
        2. 创建业务工具：把订单、库存、知识库服务包装成 LangChain Tool。
        3. 包装工具弹性：超时、重试、缓存、异常净化都在工具外层处理。
        4. 创建记忆组件：checkpointer 管短期对话历史，store 管长期业务记忆。
        5. 编译 Agent 图：``build_agent`` 内部会调用 LangChain/LangGraph 能力。
        6. 准备增强能力：反思质量门、结构化抽取、Langfuse 追踪。
        """
        if chat_model is None:
            raise AgentNotAvailableError(
                "AgentService 需要真实的 chat_model，"
                "请配置模型网关 config/model_gateway.yaml 和对应的后端 API Key。"
            )

        # ContextWindowManager 只负责“历史太长时怎么裁剪”。
        # 它不会自己保存历史；真正的消息历史由 LangGraph checkpointer 保存。
        self._context_manager = ContextWindowManager(context_config, llm=chat_model)
        self._langfuse_tracer = langfuse_tracer

        # 基础工具来自业务 service。Agent 最终看到的是 LangChain Tool，
        # 不是直接访问 repository 或数据库。
        base_realtime_tools = [
            make_order_tool(order_service),
            make_inventory_tool(inventory_service),
        ]
        base_catalog_tools = [make_knowledge_tool(knowledge_service)]
        extra_realtime_tools, extra_catalog_tools = _split_by_cache_policy(extra_tools or [])

        # Agent 看到的是包装后的工具。包装层不改变工具语义，只处理工程问题：
        # 超时、重试、缓存、异常转成可读字符串，避免一次工具失败拖垮整轮对话。
        resilient_tools = self._wrap_tools(
            realtime_tools=base_realtime_tools + extra_realtime_tools,
            catalog_tools=base_catalog_tools + extra_catalog_tools,
            max_retries=tool_max_retries,
            timeout_seconds=tool_timeout_seconds,
        )

        from app.core.config import get_settings
        from app.memory import create_long_term_memory_store

        settings = get_settings()

        # 短期记忆：保存“当前会话内”的完整消息历史。
        # 同一个 session_id 会映射到 LangGraph config.configurable.thread_id。
        # Redis 可用时持久化到 Redis；不可用时降级为 MemorySaver。
        checkpointer = create_checkpointer(
            settings.redis_url,
            ttl_seconds=settings.short_term_memory_ttl_seconds,
        )

        # 长期记忆：保存跨会话仍有价值的信息，例如用户偏好、订单处理结论。
        # 如果后端是 MySQL + Milvus，这里传入懒加载 embedding 代理，
        # 避免 FastAPI 启动时马上加载本地 embedding 权重。
        memory_embed_model = None
        long_term_backend = settings.long_term_memory_backend.strip().lower()
        if long_term_backend in {"mysql_milvus", "mysql+milvus", "mysql", "milvus"}:
            from app.graph.embed_adapter import create_lazy_embed_model

            memory_embed_model = create_lazy_embed_model(settings)
        self._memory_store = create_long_term_memory_store(
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
        )
        self._memory_governance = MemoryGovernanceService()

        # LangChain 的 checkpointer 按 thread_id 保存消息历史。
        # 所以后面每次调用只要传同一个 session_id，Agent 就能记住前几轮上下文。
        # store 则是 LangGraph 的长期记忆接口，本项目用它挂 SQLite / MySQL + Milvus 实现。
        self._agent = build_agent(
            chat_model,
            resilient_tools,
            checkpointer=checkpointer,
            store=self._memory_store,
        )

        # 反思不是 LangChain 原生 ReAct 的必要部分，而是本项目额外加的一层质量门。
        # 关闭时就是标准 ReAct；开启后会在低质量答案上自动重试。
        self._reflective_runner = (
            ReflectiveAgentRunner(
                agent=self._agent,
                max_retries=max_reflection_retries,
                threshold=reflection_threshold,
            )
            if enable_reflection
            else None
        )
        extraction_model = structured_output_model or chat_model
        self._struct_extractor = (
            self._make_structured_extractor(extraction_model) if enable_structured_output else None
        )

    async def chat(self, session_id: str, message: str, include_trace: bool = True) -> dict:
        """执行一轮非流式 Agent 对话。

        主线可以记成：
        1. 组装 LangChain config，其中 thread_id=session_id。
        2. 召回长期记忆，把相关偏好/历史决策注入当前问题。
        3. 裁剪短期消息历史，防止 prompt 无限膨胀。
        4. 运行 Agent，得到自然语言回复和工具调用记录。
        5. 可选抽取结构化履约决策。
        6. 写入长期记忆、提交 Langfuse 评分、构造 trace。
        """
        callbacks = self._callbacks()
        config = self._run_config(session_id, callbacks)

        # agent_message 是“当前用户问题 + 召回到的长期记忆”。
        # 原始 message 仍然保留，用于结构化抽取、记忆写入和 trace 展示。
        agent_message = self._message_with_memory(session_id, message)
        span_context = (
            self._langfuse_tracer.start_span(
                name="agent.chat",
                input={"session_id": session_id, "message": message},
                metadata={"include_trace": include_trace},
            )
            if self._langfuse_tracer
            else None
        )

        with (span_context if span_context is not None else nullcontext()) as langfuse_span:
            # 裁剪发生在调用 Agent 前。否则历史消息太长，会让 prompt 越来越贵，
            # 也可能超过模型上下文窗口。
            context_trimmed = self._maybe_trim_context(session_id)

            reply, tools_called, tool_call_details, reflection_info = await self._run_turn(
                session_id=session_id,
                message=agent_message,
                config=config,
                callbacks=callbacks,
            )

            # 结构化抽取失败不影响主回答，所以 _extract_decision 内部会吞掉异常并返回 None。
            decision = await self._extract_decision(message, reply)

            # 长期记忆只记录有复用价值的信息；寒暄、普通短确认会被策略过滤。
            self._remember_turn(session_id, message, reply, tools_called)
            self._score_langfuse(reply, tools_called, reflection_info)

            trace = (
                self._build_trace(session_id, message, reply, tools_called, tool_call_details)
                if include_trace
                else None
            )
            if langfuse_span is not None:
                try:
                    langfuse_span.update(
                        output={"reply": reply, "tools_called": tools_called},
                        metadata={
                            "context_trimmed": context_trimmed,
                            "reflection_score": reflection_info.score if reflection_info else None,
                        },
                    )
                except Exception:
                    pass
            return {
                "reply": reply,
                "tools_called": tools_called,
                "trace": trace,
                "context_trimmed": context_trimmed,
                "decision": decision,
                "reflection": reflection_info,
            }

    async def stream_chat(self, session_id: str, message: str) -> AsyncIterator[str]:
        """以 JSON line 形式流式返回 token、工具调用和最终指标。

        ``stream_mode="messages"`` 是 LangGraph/LangChain 的消息流模式。
        它会把模型 token、工具调用片段、工具结果都作为 message chunk 推出来。
        这里把它们翻译成前端更容易消费的事件：
        ``token``、``tool_call``、``tool_result``、``done``。

        注意：流式接口无法像普通 chat 一样一开始就知道完整 reply。
        所以这里边接收 token 边拼接 full_reply，最后在 done 事件里统一返回。
        """
        callbacks = self._callbacks()
        config = self._run_config(session_id, callbacks)
        agent_message = self._message_with_memory(session_id, message)

        full_reply = ""
        tools_in_turn: list[str] = []
        reflection_info: Optional[ReflectionInfo] = None

        if self._reflective_runner:
            # 反思评分必须先拿到完整回复才能算。如果在流式接口前置执行，
            # 前端仍然要等完整 Agent 跑完，首字响应会变慢。
            # 所以流式模式优先保证交互体验：跳过前置反思，普通 /chat 仍保留质量门。
            yield _json_line({
                "type": "status",
                "message": "流式模式已跳过前置反思，优先返回实时输出。",
            })

        try:
            async for chunk, metadata in self._agent.astream(
                {"messages": [HumanMessage(content=agent_message)]},
                config=config,
                stream_mode="messages",
            ):
                node_name = metadata.get("langgraph_node", "")

                if isinstance(chunk, AIMessageChunk):
                    if chunk.tool_call_chunks:
                        # 工具调用也会被拆成 chunk。这里只在第一次看到工具名时发事件，
                        # 避免同一个工具名随着 chunk 重复推送给前端。
                        for tool_call in chunk.tool_call_chunks:
                            name = _tool_call_name(tool_call)
                            if name and name not in tools_in_turn:
                                tools_in_turn.append(name)
                                yield _json_line({"type": "tool_call", "tool": name, "node": node_name})
                    elif chunk.content:
                        # 普通文本 token。这里直接累计起来，供最后 done 事件和长期记忆使用。
                        content = str(chunk.content)
                        full_reply += content
                        yield _json_line({"type": "token", "content": content})

                elif isinstance(chunk, ToolMessage):
                    # ToolMessage 是工具执行后的 observation。只截取前 200 字返回前端，
                    # 避免工具结果太长导致流式 UI 被大段 JSON/文本刷屏。
                    content = str(chunk.content)
                    yield _json_line({
                        "type": "tool_result",
                        "tool": getattr(chunk, "name", "unknown"),
                        "result": content[:200],
                        "truncated": len(content) > 200,
                    })

        except Exception as exc:
            yield _json_line({"type": "error", "message": str(exc)})
            return

        yield _json_line({
            "type": "done",
            "reply": full_reply,
            "tools_called": tools_in_turn,
            "reflection": (
                {
                    "score": reflection_info.score,
                    "passed": reflection_info.passed,
                    "retry_count": reflection_info.retry_count,
                }
                if reflection_info
                else None
            ),
        })
        self._remember_turn(session_id, message, full_reply, tools_in_turn)

    @staticmethod
    def _wrap_tools(
        realtime_tools: list[BaseTool],
        catalog_tools: list[BaseTool],
        max_retries: int,
        timeout_seconds: float,
    ) -> list[BaseTool]:
        """实时数据不缓存，规则/目录类数据允许缓存。

        这个函数是业务策略和通用工具包装层的分界点：
        ``AgentService`` 只决定“哪些工具能缓存”，真正怎么缓存、怎么重试，
        交给 ``wrap_all_tools``。
        """
        return (
            wrap_all_tools(
                realtime_tools,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                enable_cache=False,
            )
            + wrap_all_tools(
                catalog_tools,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                enable_cache=True,
            )
        )

    @staticmethod
    def _make_structured_extractor(chat_model: object):
        """创建“自然语言回复 -> FulfillmentDecision”的抽取链。

        ``prompt | chat_model.with_structured_output(Model)`` 是 LangChain LCEL 写法：
        prompt 先把输入组织成消息，再让模型按 Pydantic schema 输出对象。
        如果当前模型不支持 structured output，就返回 None，不影响主对话。
        """
        try:
            return _STRUCT_EXTRACTION_PROMPT | chat_model.with_structured_output(FulfillmentDecision)
        except Exception:
            return None

    def _callbacks(self) -> list:
        """把本轮要使用的 LangChain callbacks 收集起来。"""
        callbacks = []
        if self._langfuse_tracer:
            # Langfuse callback 负责把 LangChain run trace 上报到观测平台。
            callbacks.append(self._langfuse_tracer.callback)
        return callbacks

    @staticmethod
    def _run_config(session_id: str, callbacks: list) -> dict:
        """生成 LangChain/LangGraph 运行配置。

        ``thread_id`` 是记忆的 key；callbacks 是观测的入口。
        这两个都放在 config 里传给 ``ainvoke`` 或 ``astream``。
        """
        return {"configurable": {"thread_id": session_id}, "callbacks": callbacks}

    async def _run_turn(
        self,
        session_id: str,
        message: str,
        config: dict,
        callbacks: list,
    ) -> tuple[str, list[str], list[ToolCallDetail], Optional[ReflectionInfo]]:
        """统一执行一轮 Agent。

        普通模式直接调用 LangChain Agent；反思模式委托给 ``ReflectiveAgentRunner``。
        返回值统一成 reply/tools/trace/reflection，所以上层 ``chat`` 不用关心是哪种模式。
        """
        if self._reflective_runner:
            # 反思模式内部会决定是否重试，并返回最终 reply。
            run_result = await self._reflective_runner.arun(session_id, message, callbacks=callbacks)
            return (
                run_result["reply"],
                run_result["tools_called"],
                [],
                self._reflection_info(run_result),
            )

        # 标准模式：直接调用 LangChain Agent 图。
        result = await self._agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )
        reply, tools_called, tool_call_details = self._extract_reply_and_tools(result, message)
        return reply, tools_called, tool_call_details, None

    def _message_with_memory(self, session_id: str, message: str) -> str:
        """召回长期记忆，并把它作为本轮显式上下文。

        短期记忆由 checkpointer 保存完整消息历史；长期记忆只保存可复用摘要。
        模型不会自动知道该读哪条长期记忆，所以这里先 search，再注入当前问题。

        返回值仍然是一个普通 HumanMessage 的 content，只是前面多了
        “长期记忆参考”这一段。这样不用改 LangChain Agent 图结构。
        """
        memories = self._search_memories(session_id, message)
        if not memories:
            return message
        # 多条长期记忆用列表形式拼进 prompt，避免和当前问题混在一起。
        memory_text = "\n".join(f"- {item}" for item in memories)
        return f"【长期记忆参考】\n{memory_text}\n\n【当前问题】\n{message}"

    def _search_memories(self, session_id: str, message: str, limit: int = 4) -> list[str]:
        """从长期记忆里召回本轮可能有用的内容。

        召回顺序：
        1. 会话级偏好：例如用户不接受替代 SKU、优先时效。
        2. 会话级摘要：之前这个 session 里沉淀的高价值摘要。
        3. 订单级决策：如果问题里出现订单号，再查该订单历史处理结论。

        这里返回的是渲染后的短文本，不把原始 value 全量塞进 prompt。
        """
        try:
            # 先查会话级偏好，这类记忆优先级最高。
            items = list(
                self._memory_store.search(
                    ("sessions", session_id, "preferences"),
                    query=None,
                    limit=2,
                    filter={"memory_type": "user_preference"},
                )
            )
            # 再查会话级摘要，用 message 做语义检索。
            items.extend(self._memory_store.search(("sessions", session_id), query=message, limit=limit))
            if order_id := self._extract_order_id(message):
                # 如果问题中出现订单号，再查订单级历史决策。
                items.extend(self._memory_store.search(("orders", order_id), query=message, limit=2))
        except Exception:
            return []

        seen: set[str] = set()
        rendered: list[str] = []
        for item in items:
            if item.value.get("memory_status") in {"superseded", "deleted", "rejected"}:
                continue
            # store 返回的是结构化 value，先渲染成短文本。
            text = self._render_memory_item(item.value)
            if not text:
                continue
            text = str(text).strip()
            if text and text not in seen:
                # 去重并截断，避免长期记忆把 prompt 撑爆。
                seen.add(text)
                rendered.append(text[:300])
        return rendered[:limit]

    def _remember_turn(
        self,
        session_id: str,
        message: str,
        reply: str,
        tools_called: list[str],
    ) -> None:
        """把本轮对话沉淀为长期记忆。

        这里不保存完整上下文，避免长期记忆越来越噪；只保存可复用摘要、
        工具调用和显式偏好。完整逐字历史仍交给短期 checkpointer。

        写入策略：
        - 显式偏好一定写入，并用稳定 key 去重。
        - 带订单号且调用了工具的轮次，写入订单决策记忆。
        - 命中缺货、延期、风险等业务关键词的轮次，写入会话摘要。
        - 简短寒暄不会写入长期记忆。
        """
        if not reply:
            return
        # 先生成候选，再统一写入。候选生成阶段会做打分、脱敏和去重 key。
        candidates = self._build_memory_candidates(
            session_id=session_id,
            message=message,
            reply=reply,
            tools_called=tools_called,
        )
        try:
            for candidate in candidates:
                # 写入前先经过治理层：安全脱敏、冲突检测、旧偏好失效。
                self._memory_governance.govern_and_write(
                    self._memory_store,
                    namespace=candidate.namespace,
                    key=candidate.key,
                    value=candidate.value,
                )
        except Exception:
            # 长期记忆失败不影响用户主回答。
            pass

    @staticmethod
    def _render_memory_item(value: dict) -> str:
        """把不同类型的长期记忆渲染成适合放进 prompt 的短句。"""
        memory_type = value.get("memory_type")
        if memory_type == "user_preference":
            text = value.get("preference")
            return f"用户偏好：{text}" if text else ""
        if memory_type == "order_decision":
            text = value.get("summary")
            return f"历史订单决策：{text}" if text else ""
        text = value.get("summary") or value.get("assistant_reply")
        return str(text or "")

    @staticmethod
    def _build_memory_summary(message: str, reply: str, memory_type: str) -> str:
        """生成长期记忆摘要。

        摘要不是给用户看的完整回答，而是给后续 Agent 快速理解历史背景的提示。
        """
        if memory_type == "order_decision":
            return f"订单问题：{message[:120]}；处理结论：{reply[:180]}"
        return f"会话摘要：用户问 {message[:120]}；系统答 {reply[:180]}"

    @staticmethod
    def _build_memory_candidates(
        session_id: str,
        message: str,
        reply: str,
        tools_called: list[str],
    ) -> list[MemoryWriteCandidate]:
        """把一轮短期对话转换为可治理的长期记忆候选。

        这一步相当于企业里的“记忆沉淀策略层”：短期记忆保存完整过程，
        长期记忆只保存后续可能复用的偏好、订单决策、客户级策略和业务摘要。
        """
        compact = message.strip()
        if not compact or not reply:
            return []
        if len(compact) <= 8 and any(pattern == compact for pattern in LOW_VALUE_MESSAGE_PATTERNS):
            # 寒暄/确认类消息价值低，不写长期记忆。
            return []

        # 提取治理信号：订单号、客户号、显式偏好。
        order_id = AgentService._extract_order_id(message)
        customer_id = AgentService._extract_customer_id(f"{message}\n{reply}")
        preference = AgentService._extract_preference(message)
        # 计算是否值得写长期记忆。
        score = AgentService._memory_write_score(
            message=message,
            reply=reply,
            tools_called=tools_called,
            preference=preference,
            order_id=order_id,
        )
        if score < 0.55 and not preference:
            # 没达到门槛且没有明确偏好，就不写入。
            return []

        # 长期记忆入库前做脱敏和长度限制。
        safe_message = AgentService._redact_sensitive_text(message)[:500]
        safe_reply = AgentService._redact_sensitive_text(reply)[:800]
        now = datetime.now(tz=timezone.utc).isoformat()
        candidates: list[MemoryWriteCandidate] = []

        if preference:
            # 显式偏好是最高价值记忆，长期保存，不设置 TTL。
            preference_value = {
                "memory_type": "user_preference",
                "preference": preference,
                "source_message": safe_message[:300],
                "source": "agent_turn",
                "write_reason": "explicit_user_preference",
                "business_score": 0.95,
                "created_from_session": session_id,
                "updated_at": now,
                "_importance": 0.95,
                "_ttl_days": None,
            }
            candidates.append(MemoryWriteCandidate(
                namespace=("sessions", session_id, "preferences"),
                key=f"pref-{AgentService._stable_memory_key(preference)}",
                value=preference_value,
                score=0.95,
                reason="explicit_user_preference",
            ))
            if customer_id:
                # 如果能识别客户号，也写一份到客户级 namespace。
                candidates.append(MemoryWriteCandidate(
                    namespace=("customers", customer_id, "preferences"),
                    key=f"pref-{AgentService._stable_memory_key(preference)}",
                    value=preference_value | {"customer_id": customer_id},
                    score=0.95,
                    reason="customer_preference",
                ))

        if order_id and tools_called:
            # 有订单号且调用了工具，说明这是一条有证据支撑的订单决策。
            summary = AgentService._build_memory_summary(safe_message, safe_reply, "order_decision")
            candidates.append(MemoryWriteCandidate(
                namespace=("orders", order_id),
                key=f"decision-{AgentService._stable_memory_key(summary)}",
                value={
                    "memory_type": "order_decision",
                    "order_id": order_id,
                    "customer_id": customer_id,
                    "summary": summary,
                    "user_message": safe_message,
                    "assistant_reply": safe_reply,
                    "tools_called": tools_called,
                    "source": "agent_turn",
                    "write_reason": "tool_grounded_order_decision",
                    "business_score": score,
                    "created_from_session": session_id,
                    "updated_at": now,
                    "_importance": AgentService._memory_importance(message, reply, tools_called, preference, order_id),
                    "_ttl_days": 365,
                },
                score=score,
                reason="tool_grounded_order_decision",
            ))

        if score >= 0.65:
            # 高价值业务上下文写会话摘要，保留 90 天。
            summary = AgentService._build_memory_summary(safe_message, safe_reply, "conversation_summary")
            candidates.append(MemoryWriteCandidate(
                namespace=("sessions", session_id),
                key=f"summary-{AgentService._stable_memory_key(summary)}",
                value={
                    "memory_type": "conversation_summary",
                    "summary": summary,
                    "user_message": safe_message,
                    "assistant_reply": safe_reply,
                    "tools_called": tools_called,
                    "source": "agent_turn",
                    "write_reason": "high_value_business_context",
                    "business_score": score,
                    "updated_at": now,
                    "_importance": AgentService._memory_importance(message, reply, tools_called, preference, order_id),
                    "_ttl_days": 90,
                },
                score=score,
                reason="high_value_business_context",
            ))

        return candidates

    @staticmethod
    def _memory_write_score(
        message: str,
        reply: str,
        tools_called: list[str],
        preference: str | None,
        order_id: str | None,
    ) -> float:
        """计算本轮是否值得从短期记忆沉淀到长期记忆。

        这个分数是写入门槛，不是答案质量分。它更关注“后续是否可能复用”：
        有订单号、有工具证据、有业务风险/承诺/人工决策，才更值得长期保存。
        """
        combined = f"{message}\n{reply}"
        score = 0.0
        if preference:
            score += 0.5
        if order_id:
            score += 0.25
        if tools_called:
            score += min(len(tools_called), 3) * 0.12
        if any(keyword in combined for keyword in MEMORY_SIGNAL_KEYWORDS):
            score += 0.25
        if any(keyword in combined for keyword in ("人工复核", "风险", "延期", "缺货", "承诺")):
            score += 0.15
        if len(message.strip()) <= 8 and not tools_called and not preference:
            score -= 0.3
        return round(max(0.0, min(1.0, score)), 2)

    @staticmethod
    def _memory_importance(
        message: str,
        reply: str,
        tools_called: list[str],
        preference: str | None,
        order_id: str | None,
    ) -> float:
        """计算长期记忆重要性，取值 0~1。

        工具调用、订单号、显式偏好和风险词都会提高分数。
        这个分数只影响召回排序，不代表答案质量。
        """
        combined = f"{message}\n{reply}"
        score = 0.35 + min(len(tools_called), 3) * 0.1
        if preference:
            score += 0.2
        if order_id and tools_called:
            score += 0.15
        if any(keyword in combined for keyword in ("风险", "人工复核", "延期", "缺货")):
            score += 0.1
        return round(max(0.0, min(1.0, score)), 2)

    @staticmethod
    def _stable_memory_key(text: str) -> str:
        """给偏好类记忆生成稳定 key，避免同一偏好反复新增多条记录。"""
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _extract_order_id(text: str) -> str | None:
        """从用户问题中提取订单号。

        当前项目的演示订单号以 SO 开头；如果真实企业订单号规则不同，
        这里可以换成配置化正则或订单服务校验。
        """
        match = re.search(r"\bSO\d{8,}\b", text, flags=re.IGNORECASE)
        return match.group(0).upper() if match else None

    @staticmethod
    def _extract_customer_id(text: str) -> str | None:
        """抽取客户编号，用于把偏好沉淀到客户维度。

        当前兼容常见演示格式：C001、C-VIP-001、CUST-1001。
        """
        match = re.search(r"\b(?:CUST|C)(?:[-_]?[A-Z0-9]+)+\b", text, flags=re.IGNORECASE)
        return match.group(0).upper() if match else None

    @staticmethod
    def _redact_sensitive_text(text: str) -> str:
        """长期记忆入库前做轻量敏感信息脱敏。

        短期 checkpointer 可以保存完整上下文；长期记忆会跨会话复用，
        所以默认不持久化邮箱、手机号、身份证号这类直接个人标识。
        """
        redacted = text
        for pattern, replacement in SENSITIVE_TEXT_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted

    @staticmethod
    def _extract_preference(text: str) -> str | None:
        """从用户自然语言里抽取少量高确定性的偏好。

        这里故意不用 LLM 抽取，是为了让长期记忆写入可控、便宜、稳定。
        后续如果要扩展更多偏好，可以改成小模型结构化抽取。
        """
        if "不接受替代" in text or "不要替代" in text:
            return "用户不接受替代 SKU，缺货时应优先人工确认或延期处理。"
        if "接受替代" in text or "可以替代" in text:
            return "用户接受兼容替代 SKU，缺货时可优先推荐替代方案。"
        if "优先时效" in text or "尽快发货" in text:
            return "用户偏好时效优先，履约建议应优先考虑最快可发路径。"
        return None

    async def _extract_decision(
        self, message: str, reply: str
    ) -> Optional[FulfillmentDecision]:
        """从 Agent 的自然语言回复中抽取结构化履约决策。

        这里是“锦上添花”的能力：抽取失败不应该影响用户拿到自然语言回复，
        所以异常会被吞掉并返回 None。
        """
        if not self._struct_extractor or not reply:
            return None
        try:
            return await self._struct_extractor.ainvoke({
                "question": message,
                "agent_reply": reply,
            })
        except Exception:
            return None

    def _score_langfuse(
        self,
        reply: str,
        tools_called: list[str],
        reflection_info: Optional[ReflectionInfo],
    ) -> None:
        """把本轮结果提交给 Langfuse 评分。

        Langfuse 只负责观测，不参与主流程决策；没有配置时直接跳过。
        """
        if not self._langfuse_tracer:
            return
        score_after_run(
            tracer=self._langfuse_tracer,
            trace_id=self._langfuse_tracer.get_trace_id(),
            reply=reply,
            tools_called=tools_called,
            reflection_score=reflection_info.score if reflection_info else None,
            reflection_passed=reflection_info.passed if reflection_info else None,
            reflection_reason=reflection_info.reason if reflection_info else "",
        )

    @staticmethod
    def _reflection_info(run_result: dict) -> Optional[ReflectionInfo]:
        """把内部 ReflectionResult 转成 API schema。"""
        reflection = run_result.get("reflection")
        if reflection is None:
            return None
        return ReflectionInfo(
            score=reflection.total_score,
            passed=reflection.passed,
            retry_count=run_result["retry_count"],
            reason=reflection.reason,
        )

    @staticmethod
    def _build_trace(
        session_id: str,
        message: str,
        reply: str,
        tools_called: list[str],
        tool_call_details: list[ToolCallDetail],
    ) -> AgentExecutionTrace:
        """构造一轮 ReAct 轨迹。

        LangChain 不会把模型内部 thought 明文暴露出来，所以 trace 里重点记录：
        调了哪个工具、传了什么参数、工具返回了什么摘要、最后回答是什么。
        """
        if not tool_call_details and tools_called:
            tool_call_details = [
                ToolCallDetail(tool_name=name, input_args={}, output="", order=i + 1)
                for i, name in enumerate(tools_called)
            ]

        react_cycles = [
            ReActCycle(
                cycle_index=i,
                thought="",
                action=detail.tool_name,
                action_input=detail.input_args,
                observation=detail.output[:300],
            )
            for i, detail in enumerate(tool_call_details)
        ]
        return AgentExecutionTrace(
            session_id=session_id,
            user_message=message,
            tools_called=tool_call_details,
            react_cycles=react_cycles,
            model_reasoning="",
            final_reply=reply,
            execution_steps=len(tools_called),
        )

    def _maybe_trim_context(self, session_id: str) -> bool:
        """会话太长时裁剪历史，避免 prompt 持续膨胀。

        ``get_state`` 和 ``update_state`` 是 LangGraph 编译图提供的方法。
        它们让我们可以在不手动维护消息列表的情况下读写会话状态。
        """
        try:
            state = self._agent.get_state({"configurable": {"thread_id": session_id}})
            existing = state.values.get("messages", []) if state.values else []
            trimmed, result = self._context_manager.trim(existing)
            if result.messages_dropped > 0:
                self._agent.update_state(
                    {"configurable": {"thread_id": session_id}},
                    {"messages": trimmed},
                )
                return True
        except Exception:
            pass
        return False

    @staticmethod
    def _extract_reply_and_tools(
        result: dict, user_message: str
    ) -> tuple[str, list[str], list[ToolCallDetail]]:
        """从 LangChain 消息历史里提取最终回复、工具名和工具结果。

        LangChain Agent 的返回不是简单字符串，而是一段消息历史：
        - HumanMessage：用户输入
        - AIMessage(tool_calls=...)：模型决定调用哪些工具
        - ToolMessage：工具执行结果
        - AIMessage(content=...)：模型看到工具结果后的最终回复

        这个函数做的就是把这段历史压缩成 API 需要的三个东西：
        ``reply``、``tools_called``、``tool_call_details``。
        """
        messages = result.get("messages", [])
        user_idx = next(
            (i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage) and msg.content == user_message),
            -1,
        )
        current_turn = messages[user_idx + 1:] if user_idx >= 0 else messages

        ai_messages = filter_messages(current_turn, include_types=[AIMessage])
        tool_messages = filter_messages(current_turn, include_types=[ToolMessage])

        tools_called: list[str] = []
        tool_call_details: list[ToolCallDetail] = []
        reply = ""

        for message in ai_messages:
            if message.tool_calls:
                # 有 tool_calls 的 AIMessage 表示“模型还没最终回答，它正在请求工具”。
                for tool_call in message.tool_calls:
                    name = _tool_call_name(tool_call)
                    if not name:
                        continue
                    tools_called.append(name)
                    tool_call_details.append(ToolCallDetail(
                        tool_name=name,
                        input_args=_tool_call_args(tool_call),
                        output="",
                        order=len(tool_call_details) + 1,
                    ))
            else:
                # 没有 tool_calls 的 AIMessage 通常就是最终自然语言回复。
                reply = str(message.content)

        for tool_message in tool_messages:
            # ToolMessage 只告诉我们工具名和输出。这里倒序匹配，是为了在同名工具
            # 多次调用时，优先把输出填回最近一次还没有 observation 的调用记录。
            tool_name = getattr(tool_message, "name", "")
            for detail in reversed(tool_call_details):
                if detail.tool_name == tool_name and not detail.output:
                    detail.output = str(tool_message.content)[:500]
                    break

        return reply, tools_called, tool_call_details
