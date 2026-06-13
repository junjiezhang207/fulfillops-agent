"""Multi-Agent Supervisor 编排链路。

本模块实现由 Supervisor 调度库存、履约、风险三个专家 Agent 的 LangGraph 流程。
第一轮可并行收集多领域事实，后续由 Supervisor 判断是否追问或汇总。

图结构：
    START -> supervisor -> 专家节点/parallel -> barrier -> supervisor
                          -> synthesizer -> END
"""

from __future__ import annotations

import asyncio
import re
import uuid
from collections import defaultdict
from operator import add
from typing import Annotated, Any, Callable, Literal, Optional, TypedDict, Union

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, filter_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from app.agents.tools.registry import ToolServiceBundle, get_tool_registry
from app.domain.fulfillment.plan_service import FulfillmentPlanService
from app.domain.inventory.analysis import InventoryAnalysisService
from app.infrastructure.llm.model_gateway import get_model_gateway
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
from app.domain.orders.analysis import OrderAnalysisService
from app.domain.fulfillment.substitute_sku import SubstituteSkuService
from app.domain.inventory.warehouse_service import WarehouseService


SpecialistName = Literal["inventory_agent", "fulfillment_agent", "risk_agent"]
AgentName = Literal[
    "inventory_agent", "fulfillment_agent", "risk_agent", "synthesizer", "parallel", "__end__"
]
SPECIALISTS: tuple[SpecialistName, ...] = ("inventory_agent", "fulfillment_agent", "risk_agent")
ROUTABLE_NODES = (*SPECIALISTS, "synthesizer")

# 每个专家的 system prompt 要窄而明确。
# 多 Agent 不是让多个 Agent 都做同一件事，而是让它们各自负责一个稳定边界：
# 库存、履约规划、风险规则。边界越清楚，汇总时越不容易互相打架。
SYSTEM_PROMPTS: dict[SpecialistName, str] = {
    "inventory_agent": get_model_gateway().prompt_system(use_case="multi_agent", prompt_id="multi_inventory_agent"),
    "fulfillment_agent": get_model_gateway().prompt_system(use_case="multi_agent", prompt_id="multi_fulfillment_agent"),
    "risk_agent": get_model_gateway().prompt_system(use_case="multi_agent", prompt_id="multi_risk_agent"),
}
QUESTION_KEYWORDS = {
    "inventory_agent": ("库存", "现货", "仓库", "可发", "发货"),
    "fulfillment_agent": ("缺货", "不足", "替代", "方案"),
    "risk_agent": ("风险", "规则", "政策"),
}
SKU_RE = re.compile(r"\bSKU[-_A-Z0-9]+\b", re.IGNORECASE)


# Supervisor 输出结构化路由决策，避免从自然语言中解析下一跳。
class SupervisorDecision(BaseModel):
    """Supervisor 的结构化路由结果。

    follow_up 是保留多轮反馈的关键：结果不完整时，Supervisor 可以把补充问题
    精确交回某个专家，而不用从自然语言里猜下一步。
    """

    next_agent: str = Field(description="下一步节点：专业 Agent 或 synthesizer")
    follow_up: str = Field(default="", description="追加查询指令；无需补充时留空。")
    reasoning: str = Field(default="", description="路由理由，写入轨迹方便排查。")


def _merge_rounds(old: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    """LangGraph reducer：并行专家同时返回时，按 Agent 累加调用轮次。"""
    merged = dict(old)
    for agent, count in new.items():
        merged[agent] = merged.get(agent, 0) + count
    return merged


# agent_results 汇总各专家输出，供 barrier、supervisor 和 synthesizer 做 fan-in 后判断。
class MultiAgentState(TypedDict, total=False):
    """LangGraph 在各节点之间传递的共享状态。

    带 ``Annotated[..., add]`` 的字段使用 reducer。
    例如三个专家并行返回时，``agent_results`` 不会互相覆盖，而是追加到同一个列表。
    这是 LangGraph 做并行 fan-out/fan-in 时非常关键的写法。
    """

    order_id: str
    question: str
    run_id: str
    next_agent: Union[AgentName, str, None]
    follow_up_context: str
    agent_results: Annotated[list[dict[str, Any]], add]
    rounds_per_agent: Annotated[dict[str, int], _merge_rounds]
    supervisor_reasoning: Annotated[list[str], add]
    final_answer: str
    execution_log: Annotated[list[str], add]
    parallel_round_done: bool


def _prompt(name: str) -> str:
    from app.infrastructure.llm.model_gateway import get_model_gateway

    prompt_use_cases = {
        "supervisor_agent": "supervisor",
        "synthesizer_agent": "multi_agent",
    }
    return get_model_gateway().prompt_system(
        use_case=prompt_use_cases.get(name, "multi_agent"),
        prompt_id=name,
    )


def _supervisor_prompt(max_rounds: int) -> ChatPromptTemplate:
    system = (
        _prompt("supervisor_agent")
        + "\n\n反馈循环规则（每个 Agent 最多调用 {max_rounds} 次）：\n"
        + "  - 若某个 Agent 的结果不完整，可再次调用，在 follow_up 字段写明需要补充什么\n"
        + "  - 首次调用时 follow_up 留空\n"
        + "  - 当信息足够时，调 synthesizer 生成最终答案。"
    )
    return ChatPromptTemplate.from_messages([("system", system), ("human", "{context}")]).partial(
        max_rounds=max_rounds
    )


def _supervisor_context(state: MultiAgentState, max_rounds: int) -> str:
    """把当前状态整理成 Supervisor 能读懂的上下文。

    Supervisor 不直接读取 Python 对象，它看到的是一段文本。
    这里把已调用过哪些专家、每个专家说了什么、还能追加调用谁都写进去，
    让 LLM 路由时有足够信息。
    """
    parts = [f"订单 ID：{state.get('order_id', '')}", f"问题：{state.get('question', '')}"]
    results = state.get("agent_results", [])
    if results:
        parts.append("\n已收集的信息（按 Agent 分组）：")
        grouped: dict[str, list[str]] = defaultdict(list)
        for item in results:
            grouped[item["agent"]].append(item["summary"][:200])
        for agent, summaries in grouped.items():
            parts.append(f"\n  [{agent}] 已调用 {len(summaries)} 次（上限 {max_rounds}）")
            parts.extend(f"    第{i}次结果: {text}" for i, text in enumerate(summaries, 1))

    rounds = state.get("rounds_per_agent", {})
    available = [agent for agent in SPECIALISTS if rounds.get(agent, 0) < max_rounds]
    parts.append(f"\n还可追加调用的 Agent：{available}")
    return "\n".join(parts)


def _rule_based_supervisor(
    question: str,
    results: list[dict[str, Any]],
    rounds: dict[str, int],
    max_rounds: int,
) -> str:
    """LLM 不可用或输出越界时的保守路由。

    规则只补齐尚未运行的专家；多轮追问交给 LLM Supervisor。这样无 LLM
    模式仍保持“首轮三专家后汇总”的简单行为，不会自己循环。
    """
    called = {item["agent"] for item in results}
    for agent in SPECIALISTS:
        if agent not in called and rounds.get(agent, 0) < max_rounds:
            return agent
    for agent, keywords in QUESTION_KEYWORDS.items():
        if agent not in called and any(keyword in question for keyword in keywords):
            return agent
    return "synthesizer"


def _supervisor_update(next_agent: str, follow_up: str, reason: str, log: str) -> dict:
    return {
        "next_agent": next_agent,
        "follow_up_context": follow_up,
        "supervisor_reasoning": [reason],
        "execution_log": [log],
    }


# Supervisor 只负责调度，不直接处理业务事实。
def make_supervisor_node(
    llm: BaseChatModel | None,
    max_rounds_per_agent: int = 2,
) -> Callable[[MultiAgentState], dict]:
    """创建 Supervisor 节点。

    Supervisor 的任务不是回答业务问题，而是做路由决策：
    - 第一轮是否并行调用专家？
    - 某个专家结果不完整时，是否带 follow_up 再问一次？
    - 信息足够时，是否进入 synthesizer 汇总？

    这里用 ``with_structured_output(SupervisorDecision)``，是为了让 LLM 输出稳定字段，
    避免从自然语言里解析“下一步该去哪”。
    """
    chain = (
        _supervisor_prompt(max_rounds_per_agent) | llm.with_structured_output(SupervisorDecision)
        if llm
        else None
    )

    async def supervisor(state: MultiAgentState) -> dict:
        results = state.get("agent_results", [])
        rounds = state.get("rounds_per_agent", {})
        total_calls = len(results)

        if total_calls >= max_rounds_per_agent * len(SPECIALISTS):
            return _supervisor_update("synthesizer", "", "超过最大总轮次限制，强制进入汇总", "Supervisor: 强制 synthesizer")

        if not state.get("parallel_round_done", False) and not results:
            # 第一轮用并行，是因为库存、履约、风险三类事实互不依赖。
            # 并行比串行更快，也能让 synthesizer 一开始就看到完整视角。
            return _supervisor_update("parallel", "", "第一轮并行：同时调用所有专业 Agent", "Supervisor -> parallel (3 agents)")

        fallback = _rule_based_supervisor(state.get("question", ""), results, rounds, max_rounds_per_agent)
        if chain is None:
            return _supervisor_update(fallback, "", "规则模式", f"Supervisor(规则) -> {fallback}")

        try:
            decision = await chain.ainvoke({"context": _supervisor_context(state, max_rounds_per_agent)})
            next_agent, follow_up, reason = decision.next_agent, decision.follow_up, decision.reasoning
            if next_agent not in ROUTABLE_NODES:
                next_agent, follow_up, reason = fallback, "", "Supervisor 返回了未知节点，规则降级"
            elif next_agent in SPECIALISTS and rounds.get(next_agent, 0) >= max_rounds_per_agent:
                next_agent, follow_up, reason = fallback, "", f"{next_agent} 已达轮次上限，规则降级"
        except Exception:
            next_agent, follow_up, reason = fallback, "", "LLM 异常，规则降级"

        log = f"Supervisor -> {next_agent}" + (f"（追加：{follow_up[:50]}）" if follow_up else "")
        return _supervisor_update(next_agent, follow_up, f"{reason or 'Supervisor 结构化决策'}（已 {total_calls} 个结果）", log)

    return supervisor


# barrier 是并行专家 fan-in 同步点，确保一轮结果收齐后再回到 Supervisor。
def make_barrier_node() -> Callable[[MultiAgentState], dict]:
    """fan-in 同步点：只标记第一轮并行结束，不改任何业务结果。

    barrier 的作用像“集合点”：三个专家都跑完后，流程回到 supervisor。
    它不处理业务数据，只避免 supervisor 再次误判“第一轮还没并行过”。
    """
    return lambda _state: {"parallel_round_done": True}


def route_from_supervisor(state: MultiAgentState) -> Union[str, list[Send]]:
    """根据 Supervisor 的结果决定下一条边。

    LangGraph 的普通条件边返回字符串，表示去某一个节点。
    返回 ``list[Send]`` 时表示 fan-out，并行发送到多个节点。
    这里 ``parallel`` 不是一个真实节点，而是一个路由信号。
    """
    next_agent = state.get("next_agent", "synthesizer")
    if next_agent != "parallel":
        return next_agent if next_agent in ROUTABLE_NODES else "synthesizer"

    rounds = state.get("rounds_per_agent", {})
    agents = [agent for agent in SPECIALISTS if rounds.get(agent, 0) == 0]
    return [Send(agent, dict(state)) for agent in agents] if agents else "synthesizer"


def _specialist_question(state: MultiAgentState) -> str:
    question = f"订单 {state.get('order_id', '')}：{state.get('question', '')}"
    follow_up = state.get("follow_up_context", "")
    return f"{question}\n\n【追加查询指令】{follow_up}" if follow_up else question


def _last_agent_answer(result: dict[str, Any]) -> str:
    """从专家 Agent 的 LangChain 消息历史里取最终回答。"""
    ai_messages = filter_messages(result.get("messages", []), include_types=[AIMessage])
    final_messages = [message for message in ai_messages if not message.tool_calls]
    return str(final_messages[-1].content) if final_messages else "无法生成分析"


def _extract_tool_calls(result: dict[str, Any]) -> list[str]:
    """从专家 Agent 消息历史中提取本轮工具调用名。"""
    ai_messages = filter_messages(result.get("messages", []), include_types=[AIMessage])
    called: list[str] = []
    for message in ai_messages:
        for tool_call in message.tool_calls or []:
            name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
            if name:
                called.append(str(name))
    return called


def _extract_sku(question: str) -> str | None:
    match = SKU_RE.search(question)
    return match.group(0).upper() if match else None


def _tool_args(tool_name: str, order_id: str, question: str) -> dict[str, Any] | None:
    """为无 LLM 规则降级构造工具参数。

    规则模式没有模型帮我们填参，所以这里只处理项目里已知工具的最小参数集。
    SKU 粒度工具在问题里没有 SKU 时跳过，避免用错误参数硬调失败。
    """
    if tool_name in {"analyze_order", "check_inventory", "generate_fulfillment_plan"}:
        return {"order_id": order_id}
    if tool_name == "retrieve_knowledge":
        return {"order_id": order_id, "question": question, "categories": ""}
    if tool_name in {"search_warehouse_inventory", "find_substitute_sku"}:
        sku = _extract_sku(question)
        return {"sku_id": sku} if sku else None
    return {}


def _run_tools_for_summary(tools: list, order_id: str, question: str) -> tuple[str, list[str]]:
    """无 LLM 时的规则降级：直接按工具顺序跑一遍生成摘要。"""
    outputs = []
    called: list[str] = []
    for tool in tools:
        args = _tool_args(tool.name, order_id, question)
        if args is None:
            outputs.append(f"[{tool.name}] 跳过: 问题中未提供 SKU")
            continue
        try:
            outputs.append(f"[{tool.name}] {str(tool.invoke(args))[:300]}")
            called.append(tool.name)
        except Exception as exc:
            outputs.append(f"[{tool.name}] 失败: {exc}")
            called.append(tool.name)
    return "\n".join(outputs), called


async def _run_tools_for_summary_async(tools: list, order_id: str, question: str) -> tuple[str, list[str]]:
    """无 LLM 时的异步规则降级，优先使用 LangChain tool.ainvoke。"""
    outputs = []
    called: list[str] = []
    for tool in tools:
        args = _tool_args(tool.name, order_id, question)
        if args is None:
            outputs.append(f"[{tool.name}] 跳过: 问题中未提供 SKU")
            continue
        try:
            outputs.append(f"[{tool.name}] {str(await tool.ainvoke(args))[:300]}")
            called.append(tool.name)
        except Exception as exc:
            outputs.append(f"[{tool.name}] 失败: {exc}")
            called.append(tool.name)
    return "\n".join(outputs), called


def _agent_update(agent_name: str, question: str, summary: str, log: str, tools_called: list[str]) -> dict:
    return {
        "agent_results": [{
            "agent": agent_name,
            "question": question,
            "summary": summary,
            "tools_called": tools_called,
        }],
        "rounds_per_agent": {agent_name: 1},
        "execution_log": [log],
    }


def make_specialist_agent_node(
    agent_name: SpecialistName,
    system_prompt: str,
    tools: list,
    llm: BaseChatModel | None,
    checkpointer=None,
    store=None,
) -> Callable[[MultiAgentState], dict]:
    """创建一个专家 Agent 节点。

    有 LLM 时，每个专家本身也是一个 ReAct Agent，可以在自己的工具集合里选择调用。
    没有 LLM 时，走规则降级，直接执行该专家拥有的工具。
    """
    specialist_kwargs = {
        "model": llm,
        "tools": tools,
        "system_prompt": system_prompt,
        "checkpointer": checkpointer,
    }
    if checkpointer is None:
        raise RuntimeError("Multi-Agent 专家 Agent 必须显式传入 Redis checkpointer。")
    if store is not None:
        specialist_kwargs["store"] = store
    specialist = create_agent(**specialist_kwargs) if llm else None

    async def node(state: MultiAgentState) -> dict:
        question = _specialist_question(state) if specialist else state.get("question", "")
        current_round = state.get("rounds_per_agent", {}).get(agent_name, 0) + 1

        if specialist:
            # 每个专家每一轮使用独立 thread_id，避免不同专家之间共享私有上下文。
            # 专家输出会通过 MultiAgentState 汇总，而不是靠各自记忆互相通信。
            run_id = state.get("run_id", uuid.uuid4().hex[:8])
            thread_id = f"{run_id}_{agent_name}_{state.get('order_id', '')}_round{current_round}"
            try:
                result = await specialist.ainvoke(
                    {"messages": [HumanMessage(content=question)]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                answer = _last_agent_answer(result)
                tools_called = _extract_tool_calls(result)
            except Exception as exc:
                answer = f"[{agent_name}] 执行异常：{exc}"
                tools_called = []
        else:
            answer, tools_called = await _run_tools_for_summary_async(
                tools,
                state.get("order_id", ""),
                state.get("question", ""),
            )
            if follow_up := state.get("follow_up_context", ""):
                answer += f"\n【追加指令】{follow_up}"

        follow_up = state.get("follow_up_context", "")
        mode = f"第{current_round}轮" + (f"（追加：{follow_up[:30]}）" if follow_up else "")
        log = f"{agent_name} {mode} 完成" if specialist else f"{agent_name} 完成（规则模式）"
        return _agent_update(agent_name, question, answer, log, tools_called)

    return node


def _synthesis_context(question: str, results: list[dict[str, Any]]) -> str:
    reports = "\n".join(
        f"\n[{item['agent']} | tools={item.get('tools_called', [])}]\n{item['summary']}"
        for item in results
    )
    return f"用户问题：{question}\n\n各专业 Agent 报告：{reports}"


def _fallback_synthesis(results: list[dict[str, Any]], limit: int = 100) -> str:
    return "综合：\n" + "\n".join(f"[{item['agent']}] {item['summary'][:limit]}" for item in results)


def make_synthesizer_node(llm: BaseChatModel | None) -> Callable[[MultiAgentState], dict]:
    """创建汇总节点。

    synthesizer 不再调用业务工具，它只阅读各专家报告，生成面向用户的最终答案。
    这能把“收集事实”和“组织答案”拆开，复杂问题会更清晰。
    """
    chain = (
        ChatPromptTemplate.from_messages([("system", _prompt("synthesizer_agent")), ("human", "{context}")])
        | llm
        | StrOutputParser()
        if llm
        else None
    )

    async def synthesizer(state: MultiAgentState) -> dict:
        results = state.get("agent_results", [])
        if not results:
            return {"final_answer": "未能收集到足够信息。", "execution_log": ["Synthesizer: 无 Agent 输出"]}
        if chain is None:
            answer = _fallback_synthesis(results, limit=150)
        else:
            try:
                answer = await chain.ainvoke({"context": _synthesis_context(state.get("question", ""), results)})
            except Exception:
                answer = _fallback_synthesis(results)
        return {"final_answer": answer, "execution_log": ["Synthesizer 完成汇总"]}

    return synthesizer


# Multi-Agent 适合跨多个领域且可并行收集事实的问题。
class MultiAgentOrchestrator:
    """对外的多 Agent 编排器。

    第一轮固定并行跑库存、履约、风险三位专家；随后由 Supervisor 判断是否
    需要追问。每个专家有轮次上限，防止反馈循环失控。
    """

    def __init__(
        self,
        order_service: OrderAnalysisService,
        inventory_service: InventoryAnalysisService,
        knowledge_service: KnowledgeRetrievalService,
        warehouse_service: WarehouseService,
        substitute_service: SubstituteSkuService,
        fulfillment_service: FulfillmentPlanService,
        llm: Optional[BaseChatModel] = None,
        max_rounds_per_agent: int = 2,
        checkpointer=None,
        store=None,
    ):
        self.llm = llm
        self.max_rounds_per_agent = max_rounds_per_agent
        if checkpointer is None:
            raise RuntimeError("Multi-Agent 编排器必须显式传入 Redis checkpointer。")
        self._checkpointer = checkpointer
        self._store = store
        self._graph = self._build_graph(
            order_service,
            inventory_service,
            knowledge_service,
            warehouse_service,
            substitute_service,
            fulfillment_service,
        )

    def _build_graph(
        self,
        order_svc: OrderAnalysisService,
        inv_svc: InventoryAnalysisService,
        know_svc: KnowledgeRetrievalService,
        wh_svc: WarehouseService,
        sub_svc: SubstituteSkuService,
        ful_svc: FulfillmentPlanService,
    ) -> CompiledStateGraph:
        """组装 LangGraph。

        读图时重点看三类边：
        1. START -> supervisor：所有请求先交给调度器。
        2. supervisor 的条件边：可能去某个专家、并行多个专家，或去 synthesizer。
        3. 专家 -> barrier -> supervisor：专家结果回流后再次判断是否需要补问。
        """
        graph = StateGraph(MultiAgentState)
        tool_services = ToolServiceBundle(
            order_service=order_svc,
            inventory_service=inv_svc,
            knowledge_service=know_svc,
            warehouse_service=wh_svc,
            substitute_service=sub_svc,
            fulfillment_service=ful_svc,
        )
        tool_registry = get_tool_registry()
        tools_by_agent = {
            "inventory_agent": tool_registry.build_tools(
                services=tool_services,
                use_case="multi_agent",
                group="multi_agent.inventory_agent",
            ),
            "fulfillment_agent": tool_registry.build_tools(
                services=tool_services,
                use_case="multi_agent",
                group="multi_agent.fulfillment_agent",
            ),
            "risk_agent": tool_registry.build_tools(
                services=tool_services,
                use_case="multi_agent",
                group="multi_agent.risk_agent",
            ),
        }

        graph.add_node("supervisor", make_supervisor_node(self.llm, self.max_rounds_per_agent))
        for agent_name, tools in tools_by_agent.items():
            graph.add_node(
                agent_name,
                make_specialist_agent_node(
                    agent_name,
                    SYSTEM_PROMPTS[agent_name],
                    tools,
                    self.llm,
                    checkpointer=self._checkpointer,
                    store=self._store,
                ),
            )
        graph.add_node("barrier", make_barrier_node())
        graph.add_node("synthesizer", make_synthesizer_node(self.llm))

        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {node: node for node in ROUTABLE_NODES},
        )
        for agent_name in SPECIALISTS:
            graph.add_edge(agent_name, "barrier")
        graph.add_edge("barrier", "supervisor")
        graph.add_edge("synthesizer", END)
        compile_kwargs = {"checkpointer": self._checkpointer}
        if self._store is not None:
            compile_kwargs["store"] = self._store
        return graph.compile(**compile_kwargs)

    def run(self, order_id: str, question: str) -> dict:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(order_id=order_id, question=question))
        raise RuntimeError("当前已有事件循环，请使用 await arun(...)")

    async def arun(self, order_id: str, question: str) -> dict:
        run_id = uuid.uuid4().hex[:12]
        final = await self._graph.ainvoke(
            {
                "order_id": order_id,
                "question": question,
                "run_id": run_id,
                "agent_results": [],
                "rounds_per_agent": {},
                "follow_up_context": "",
                "supervisor_reasoning": [],
                "execution_log": [],
            },
            config={"configurable": {"thread_id": f"multi_agent_{run_id}"}},
        )
        results = final.get("agent_results", [])
        return {
            "final_answer": final.get("final_answer", ""),
            "agent_results": results,
            "execution_log": final.get("execution_log", []),
            "agents_called": [item["agent"] for item in results],
            "supervisor_reasoning": final.get("supervisor_reasoning", []),
            "rounds_per_agent": final.get("rounds_per_agent", {}),
        }
