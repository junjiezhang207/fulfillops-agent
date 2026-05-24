"""文件作用摘要：Plan-and-Execute Agent，先规划再执行。

这个文件实现一种不同于 ReAct 的 Agent 执行模式。普通 ReAct 是“走一步看一步”，
每一步根据当前观察决定下一个工具；Plan-and-Execute 是先生成全局计划，再按
计划逐步执行，执行后由 Replanner 判断是否还需要调整计划。

主要做的事：
1. ``Plan``：Planner 输出的结构化计划，包含按顺序执行的步骤。
2. ``PlanExecuteState``：LangGraph 图里的状态，记录计划、已执行步骤和最终答案。
3. ``_make_planner_node``：根据用户问题生成初始计划。
4. ``_make_executor_node``：用子 Agent 执行当前计划步骤。
5. ``_make_replanner_node``：根据执行结果判断剩余步骤或直接结束。
6. ``_make_synthesizer_node``：把所有步骤结果汇总成最终答案。
7. ``build_plan_execute_agent``：组装 Planner -> Executor -> Replanner -> Synthesizer 图。

适用场景：
- 需要先查订单，再查库存，再找替代品，最后生成履约方案。
- 用户要求“完整分析”“列出所有可行路径”“按步骤给方案”。

不适合场景：
- 简单单步查询，普通 ReAct 更轻。
- 多领域专家并行讨论，Multi-Agent Supervisor 更合适。

学习时先看：
1. ``PlanExecuteState``：计划执行图里保存什么。
2. ``build_plan_execute_agent``：整张图怎么循环。
3. Planner / Executor / Replanner 三个节点：理解计划如何被执行和修正。
"""

from __future__ import annotations

import logging
from operator import add
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, filter_messages, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Pydantic 结构化输出模型 ───────────────────────────────────────────────────

# 面试官可能问：Plan-and-Execute 为什么要让 Planner 输出结构化 Plan？
# 回答：结构化计划比自由文本更容易让程序读取和控制循环。Executor 可以按
# steps 逐条执行，Replanner 也能明确判断还剩哪些步骤，而不是解析一段随意文本。
class Plan(BaseModel):
    """LLM 生成的执行计划。"""
    steps: list[str] = Field(
        description="按执行顺序排列的步骤列表，每步明确说明：调用哪个工具、查询什么、期望得到什么。最多 5 步。"
    )


class ReplanDecision(BaseModel):
    """Replanner 的决策结果。"""
    remaining_steps: list[str] = Field(
        description=(
            "仍需执行的步骤列表。"
            "如果已收集到足够信息可以生成最终答复，返回空列表 []。"
        )
    )


# ── 状态定义 ─────────────────────────────────────────────────────────────────

# 面试官可能问：PlanExecuteState 里为什么要保存 past_steps？
# 回答：past_steps 让 Replanner 知道哪些步骤已经做过、结果是什么，避免重复执行；
# 最终 Synthesizer 也需要基于完整步骤结果生成有依据的综合答案。
class PlanExecuteState(TypedDict, total=False):
    """Plan-and-Execute 工作流共享状态。

    past_steps 使用 operator.add reducer：每个 executor 节点追加一条记录，
    不会覆盖前面的结果，这是 LangGraph reducer 模式的标准用法。
    """
    input: str                                           # 用户原始问题
    order_id: str                                        # 订单 ID（供工具调用）
    run_id: str                                          # 单次 Plan-and-Execute 运行 ID
    plan: list[str]                                      # 当前待执行步骤（动态更新）
    past_steps: Annotated[list[tuple[str, str]], add]    # 已完成：[(步骤描述, 结果)]
    final_answer: str                                    # 最终综合答复
    iterations: int                                      # 执行轮次（防无限循环）


# ── Prompt 定义 ───────────────────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
你是供应链履约专家，负责将用户的复杂问题分解为有序的执行步骤。

步骤设计原则：
1. 按依赖关系排序（必须先查订单信息，才能查对应的库存）
2. 每步明确说明调用哪个工具、查询什么
3. 最多 5 步（避免过度分解）
4. 步骤间传递上下文（如"根据上一步查到的缺货 SKU，查找替代品"）

可用工具：analyze_order / check_inventory / search_warehouse_inventory /
         find_substitute_sku / generate_fulfillment_plan / retrieve_knowledge
"""

_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _PLANNER_SYSTEM),
    ("human", "订单ID: {order_id}\n用户问题: {input}"),
])

_REPLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "你是供应链履约专家。根据用户目标和已执行步骤的结果，判断是否还需要继续执行更多步骤。\n"
        "如果已有足够信息可以给出最终答复，返回空列表。\n"
        "如果仍需继续，只保留真正必要的剩余步骤（删除已被之前结果覆盖的步骤）。"
    )),
    ("human", (
        "用户目标: {input}\n\n"
        "已完成步骤及结果:\n{past_steps_text}\n\n"
        "当前剩余计划:\n{plan_text}\n\n"
        "请决定是否继续，并更新剩余步骤列表。"
    )),
])

_SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "你是供应链履约专家。根据各执行步骤收集到的数据，"
        "生成最终的综合答复。要求：简洁（不超过 300 字）、准确、可操作，"
        "包含关键数据和明确建议。"
    )),
    ("human", "用户问题: {input}\n\n各步骤结果:\n{past_steps_text}"),
])

_EXECUTOR_SYSTEM = (
    "你是供应链履约执行助手。根据给定的具体任务，调用相应工具获取信息并返回结果。"
    "只需完成当前任务，不要超出范围。结果要包含具体数据（数字、SKU、仓库名等）。"
)


# ── 节点工厂 ──────────────────────────────────────────────────────────────────

def _make_planner_node(llm: BaseChatModel):
    """Planner 节点：LLM 生成结构化执行计划。"""
    planner_chain = _PLANNER_PROMPT | llm.with_structured_output(Plan)

    async def planner(state: PlanExecuteState) -> dict:
        try:
            plan_obj: Plan = await planner_chain.ainvoke({
                "input": state.get("input", ""),
                "order_id": state.get("order_id", ""),
            })
            logger.info("[PlanExecute] 计划生成完成，共 %d 步", len(plan_obj.steps))
            for i, step in enumerate(plan_obj.steps, 1):
                logger.debug("[PlanExecute] 步骤 %d: %s", i, step)
            return {"plan": plan_obj.steps, "iterations": 0}
        except Exception as exc:
            logger.warning("[PlanExecute] 计划生成失败，使用默认计划：%s", exc)
            # 降级：生成默认计划
            order_id = state.get("order_id", "")
            return {
                "plan": [
                    f"使用 analyze_order 分析订单 {order_id} 的详情",
                    f"使用 check_inventory 检查订单 {order_id} 的库存状态",
                    "使用 generate_fulfillment_plan 生成最优履约方案",
                ],
                "iterations": 0,
            }

    return planner


def _make_executor_node(sub_agent):
    """Executor 节点：执行计划中的第一个待完成步骤。"""

    async def executor(state: PlanExecuteState) -> dict:
        plan = state.get("plan", [])
        if not plan:
            return {"past_steps": [("(无步骤)", "计划已全部完成")], "iterations": state.get("iterations", 0) + 1}

        current_step = plan[0]

        # 把前几步的结果作为上下文传给 executor
        past = state.get("past_steps", [])
        context_lines = [f"- [{s}] → {r[:200]}" for s, r in past[-3:]]  # 只取最近 3 步
        context = "\n".join(context_lines)

        task_msg = current_step
        if context:
            task_msg = f"已知上下文：\n{context}\n\n当前任务：{current_step}"

        try:
            run_id = state.get("run_id") or str(id(state))
            iteration = state.get("iterations", 0)
            result = await sub_agent.ainvoke(
                {"messages": [HumanMessage(content=task_msg)]},
                config={"configurable": {"thread_id": f"pe_exec_{run_id}_{iteration}"}},
            )
            ai_msgs = filter_messages(result.get("messages", []), include_types=[AIMessage])
            final_replies = [m for m in ai_msgs if not m.tool_calls]
            reply = str(final_replies[-1].content) if final_replies else "步骤执行完成，但无法提取结果。"
        except Exception as exc:
            reply = f"步骤执行异常：{exc}"
            logger.warning("[PlanExecute] Executor 异常：%s", exc)

        logger.info("[PlanExecute] 步骤完成：%s", current_step[:60])
        return {
            "past_steps": [(current_step, reply)],
            "plan": plan[1:],   # 移除已执行的步骤
            "iterations": state.get("iterations", 0) + 1,
        }

    return executor


def _make_replanner_node(llm: BaseChatModel):
    """Replanner 节点：根据已有结果判断是否继续、以及更新剩余计划。"""
    replanner_chain = _REPLANNER_PROMPT | llm.with_structured_output(ReplanDecision)

    async def replanner(state: PlanExecuteState) -> dict:
        past = state.get("past_steps", [])
        past_text = "\n".join(
            f"{i+1}. [{step}]\n   结果：{result[:300]}"
            for i, (step, result) in enumerate(past)
        )
        plan_text = "\n".join(f"- {s}" for s in state.get("plan", [])) or "（无剩余步骤）"

        try:
            decision: ReplanDecision = await replanner_chain.ainvoke({
                "input": state.get("input", ""),
                "past_steps_text": past_text,
                "plan_text": plan_text,
            })
            return {"plan": decision.remaining_steps}
        except Exception as exc:
            logger.warning("[PlanExecute] Replanner 异常，清空剩余计划：%s", exc)
            return {"plan": []}   # 发生异常时结束循环，进入综合

    return replanner


def _make_synthesizer_node(llm: BaseChatModel):
    """Synthesizer 节点：综合所有步骤结果生成最终答复。"""
    chain = _SYNTHESIZER_PROMPT | llm | StrOutputParser()

    async def synthesizer(state: PlanExecuteState) -> dict:
        past = state.get("past_steps", [])
        past_text = "\n".join(
            f"{i+1}. [{step}]\n   {result[:400]}"
            for i, (step, result) in enumerate(past)
        )
        try:
            answer = await chain.ainvoke({
                "input": state.get("input", ""),
                "past_steps_text": past_text,
            })
        except Exception as exc:
            logger.warning("[PlanExecute] Synthesizer 异常：%s", exc)
            answer = "综合结果：\n" + "\n".join(f"• [{s}]：{r[:150]}" for s, r in past)
        return {"final_answer": answer}

    return synthesizer


# ── 路由函数 ──────────────────────────────────────────────────────────────────

def _make_route_fn(max_iterations: int):
    """条件边路由：plan 非空且未超过最大迭代次数 → 继续执行；否则 → 综合。"""

    def route(state: PlanExecuteState) -> str:
        has_plan = bool(state.get("plan"))
        under_limit = state.get("iterations", 0) < max_iterations
        if has_plan and under_limit:
            return "executor"
        if state.get("iterations", 0) >= max_iterations:
            logger.warning("[PlanExecute] 达到最大迭代次数 %d，强制进入综合", max_iterations)
        return "synthesizer"

    return route


# ── 图组装 ────────────────────────────────────────────────────────────────────

# 面试官可能问：什么时候用 Plan-and-Execute，不用普通 ReAct？
# 回答：当问题有明显前后依赖、需要先规划再执行时使用，比如完整履约方案分析。
# 普通 ReAct 更适合开放追问和局部决策，简单问题用 Plan-and-Execute 会显得重。
def build_plan_execute_agent(
    llm: BaseChatModel,
    tools: list[BaseTool],
    max_iterations: int = 5,
    checkpointer=None,
    store=None,
) -> CompiledStateGraph:
    """构建 Plan-and-Execute LangGraph 图。

    Args:
        llm:            LLM 模型（用于 Planner / Replanner / Synthesizer）
        tools:          工具列表（Executor 子 Agent 使用）
        max_iterations: 最大执行轮次（防死循环，默认 5）
        checkpointer:   状态持久化（默认 MemorySaver）
    """
    _checkpointer = checkpointer or MemorySaver()

    # Executor 使用的子 Agent（mini ReAct，执行单个步骤）
    sub_agent_kwargs = {
        "model": llm,
        "tools": tools,
        "system_prompt": _EXECUTOR_SYSTEM,
        "checkpointer": _checkpointer,
    }
    if store is not None:
        sub_agent_kwargs["store"] = store
    sub_agent = create_agent(**sub_agent_kwargs)

    graph = StateGraph(PlanExecuteState)
    graph.add_node("planner", _make_planner_node(llm))
    graph.add_node("executor", _make_executor_node(sub_agent))
    graph.add_node("replanner", _make_replanner_node(llm))
    graph.add_node("synthesizer", _make_synthesizer_node(llm))

    # 边：固定顺序
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "replanner")

    # 条件边：有剩余步骤 → executor 循环；无剩余 → synthesizer
    graph.add_conditional_edges(
        "replanner",
        _make_route_fn(max_iterations),
        {"executor": "executor", "synthesizer": "synthesizer"},
    )
    graph.add_edge("synthesizer", END)

    compile_kwargs = {"checkpointer": _checkpointer}
    if store is not None:
        compile_kwargs["store"] = store
    return graph.compile(**compile_kwargs)
