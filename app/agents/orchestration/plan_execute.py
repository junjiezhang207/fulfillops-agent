"""Plan-and-Execute Agent 编排。

本模块实现“先生成计划，再按步骤执行，并在每步后重新评估计划”的
LangGraph 流程。它适合有明确步骤依赖的复杂履约分析。

图结构：
    planner -> executor -> replanner -> executor/synthesizer

主要节点：
1. ``Planner``：生成结构化执行计划。
2. ``Executor``：用子 ReAct Agent 执行当前步骤。
3. ``Replanner``：根据已完成结果更新剩余步骤。
4. ``Synthesizer``：汇总最终答案。
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
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from app.infrastructure.llm.model_gateway import get_model_gateway

logger = logging.getLogger(__name__)

# ── Pydantic 结构化输出模型 ───────────────────────────────────────────────────

# Planner 使用结构化输出，避免后续节点解析自由文本计划。
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

# past_steps 保存已执行步骤和结果，供 Replanner 避免重复执行，
# 也供 Synthesizer 生成有依据的综合答案。
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

_PLANNER_SYSTEM = get_model_gateway().prompt_system(
    use_case="plan_execute",
    prompt_id="plan_execute_planner",
)

_PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _PLANNER_SYSTEM),
    ("human", "订单ID: {order_id}\n用户问题: {input}"),
])

_REPLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", get_model_gateway().prompt_system(use_case="plan_execute", prompt_id="plan_execute_replanner")),
    ("human", (
        "用户目标: {input}\n\n"
        "已完成步骤及结果:\n{past_steps_text}\n\n"
        "当前剩余计划:\n{plan_text}\n\n"
        "请决定是否继续，并更新剩余步骤列表。"
    )),
])

_SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", get_model_gateway().prompt_system(use_case="plan_execute", prompt_id="plan_execute_synthesizer")),
    ("human", "用户问题: {input}\n\n各步骤结果:\n{past_steps_text}"),
])

_EXECUTOR_SYSTEM = get_model_gateway().prompt_system(
    use_case="plan_execute",
    prompt_id="plan_execute_executor",
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

# Plan-and-Execute 适合有前后依赖的复杂任务；简单查询仍优先使用普通 ReAct。
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
        checkpointer:   PostgreSQL 状态持久化（生产模式必传）
    """
    if checkpointer is None:
        raise RuntimeError("Plan-and-Execute 必须显式传入 PostgreSQL checkpointer。")

    # Executor 使用的子 Agent（mini ReAct，执行单个步骤）
    sub_agent_kwargs = {
        "model": llm,
        "tools": tools,
        "system_prompt": _EXECUTOR_SYSTEM,
        "checkpointer": checkpointer,
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

    compile_kwargs = {"checkpointer": checkpointer}
    if store is not None:
        compile_kwargs["store"] = store
    return graph.compile(**compile_kwargs)
