"""LangGraph 工作流节点函数集合。

为什么把节点组织在一个类里？
    LangGraph 的节点本质是"接受 state，返回 state 增量"的函数。
    但节点需要调用已有 service（订单/库存/知识/LLM），
    最自然的做法就是把 service 作为类成员，节点是类方法。
    这样：
      1. service 依赖只在构造时注入一次
      2. 每个方法都是天然的闭包，天生适配 graph.add_node
      3. 单元测试时只需 mock service 属性即可

节点实现约束（全文件统一）：
    1. 只接受 `state: GraphState`，只返回 dict 增量。
    2. 捕获异常 → 写 errors；不 raise，不阻断主流程
       （dispatch_node 的参数缺失例外，它会直接抛 ValueError 让上层挡住）。
    3. 每个节点无论成功失败，都必须写一条 TraceEvent。

学习时可以把每个节点看成“流水线上的工位”：
    dispatch             只检查入参、记录启动。
    order_analysis       查订单事实。
    inventory_analysis   查库存，并决定快路径/缺货路径；必要时触发人工审批。
    knowledge_retrieval  缺货时检索规则。
    finalize             把前面所有事实组织成最终答案。
"""

from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command, interrupt

from app.workflows.fulfillment.risk_evaluator import RiskEvaluator
from app.workflows.fulfillment.state import GraphState
from app.workflows.fulfillment.trace import ErrorEvent, build_trace_event
from app.schemas.workflow import FinalAnswer
from app.domain.inventory.analysis import InventoryAnalysisService
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
from app.domain.orders.analysis import (
    OrderAnalysisService,
    OrderNotFoundError,
)

# ── finalize 节点的 LLM Prompt ───────────────────────────────────────────────
# 替代旧版的 LLMPort.summarize() 调用；使用标准 LCEL 链：
#   _FINALIZE_PROMPT | chat_model | StrOutputParser()
_FINALIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个供应链履约决策助手。"
        "你会收到订单信息、库存状态和可参考的履约规则，"
        "请根据这些信息用 2~3 句话给出简洁、可操作的履约结论和首要建议。"
        "不要重复输入内容，直接给出结论。",
    ),
    ("human", "{context}"),
])


class WorkflowNodes:
    """工作流节点集合。

    每个 public 方法都是一个 LangGraph 节点，签名统一为：
        (state: GraphState) -> dict
    """

    def __init__(
        self,
        order_service: OrderAnalysisService,
        inventory_service: InventoryAnalysisService,
        knowledge_service: KnowledgeRetrievalService,
        chat_model: BaseChatModel | None = None,
    ) -> None:
        """把外部 service 注入到节点集合。

        LangGraph 本身不负责依赖注入；它只会调用节点函数。
        所以这里先把 service 存成成员变量，节点执行时再使用。
        """
        self.order_service = order_service
        self.inventory_service = inventory_service
        self.knowledge_service = knowledge_service
        self._chat_model = chat_model

    # ------------------------------------------------------------------ #
    # 节点 1：dispatch_node
    # ------------------------------------------------------------------ #
    def dispatch(self, state: GraphState) -> dict:
        """入口调度节点。

        当前职责（本回合）：
          1. 校验入参（order_id 必填）
          2. 写入一条"工作流启动"的 trace

        未来扩展（下一回合 / 更远）：
          - 调用 LLM 做真正的"意图识别"并决定跳过哪些节点
          - 初始化 checkpointer 上下文
          - Human-in-the-loop 鉴权

        输入依赖：order_id / question / filter_categories
        产出字段：trace
        """
        start_ts = datetime.now()

        order_id = state.get("order_id")
        if not order_id:
            # dispatch_node 的参数错误是唯一一种"允许抛异常"的情况，
            # 因为此时整条链路不应当继续推进。
            raise ValueError("order_id 不能为空。")

        end_ts = datetime.now()
        note = (
            f"工作流启动；question={'有' if state.get('question') else '无'}；"
            f"filter_categories={state.get('filter_categories') or []}"
        )
        # 节点只返回增量，不返回整个 state。
        # LangGraph 会把 trace 追加到 GraphState.trace，因为 trace 字段配置了 add reducer。
        return {
            "trace": [
                build_trace_event(
                    node="dispatch", start_ts=start_ts, end_ts=end_ts, status="ok", note=note
                )
            ],
        }

    # ------------------------------------------------------------------ #
    # 节点 2：order_analysis_node
    # ------------------------------------------------------------------ #
    def order_analysis(self, state: GraphState) -> Command | dict:
        """调用订单分析服务。

        错误恢复策略（LangGraph Command API）：
          OrderNotFoundError → 不可恢复，Command(goto="finalize") 跳过中间所有节点，
                               直接带 error 信息落地，而不是继续走空状态。
          其他异常            → 写 errors 继续推进（下游节点有空值兜底）。

        输入依赖：state["order_id"]
        产出字段：state["order_result"]

        这里返回类型是 Command | dict：
          - dict 表示“正常更新 state，然后按图上的边继续走”。
          - Command(goto=...) 表示“更新 state 后跳到指定节点”，适合异常兜底。
        """
        start_ts = datetime.now()
        try:
            result = self.order_service.analyze_order(state["order_id"])
        except OrderNotFoundError as exc:
            end_ts = datetime.now()
            # 订单不存在是不可恢复错误：用 Command 直接跳到 finalize，
            # 跳过 inventory_analysis / knowledge_retrieval，避免空状态传播。
            return Command(
                goto="finalize",
                update={
                    "errors": [
                        ErrorEvent(
                            node="order_analysis",
                            message=str(exc),
                            exception_type=type(exc).__name__,
                        )
                    ],
                    "trace": [
                        build_trace_event(
                            node="order_analysis",
                            start_ts=start_ts,
                            end_ts=end_ts,
                            status="error",
                            note="订单不存在，跳过中间节点直达 finalize",
                        )
                    ],
                },
            )
        except Exception as exc:
            end_ts = datetime.now()
            # 普通异常不强制跳转，返回 errors 后让流程继续。
            # 下游节点通过空值判断自行降级，finalize 负责把错误体现在结果里。
            return {
                "errors": [
                    ErrorEvent(
                        node="order_analysis",
                        message=str(exc),
                        exception_type=type(exc).__name__,
                    )
                ],
                "trace": [
                    build_trace_event(
                        node="order_analysis",
                        start_ts=start_ts,
                        end_ts=end_ts,
                        status="error",
                    )
                ],
            }

        end_ts = datetime.now()
        note = f"已提取 {result.item_count} 行商品，总件数 {result.total_quantity}"
        return {
            "order_result": result,
            "trace": [
                build_trace_event(
                    node="order_analysis",
                    start_ts=start_ts,
                    end_ts=end_ts,
                    status="ok",
                    note=note,
                )
            ],
        }

    # ------------------------------------------------------------------ #
    # 节点 3：inventory_analysis_node
    # ------------------------------------------------------------------ #
    def inventory_analysis(self, state: GraphState) -> dict:
        """调用库存判断服务，并根据结果设置 fulfillment_branch。

        这个节点比其他节点多做一件事：
          把 fulfillment_branch 字段写入 state。
          它不是业务结果的一部分，而是给"条件边"读的"路由指示灯"。

        为什么在这个节点写分支字段，而不是在条件边函数里算？
          1. 条件边函数应保持极简（只读 state、返回节点名）。
          2. 把业务判定语义放在节点内部，方便测试和单独复用。

        输入依赖：order_result / order_id
        产出字段：inventory_result / fulfillment_branch / risk_level / risk_signals
        """
        start_ts = datetime.now()

        # 如果上游节点已经失败，跳过本节点。
        # 这是简单的"短路"策略：错误会被 finalize 感知。
        if state.get("order_result") is None:
            end_ts = datetime.now()
            return {
                "trace": [
                    build_trace_event(
                        node="inventory_analysis",
                        start_ts=start_ts,
                        end_ts=end_ts,
                        status="skipped",
                        note="上游 order_analysis 未产出结果",
                    )
                ],
                # 分支兜底为缺货方向，交给下游处理
                "fulfillment_branch": "stockout",
            }

        try:
            result = self.inventory_service.analyze_inventory(state["order_id"])
        except Exception as exc:
            end_ts = datetime.now()
            return {
                "errors": [
                    ErrorEvent(
                        node="inventory_analysis",
                        message=str(exc),
                        exception_type=type(exc).__name__,
                    )
                ],
                "trace": [
                    build_trace_event(
                        node="inventory_analysis",
                        start_ts=start_ts,
                        end_ts=end_ts,
                        status="error",
                    )
                ],
                "fulfillment_branch": "stockout",
            }

        branch = "fulfillable" if result.fulfillment_ready else "stockout"
        end_ts = datetime.now()
        note = (
            "库存充足，可直接履约"
            if result.fulfillment_ready
            else f"库存不足 SKU：{'、'.join(result.insufficient_skus)}"
        )

        # ----- Human-in-the-Loop：多规则风险评估驱动的中断 -----
        # 这一段是 workflow 和普通函数链最大的不同点之一：
        # 节点可以调用 interrupt() 暂停整张图，把审批问题交给人。
        # 审批完成后，上层用 Command(resume="approved"|"rejected"|...) 回到这里继续。
        human_decision = state.get("human_decision")
        if not human_decision:
            evaluator = RiskEvaluator()
            # 构建风险上下文（从 order_result 和 inventory_result 提取关键指标）。
            # RiskEvaluator 不依赖 LangGraph，它只是一个纯业务规则评估器。
            order_r = state.get("order_result")
            risk_context = {
                "order_id": state["order_id"],
                "order_value": getattr(order_r, "total_amount", 0) if order_r else 0,
                "customer_level": getattr(order_r, "customer_level", "") if order_r else "",
                "hours_to_deadline": getattr(order_r, "hours_to_deadline", 999) if order_r else 999,
                "stock_gap_ratio": (
                    len(result.insufficient_skus) / max(result.item_count, 1)
                    if not result.fulfillment_ready else 0.0
                ),
                "split_order_required": len(result.insufficient_skus) > 0 and result.fulfillment_ready is False,
                "sku_types": getattr(order_r, "sku_types", []) if order_r else [],
                "cross_region": getattr(order_r, "cross_region", False) if order_r else False,
            }

            assessment = evaluator.evaluate(risk_context)

            if assessment.needs_human_review:
                # HIGH / CRITICAL：必须人工审批。
                # interrupt_info 是给前端/审批台看的结构化信息。
                interrupt_context = evaluator.build_interrupt_context(
                    order_id=state["order_id"],
                    assessment=assessment,
                    inventory_summary=result.summary,
                    order_summary=getattr(order_r, "summary", "") if order_r else "",
                )
                interrupt_info = {
                    "type": assessment.interrupt_type,
                    "node": "inventory_analysis",
                    "prompt": self._build_interrupt_prompt(assessment),
                    "context": interrupt_context,
                    "options": ["approved", "rejected", "escalate"],
                    "risk_level": assessment.max_level.name,
                    "risk_signals": assessment.signal_names,
                    "timeout_seconds": assessment.timeout_seconds(),
                }
                try:
                    decision = interrupt(interrupt_info)
                except RuntimeError as exc:
                    # 单元测试或脚本直接调用节点函数时，不存在 LangGraph runnable
                    # context，interrupt() 无法暂停图。真实 workflow 运行时仍走
                    # interrupt；直接调用时返回 pending 信息并保持缺货分支兜底。
                    if "outside of a runnable context" not in str(exc):
                        raise
                    return {
                        "interrupt_info": interrupt_info,
                        "risk_level": assessment.max_level.name,
                        "risk_signals": assessment.signal_names,
                        "inventory_result": result,
                        "fulfillment_branch": "stockout",
                        "trace": [
                            build_trace_event(
                                node="inventory_analysis",
                                start_ts=start_ts,
                                end_ts=datetime.now(),
                                status="ok",
                                note=f"等待人工审批（{assessment.max_level.name}）",
                            )
                        ],
                    }
                # interrupt 返回后，说明 workflow 已经被 resume。
                # decision 就是审批员给出的结果，后面根据它改 fulfillment_branch。
                return {
                    "interrupt_info": interrupt_info,
                    "risk_level": assessment.max_level.name,
                    "risk_signals": assessment.signal_names,
                    "human_decision": {"decision": decision, "node": "inventory_analysis"},
                    "inventory_result": result,
                    "fulfillment_branch": "fulfillable" if decision == "approved" else "stockout",
                    "trace": [
                        build_trace_event(
                            node="inventory_analysis",
                            start_ts=start_ts,
                            end_ts=datetime.now(),
                            status="ok",
                            note=f"人工决策（{assessment.max_level.name}）：{decision}",
                        )
                    ],
                }
            else:
                # LOW / MEDIUM：自动放行，打标记但不打断流程。
                # 这样最终结果里仍能看到风险信号，但用户不会被要求审批。
                note = (
                    f"自动放行（{assessment.max_level.name}）"
                    + (f"，低风险信号：{assessment.summary}" if assessment.signals else "，无风险信号")
                )
                return {
                    "inventory_result": result,
                    "risk_level": assessment.max_level.name,
                    "risk_signals": assessment.signal_names,
                    "fulfillment_branch": branch,
                    "trace": [
                        build_trace_event(
                            node="inventory_analysis",
                            start_ts=start_ts,
                            end_ts=datetime.now(),
                            status="ok",
                            note=note,
                        )
                    ],
                }

        return {
            "inventory_result": result,
            "fulfillment_branch": branch,
            "trace": [
                build_trace_event(
                    node="inventory_analysis",
                    start_ts=start_ts,
                    end_ts=end_ts,
                    status="ok",
                    note=note,
                )
            ],
        }

    # ------------------------------------------------------------------ #
    # 节点 4：knowledge_retrieval_node
    # ------------------------------------------------------------------ #
    def knowledge_retrieval(self, state: GraphState) -> dict:
        """调用知识检索服务。

        进入本节点的两种情况：
          1. 库存不足，走完整路径
          2. （未来扩展）用户强制指定必须检索知识

        当 question 为 None 时，使用一个兜底提问，
        确保知识检索服务能执行完整流程。

        输入依赖：order_id / question / filter_categories
        产出字段：knowledge_result
        """
        start_ts = datetime.now()
        question = state.get("question") or "当前订单缺货时应遵循什么履约规则？"
        filter_categories = state.get("filter_categories") or []
        try:
            # 这里调用的是项目的 RAG 服务。
            # Workflow 不关心 RAG 内部怎么检索，只依赖它返回 KnowledgeRetrieveResult。
            result = self.knowledge_service.retrieve(
                order_id=state["order_id"],
                question=question,
                filter_categories=filter_categories,
            )
        except Exception as exc:
            end_ts = datetime.now()
            return {
                "errors": [
                    ErrorEvent(
                        node="knowledge_retrieval",
                        message=str(exc),
                        exception_type=type(exc).__name__,
                    )
                ],
                "trace": [
                    build_trace_event(
                        node="knowledge_retrieval",
                        start_ts=start_ts,
                        end_ts=end_ts,
                        status="error",
                    )
                ],
            }

        end_ts = datetime.now()
        note = (
            f"命中 {len(result.hits)} 条规则，"
            f"覆盖类别：{'、'.join(result.matched_categories) or '无'}"
        )
        return {
            "knowledge_result": result,
            "trace": [
                build_trace_event(
                    node="knowledge_retrieval",
                    start_ts=start_ts,
                    end_ts=end_ts,
                    status="ok",
                    note=note,
                )
            ],
        }

    # ------------------------------------------------------------------ #
    # 节点 5：finalize_node
    # ------------------------------------------------------------------ #
    def finalize(self, state: GraphState) -> dict:
        """汇总节点：把三个中间结果整合成最终答复。

        教学重点：
          finalize 是“读 state、组织答案”的出口节点，不重新查订单、不重新查库存、
          也不自己做 RAG。这样可以保证最终答案的依据都能回溯到前面节点的 trace。

        当前执行策略：
          - key_evidences：从订单、库存、知识检索结果中抽取证据。
          - suggested_actions：缺货路径优先使用知识库 SOP；快路径用库存状态派生建议。
          - conclusion：优先调用注入的 LangChain chat_model；没有模型或模型失败时，
            使用规则模板 fallback，保证 workflow 在本地演示环境仍然可用。

        输入依赖：order_result / inventory_result / knowledge_result / errors
        产出字段：final_answer / trace
        """
        start_ts = datetime.now()

        order_result = state.get("order_result")
        inventory_result = state.get("inventory_result")
        knowledge_result = state.get("knowledge_result")

        # 根据是否走过知识检索，标记路径
        # 这个字段方便前端或评测系统判断本次执行是快路径还是缺货知识路径。
        routing_path = "knowledge_path" if knowledge_result is not None else "fast_path"

        key_evidences = self._collect_key_evidences(
            order_result, inventory_result, knowledge_result
        )
        suggested_actions = self._collect_suggested_actions(
            inventory_result, knowledge_result
        )
        conclusion = self._build_conclusion(
            order_result, inventory_result, knowledge_result, routing_path
        )

        final_answer = FinalAnswer(
            conclusion=conclusion,
            key_evidences=key_evidences,
            suggested_actions=suggested_actions,
            routing_path=routing_path,
        )

        end_ts = datetime.now()
        return {
            "final_answer": final_answer,
            "trace": [
                build_trace_event(
                    node="finalize",
                    start_ts=start_ts,
                    end_ts=end_ts,
                    status="ok",
                    note=f"路径={routing_path}",
                )
            ],
        }

    # ------------------------------------------------------------------ #
    # HITL 辅助方法（非节点）
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_interrupt_prompt(assessment) -> str:
        """根据风险评估结果生成给审批员的提示文本。"""
        lines = [f"[{assessment.max_level.name}] 本订单触发以下风险规则，需要您审批："]
        for sig in assessment.signals:
            lines.append(f"  - {sig.reason}")
            if sig.suggested_action:
                lines.append(f"    建议：{sig.suggested_action}")
        lines.append("\n请选择处理方式：")
        lines.append("  approved  — 确认履约，按推荐方案执行")
        lines.append("  rejected  — 拒绝，退回重新处理")
        lines.append("  escalate  — 上报，转交高级别处理")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # finalize 的内部辅助方法（非节点）
    # ------------------------------------------------------------------ #
    def _collect_key_evidences(
        self,
        order_result,
        inventory_result,
        knowledge_result,
    ) -> list[str]:
        """收集最终答案里的“依据”。

        finalize 不应该重新发明事实，只应该引用前面节点已经产出的事实。
        所以这里从 order/inventory/knowledge 三个结果中抽取摘要和关键规则。
        """
        evidences: list[str] = []
        if order_result is not None:
            evidences.append(f"订单事实：{order_result.summary}")
        if inventory_result is not None:
            evidences.append(f"库存事实：{inventory_result.summary}")
        if knowledge_result is not None and knowledge_result.answer_summary:
            # 只取知识层的 key_rules 前 3 条，避免最终结论冗长
            for rule in knowledge_result.answer_summary.key_rules[:3]:
                evidences.append(f"规则依据：{rule}")
        return evidences

    def _collect_suggested_actions(
        self,
        inventory_result,
        knowledge_result,
    ) -> list[str]:
        """收集最终答案里的“建议动作”。

        缺货路径优先采用知识库建议，因为它更贴近业务 SOP；
        快路径没有知识结果时，就根据库存状态给一个保守建议。
        """
        # 优先用知识检索的建议（业务语义最强）
        if knowledge_result is not None and knowledge_result.answer_summary:
            return list(knowledge_result.answer_summary.suggested_actions)

        # fast_path：没有知识检索结果时，退化为简单的基于库存的建议
        if inventory_result is None:
            return []
        if inventory_result.fulfillment_ready:
            return ["按标准区域仓配策略发货，并监控时效。"]
        return ["进入缺货处理 SOP 并提示人工复核。"]

    def _build_conclusion(
        self,
        order_result,
        inventory_result,
        knowledge_result,
        routing_path: str,
    ) -> str:
        """生成最终自然语言结论。

        这不是一个“让 LLM 自由发挥”的节点：
          1. LLM 只能看到 _build_context_for_llm 组织出的事实上下文。
          2. Prompt 要求简洁给出履约结论，不要求模型重新推理库存。
          3. 模型失败时静默降级到规则模板，避免最终出口因为话术模型不可用而失败。
        """
        context = self._build_context_for_llm(order_result, inventory_result, knowledge_result)

        # LangChain LCEL 链：_FINALIZE_PROMPT | chat_model | StrOutputParser()
        # 替代旧版的 LLMPort.summarize()，直接使用标准 LangChain 接口。
        # chat_model 为 None（未配置 LLM）时走规则模板 fallback。
        if self._chat_model is not None:
            try:
                chain = _FINALIZE_PROMPT | self._chat_model | StrOutputParser()
                llm_output = chain.invoke({"context": context})
                if llm_output:
                    return llm_output
            except Exception:
                pass  # LLM 调用失败时降级到规则模板

        # 规则模板 fallback（无 LLM 或调用失败时）
        if inventory_result is None:
            return "当前订单信息不完整，主链路未能给出完整结论，请检查上游节点错误。"
        if inventory_result.fulfillment_ready:
            return (
                f"订单 {inventory_result.order_id} 库存充足，"
                f"已走快速路径（{routing_path}），建议按区域仓配策略尽快发货。"
            )
        return (
            f"订单 {inventory_result.order_id} 存在缺货风险，"
            f"已走知识检索路径（{routing_path}），"
            f"请参考上述规则依据与建议动作执行缺货履约流程。"
        )

    def _build_context_for_llm(
        self,
        order_result,
        inventory_result,
        knowledge_result,
    ) -> str:
        """把三个模块的结果组织成带结构标签的 LLM 输入文本。

        为什么要加【标签】？
            让模型清楚地知道"这段是订单信息"、"这段是库存状态"、"这段是规则参考"，
            避免把三段内容当成无结构的流水文本处理，输出质量更稳定。
        """
        parts: list[str] = []
        if order_result is not None:
            parts.append(f"【订单信息】\n{order_result.summary}")
        if inventory_result is not None:
            parts.append(f"【库存状态】\n{inventory_result.summary}")
        if knowledge_result is not None and knowledge_result.answer_summary:
            rules = knowledge_result.answer_summary.key_rules
            if rules:
                rules_text = "\n".join(f"- {r}" for r in rules)
                parts.append(f"【履约规则参考】\n{rules_text}")
        return "\n\n".join(parts)
