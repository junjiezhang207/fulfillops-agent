"""LangGraph 工作流节点函数集合。

节点组织在类中，便于在构造时注入订单、库存、知识库和 LLM 等服务。
每个 public 方法都是可注册到 LangGraph 的节点函数，单元测试时也可以直接
mock 对应 service 成员。

节点实现约束（全文件统一）：
    1. 只接受 `state: GraphState`，只返回 dict 增量。
    2. 捕获异常 → 写 errors；不 raise，不阻断主流程
       （dispatch_node 的参数缺失例外，它会直接抛 ValueError 让上层挡住）。
    3. 每个节点无论成功失败，都必须写一条 TraceEvent。

节点职责：
    dispatch             只检查入参、记录启动。
    order_analysis       查订单事实。
    inventory_analysis   查库存，并决定快路径/缺货路径；必要时触发人工审批。
    knowledge_retrieval  缺货时检索规则。
    finalize             把前面所有事实组织成最终答案。
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command, interrupt

from app.agents.tools.read_only_gateway import ReadOnlyToolGateway
from app.agents.tools.registry import ToolServiceBundle
from app.application.routing.order_context_service import OrderContextService
from app.workflows.fulfillment.risk_evaluator import RiskEvaluator
from app.workflows.fulfillment.state import GraphState
from app.workflows.fulfillment.trace import ErrorEvent, build_trace_event
from app.infrastructure.llm.model_gateway import get_model_gateway
from app.observability.business_trace import add_trace_step
from app.schemas.workflow import (
    ExecutionProposal,
    FinalAnswer,
    PreflightValidation,
    ProposalAction,
)
from app.domain.inventory.analysis import InventoryAnalysisService
from app.domain.fulfillment.context_builder import OrderContextBuilder
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
from app.rag.rag_context_service import PlannerRAGContextService
from app.domain.orders.analysis import (
    OrderAnalysisService,
    OrderNotFoundError,
)


ALLOWED_PROPOSAL_ACTION_TYPES = {
    "ship_from_warehouse",
    "switch_warehouse",
    "split_order",
    "merge_order",
    "inventory_transfer",
    "change_carrier",
    "logistics_exception",
    "replenishment",
    "stockout_resolution",
    "customer_complaint",
}

FORBIDDEN_DIRECT_MUTATION_ACTION_TYPES = {
    "direct_order_mutation",
    "direct_inventory_mutation",
    "direct_split_order",
    "direct_inventory_transfer",
    "direct_waybill_mutation",
    "direct_carrier_core_data_mutation",
    "modify_order",
    "update_inventory",
    "cancel_order",
    "create_waybill",
    "direct_ship",
}

PROGRESSIVE_CONTEXT_TOOL_BY_SOURCE = {
    "OMS": "get_order_detail",
    "WMS": "get_inventory_warehouse_detail",
    "TMS": "get_shipping_detail",
    "ERP": "get_supply_chain_detail",
    "PIM": "get_product_constraints",
    "CRM": "get_customer_case_context",
}

PROGRESSIVE_CONTEXT_SOURCE_BY_FIELD = {
    "order_main": "OMS",
    "order_items": "OMS",
    "fulfillment_state": "OMS",
    "candidate_warehouses": "WMS",
    "sku_warehouse_inventory": "WMS",
    "logistics_options": "TMS",
    "replenishment_options": "ERP",
    "product_restrictions": "PIM",
    "customer_risk": "CRM",
}

# ── finalize 节点的 LLM Prompt ───────────────────────────────────────────────
# 替代旧版的 LLMPort.summarize() 调用；使用标准 LCEL 链：
#   _FINALIZE_PROMPT | chat_model | StrOutputParser()
_FINALIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        get_model_gateway().prompt_system(use_case="workflow_finalize"),
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
        self.context_builder = OrderContextBuilder(
            logistics_quote=self._logistics_quote,
            eta_label=self._eta_label,
        )
        self.context_service = OrderContextService(
            order_service=order_service,
            inventory_service=inventory_service,
            context_builder=self.context_builder,
        )
        self.read_only_tool_gateway = ReadOnlyToolGateway()
        self.rag_context_service = PlannerRAGContextService(knowledge_service)

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
        """调用库存判断服务并写入风险评分。

        这个节点只负责事实读取和风险判断，不再直接触发人工中断。
        提案会在 proposal_generation 节点写入 checkpoint，human_approval 节点
        再基于该快照 interrupt，这样审批恢复后才能做旧方案失效判断。
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
                # 分支兜底为缺货方向，交给下游处理。
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

        evaluator = RiskEvaluator()
        order_r = state.get("order_result")
        risk_context = {
            "order_id": state["order_id"],
            "order_value": self._order_value(order_r),
            "customer_level": getattr(order_r, "customer_level", "") if order_r else "",
            "hours_to_deadline": getattr(order_r, "hours_to_deadline", 999) if order_r else 999,
            "stock_gap_ratio": (
                len(result.insufficient_skus) / max(len(result.sku_checks), 1)
                if not result.fulfillment_ready else 0.0
            ),
            "split_order_required": bool(result.insufficient_skus),
            "sku_types": getattr(order_r, "sku_types", []) if order_r else [],
            "cross_region": getattr(order_r, "cross_region", False) if order_r else False,
        }
        assessment = evaluator.evaluate(risk_context)
        if assessment.signal_names:
            note = f"{note}；风险={assessment.max_level.name}；信号={','.join(assessment.signal_names)}"
        return {
            "inventory_result": result,
            "fulfillment_branch": branch,
            "risk_level": assessment.max_level.name,
            "risk_signals": assessment.signal_names,
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
    # 节点 4：proposal_generation_node
    # ------------------------------------------------------------------ #
    def proposal_generation(self, state: GraphState) -> dict:
        """生成可审批的履约执行提案。"""
        start_ts = datetime.now()
        order_result = state.get("order_result")
        inventory_result = state.get("inventory_result")
        if order_result is None or inventory_result is None:
            return {
                "trace": [
                    build_trace_event(
                        node="proposal_generation",
                        start_ts=start_ts,
                        end_ts=datetime.now(),
                        status="skipped",
                        note="订单或库存结果缺失，跳过执行提案生成",
                    )
                ],
            }

        proposal = self._build_execution_proposal(
            order_result,
            inventory_result,
            question=state.get("question"),
            session_memory=state.get("session_memory"),
        )
        return {
            "execution_proposal": proposal,
            "trace": [
                build_trace_event(
                    node="proposal_generation",
                    start_ts=start_ts,
                    end_ts=datetime.now(),
                    status="ok",
                    note=f"生成执行提案：{proposal.title}，动作 {len(proposal.actions)} 个",
                )
            ],
        }

    # ------------------------------------------------------------------ #
    # 节点 5：human_approval_node
    # ------------------------------------------------------------------ #
    def human_approval(self, state: GraphState) -> dict:
        """HITL 审批节点；批准后执行二次校验。"""
        start_ts = datetime.now()
        proposal = state.get("execution_proposal")
        if proposal is None:
            return {
                "fulfillment_branch": state.get("fulfillment_branch") or "stockout",
                "trace": [
                    build_trace_event(
                        node="human_approval",
                        start_ts=start_ts,
                        end_ts=datetime.now(),
                        status="skipped",
                        note="无执行提案，跳过人工审批",
                    )
                ],
            }

        risk_level = state.get("risk_level") or "LOW"
        needs_review = proposal.approval_required or risk_level in {"HIGH", "CRITICAL"}
        if not needs_review:
            return {
                "fulfillment_branch": state.get("fulfillment_branch") or "fulfillable",
                "trace": [
                    build_trace_event(
                        node="human_approval",
                        start_ts=start_ts,
                        end_ts=datetime.now(),
                        status="ok",
                        note="低风险提案自动放行",
                    )
                ],
            }

        interrupt_info = {
            "type": "fulfillment_action_approval",
            "node": "human_approval",
            "prompt": self._build_action_card_prompt(proposal, risk_level, state.get("risk_signals") or []),
            "context": {
                "order_id": proposal.order_id,
                "proposal": proposal.model_dump(mode="json"),
                "inventory_summary": getattr(state.get("inventory_result"), "summary", ""),
                "order_summary": getattr(state.get("order_result"), "summary", ""),
                "preflight_required": proposal.preflight_checks,
            },
            "options": ["approved", "rejected", "modify", "ask_followup"],
            "risk_level": risk_level,
            "risk_signals": state.get("risk_signals") or [],
            "timeout_seconds": 1800 if risk_level != "CRITICAL" else 900,
        }
        try:
            decision = interrupt(interrupt_info)
        except RuntimeError as exc:
            if "outside of a runnable context" not in str(exc):
                raise
            return {
                "interrupt_info": interrupt_info,
                "fulfillment_branch": "stockout",
                "trace": [
                    build_trace_event(
                        node="human_approval",
                        start_ts=start_ts,
                        end_ts=datetime.now(),
                        status="ok",
                        note="等待人工审批执行提案",
                    )
                ],
            }

        if decision != "approved":
            validation = PreflightValidation(
                status="rejected",
                checked_at=datetime.now(timezone.utc).isoformat(),
                checks=[{"name": "human_decision", "status": decision}],
                old_fingerprint=proposal.data_fingerprint,
                new_fingerprint=None,
                message="人工未批准执行提案，流程转入缺货/人工处置路径。",
            )
            return {
                "human_decision": {"decision": decision, "node": "human_approval"},
                "preflight_validation": validation,
                "fulfillment_branch": "stockout",
                "trace": [
                    build_trace_event(
                        node="human_approval",
                        start_ts=start_ts,
                        end_ts=datetime.now(),
                        status="ok",
                        note=f"人工决策：{decision}",
                    )
                ],
            }

        validation = self._preflight_validate(proposal)
        branch = "fulfillable" if validation.status == "valid" else "stockout"
        return {
            "human_decision": {"decision": decision, "node": "human_approval"},
            "preflight_validation": validation,
            "execution_proposal": validation.replacement_proposal or proposal,
            "fulfillment_branch": branch,
            "trace": [
                build_trace_event(
                    node="human_approval",
                    start_ts=start_ts,
                    end_ts=datetime.now(),
                    status="ok",
                    note=f"人工批准后二次校验：{validation.status}",
                )
            ],
        }

    # ------------------------------------------------------------------ #
    # 节点 6：knowledge_retrieval_node
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

        设计说明：
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
        execution_proposal = state.get("execution_proposal")
        preflight_validation = state.get("preflight_validation")

        # 根据是否走过知识检索，标记路径
        # 这个字段方便前端或评测系统判断本次执行是快路径还是缺货知识路径。
        routing_path = "knowledge_path" if knowledge_result is not None else "fast_path"

        key_evidences = self._collect_key_evidences(
            order_result, inventory_result, knowledge_result, execution_proposal, preflight_validation
        )
        suggested_actions = self._collect_suggested_actions(
            inventory_result, knowledge_result, execution_proposal, preflight_validation
        )
        conclusion = self._build_conclusion(
            order_result,
            inventory_result,
            knowledge_result,
            routing_path,
            execution_proposal,
            preflight_validation,
        )

        final_answer = FinalAnswer(
            conclusion=conclusion,
            key_evidences=key_evidences,
            suggested_actions=suggested_actions,
            routing_path=routing_path,
            execution_proposal=execution_proposal,
            preflight_validation=preflight_validation,
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

    @staticmethod
    def _order_value(order_result) -> float:
        if order_result is None:
            return 0.0
        return round(
            sum(item.quantity * item.unit_price for item in getattr(order_result, "items", []) or []),
            2,
        )

    def _build_execution_proposal(
        self,
        order_result,
        inventory_result,
        *,
        question: str | None = None,
        intent: str | None = None,
        plan_version: int = 1,
        session_memory: dict | None = None,
    ) -> ExecutionProposal:
        """把订单和库存事实转换成前端 Action Card 可消费的执行提案。"""
        context_intent = intent or self._proposal_context_intent(question)
        planning_use_case = "replanner" if plan_version > 1 else "planner"
        planner_governance = self._planner_gateway_metadata(planning_use_case)
        context_envelope = self.context_builder.build_from_results(
            order_result,
            inventory_result,
            intent=context_intent,
            question=question,
        )
        decision_context = context_envelope.selected_context
        decision_context["context_intent"] = context_intent
        decision_context["source_question"] = question
        decision_context["planner_governance"] = planner_governance
        if session_memory:
            decision_context["session_memory"] = session_memory
        rag_context = self._build_planner_rag_context(
            order_id=order_result.order_id,
            question=question or "请生成履约执行提案。",
            session_memory=session_memory,
            current_business_context=decision_context,
        )
        decision_context["rag_context"] = rag_context.model_dump(mode="json")
        decision_context["progressive_context_loading"] = self._attempt_progressive_context_loading(
            order_id=order_result.order_id,
            intent=context_intent,
            question=question,
            decision_context=decision_context,
        )
        actions: list[ProposalAction] = []
        capabilities: set[str] = set()
        total_shipping_cost = 0.0
        max_eta_hours = 0
        inventory_snapshot: list[dict] = []
        rule_citations = self._proposal_rule_citations(
            order_result.order_id,
            inventory_result,
            rag_context=rag_context,
        )

        for item in order_result.items:
            check = next((sku for sku in inventory_result.sku_checks if sku.sku_id == item.sku_id), None)
            if check is None:
                continue
            warehouses = sorted(
                check.warehouse_records,
                key=lambda record: record.available_stock,
                reverse=True,
            )
            eligible_warehouses = [
                record for record in warehouses
                if self._warehouse_can_fulfill_item(record, item)
            ]
            inventory_snapshot.append({
                "sku_id": item.sku_id,
                "required_quantity": max(item.quantity - item.shipped_quantity, 0),
                "total_available_stock": check.total_available_stock,
                "warehouses": [
                    {
                        "warehouse_id": record.warehouse_id,
                        "warehouse_name": record.warehouse_name,
                        "available_stock": record.available_stock,
                        "locked_stock": record.locked_stock,
                        "updated_at": record.updated_at.isoformat(),
                    }
                    for record in warehouses[:5]
                ],
            })

            remaining = max(item.quantity - item.shipped_quantity, 0)
            if remaining == 0:
                continue
            single_warehouse = next((record for record in eligible_warehouses if record.available_stock >= remaining), None)
            if single_warehouse is not None:
                carrier = self._select_carrier(order_result, split=False)
                quote = self._logistics_quote(order_result.region, carrier, remaining)
                total_shipping_cost += quote["price"]
                max_eta_hours = max(max_eta_hours, int(quote["eta_hours"]))
                actions.append(ProposalAction(
                    action_id=f"act-{item.sku_id}-ship",
                    action_type="ship_from_warehouse",
                    sku_id=item.sku_id,
                    quantity=remaining,
                    from_warehouse=single_warehouse.warehouse_id,
                    carrier=carrier,
                    cost_delta=quote["price"],
                    eta_hours=int(quote["eta_hours"]),
                    reason=f"{single_warehouse.warehouse_name} 可一次性覆盖未发货 {remaining} 件需求。",
                ))
                capabilities.add("换仓履约")
                continue

            if not item.split_allowed:
                actions.append(ProposalAction(
                    action_id=f"act-{item.sku_id}-split-blocked",
                    action_type="stockout_resolution",
                    sku_id=item.sku_id,
                    quantity=remaining,
                    eta_hours=168,
                    reason="该 SKU 不允许拆单，且没有单仓可覆盖未发货数量，需要客户确认延期、调拨合仓或取消缺货行。",
                    reversible=False,
                ))
                capabilities.add("缺货处置")
                max_eta_hours = max(max_eta_hours, 168)
                continue

            for record in eligible_warehouses:
                if remaining <= 0 or record.available_stock <= 0:
                    continue
                allocated = min(remaining, record.available_stock)
                carrier = self._select_carrier(order_result, split=True)
                quote = self._logistics_quote(order_result.region, carrier, allocated)
                total_shipping_cost += quote["price"]
                max_eta_hours = max(max_eta_hours, int(quote["eta_hours"]))
                actions.append(ProposalAction(
                    action_id=f"act-{item.sku_id}-{record.warehouse_id}-split",
                    action_type="split_order",
                    sku_id=item.sku_id,
                    quantity=allocated,
                    from_warehouse=record.warehouse_id,
                    carrier=carrier,
                    cost_delta=quote["price"],
                    eta_hours=int(quote["eta_hours"]),
                    reason=f"{record.warehouse_name} 可先发 {allocated} 件，降低整单等待时间。",
                ))
                remaining -= allocated
                capabilities.update({"拆单", "物流渠道变更"})

            if remaining > 0:
                actions.append(ProposalAction(
                    action_id=f"act-{item.sku_id}-stockout",
                    action_type="stockout_resolution",
                    sku_id=item.sku_id,
                    quantity=remaining,
                    cost_delta=0.0,
                    eta_hours=168,
                    reason="当前可售库存或可履约仓能力不足，需要客户确认延期、替代 SKU、取消缺货行或等待补货。",
                    reversible=False,
                ))
                capabilities.add("缺货处置")
                max_eta_hours = max(max_eta_hours, 168)
            elif len([action for action in actions if action.sku_id == item.sku_id]) > 1:
                target = next((record for record in eligible_warehouses if record.available_stock > 0), None)
                if target is not None:
                    actions.append(ProposalAction(
                        action_id=f"act-{item.sku_id}-transfer-review",
                        action_type="inventory_transfer",
                        sku_id=item.sku_id,
                        quantity=item.quantity,
                        to_warehouse=target.warehouse_id,
                        cost_delta=12.0,
                        eta_hours=24,
                        reason="可选择跨仓调货后合并出库，适合客户不接受多包裹时使用。",
                    ))
                    capabilities.add("库存调拨/跨仓调货")

        primary_ship_actions = [
            action for action in actions
            if action.action_type in {"ship_from_warehouse", "split_order"}
        ]
        unique_ship_warehouses = {
            action.from_warehouse for action in primary_ship_actions if action.from_warehouse
        }
        if len(order_result.items) > 1 and len(unique_ship_warehouses) == 1:
            warehouse_id = next(iter(unique_ship_warehouses))
            actions.insert(0, ProposalAction(
                action_id="act-order-consolidate",
                action_type="merge_order",
                quantity=sum(item.quantity for item in order_result.items),
                from_warehouse=warehouse_id,
                carrier=self._select_carrier(order_result, split=False),
                cost_delta=-8.0,
                eta_hours=max_eta_hours or 72,
                reason="多个 SKU 可由同一仓发出，建议合单履约以减少包裹和客服解释成本。",
            ))
            capabilities.add("合单")

        if not actions:
            actions.append(ProposalAction(
                action_id="act-manual-review",
                action_type="stockout_resolution",
                quantity=0,
                reason="未能生成确定动作，请人工补充订单、库存或物流数据后重算。",
                reversible=False,
            ))
            capabilities.add("缺货处置")

        actions = self._enrich_proposal_actions(actions, rule_citations, order_result, inventory_result)
        proposal_status = "pending_approval"
        summary = self._proposal_summary(order_result.order_id, actions)
        data_conflict = decision_context.get("context_completeness", {}).get("status") == "DATA_CONFLICT"
        invalidation_reason = None
        preflight_checks = ["订单状态仍可履约", "SKU 可售库存仍覆盖动作数量", "物流渠道报价与时效仍有效"]
        if data_conflict:
            proposal_status = "invalidated"
            invalidation_reason = "固定必查业务上下文不完整或冲突，请重新拉取业务源系统或人工核实后再生成方案。"
            summary = f"{summary} 当前 OrderContext 处于 DATA_CONFLICT，禁止直接下发执行。"
            preflight_checks.insert(0, "业务上下文完整性必须恢复为 complete")
        goal_type = self._goal_type(actions)
        action_dag = self._build_action_dag(actions)
        success_criteria = self._success_criteria_template(goal_type, actions)
        freshness = {
            "order_checked_at": datetime.now(timezone.utc).isoformat(),
            "inventory_checked_at": max(
                (
                    record.updated_at.isoformat()
                    for check in inventory_result.sku_checks
                    for record in check.warehouse_records
                ),
                default=datetime.now(timezone.utc).isoformat(),
            ),
            "logistics_quote_checked_at": datetime.now(timezone.utc).isoformat(),
        }
        fingerprint_payload = {
            "order_main": decision_context["order_main"],
            "order_items": decision_context["order_items"],
            "sku_warehouse_inventory": decision_context["sku_warehouse_inventory"],
            "logistics_options": decision_context["logistics_options"],
            "shipping_cost": round(total_shipping_cost, 2),
            "eta_hours": max_eta_hours,
        }
        fingerprint = self._fingerprint(fingerprint_payload)
        context_version = f"ctx-{fingerprint[:12]}"
        proposal = ExecutionProposal(
            proposal_id=f"prop-{order_result.order_id}-{fingerprint[:10]}",
            order_id=order_result.order_id,
            title=self._proposal_title(actions),
            summary=summary,
            status=proposal_status,
            capabilities=sorted(capabilities),
            actions=actions,
            decision_context=decision_context,
            inventory_snapshot=inventory_snapshot,
            cost_breakdown={
                "shipping_cost": round(total_shipping_cost, 2),
                "handling_cost": round(max(0, len(primary_ship_actions) - 1) * 6.0, 2),
                "estimated_total_delta": round(total_shipping_cost + max(0, len(primary_ship_actions) - 1) * 6.0, 2),
            },
            eta={
                "eta_hours": max_eta_hours,
                "eta_label": self._eta_label(max_eta_hours),
            },
            rule_citations=rule_citations,
            data_fingerprint=fingerprint,
            freshness=freshness,
            approval_required=self._proposal_needs_approval(actions, inventory_result),
            preflight_checks=preflight_checks,
            invalidation_reason=invalidation_reason,
            goal_type=goal_type,
            action_dag=action_dag,
            success_criteria=success_criteria,
            plan_version=plan_version,
            context_version=context_version,
            expires_at=(datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        )
        policy_failures = self._validate_execution_proposal_policy(proposal)
        if policy_failures:
            reasons = [str(item["detail"]) for item in policy_failures]
            merged_reason = "; ".join(reasons)
            if proposal.invalidation_reason:
                merged_reason = f"{proposal.invalidation_reason}; {merged_reason}"
            policy_preflight = "Policy/Schema Validator 必须通过：Plan Schema、Action DAG、动作参数、业务硬约束和 Success Criteria"
            preflight_checks = list(proposal.preflight_checks)
            if policy_preflight not in preflight_checks:
                preflight_checks.insert(0, policy_preflight)
            proposal = proposal.model_copy(update={
                "status": "invalidated",
                "invalidation_reason": merged_reason,
                "preflight_checks": preflight_checks,
                "decision_context": {
                    **proposal.decision_context,
                    "policy_schema_validation": {
                        "status": "fail",
                        "checks": policy_failures,
                    },
                },
            })
        proposal = proposal.model_copy(update={
            "decision_context": {
                **proposal.decision_context,
                "planner_governance": {
                    **planner_governance,
                    "policy_validation": {
                        "status": "fail" if policy_failures else "pass",
                        "failure_count": len(policy_failures),
                    },
                },
            },
        })
        self._record_planner_governance_trace(
            proposal=proposal,
            governance=planner_governance,
            policy_failures=policy_failures,
        )
        return proposal

    def _preflight_validate(self, proposal: ExecutionProposal) -> PreflightValidation:
        """人工批准后重新读取实时事实，避免批准旧快照。"""
        checked_at = datetime.now(timezone.utc).isoformat()
        if proposal.status == "invalidated":
            return PreflightValidation(
                status="invalidated",
                checked_at=checked_at,
                checks=[
                    {
                        "name": "policy_schema_validator",
                        "status": "fail",
                        "detail": proposal.invalidation_reason or "执行提案已被标记失效。",
                    }
                ],
                old_fingerprint=proposal.data_fingerprint,
                new_fingerprint=None,
                message="执行提案未通过策略/结构校验，禁止进入二次实时校验和外部任务下发。",
            )
        expiry_failure = self._proposal_expiry_failure(proposal)
        if expiry_failure is not None:
            return PreflightValidation(
                status="invalidated",
                checked_at=checked_at,
                checks=[
                    {
                        "name": "proposal_expiry",
                        "status": "fail",
                        "detail": expiry_failure,
                    }
                ],
                old_fingerprint=proposal.data_fingerprint,
                new_fingerprint=None,
                message="执行提案已过期或过期时间无效，必须重新读取最新业务上下文并重新规划。",
            )
        try:
            latest_order = self.order_service.analyze_order(proposal.order_id)
            latest_inventory = self.inventory_service.analyze_inventory(proposal.order_id)
        except Exception as exc:
            return PreflightValidation(
                status="invalidated",
                checked_at=checked_at,
                checks=[{"name": "runtime_lookup", "status": "failed", "detail": str(exc)}],
                old_fingerprint=proposal.data_fingerprint,
                new_fingerprint=None,
                message=f"二次校验读取实时数据失败，旧提案自动失效：{exc}",
            )

        original_context = proposal.decision_context or {}
        replacement = self._build_execution_proposal(
            latest_order,
            latest_inventory,
            question=original_context.get("source_question"),
            intent=original_context.get("context_intent"),
            plan_version=proposal.plan_version,
            session_memory=original_context.get("session_memory"),
        )
        order_status_ok = latest_order.order_status in {
            "paid",
            "paid_waiting_fulfillment",
            "pending_fulfillment",
            "待发货",
            "待履约",
            "已支付",
        }
        checks = [
            {
                "name": "order_status",
                "status": "pass" if order_status_ok else "fail",
                "value": latest_order.order_status,
            },
        ]
        checks.extend(self._preflight_inventory_checks(proposal, latest_inventory))
        checks.extend(self._preflight_logistics_checks(proposal, latest_order))
        fingerprint_changed = proposal.data_fingerprint != replacement.data_fingerprint
        has_failed_check = any(check.get("status") == "fail" for check in checks)
        if fingerprint_changed or has_failed_check:
            replacement = replacement.model_copy(update={
                "status": "invalidated",
                "invalidation_reason": "实时订单、库存或物流报价已变化，请基于新提案重新审批。",
                "plan_version": proposal.plan_version + 1,
            })
            return PreflightValidation(
                status="invalidated",
                checked_at=checked_at,
                checks=checks,
                old_fingerprint=proposal.data_fingerprint,
                new_fingerprint=replacement.data_fingerprint,
                message="批准后发现实时数据已变化，旧方案自动失效，已重新计算执行提案。",
                replacement_proposal=replacement,
            )
        return PreflightValidation(
            status="valid",
            checked_at=checked_at,
            checks=checks,
            old_fingerprint=proposal.data_fingerprint,
            new_fingerprint=replacement.data_fingerprint,
            message="二次校验通过：订单状态、库存和物流报价仍匹配审批时提案，可交给 OMS/WMS/TMS 执行。",
        )

    @staticmethod
    def _preflight_inventory_checks(proposal: ExecutionProposal, latest_inventory) -> list[dict]:
        checks: list[dict] = []
        for action in proposal.actions:
            if action.action_type not in {"ship_from_warehouse", "split_order"}:
                continue
            check = next((item for item in latest_inventory.sku_checks if item.sku_id == action.sku_id), None)
            record = None
            if check is not None:
                record = next(
                    (
                        warehouse
                        for warehouse in check.warehouse_records
                        if warehouse.warehouse_id == action.from_warehouse
                    ),
                    None,
                )
            available = record.available_stock if record else 0
            checks.append({
                "name": f"inventory:{action.sku_id}:{action.from_warehouse}",
                "status": "pass" if available >= action.quantity else "fail",
                "required_quantity": action.quantity,
                "available_stock": available,
                "inventory_version": getattr(record, "inventory_version", None) if record else None,
                "updated_at": record.updated_at.isoformat() if record else None,
            })
        if not checks:
            checks.append({
                "name": "inventory:no_direct_ship_action",
                "status": "pass",
                "detail": "当前提案不包含直接发货或拆单发货动作。",
            })
        return checks

    def _preflight_logistics_checks(self, proposal: ExecutionProposal, latest_order) -> list[dict]:
        checks: list[dict] = []
        for action in proposal.actions:
            if action.action_type not in {"ship_from_warehouse", "split_order"} or not action.carrier:
                continue
            quote = self._logistics_quote(latest_order.region, action.carrier, action.quantity)
            checks.append({
                "name": f"logistics:{action.from_warehouse}:{action.carrier}",
                "status": "pass" if int(quote["eta_hours"]) == action.eta_hours else "fail",
                "carrier": action.carrier,
                "shipping_cost": quote["price"],
                "eta_hours": quote["eta_hours"],
                "approved_eta_hours": action.eta_hours,
            })
        if not checks:
            checks.append({
                "name": "logistics:no_direct_ship_action",
                "status": "pass",
                "detail": "当前提案不包含需要即时发运的物流动作。",
            })
        return checks

    def _build_decision_context(self, order_result, inventory_result) -> dict:
        """Compatibility wrapper for tests and debug tools.

        The real implementation lives in OrderContextBuilder so program-loaded
        fields, field groups, and progressive detail lookup stay centralized.
        """
        return self.context_builder.build_from_results(
            order_result,
            inventory_result,
            intent="fulfillment_action",
        ).selected_context

    @staticmethod
    def _proposal_context_intent(question: str | None) -> str:
        text = (question or "").lower()
        if any(marker in text for marker in ("复盘", "摘要", "案件", "case review", "review")):
            return "case_review"
        if any(marker in text for marker in ("物流", "渠道", "快递", "carrier", "channel", "tms")):
            return "logistics_exception"
        if any(marker in text for marker in ("缺货", "调拨", "补货", "在途", "替代", "stockout", "replenishment")):
            return "stockout_resolution"
        return "fulfillment_action"

    @staticmethod
    def _enrich_proposal_actions(
        actions: list[ProposalAction],
        rule_citations: list[str],
        order_result=None,
        inventory_result=None,
    ) -> list[ProposalAction]:
        return [
            action.model_copy(
                update={
                    "responsibility_domain": action.responsibility_domain
                    or WorkflowNodes._responsibility_domain(action.action_type),
                    "business_evidence": action.business_evidence
                    or WorkflowNodes._action_business_evidence(
                        action,
                        order_result,
                        inventory_result,
                        rule_citations,
                    ),
                }
            )
            for action in actions
        ]

    @staticmethod
    def _action_business_evidence(
        action: ProposalAction,
        order_result=None,
        inventory_result=None,
        rule_citations: list[str] | None = None,
    ) -> list[str]:
        evidence: list[str] = []
        if order_result is not None:
            order_status = getattr(order_result, "order_status", "") or "unknown"
            priority = getattr(order_result, "priority", "") or "normal"
            region = getattr(order_result, "region", "") or "unknown"
            evidence.append(f"OMS订单事实：状态={order_status}，优先级={priority}，区域={region}")
            item = next(
                (
                    candidate for candidate in getattr(order_result, "items", []) or []
                    if getattr(candidate, "sku_id", "") == action.sku_id
                ),
                None,
            )
            if item is not None:
                remaining = max(int(getattr(item, "quantity", 0)) - int(getattr(item, "shipped_quantity", 0)), 0)
                evidence.append(f"OMS商品事实：SKU={action.sku_id}，未发货数量={remaining}")
        if inventory_result is not None and action.sku_id:
            check = next(
                (
                    candidate for candidate in getattr(inventory_result, "sku_checks", []) or []
                    if getattr(candidate, "sku_id", "") == action.sku_id
                ),
                None,
            )
            if check is not None:
                evidence.append(
                    f"WMS库存事实：SKU={action.sku_id}，可售总数={getattr(check, 'total_available_stock', 0)}"
                )
                if action.from_warehouse:
                    warehouse = next(
                        (
                            record for record in getattr(check, "warehouse_records", []) or []
                            if getattr(record, "warehouse_id", "") == action.from_warehouse
                        ),
                        None,
                    )
                    if warehouse is not None:
                        evidence.append(
                            f"WMS仓库事实：{action.from_warehouse} 可售={getattr(warehouse, 'available_stock', 0)}"
                        )
        if action.carrier or action.eta_hours:
            evidence.append(f"TMS物流事实：承运商={action.carrier or '待定'}，ETA={action.eta_hours}小时")
        evidence.extend((rule_citations or [])[:3])
        return evidence[:6]

    @staticmethod
    def _responsibility_domain(action_type: str) -> str:
        if action_type in {"change_carrier", "logistics_exception"}:
            return "TMS"
        if action_type == "replenishment":
            return "ERP"
        if action_type in {"stockout_resolution", "customer_complaint"}:
            return "CRM"
        return "WMS"

    @staticmethod
    def _goal_type(actions: list[ProposalAction]) -> str:
        action_types = {action.action_type for action in actions}
        if "stockout_resolution" in action_types and not action_types & {"ship_from_warehouse", "split_order"}:
            return "stockout_customer_resolution"
        if "inventory_transfer" in action_types:
            return "cross_warehouse_fulfillment"
        if "split_order" in action_types:
            return "split_fulfillment"
        if "merge_order" in action_types:
            return "merged_fulfillment"
        if "change_carrier" in action_types or "logistics_exception" in action_types:
            return "logistics_recovery"
        return "warehouse_fulfillment"

    @staticmethod
    def _build_action_dag(actions: list[ProposalAction]) -> dict:
        nodes = [
            {
                "action_id": action.action_id,
                "action_type": action.action_type,
                "responsibility_domain": action.responsibility_domain,
                "dependency_ids": action.depends_on,
                "status": "READY" if not action.depends_on else "BLOCKED",
            }
            for action in actions
        ]
        edges = [
            {"from": dependency_id, "to": action.action_id}
            for action in actions
            for dependency_id in action.depends_on
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "scheduling_hint": "parallel_without_dependencies" if not edges else "respect_dependencies",
        }

    @staticmethod
    def _success_criteria_template(goal_type: str, actions: list[ProposalAction]) -> dict:
        criteria = [
            {
                "name": "fresh_business_state_reloaded",
                "description": "VERIFYING 阶段必须重新读取 OMS/WMS/TMS/ERP/PIM/CRM 最新状态。",
                "source_systems": ["OMS", "WMS", "TMS", "ERP", "PIM", "CRM"],
            },
            {
                "name": "external_task_business_evidence_present",
                "description": "每个外部任务回调必须携带目标系统业务状态证明，而不是只返回任务完成。",
                "required_for_actions": [action.action_id for action in actions],
            },
            {
                "name": "no_active_fulfillment_tasks",
                "description": "不存在正在执行的旧分仓、占库存、调拨或拆单任务。",
                "source_systems": ["OMS", "WMS"],
            },
        ]
        if any(action.action_type in {"ship_from_warehouse", "split_order", "merge_order"} for action in actions):
            criteria.append({
                "name": "direct_fulfillment_materialized",
                "description": "订单已真实形成占库存、包裹、运单或出库状态之一。",
                "source_systems": ["OMS", "WMS", "TMS"],
            })
        if goal_type == "stockout_customer_resolution":
            criteria.append({
                "name": "customer_confirmation_recorded",
                "description": "CRM/客服工单已记录延期、替代 SKU、取消缺货行或客户确认结果。",
                "source_systems": ["CRM"],
            })
        return {
            "goal_type": goal_type,
            "criteria": criteria,
            "max_replan_count": 3,
        }

    def _attempt_progressive_context_loading(
        self,
        *,
        order_id: str,
        intent: str,
        question: str | None,
        decision_context: dict,
    ) -> dict:
        completeness = decision_context.get("context_completeness") or {}
        if completeness.get("status") != "DATA_CONFLICT":
            return {
                "status": "skipped",
                "reason": "context_already_complete",
                "tool_calls": [],
            }

        sources = self._progressive_context_sources(completeness)
        if not sources:
            sources = ["OMS", "WMS", "TMS", "ERP", "PIM", "CRM"]
        tool_services = ToolServiceBundle(context_service=self.context_service)
        tool_calls: list[dict] = []
        updated_fields: list[str] = []

        for source in sources:
            tool_name = PROGRESSIVE_CONTEXT_TOOL_BY_SOURCE.get(source)
            if not tool_name:
                continue
            try:
                result = self.read_only_tool_gateway.execute_json(
                    tool_name=tool_name,
                    args={
                        "order_id": order_id,
                        "intent": intent,
                        "question": question or "",
                    },
                    services=tool_services,
                )
            except Exception as exc:  # noqa: BLE001 - progressive loading must be traceable, not fatal.
                tool_calls.append({
                    "source_system": source,
                    "tool_name": tool_name,
                    "status": "error",
                    "error_code": exc.__class__.__name__,
                    "summary": str(exc),
                })
                continue

            data = result.get("data") if isinstance(result, dict) else {}
            details = data.get("details") if isinstance(data, dict) else {}
            loaded_paths: list[str] = []
            if result.get("status") == "ok" and isinstance(details, dict):
                for path, value in details.items():
                    if value not in (None, [], {}):
                        decision_context[path] = value
                        loaded_paths.append(path)
                updated_fields.extend(loaded_paths)
            error = result.get("error") if isinstance(result.get("error"), dict) else {}
            tool_calls.append({
                "source_system": source,
                "tool_name": tool_name,
                "status": result.get("status"),
                "summary": result.get("summary", ""),
                "loaded_paths": loaded_paths,
                "error_code": error.get("code"),
            })

        refreshed = self.context_builder._validate_completeness(decision_context)
        decision_context["context_completeness"] = refreshed
        status = "recovered" if refreshed.get("status") == "complete" else "unresolved"
        result = {
            "status": status,
            "trigger": "DATA_CONFLICT",
            "sources_requested": sources,
            "updated_fields": sorted(set(updated_fields)),
            "tool_calls": tool_calls,
            "context_completeness": refreshed,
        }
        add_trace_step(
            step_type="context",
            name="progressive_context_loading",
            status="success" if status == "recovered" else "failed",
            summary=(
                "Planner 已通过只读 Tool Gateway 渐进加载缺失业务源。"
                if status == "recovered"
                else "Planner 已尝试只读 Tool Gateway 渐进加载，业务上下文仍不完整。"
            ),
            input_summary={
                "order_id": order_id,
                "intent": intent,
                "sources_requested": sources,
                "missing_required_fields": completeness.get("missing_required_fields", []),
            },
            output_summary={
                "status": status,
                "updated_fields": sorted(set(updated_fields)),
                "context_status": refreshed.get("status"),
            },
            metadata={
                "tool_calls": tool_calls,
                "read_only": True,
                "gateway": "ReadOnlyToolGateway",
            },
        )
        return result

    @staticmethod
    def _progressive_context_sources(completeness: dict) -> list[str]:
        sources: list[str] = []
        for failure in completeness.get("adapter_failures") or []:
            if isinstance(failure, dict):
                source = str(failure.get("source_system") or "").upper()
                if source in PROGRESSIVE_CONTEXT_TOOL_BY_SOURCE and source not in sources:
                    sources.append(source)
        for field in completeness.get("missing_required_fields") or []:
            source = PROGRESSIVE_CONTEXT_SOURCE_BY_FIELD.get(str(field))
            if source and source not in sources:
                sources.append(source)
        return sources

    @staticmethod
    def _planner_gateway_metadata(use_case: str) -> dict:
        gateway = get_model_gateway()
        try:
            prompt = gateway.load_prompt(use_case=use_case)
            prompt_metadata = {
                "prompt_id": prompt.id,
                "prompt_version": prompt.version,
                "prompt_source": prompt.source,
                "prompt_hash": hashlib.sha256(prompt.system.encode("utf-8")).hexdigest()[:16],
                "output_contract": prompt.output_contract,
            }
        except Exception as exc:
            prompt_metadata = {
                "prompt_id": use_case,
                "prompt_version": "unknown",
                "prompt_source": "unavailable",
                "prompt_error": exc.__class__.__name__,
            }
        try:
            active_model = gateway.active_model(use_case=use_case)
        except Exception as exc:
            active_model = {"status": "unavailable", "error": exc.__class__.__name__}
        return {
            "use_case": use_case,
            "model_gateway": {
                "active_model": active_model,
                "actual_model_call": False,
                "invocation_mode": "deterministic_planner_fallback",
            },
            "prompt": prompt_metadata,
            "decision_priority": [
                "Current Business Data",
                "Current SOP",
                "Historical Case",
            ],
            "case_reference_policy": "Historical Case is advisory and ignored when it conflicts with fresh business data or active SOP.",
        }

    @staticmethod
    def _record_planner_governance_trace(
        *,
        proposal: ExecutionProposal,
        governance: dict,
        policy_failures: list[dict],
    ) -> None:
        gateway_metadata = governance.get("model_gateway") or {}
        prompt_metadata = governance.get("prompt") or {}
        active_model = gateway_metadata.get("active_model") or {}
        add_trace_step(
            step_type="model_gateway",
            name=f"{governance.get('use_case')}:governance_snapshot",
            status="skipped",
            summary="Planner 使用 Model Gateway 管理 prompt/model 路由；当前走确定性规划 fallback。",
            metadata={
                "use_case": governance.get("use_case"),
                "actual_model_call": False,
                "invocation_mode": gateway_metadata.get("invocation_mode"),
                "prompt_id": prompt_metadata.get("prompt_id"),
                "prompt_version": prompt_metadata.get("prompt_version"),
                "prompt_source": prompt_metadata.get("prompt_source"),
                "prompt_hash": prompt_metadata.get("prompt_hash"),
                "active_model_id": active_model.get("id") if isinstance(active_model, dict) else None,
                "fallback_model_ids": active_model.get("fallback_model_ids") if isinstance(active_model, dict) else [],
            },
        )
        add_trace_step(
            step_type="planner",
            name="execution_proposal_planned",
            status="success" if proposal.status != "invalidated" else "failed",
            summary=f"生成 {proposal.goal_type} 提案，动作数 {len(proposal.actions)}，状态 {proposal.status}。",
            input_summary={
                "order_id": proposal.order_id,
                "use_case": governance.get("use_case"),
                "decision_priority": governance.get("decision_priority"),
            },
            output_summary={
                "proposal_id": proposal.proposal_id,
                "plan_version": proposal.plan_version,
                "context_version": proposal.context_version,
                "action_count": len(proposal.actions),
                "policy_validation": "fail" if policy_failures else "pass",
            },
            metadata={
                "goal_type": proposal.goal_type,
                "plan_version": proposal.plan_version,
                "context_version": proposal.context_version,
                "action_dag_node_count": len(proposal.action_dag.get("nodes") or []),
                "action_dag_edge_count": len(proposal.action_dag.get("edges") or []),
                "policy_validation": {
                    "status": "fail" if policy_failures else "pass",
                    "failure_count": len(policy_failures),
                    "failures": policy_failures[:10],
                },
                "success_criteria_names": [
                    item.get("name")
                    for item in proposal.success_criteria.get("criteria", [])
                    if isinstance(item, dict)
                ],
            },
        )

    @staticmethod
    def _validate_execution_proposal_policy(proposal: ExecutionProposal) -> list[dict]:
        """Validate planner output before it can become executable external tasks."""

        failures: list[dict] = []
        actions = proposal.actions
        action_ids = [action.action_id for action in actions]
        action_id_set = set(action_ids)

        if not actions:
            failures.append({
                "name": "plan_schema.actions",
                "status": "fail",
                "detail": "执行提案至少需要包含一个可路由动作。",
            })
        expiry_failure = WorkflowNodes._proposal_expiry_failure(proposal)
        if expiry_failure is not None:
            failures.append({
                "name": "plan_schema.expires_at",
                "status": "fail",
                "detail": expiry_failure,
            })
        if len(action_ids) != len(action_id_set):
            failures.append({
                "name": "plan_schema.action_id_unique",
                "status": "fail",
                "detail": "动作 action_id 必须唯一。",
            })
        for index, action in enumerate(actions, start=1):
            label = action.action_id or f"#{index}"
            if not action.action_id.strip():
                failures.append({
                    "name": f"plan_schema.action_id:{index}",
                    "status": "fail",
                    "detail": "动作 action_id 不能为空。",
                })
            if action.action_type in FORBIDDEN_DIRECT_MUTATION_ACTION_TYPES:
                failures.append({
                    "name": f"hard_constraint.no_direct_business_mutation:{label}",
                    "status": "fail",
                    "detail": f"动作 {label} 使用了直接业务突变类型 {action.action_type}，只能创建外部协同请求/任务。",
                })
            elif action.action_type not in ALLOWED_PROPOSAL_ACTION_TYPES:
                failures.append({
                    "name": f"plan_schema.action_type:{label}",
                    "status": "fail",
                    "detail": f"动作 {label} 的 action_type={action.action_type} 不在受控动作白名单中。",
                })
            failures.extend(WorkflowNodes._validate_action_required_params(action, label))
            if not action.responsibility_domain:
                failures.append({
                    "name": f"hard_constraint.responsibility_domain:{label}",
                    "status": "fail",
                    "detail": f"动作 {label} 缺少责任域，无法安全路由到 WMS/TMS/ERP/CRM。",
                })
            if not action.business_evidence:
                failures.append({
                    "name": f"hard_constraint.business_evidence:{label}",
                    "status": "fail",
                    "detail": f"动作 {label} 缺少业务依据，不能只凭模型自由生成执行项。",
                })
            dangling_dependencies = [item for item in action.depends_on if item not in action_id_set]
            if dangling_dependencies:
                failures.append({
                    "name": f"action_dag.dependencies:{label}",
                    "status": "fail",
                    "detail": f"动作 {label} 依赖不存在的前置动作：{', '.join(dangling_dependencies)}。",
                })

        failures.extend(WorkflowNodes._validate_action_dag(proposal.action_dag, action_id_set))
        failures.extend(WorkflowNodes._validate_success_criteria(proposal.success_criteria, proposal.goal_type, action_ids))
        return failures

    @staticmethod
    def _validate_action_required_params(action: ProposalAction, label: str) -> list[dict]:
        failures: list[dict] = []

        def require(condition: bool, name: str, detail: str) -> None:
            if not condition:
                failures.append({
                    "name": f"plan_schema.params.{name}:{label}",
                    "status": "fail",
                    "detail": detail,
                })

        if action.action_type in {"ship_from_warehouse", "split_order"}:
            require(bool(action.sku_id), "sku_id", f"动作 {label} 必须指定 SKU。")
            require(action.quantity > 0, "quantity", f"动作 {label} 的数量必须大于 0。")
            require(bool(action.from_warehouse), "from_warehouse", f"动作 {label} 必须指定发货仓。")
            require(bool(action.carrier), "carrier", f"动作 {label} 必须指定物流渠道。")
        elif action.action_type == "merge_order":
            require(action.quantity > 0, "quantity", f"动作 {label} 的合单数量必须大于 0。")
            require(bool(action.from_warehouse), "from_warehouse", f"动作 {label} 必须指定合单发货仓。")
        elif action.action_type in {"switch_warehouse", "inventory_transfer"}:
            require(bool(action.sku_id), "sku_id", f"动作 {label} 必须指定 SKU。")
            require(action.quantity > 0, "quantity", f"动作 {label} 的调拨/换仓数量必须大于 0。")
            require(bool(action.to_warehouse), "to_warehouse", f"动作 {label} 必须指定目标仓。")
        elif action.action_type in {"change_carrier", "logistics_exception"}:
            require(bool(action.carrier), "carrier", f"动作 {label} 必须指定物流渠道或承运方。")
        elif action.action_type == "replenishment":
            require(bool(action.sku_id), "sku_id", f"动作 {label} 必须指定补货 SKU。")
            require(action.quantity > 0, "quantity", f"动作 {label} 的补货数量必须大于 0。")
        elif action.action_type in {"stockout_resolution", "customer_complaint"}:
            require(bool(action.reason.strip()), "reason", f"动作 {label} 必须说明客服/客户处理原因。")
        return failures

    @staticmethod
    def _validate_action_dag(action_dag: dict, action_id_set: set[str]) -> list[dict]:
        failures: list[dict] = []
        nodes = action_dag.get("nodes") if isinstance(action_dag, dict) else None
        edges = action_dag.get("edges") if isinstance(action_dag, dict) else None
        if not isinstance(nodes, list) or not isinstance(edges, list):
            return [{
                "name": "action_dag.schema",
                "status": "fail",
                "detail": "Action DAG 必须包含 nodes 和 edges 列表。",
            }]

        dag_node_ids = {
            str(node.get("action_id"))
            for node in nodes
            if isinstance(node, dict) and node.get("action_id")
        }
        if dag_node_ids != action_id_set:
            failures.append({
                "name": "action_dag.nodes_match_actions",
                "status": "fail",
                "detail": "Action DAG nodes 必须与 actions 的 action_id 完全一致。",
            })

        adjacency: dict[str, list[str]] = {action_id: [] for action_id in action_id_set}
        for edge in edges:
            if not isinstance(edge, dict):
                failures.append({
                    "name": "action_dag.edge_schema",
                    "status": "fail",
                    "detail": "Action DAG edge 必须是包含 from/to 的对象。",
                })
                continue
            source = str(edge.get("from") or "")
            target = str(edge.get("to") or "")
            if source not in action_id_set or target not in action_id_set:
                failures.append({
                    "name": "action_dag.edge_reference",
                    "status": "fail",
                    "detail": f"Action DAG edge 引用了不存在的动作：{source}->{target}。",
                })
                continue
            adjacency[source].append(target)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(action_id: str) -> bool:
            if action_id in visiting:
                return True
            if action_id in visited:
                return False
            visiting.add(action_id)
            for next_id in adjacency.get(action_id, []):
                if visit(next_id):
                    return True
            visiting.remove(action_id)
            visited.add(action_id)
            return False

        if any(visit(action_id) for action_id in action_id_set):
            failures.append({
                "name": "action_dag.acyclic",
                "status": "fail",
                "detail": "Action DAG 不能包含循环依赖。",
            })
        return failures

    @staticmethod
    def _validate_success_criteria(success_criteria: dict, goal_type: str, action_ids: list[str]) -> list[dict]:
        failures: list[dict] = []
        if not isinstance(success_criteria, dict):
            return [{
                "name": "success_criteria.schema",
                "status": "fail",
                "detail": "Success Criteria 必须是结构化对象。",
            }]
        if success_criteria.get("goal_type") != goal_type:
            failures.append({
                "name": "success_criteria.goal_type",
                "status": "fail",
                "detail": "Success Criteria 的 goal_type 必须与执行提案一致。",
            })
        if success_criteria.get("max_replan_count") != 3:
            failures.append({
                "name": "success_criteria.max_replan_count",
                "status": "fail",
                "detail": "Success Criteria 必须声明最多 Replan 3 次。",
            })
        criteria = success_criteria.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            failures.append({
                "name": "success_criteria.criteria",
                "status": "fail",
                "detail": "Success Criteria 必须包含非空 criteria 列表。",
            })
            return failures
        names = {criterion.get("name") for criterion in criteria if isinstance(criterion, dict)}
        missing = sorted({
            "fresh_business_state_reloaded",
            "external_task_business_evidence_present",
        } - names)
        if missing:
            failures.append({
                "name": "success_criteria.required_names",
                "status": "fail",
                "detail": f"Success Criteria 缺少必需校验项：{', '.join(missing)}。",
            })
        evidence_criteria = next(
            (
                criterion
                for criterion in criteria
                if isinstance(criterion, dict)
                and criterion.get("name") == "external_task_business_evidence_present"
            ),
            {},
        )
        required_for_actions = set(evidence_criteria.get("required_for_actions") or [])
        if required_for_actions != set(action_ids):
            failures.append({
                "name": "success_criteria.evidence_scope",
                "status": "fail",
                "detail": "外部任务业务证明校验必须覆盖所有 action_id。",
            })
        return failures

    @staticmethod
    def _proposal_expiry_failure(proposal: ExecutionProposal) -> str | None:
        if not proposal.expires_at:
            return "执行提案缺少 expires_at，不能证明计划仍在有效审批窗口内。"
        raw = proposal.expires_at.strip().replace("Z", "+00:00")
        try:
            expires_at = datetime.fromisoformat(raw)
        except ValueError:
            return f"执行提案 expires_at 格式无效：{proposal.expires_at}。"
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= datetime.now(timezone.utc):
            return "执行提案已过期，禁止直接执行。"
        return None

    def _build_planner_rag_context(
        self,
        *,
        order_id: str,
        question: str,
        session_memory: dict | None,
        current_business_context: dict | None = None,
    ):
        return self.rag_context_service.build(
            order_id=order_id,
            question=question,
            session_memory=session_memory,
            current_business_context=current_business_context,
        )

    def _proposal_rule_citations(self, order_id: str, inventory_result, rag_context=None) -> list[str]:
        if rag_context is not None:
            rules = [f"SOP：{item}" for item in getattr(rag_context.sop_evidence, "key_points", [])[:4]]
            for hit in getattr(rag_context.sop_evidence, "evidence", [])[:2]:
                if hit.source_file:
                    rules.append(f"SOP来源：{hit.source_file} / {hit.category}")
            for hit in getattr(rag_context.similar_cases, "evidence", [])[:2]:
                label = hit.source_case_id or hit.source_file
                if label:
                    rules.append(f"历史案例参考：{label} / {hit.category}")
            if rules:
                return rules[:6]

        question = (
            "请为履约执行提案提取可引用规则：换仓履约、拆单/合单、物流渠道变更、"
            "库存调拨/跨仓调货和缺货处置。"
        )
        if inventory_result.insufficient_skus:
            question += f" 缺货 SKU：{'、'.join(inventory_result.insufficient_skus)}。"
        try:
            result = self.knowledge_service.retrieve(
                order_id=order_id,
                question=question,
                filter_categories=[],
            )
        except Exception:
            return [
                "缺货或跨仓动作执行前必须保留人工审批记录。",
                "拆单、替代 SKU、延期承诺需先确认客户接受。",
                "执行前必须重新校验库存、订单状态和物流报价。",
            ]
        rules = list(getattr(result.answer_summary, "key_rules", [])[:4])
        if not rules:
            rules = [getattr(result.answer_summary, "conclusion", "按履约 SOP 执行人工审批。")]
        for hit in getattr(result, "hits", [])[:3]:
            source = getattr(hit, "source_file", "")
            category = getattr(hit, "category", "")
            if source:
                rules.append(f"来源：{source} / {category}")
        return rules[:6]

    @staticmethod
    def _build_action_card_prompt(
        proposal: ExecutionProposal,
        risk_level: str,
        risk_signals: list[str],
    ) -> str:
        lines = [
            f"[{risk_level}] 请审批订单 {proposal.order_id} 的执行提案：{proposal.title}",
            proposal.summary,
            "可选操作：approved=批准执行；rejected=拒绝；modify=修改方案；ask_followup=继续追问。",
        ]
        if risk_signals:
            lines.append("风险信号：" + "、".join(risk_signals))
        return "\n".join(lines)

    @staticmethod
    def _proposal_title(actions: list[ProposalAction]) -> str:
        types = {action.action_type for action in actions}
        if "stockout_resolution" in types:
            return "缺货处置与拆单履约提案"
        if "split_order" in types:
            return "跨仓拆单履约提案"
        if "merge_order" in types:
            return "合单履约提案"
        return "换仓履约提案"

    @staticmethod
    def _proposal_summary(order_id: str, actions: list[ProposalAction]) -> str:
        ship_count = len([action for action in actions if action.action_type in {"ship_from_warehouse", "split_order"}])
        stockout_count = len([action for action in actions if action.action_type == "stockout_resolution"])
        return (
            f"订单 {order_id} 生成 {len(actions)} 个可审批动作："
            f"{ship_count} 个发货/拆单动作，{stockout_count} 个缺货处置动作。"
        )

    @staticmethod
    def _proposal_needs_approval(actions: list[ProposalAction], inventory_result) -> bool:
        controlled_actions = {
            "split_order",
            "merge_order",
            "change_carrier",
            "inventory_transfer",
            "stockout_resolution",
        }
        return (
            bool(inventory_result.insufficient_skus)
            or any(action.action_type in controlled_actions for action in actions)
            or any(not action.reversible for action in actions)
        )

    @staticmethod
    def _warehouse_can_fulfill_item(record, item) -> bool:
        if record.available_stock <= 0:
            return False
        if record.warehouse_status != "normal":
            return False
        if record.capacity_status not in {"normal", "available", "ok"}:
            return False
        supported = set(record.supported_sku_types or ["standard"])
        sku_type = item.sku_type or "standard"
        if sku_type not in supported and "all" not in supported:
            return False
        if item.special_storage and item.special_storage not in supported and "all" not in supported:
            return False
        return True

    @staticmethod
    def _select_carrier(order_result, split: bool) -> str:
        priority = str(getattr(order_result, "priority", "")).lower()
        if "vip" in priority or "urgent" in priority:
            return "顺丰标快"
        if split:
            return "京东快运"
        return "中通标准"

    @staticmethod
    def _logistics_quote(region: str, carrier: str, quantity: int) -> dict:
        base = {
            "顺丰标快": (18.0, 36),
            "京东快运": (14.0, 48),
            "中通标准": (9.0, 72),
        }.get(carrier, (10.0, 72))
        remote_delta = 6.0 if any(marker in str(region) for marker in ("新疆", "西藏", "内蒙古", "海南")) else 0.0
        price = base[0] + remote_delta + max(quantity - 1, 0) * 2.0
        return {"carrier": carrier, "price": round(price, 2), "eta_hours": base[1]}

    @staticmethod
    def _eta_label(hours: int) -> str:
        if hours <= 0:
            return "待确认"
        if hours < 24:
            return f"{hours} 小时内"
        days = max(1, round(hours / 24))
        return f"{days} 天左右"

    @staticmethod
    def _fingerprint(payload: dict) -> str:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ #
    # finalize 的内部辅助方法（非节点）
    # ------------------------------------------------------------------ #
    def _collect_key_evidences(
        self,
        order_result,
        inventory_result,
        knowledge_result,
        execution_proposal=None,
        preflight_validation=None,
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
        if execution_proposal is not None:
            evidences.append(f"执行提案：{execution_proposal.title}，快照指纹 {execution_proposal.data_fingerprint[:10]}")
        if preflight_validation is not None:
            evidences.append(f"二次校验：{preflight_validation.status}，{preflight_validation.message}")
        return evidences

    def _collect_suggested_actions(
        self,
        inventory_result,
        knowledge_result,
        execution_proposal=None,
        preflight_validation=None,
    ) -> list[str]:
        """收集最终答案里的“建议动作”。

        缺货路径优先采用知识库建议，因为它更贴近业务 SOP；
        快路径没有知识结果时，就根据库存状态给一个保守建议。
        """
        # 优先用知识检索的建议（业务语义最强）
        actions: list[str] = []
        if execution_proposal is not None:
            actions.append(f"审批执行提案：{execution_proposal.summary}")
            actions.extend(
                f"{item.action_type}：{item.reason}"
                for item in execution_proposal.actions[:5]
            )
        if preflight_validation is not None:
            actions.append(preflight_validation.message)
        if actions:
            return actions

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
        execution_proposal=None,
        preflight_validation=None,
    ) -> str:
        """生成最终自然语言结论。

        这不是一个“让 LLM 自由发挥”的节点：
          1. LLM 只能看到 _build_context_for_llm 组织出的事实上下文。
          2. Prompt 要求简洁给出履约结论，不要求模型重新推理库存。
          3. 模型失败时静默降级到规则模板，避免最终出口因为话术模型不可用而失败。
        """
        context = self._build_context_for_llm(order_result, inventory_result, knowledge_result)
        if inventory_result is None:
            return "当前订单信息不完整，主链路未能给出完整结论，请检查上游节点错误。"
        if preflight_validation is not None:
            if preflight_validation.status == "valid":
                return (
                    f"订单 {inventory_result.order_id} 的执行提案已获人工批准，并通过执行前二次校验。"
                    "订单状态、库存和物流报价仍匹配审批快照，可交由 OMS/WMS/TMS 执行动作。"
                )
            if preflight_validation.status == "invalidated":
                return (
                    f"订单 {inventory_result.order_id} 的旧执行提案已自动失效。"
                    "审批后重新查询发现订单、库存或物流报价发生变化，Agent 已重新计算提案，需再次人工确认。"
                )
            if preflight_validation.status == "rejected":
                return (
                    f"订单 {inventory_result.order_id} 的执行提案未被人工批准，"
                    "请进入缺货处置、方案修改或继续追问流程。"
                )
        if execution_proposal is not None and execution_proposal.approval_required:
            return (
                f"订单 {inventory_result.order_id} 已生成「{execution_proposal.title}」，"
                "需要运营在 Action Card 中批准、拒绝、修改方案或继续追问。"
            )

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

        结构标签用于区分订单信息、库存状态和规则参考，
        避免模型把多段事实当成无结构文本处理。
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
