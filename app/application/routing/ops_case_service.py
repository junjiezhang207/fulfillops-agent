"""运营异常案件分析服务。

这个服务的定位不是替代 OMS/WMS/TMS/ERP 执行动作，而是把这些系统已经能
提供的事实，整理成运营人员真正需要的案件材料：

- 当前异常为什么重要。
- 适配了哪些 SOP。
- 应该分流给谁。
- 客服/仓配/主管可以怎么沟通。
- 哪些问题值得沉淀成自动规则。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.domain.inventory.analysis import InventoryAnalysisService
from app.domain.orders.analysis import OrderAnalysisService
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
from app.schemas.inventory import InventoryAnalysisResult, SkuInventoryCheckResult
from app.schemas.knowledge import KnowledgeRetrieveResult
from app.schemas.orders import OrderAnalysisResult


@dataclass
class OpsCaseResult:
    """运营异常案件分析结果。"""

    order_id: str
    case_type: str
    severity: str
    owner_team: str
    decision_summary: str
    business_impact: list[str] = field(default_factory=list)
    sop_fit: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    communication_drafts: list[str] = field(default_factory=list)
    automation_candidates: list[str] = field(default_factory=list)
    human_checkpoints: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    source_systems: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """把结构化结果渲染成前端当前可直接展示的运营报告。"""

        sections = [
            ("案件结论", [self.decision_summary]),
            ("案件摘要", [f"类型：{self.case_type}", f"严重度：{self.severity}", f"建议负责人：{self.owner_team}"]),
            ("商业影响", self.business_impact),
            ("SOP 适配", self.sop_fit),
            ("分流建议", self.recommended_actions),
            ("沟通草稿", self.communication_drafts),
            ("人工确认", self.human_checkpoints),
            ("复盘建议", self.automation_candidates),
            ("关键依据", self.evidence),
        ]
        rendered: list[str] = []
        for title, lines in sections:
            useful_lines = [line for line in lines if line]
            if not useful_lines:
                continue
            body = "\n".join(f"- {line}" for line in useful_lines)
            rendered.append(f"{title}：\n{body}")
        return "\n\n".join(rendered)


class OpsCaseAnalysisService:
    """把订单异常转成运营案件材料的纯业务服务。"""

    def __init__(
        self,
        order_service: OrderAnalysisService,
        inventory_service: InventoryAnalysisService,
        knowledge_service: KnowledgeRetrievalService,
    ) -> None:
        self.order_service = order_service
        self.inventory_service = inventory_service
        self.knowledge_service = knowledge_service

    def analyze(self, order_id: str, question: str = "") -> OpsCaseResult:
        """分析一个订单对应的运营异常案件。

        这里会读取 OMS 订单、WMS 库存和 SOP/RAG 结果，但不会执行任何履约动作。
        """

        order = self.order_service.analyze_order(order_id)
        inventory = self.inventory_service.analyze_inventory(order_id)
        knowledge = self._retrieve_sop(order_id, question, inventory)

        stockout_ratio = self._stockout_ratio(inventory)
        order_value = self._order_value(order)
        case_type = self._case_type(inventory, order_value)
        severity = self._severity(order, inventory, order_value, stockout_ratio)
        owner_team = self._owner_team(order, inventory, order_value, severity)
        recommended_actions = self._recommended_actions(order, inventory, knowledge, owner_team)
        human_checkpoints = self._human_checkpoints(order, inventory, order_value, severity)

        return OpsCaseResult(
            order_id=order_id,
            case_type=case_type,
            severity=severity,
            owner_team=owner_team,
            decision_summary=self._decision_summary(order_id, case_type, severity, owner_team),
            business_impact=self._business_impact(order, inventory, order_value, stockout_ratio),
            sop_fit=self._sop_fit(knowledge, inventory),
            recommended_actions=recommended_actions,
            communication_drafts=self._communication_drafts(order, inventory, owner_team),
            automation_candidates=self._automation_candidates(order, inventory, case_type),
            human_checkpoints=human_checkpoints,
            evidence=self._evidence(order, inventory, knowledge),
            source_systems=["OMS 订单", "WMS 库存", "SOP 知识库"],
        )

    def _retrieve_sop(
        self,
        order_id: str,
        question: str,
        inventory: InventoryAnalysisResult,
    ) -> KnowledgeRetrieveResult | None:
        sop_question = question.strip() or "请根据当前订单异常匹配缺货、分流、客服沟通和人工复核 SOP。"
        if inventory.insufficient_skus:
            sop_question += f"\n库存不足 SKU：{'、'.join(inventory.insufficient_skus)}。"
        try:
            return self.knowledge_service.retrieve(
                order_id=order_id,
                question=sop_question,
                filter_categories=[],
            )
        except Exception:
            return None

    @staticmethod
    def _order_value(order: OrderAnalysisResult) -> float:
        return round(sum(item.quantity * item.unit_price for item in order.items), 2)

    @staticmethod
    def _stockout_ratio(inventory: InventoryAnalysisResult) -> float:
        total = len(inventory.sku_checks) or 1
        return len(inventory.insufficient_skus) / total

    @staticmethod
    def _shortage_qty(check: SkuInventoryCheckResult) -> int:
        return max(check.required_quantity - check.total_available_stock, 0)

    def _case_type(self, inventory: InventoryAnalysisResult, order_value: float) -> str:
        if inventory.insufficient_skus and order_value >= 100_000:
            return "高价值缺货异常"
        if inventory.insufficient_skus:
            return "缺货履约异常"
        if order_value >= 100_000:
            return "高价值订单复核"
        return "运营咨询案件"

    def _severity(
        self,
        order: OrderAnalysisResult,
        inventory: InventoryAnalysisResult,
        order_value: float,
        stockout_ratio: float,
    ) -> str:
        priority = str(order.priority).lower()
        if order_value >= 100_000 and (stockout_ratio >= 0.5 or "vip" in priority or "urgent" in priority):
            return "critical"
        if order_value >= 100_000 or stockout_ratio >= 0.5 or "vip" in priority:
            return "high"
        if inventory.insufficient_skus:
            return "medium"
        return "low"

    def _owner_team(
        self,
        order: OrderAnalysisResult,
        inventory: InventoryAnalysisResult,
        order_value: float,
        severity: str,
    ) -> str:
        priority = str(order.priority).lower()
        if severity == "critical":
            return "运营主管"
        if order_value >= 100_000:
            return "财务/运营主管"
        if inventory.insufficient_skus:
            return "仓配运营"
        if "vip" in priority:
            return "客服负责人"
        return "一线运营"

    def _business_impact(
        self,
        order: OrderAnalysisResult,
        inventory: InventoryAnalysisResult,
        order_value: float,
        stockout_ratio: float,
    ) -> list[str]:
        impacts = [
            f"订单金额约 ￥{order_value:,.2f}，涉及 {order.item_count} 个 SKU、{order.total_quantity} 件商品。",
            f"收货区域为 {order.region}，当前订单状态为 {order.order_status}。",
        ]
        if inventory.insufficient_skus:
            impacts.append(
                f"缺货 SKU 占比 {stockout_ratio:.0%}，会影响 SLA、客服承诺和客户体验。"
            )
            shortage_parts = [
                f"{check.sku_id} 缺 {self._shortage_qty(check)} 件"
                for check in inventory.sku_checks
                if not check.fulfillment_ready
            ]
            if shortage_parts:
                impacts.append("缺口明细：" + "；".join(shortage_parts) + "。")
        else:
            impacts.append("库存侧未发现缺口，重点应转向规则解释、客户承诺或内部复核。")
        return impacts

    def _sop_fit(
        self,
        knowledge: KnowledgeRetrieveResult | None,
        inventory: InventoryAnalysisResult,
    ) -> list[str]:
        if knowledge is None:
            return ["未成功读取 SOP 知识库，建议运营按当前企业缺货/客服升级制度人工确认。"]
        rules = list(knowledge.answer_summary.key_rules[:4])
        if inventory.insufficient_skus and not any("缺货" in rule or "库存" in rule for rule in rules):
            rules.append("当前案件存在库存缺口，应优先适配缺货处理、跨仓调拨、替代 SKU 或客户确认 SOP。")
        return rules or [knowledge.answer_summary.conclusion]

    def _recommended_actions(
        self,
        order: OrderAnalysisResult,
        inventory: InventoryAnalysisResult,
        knowledge: KnowledgeRetrieveResult | None,
        owner_team: str,
    ) -> list[str]:
        actions = [f"将案件分流给{owner_team}，由其确认下一步是否释放系统动作。"]
        if inventory.insufficient_skus:
            actions.append("先由程序继续查询各仓可售库存、在途库存和可替代 SKU，人工只判断客户影响和例外处理。")
            actions.append("在执行拆单、替代、延期或取消前，保留客户确认记录和内部审批记录。")
        else:
            actions.append("不重复执行系统已能完成的履约动作，只整理客户沟通、规则解释或复核材料。")
        if knowledge is not None:
            actions.extend(knowledge.answer_summary.suggested_actions[:3])
        return list(dict.fromkeys(actions))

    def _communication_drafts(
        self,
        order: OrderAnalysisResult,
        inventory: InventoryAnalysisResult,
        owner_team: str,
    ) -> list[str]:
        if inventory.insufficient_skus:
            sku_text = "、".join(inventory.insufficient_skus)
            return [
                f"客服对客户：您订单中的 {sku_text} 当前库存不足，我们正在确认跨仓调拨、替代商品或分批发货方案，确认后会给出新的履约时间。",
                f"内部备注：{owner_team} 已接管订单 {order.order_id}，请在客户确认前不要释放不可逆履约动作。",
            ]
        return [
            f"客服对客户：订单 {order.order_id} 当前未发现库存缺口，我们会按既定履约承诺推进。",
            "内部备注：本单主要为规则解释或高价值复核，不需要 Agent 执行系统动作。",
        ]

    def _automation_candidates(
        self,
        order: OrderAnalysisResult,
        inventory: InventoryAnalysisResult,
        case_type: str,
    ) -> list[str]:
        candidates = [
            "可沉淀规则：同类异常再次出现时，由系统自动打标签并推送给对应 owner_team。",
        ]
        if inventory.insufficient_skus:
            candidates.append("可沉淀规则：库存缺口超过 50% 或缺货 SKU 超过 1 个时，自动生成客服确认任务。")
        if self._order_value(order) >= 100_000:
            candidates.append("可沉淀规则：订单金额超过 10 万时，自动附加财务/主管复核材料。")
        candidates.append(f"复盘主题：{case_type} 是否可以通过库存预警、补货策略或 SOP 字段结构化减少人工判断。")
        return candidates

    def _human_checkpoints(
        self,
        order: OrderAnalysisResult,
        inventory: InventoryAnalysisResult,
        order_value: float,
        severity: str,
    ) -> list[str]:
        checkpoints: list[str] = []
        if inventory.insufficient_skus:
            checkpoints.append("确认客户是否接受延期、拆单或替代 SKU。")
        if order_value >= 100_000:
            checkpoints.append("确认是否需要财务或主管审批，避免大额订单误放行。")
        if severity in {"high", "critical"}:
            checkpoints.append("确认本次处理是否会造成平台处罚、VIP 投诉或赔付。")
        return checkpoints or ["无需额外人工审批，但建议保留本次 SOP 匹配记录。"]

    def _evidence(
        self,
        order: OrderAnalysisResult,
        inventory: InventoryAnalysisResult,
        knowledge: KnowledgeRetrieveResult | None,
    ) -> list[str]:
        items = [f"订单事实：{order.summary}", f"库存事实：{inventory.summary}"]
        if knowledge is not None:
            items.append(f"SOP 覆盖：{knowledge.answer_summary.coverage_note}")
            for hit in knowledge.hits[:3]:
                items.append(f"SOP 来源：{hit.source_file} / {hit.category} / score {hit.score:.2f}")
        return items

    @staticmethod
    def _decision_summary(order_id: str, case_type: str, severity: str, owner_team: str) -> str:
        return (
            f"订单 {order_id} 被整理为「{case_type}」，严重度 {severity}。"
            f"建议由{owner_team}接管判断，Agent 只提供依据、话术和复盘建议，不直接执行系统动作。"
        )

