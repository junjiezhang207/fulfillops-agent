"""RAG 答案摘要与对外结果组装（学习版注释）。

这个文件位于 RAG 链路的最后一段：它不负责检索，也不调用 LLM。
它只把已经排序好的 KnowledgeHit 转成更适合 Agent / Workflow 使用的结构化结果。

为什么需要 AnswerBuilder？
  - 原始 chunk 对模型来说噪声较多，直接拼进 prompt 会降低回答稳定性。
  - 上层 Agent 更需要“结论、关键规则、建议动作、覆盖说明”这类结构化输入。
  - API 也需要一个 summary 字段，方便前端和调试日志快速看懂本次检索质量。
"""

from app.schemas.knowledge import (
    KnowledgeAnswerSummary,
    KnowledgeHit,
    KnowledgeRetrieveResult,
    QueryIntent,
    QueryIntentType,
)
from app.services.rag_query_planner import RetrievalInputs


# 业务术语命中表。
# 这里只做轻量解释用途：告诉调用方本次命中片段里有哪些词同时出现在 query 和文本中。
MATCH_TERMS = [
    "缺货",
    "高优先级",
    "区域",
    "跨仓",
    "跨区域",
    "调拨",
    "延迟发货",
    "履约",
    "人工介入",
    "仓配",
    "拆单",
    "合单",
    "补发",
    "退货",
    "售后",
]


class RAGAnswerBuilder:
    """把 RAG 命中片段转成 API 输出和可读摘要。

    这个类不持有状态，所有方法都是确定性加工。
    这样便于单测，也避免并发请求之间互相影响。
    """

    def build_result(
        self,
        order_id: str,
        question: str,
        prepared: RetrievalInputs,
        expanded_queries: list[str],
        hits: list[KnowledgeHit],
    ) -> KnowledgeRetrieveResult:
        """组装 KnowledgeRetrieveResult。

        这是对外的最终结果对象，包含：
          - 原始问题和订单号。
          - 意图、业务上下文、扩展 query。
          - 命中片段 hits。
          - 结构化 answer_summary。
          - 一句话 summary。
        """
        # 库存结果来自 QueryPlanner.prepare 阶段，这里直接复用。
        inventory_result = prepared.inventory_result
        # 从命中片段里统计实际覆盖到的知识类别。
        matched_categories = sorted({hit.category for hit in hits})
        # KnowledgeRetrieveResult 是 RAG API/Agent 使用的统一返回结构。
        return KnowledgeRetrieveResult(
            order_id=order_id,
            question=question,
            intent=prepared.intent,
            retrieval_context=prepared.retrieval_context,
            expanded_queries=expanded_queries,
            applied_filters=prepared.active_filters,
            matched_categories=matched_categories,
            hits=hits,
            answer_summary=self.build_answer_summary(
                question=question,
                intent=prepared.intent,
                hits=hits,
                matched_categories=matched_categories,
                insufficient_skus=inventory_result.insufficient_skus,
                fulfillment_ready=inventory_result.fulfillment_ready,
            ),
            summary=self.build_summary(
                hits,
                question,
                matched_categories,
                prepared.intent,
                prepared.active_filters,
            ),
        )

    def build_answer_summary(
        self,
        question: str,
        intent: QueryIntent,
        hits: list[KnowledgeHit],
        matched_categories: list[str],
        insufficient_skus: list[str],
        fulfillment_ready: bool,
    ) -> KnowledgeAnswerSummary:
        """生成结构化答案摘要。

        注意：这里不是最终自然语言回答。
        它是给 Agent/Workflow 的“证据摘要”，让后续回答更稳。
        """
        if not hits:
            # 没有命中时，给出明确的补救建议，而不是返回空字符串。
            return KnowledgeAnswerSummary(
                conclusion=f"未找到足够回答 [{question}] 的规则片段。",
                key_rules=[],
                suggested_actions=["检查知识标签过滤条件", "补充相关 SOP 后重建索引"],
                coverage_note="当前检索未命中有效规则片段。",
            )
        # 从 top hits 中抽取少量关键规则作为证据摘要。
        key_rules = self.extract_key_rules(hits)
        # coverage_note 用来说明本次检索覆盖了哪些知识域。
        coverage_note = (
            f"共覆盖 {len(matched_categories)} 个知识类别："
            f"{'、'.join(matched_categories) if matched_categories else '无'}。"
        )
        # conclusion/actions/coverage 分开，便于 Agent 后续组织自然语言回答。
        return KnowledgeAnswerSummary(
            conclusion=self.build_conclusion(intent, insufficient_skus, fulfillment_ready, key_rules),
            key_rules=key_rules,
            suggested_actions=self.suggest_actions(
                intent, insufficient_skus, fulfillment_ready, matched_categories
            ),
            coverage_note=coverage_note,
        )

    def extract_key_rules(self, hits: list[KnowledgeHit]) -> list[str]:
        """从命中片段里抽取少量关键规则文本。

        当前采用保守策略：只取 top3 hit 的前两行非空文本。
        这样不会过度加工原始证据，也能避免摘要层生成不存在的规则。
        """
        rules: list[str] = []
        for hit in hits[:3]:
            # 按行拆开，过滤空行。
            lines = [line.strip() for line in hit.text.splitlines() if line.strip()]
            for line in lines[:2]:
                # 去重，避免多个 chunk 开头重复。
                if line not in rules:
                    rules.append(line)
        return rules[:5]

    def suggest_actions(
        self,
        intent: QueryIntent,
        insufficient_skus: list[str],
        fulfillment_ready: bool,
        matched_categories: list[str],
    ) -> list[str]:
        """结合意图、库存状态和命中类别给出建议动作。

        这些动作是规则化生成，不依赖 LLM。
        优点是可解释、稳定；缺点是表达不如模型灵活。
        """
        actions: list[str] = []
        if insufficient_skus:
            # 缺货时优先建议同区补货、跨仓调拨、人工复核。
            actions += [
                "优先检查同区域仓是否存在可替代库存",
                "必要时评估跨仓调拨或跨区域履约",
                "若仍不足，进入缺货订单 SOP 并提示人工复核",
            ]
        elif fulfillment_ready:
            # 库存充足时，重点是选择合适仓配路径。
            actions.append("优先选择同区域或最近区域仓发货")
        if intent.primary_intent == QueryIntentType.PRIORITY_FULFILLMENT:
            # 高优先级订单要强调时效和人工关注。
            actions.append("对高优先级订单增加人工关注和时效复核")
        if intent.primary_intent == QueryIntentType.SPLIT_MERGE:
            # 拆合单影响时效和物流成本。
            actions.append("评估拆单是否影响时效与物流成本")
        if intent.primary_intent == QueryIntentType.AFTER_SALES:
            # 售后场景需要关注库存回写/补发规则。
            actions.append("确认补发或退货规则是否需要触发库存更新")
        if "regional_strategy" in matched_categories:
            # 命中了区域策略知识，就补充仓配路径建议。
            actions.append("结合区域仓配策略选择最优发货路径")
        # 去重并限制数量，避免建议过长。
        return list(dict.fromkeys(actions))[:5]

    def build_conclusion(
        self,
        intent: QueryIntent,
        insufficient_skus: list[str],
        fulfillment_ready: bool,
        key_rules: list[str],
    ) -> str:
        """生成一句业务结论。

        优先级：
          1. 有缺货 SKU：直接提醒缺货风险和 SOP。
          2. 库存可履约：强调区域仓配和优先级规则。
          3. 有关键规则：说明可据此决策。
          4. 否则提示证据不足。
        """
        prefix = f"系统将当前问题识别为 [{intent.primary_intent.value}] 场景。"
        if insufficient_skus:
            return f"{prefix} 当前订单存在缺货风险，应优先参考缺货处理、跨仓调拨规则。"
        if fulfillment_ready:
            return f"{prefix} 当前库存可支持履约，应参考区域仓配、优先级处理规则。"
        if key_rules:
            return f"{prefix} 已提取关键规则，可据此做履约决策。"
        return f"{prefix} 已完成知识检索，需要更多规则片段支持结论。"

    def build_summary(
        self,
        hits: list[KnowledgeHit],
        question: str,
        matched_categories: list[str],
        intent: QueryIntent,
        applied_filters: list[str],
    ) -> str:
        """生成适合 API/日志展示的一句话检索摘要。"""
        if not hits:
            # 无命中时，把意图和过滤条件写清楚，方便排查是否 filter 太窄。
            return (
                f"未检索到与问题 [{question}] 强相关的规则片段。"
                f"识别意图：{intent.primary_intent.value}，过滤条件：{applied_filters or ['无']}。"
            )
        # 有命中时，展示来源文件和类别，方便判断召回质量。
        source_files = "、".join(sorted({h.source_file for h in hits}))
        return (
            f"本次检索围绕问题 [{question}]，识别意图 {intent.primary_intent.value}，"
            f"命中 {len(hits)} 条规则，来源：{source_files}，"
            f"类别：{'、'.join(matched_categories)}，过滤：{'、'.join(applied_filters) or '无'}。"
        )

    def extract_matched_terms(self, text: str, query: str) -> list[str]:
        """提取同时出现在文本和 query 中的业务术语。"""
        # 这个方法只做轻量解释，不参与实际排序。
        return [
            term
            for term in MATCH_TERMS
            if term.lower() in text.lower() and term.lower() in query.lower()
        ]
