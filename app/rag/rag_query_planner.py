"""RAG 查询规划（学习版注释）：业务上下文、意图识别、query 扩展。

这个文件负责 RAG 链路的“检索前准备”，不直接访问向量库。

为什么要单独拆出来？
  - KnowledgeRetrievalService 只应该负责索引、召回、重排和结果组装。
  - 意图识别、库存上下文拼接、query 扩展属于“检索策略”，变化频率更高。
  - 单独拆开后，后续可以把规则意图识别替换成小模型分类，而不用动主检索服务。

输入：
  order_id + question + filter_categories

输出：
  RetrievalInputs，其中包含库存分析结果、识别意图、业务检索上下文和过滤条件。
"""

from dataclasses import dataclass

from app.schemas.inventory import InventoryAnalysisResult
from app.schemas.knowledge import QueryIntent, QueryIntentType
from app.domain.inventory.analysis import InventoryAnalysisService
from app.rag.query_rewriter import QueryRewriter


# 轻量规则意图识别关键词。
# 当前不用 LLM 分类，是为了保证本地演示稳定、便宜、可解释。
# 后续如果换成小模型分类，可以保留这些关键词作为 fallback。
INTENT_KEYWORDS = {
    QueryIntentType.STOCKOUT_HANDLING: ["缺货", "库存不足", "延迟发货", "调拨", "人工介入"],
    QueryIntentType.PRIORITY_FULFILLMENT: ["高优先级", "优先级", "时效", "会员", "大促"],
    QueryIntentType.REGIONAL_STRATEGY: ["区域", "华东", "华南", "华北", "跨区域", "仓配"],
    QueryIntentType.SPLIT_MERGE: ["拆单", "合单", "子单"],
    QueryIntentType.AFTER_SALES: ["售后", "补发", "退货", "质检"],
}

# 每类意图对应的补充检索 query。
# 用户原话往往很短，例如“这个订单怎么办”，扩展后能召回到 SOP 里的正式术语。
INTENT_QUERY_EXPANSIONS = {
    QueryIntentType.STOCKOUT_HANDLING: ["缺货履约规则", "缺货订单 SOP"],
    QueryIntentType.PRIORITY_FULFILLMENT: ["高优先级订单履约规范"],
    QueryIntentType.REGIONAL_STRATEGY: ["区域仓配策略", "跨区域履约规则"],
    QueryIntentType.SPLIT_MERGE: ["拆单与合单履约规则"],
    QueryIntentType.AFTER_SALES: ["售后与补发履约规则"],
    QueryIntentType.GENERAL: ["通用履约规则"],
}


def unique_nonempty(items: list[str]) -> list[str]:
    """去除空字符串并保持原顺序去重。"""
    # dict.fromkeys 在 Python 3.7+ 会保留插入顺序。
    return list(dict.fromkeys(item.strip() for item in items if item.strip()))


@dataclass
class RetrievalInputs:
    """召回前准备好的业务上下文。

    active_filters：调用方显式传入的知识类别过滤。
    inventory_result：订单库存分析结果，影响意图识别和 query 扩展。
    intent：识别出的主意图/次意图。
    retrieval_context：拼接后的检索上下文，会作为一条 query 参与召回。
    """

    active_filters: list[str]
    inventory_result: InventoryAnalysisResult
    intent: QueryIntent
    retrieval_context: str


class RAGQueryPlanner:
    """把订单状态和用户问题转成检索计划。

    Planner 的核心思想是：不要只拿用户原句去查知识库。
    供应链问题往往依赖订单状态，例如库存是否不足、哪些 SKU 缺货、
    订单是否已经可履约。把这些状态放入 query，召回质量会明显更稳定。
    """

    def __init__(
        self,
        inventory_analysis_service: InventoryAnalysisService,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        # 库存分析用于把订单状态加入检索计划。
        self.inventory_analysis_service = inventory_analysis_service
        # 可选 LLM 改写器；没配置也不影响规则扩展。
        self.query_rewriter = query_rewriter

    def prepare(
        self,
        order_id: str,
        question: str,
        filter_categories: list[str] | None,
    ) -> RetrievalInputs:
        """准备一次检索所需的全部业务上下文。

        这里会先查库存，再识别意图。
        这样做的原因是：同一个问题“怎么办”，在库存充足和库存不足时应该检索不同规则。
        """
        # 先分析库存，因为库存状态会影响意图和扩展 query。
        inventory_result = self.inventory_analysis_service.analyze_inventory(order_id)
        # 结合用户问题和库存状态识别意图。
        intent = self.recognize_intent(
            question=question,
            insufficient_skus=inventory_result.insufficient_skus,
            fulfillment_ready=inventory_result.fulfillment_ready,
        )
        # 构建一条“高信息密度”的检索上下文 query。
        retrieval_context = self.build_retrieval_context(
            question=question,
            inventory_summary=inventory_result.summary,
            order_summary=inventory_result.order_summary,
            insufficient_skus=inventory_result.insufficient_skus,
            intent=intent,
        )
        # active_filters 是调用方指定的知识类别过滤，没有就用空列表。
        return RetrievalInputs(filter_categories or [], inventory_result, intent, retrieval_context)

    def build_queries(self, question: str, prepared: RetrievalInputs) -> list[str]:
        """构建同步检索 query 列表。

        返回内容包含：
          - 原始问题
          - 订单/库存/意图组成的业务上下文 query
          - 基于规则的领域扩展 query
          - 可选 LLM query rewrite 结果
        """
        # 规则扩展永远可用，是 RAG 的基础兜底能力。
        rule_queries = self.rule_expanded_queries(question, prepared)
        if self.query_rewriter is None:
            return rule_queries
        # 有 LLM rewriter 时，把模型改写和规则扩展合并去重。
        return unique_nonempty([
            *self.query_rewriter.rewrite(question, n=3),
            *rule_queries,
        ])

    async def abuild_queries(self, question: str, prepared: RetrievalInputs) -> list[str]:
        """异步构建检索 query 列表。"""
        rule_queries = self.rule_expanded_queries(question, prepared)
        if self.query_rewriter is None:
            return rule_queries
        # 异步调用 LLM 改写器，避免阻塞上层 async API。
        rewritten = await self.query_rewriter.arewrite(question, n=3)
        return unique_nonempty([*rewritten, *rule_queries])

    def rule_expanded_queries(self, question: str, prepared: RetrievalInputs) -> list[str]:
        """只使用规则生成扩展 query。

        即使没有配置 query_rewriter，也能保证 RAG 至少有业务规则扩展能力。
        """
        # 这里读取 prepare 阶段已经算好的库存结果，避免重复查库存。
        inventory_result = prepared.inventory_result
        return self.build_expanded_queries(
            question=question,
            retrieval_context=prepared.retrieval_context,
            insufficient_skus=inventory_result.insufficient_skus,
            fulfillment_ready=inventory_result.fulfillment_ready,
            intent=prepared.intent,
        )

    def recognize_intent(
        self, question: str, insufficient_skus: list[str], fulfillment_ready: bool
    ) -> QueryIntent:
        """识别用户问题的业务意图。

        评分由两部分组成：
          1. 用户问题里的关键词命中。
          2. 库存状态带来的业务修正。

        例如只要存在 insufficient_skus，即使用户没说“缺货”，也会提高缺货处理意图分。
        """
        # 统一小写后做关键词匹配，中文不受大小写影响，英文 SKU/词也能兼容。
        normalized = question.lower()
        # 统计每个意图类别命中了几个关键词。
        scores = {
            intent_type: sum(1 for kw in keywords if kw.lower() in normalized)
            for intent_type, keywords in INTENT_KEYWORDS.items()
        }
        if insufficient_skus:
            # 有缺货 SKU 时，即使用户没明说“缺货”，也应强烈倾向缺货处理。
            scores[QueryIntentType.STOCKOUT_HANDLING] += 2
        if fulfillment_ready:
            # 库存充足时，更多是仓配/区域履约策略问题。
            scores[QueryIntentType.REGIONAL_STRATEGY] += 1

        # 只保留得分大于 0 的意图。
        scored = [(t, s) for t, s in scores.items() if s > 0]
        if not scored:
            # 没有明显命中时，给通用意图和较低置信度。
            return QueryIntent(
                primary_intent=QueryIntentType.GENERAL,
                confidence=0.45,
                reasoning="未命中明显领域术语，归类为通用意图。",
                secondary_intents=[],
            )
        # 得分最高的是主意图，其余前两个作为次意图。
        scored.sort(key=lambda x: x[1], reverse=True)
        primary, primary_score = scored[0]
        total = sum(s for _, s in scored)
        return QueryIntent(
            primary_intent=primary,
            confidence=round(primary_score / max(total, 1), 2),
            reasoning=f"命中 {primary.value} 相关术语，并结合库存状态修正。",
            secondary_intents=[t for t, _ in scored[1:3]],
        )

    def build_retrieval_context(
        self,
        question: str,
        inventory_summary: str,
        order_summary: str,
        insufficient_skus: list[str],
        intent: QueryIntent,
    ) -> str:
        """把用户问题、订单摘要、库存摘要和意图拼成检索上下文。

        这段文本不是给用户看的，而是作为一条高信息密度 query 参与召回。
        它能把“这个订单怎么处理”这类短问题补全成带业务状态的检索请求。
        """
        # 缺货 SKU 列表会显著影响召回，例如触发缺货 SOP。
        insufficient_text = (
            "无库存不足 SKU" if not insufficient_skus
            else f"库存不足 SKU：{'、'.join(insufficient_skus)}"
        )
        # 这段上下文会作为 query 参与召回，不是展示给最终用户的回答。
        return (
            f"用户问题：{question}\n"
            f"识别意图：{intent.primary_intent.value}\n"
            f"订单摘要：{order_summary}\n"
            f"库存摘要：{inventory_summary}\n"
            f"库存风险：{insufficient_text}\n"
            "请检索最相关的履约规则、优先级规范、缺货处理 SOP、区域仓配策略、拆单合单规则或售后规则。"
        )

    def build_expanded_queries(
        self,
        question: str,
        retrieval_context: str,
        insufficient_skus: list[str],
        fulfillment_ready: bool,
        intent: QueryIntent,
    ) -> list[str]:
        """生成最终的规则扩展 query 列表。

        扩展策略：
          - 原始问题永远保留，避免扩展方向跑偏。
          - 主意图和次意图都加入对应 SOP 关键词。
          - 根据库存是否充足补充不同履约场景 query。
        """
        # 原始问题 + 业务上下文是基础 query。
        queries = [question, retrieval_context]
        # 主意图扩展，例如缺货 -> 缺货履约规则、缺货订单 SOP。
        queries.extend(INTENT_QUERY_EXPANSIONS.get(intent.primary_intent, []))
        for secondary_intent in intent.secondary_intents:
            # 次意图也加入扩展，避免跨领域问题漏召回。
            queries.extend(INTENT_QUERY_EXPANSIONS.get(secondary_intent, []))
        if insufficient_skus:
            # 缺货场景补充具体 SKU，让检索更贴近当前订单。
            queries.append(f"库存不足 SKU：{'、'.join(insufficient_skus)}，优先检索缺货处理与跨仓规则")
        if fulfillment_ready:
            queries.append("库存充足场景下的区域优先发货策略")
        else:
            queries.append("库存不足时跨仓调拨、跨区域履约和拆单建议")
        # 去重并去掉空字符串。
        return unique_nonempty(queries)
