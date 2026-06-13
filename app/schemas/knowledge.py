"""知识检索模块数据模型。

设计理念："检索结果全透明、全可解释"。

- QueryIntentType / QueryIntent：先识别意图，再决定检索策略与扩展方式。
- KnowledgeRetrieveRequest：支持标签过滤，调用方可主动约束知识域。
- HybridScoreDetail：保留多路分数明细字段，让排名调试更透明。
- KnowledgeHit：每个命中片段携带分数明细、类别、术语，零黑盒。
- KnowledgeAnswerSummary：答案摘要层，从命中片段提炼结论、规则、建议。
- KnowledgeRetrieveResult：汇聚全部增强信息，为 LangGraph / LangChain 提供结构化输入。
"""

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. 查询意图
# ---------------------------------------------------------------------------


class QueryIntentType(str, Enum):
    """查询意图类型枚举。

    使用枚举而不是裸字符串的三个理由：
    1. 编译期可检查合法性，防止拼写错误导致的静默 bug。
    2. 后续 LangGraph 做条件路由时，可直接用枚举值做 if/match 分支。
    3. API 返回值规范，前端和下游系统可精确处理每种意图。

    六种意图说明：
    - STOCKOUT_HANDLING      问题核心是"库存不足怎么办、缺货 SOP"。
    - PRIORITY_FULFILLMENT   问题核心是"高优先级订单如何特殊处理"。
    - REGIONAL_STRATEGY      问题核心是"该从哪个区域仓发货、仓配路由"。
    - SPLIT_MERGE            问题核心是"是否拆单或合单"。
    - AFTER_SALES            问题核心是"售后、补发、退货"。
    - GENERAL                无法归入上面五类的通用履约问题。
    """

    STOCKOUT_HANDLING = "stockout_handling"
    PRIORITY_FULFILLMENT = "priority_fulfillment"
    REGIONAL_STRATEGY = "regional_strategy"
    SPLIT_MERGE = "split_merge"
    AFTER_SALES = "after_sales"
    GENERAL = "general"


class QueryIntent(BaseModel):
    """查询意图识别结果。

    为什么不只返回意图类型，还需要置信度和推理依据？

    - 高置信度（> 0.75）：检索策略可以聚焦在主意图对应的知识域。
    - 中等置信度（0.4 ~ 0.75）：需要同时检索主意图 + 次要意图对应知识域。
    - 低置信度（< 0.4）：退化为通用检索，不做意图定向收窄。

    secondary_intents 的意义：
    真实业务问题往往跨越多个领域。例如"高优先级订单的缺货处理"
    同时涉及 PRIORITY_FULFILLMENT 和 STOCKOUT_HANDLING，
    secondary_intents 避免了"非此即彼"的误伤，让检索更全面。
    """

    primary_intent: QueryIntentType = Field(
        ...,
        description="主意图类型。",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="意图识别置信度，范围 0.0 ~ 1.0。",
    )
    reasoning: str = Field(
        ...,
        description="识别为该意图的判定理由，说明触发了哪些关键词或业务信号。",
    )
    secondary_intents: list[QueryIntentType] = Field(
        default_factory=list,
        description="次要意图列表，当问题跨越多个领域时包含多个值。",
    )


# ---------------------------------------------------------------------------
# 2. 请求模型
# ---------------------------------------------------------------------------


class KnowledgeRetrieveRequest(BaseModel):
    """知识检索请求模型。

    filter_categories 是这一版的核心新增能力。

    标签过滤的工作原理：
    1. 召回阶段：使用全部知识文档做向量检索，保证召回率。
    2. 过滤阶段：在重排之前，根据 filter_categories 约束命中结果的类别。
    3. 重排阶段：对过滤后的结果做混合打分与排序。

    为什么"先召回再过滤"而不是"只召回指定类别"？
    - 先召回更宽，不会因为分类不准确而漏掉相关结果。
    - 过滤在重排前，保证最终结果仍然是按质量排序的。
    """

    order_id: str | None = Field(
        default=None,
        description="订单编号。可选；不传时只检索知识库，不做订单和库存上下文增强。",
    )
    question: str = Field(..., description="用户希望检索的业务问题。")
    filter_categories: list[str] = Field(
        default_factory=list,
        description=(
            "标签过滤条件。若不为空，只返回属于这些类别的知识片段。"
            "合法值：stockout_rule, priority_rule, regional_strategy, "
            "split_merge_rule, after_sales_rule, general。"
        ),
    )


# ---------------------------------------------------------------------------
# 3. 混合检索分数明细
# ---------------------------------------------------------------------------


class HybridScoreDetail(BaseModel):
    """混合检索分数明细。

    为什么需要分数明细而不是只给一个总分？

    单一总分是黑盒，调试时无法定位问题：
    - 是语义没打中？→ 看 semantic_score
    - 是关键词没覆盖？→ 看 keyword_score
    - 是业务规则分低？→ 看 business_rule_score
    - 是 rerank 模型改变了排序？→ 看 rerank_score

    当前实现：
    - semantic_score：LlamaIndex 混合召回返回的基础相关性分。
    - keyword_score：query 与 chunk 的业务关键词重合度。
    - business_rule_score：意图类别命中后的业务加权分。
    - rerank_score：Cross-Encoder/DashScope reranker 的归一化分。
    - final_score：上面信号参与排序后的最终分。
    """

    semantic_score: float = Field(
        ...,
        description="语义相似度分数，来自 LlamaIndex dense 召回。",
    )
    keyword_score: float = Field(
        ...,
        description="关键词重合分，表示 query 和 chunk 是否共享关键业务术语。",
    )
    business_rule_score: float = Field(
        ...,
        description="业务规则加权分，来自查询意图与知识类别的匹配。",
    )
    rerank_score: float = Field(
        default=0.0,
        description="rerank 归一化分，来自 Cross-Encoder/DashScope 等重排模型。",
    )
    final_score: float = Field(
        ...,
        description="最终融合分数。",
    )


class KnowledgeMetadata(BaseModel):
    """知识片段细粒度 metadata。"""

    document_id: str = Field(..., description="文档稳定 ID。")
    chunk_id: str = Field(..., description="知识切片稳定 ID。")
    title: str = Field(..., description="文档标题。")
    section_path: list[str] = Field(..., description="片段所在章节路径。")
    tags: list[str] = Field(..., description="知识标签。")
    source_file: str = Field(..., description="来源文件名。")
    source_path: str = Field(..., description="来源文件路径。")
    version: str = Field(default="", description="知识版本号，来自 front matter。")
    owner: str = Field(default="", description="知识负责人，来自 front matter。")
    effective_date: str = Field(default="", description="生效日期，来自 front matter。")
    expires_at: str = Field(default="", description="规则过期日期，来自 front matter；过期规则不会进入最终检索结果。")
    region: str = Field(default="", description="适用区域，来自 front matter。")
    business_scope: list[str] = Field(default_factory=list, description="适用业务范围。")


# ---------------------------------------------------------------------------
# 4. 命中片段模型
# ---------------------------------------------------------------------------


class KnowledgeHit(BaseModel):
    """知识命中片段模型。

    每个命中片段不只是"一段文本 + 一个分数"，
    而是携带了足够的上下文让调用方完全理解"为什么这条被选中"。
    """

    score: float = Field(
        ...,
        description="最终混合分数，与 score_detail.final_score 一致。",
    )
    score_detail: HybridScoreDetail = Field(
        ...,
        description="三维度分数明细，语义分 + 关键词分 + 业务规则分。",
    )
    source_file: str = Field(..., description="来源文件名。")
    category: str = Field(..., description="命中文档所属知识类别。")
    metadata: KnowledgeMetadata = Field(
        ...,
        description="知识片段细粒度 metadata，包括文档 ID、切片 ID、章节路径、标签等。",
    )
    retrieval_channels: list[str] = Field(
        default_factory=list,
        description="命中该片段的召回通道，例如 semantic、bm25。",
    )
    matched_terms: list[str] = Field(
        ...,
        description="本次命中时匹配到的业务关键术语列表。",
    )
    text: str = Field(..., description="命中文本片段内容。")


# ---------------------------------------------------------------------------
# 5. 知识答案摘要层
# ---------------------------------------------------------------------------


class KnowledgeAnswerSummary(BaseModel):
    """知识答案摘要层。

    这是整个知识检索模块价值最高的输出层，也是这次最重要的新增能力。

    为什么必须有这一层？

    问题：5 条原始命中片段摆在那里，调用方（LangChain / LangGraph）该怎么用？
    - 方案 A：直接把 5 条片段拼成 prompt 喂给模型 → 效果差，信息噪声大。
    - 方案 B：这一层先做结构化归纳，输出"结论 + 规则 + 行动" → 模型输入更干净。

    所以这一层的定位是：
    "大模型生成答案的高质量结构化输入"，而不是"给人看的摘要"。

    四个字段的设计意图：

    1. conclusion（直接结论）
       对用户问题的一句话直接回答，来自命中片段的归纳。
       例如："当前订单存在缺货 SKU，应按照缺货 SOP 执行跨仓调拨或延迟发货。"

    2. key_rules（关键规则条目）
       从命中片段中提炼的操作级规则，不是原文照搬，而是精炼成"步骤 N"格式。
       例如：["步骤1：先检查同区域其他仓库存", "步骤2：无同区域库存时允许跨区域调拨"]

    3. suggested_actions（建议操作）
       结合当前订单和库存状态推导出的具体建议，是"知识 × 业务状态"的交叉推导。
       例如：["建议检查华东区域其他仓 SKU 库存", "建议对高优先级订单启动人工复核"]

    4. coverage_note（覆盖度说明）
       告诉调用方"这次检索覆盖了哪些知识域、是否有遗漏"。
       这是排查检索质量问题的重要依据。
    """

    conclusion: str = Field(
        ...,
        description="基于命中知识对用户问题的直接回答摘要。",
    )
    key_rules: list[str] = Field(
        ...,
        description="从命中片段中提炼的关键规则条目列表，操作级。",
    )
    suggested_actions: list[str] = Field(
        ...,
        description="结合当前业务状态推导出的建议操作列表。",
    )
    coverage_note: str = Field(
        ...,
        description="当前检索知识覆盖度说明，是否存在未覆盖的知识领域。",
    )


# ---------------------------------------------------------------------------
# 6. 最终检索结果模型
# ---------------------------------------------------------------------------


class KnowledgeRetrieveResult(BaseModel):
    """知识检索最终结果模型。

    这个模型汇聚了整条检索链路的全部输出，字段顺序对应检索流程顺序：

    1. intent          先识别意图
    2. retrieval_context  再构造业务上下文
    3. expanded_queries   基于意图扩展查询
    4. applied_filters    标签过滤
    5. hits            命中结果（带三维度分数）
    6. matched_categories 命中的知识域
    7. answer_summary  结构化答案摘要
    8. summary         一句话摘要
    """

    order_id: str = Field(..., description="订单编号。")
    question: str = Field(..., description="原始业务问题。")
    intent: QueryIntent = Field(
        ...,
        description="识别出的查询意图，包含主意图、置信度、推理依据。",
    )
    retrieval_context: str = Field(
        ...,
        description="用于检索的业务组合上下文（订单 + 库存 + 问题拼接）。",
    )
    expanded_queries: list[str] = Field(
        ...,
        description="基于意图和业务状态扩展后的多检索查询列表。",
    )
    applied_filters: list[str] = Field(
        default_factory=list,
        description="实际生效的标签过滤条件列表。",
    )
    matched_categories: list[str] = Field(
        ...,
        description="本次命中结果涉及的知识类别列表。",
    )
    hits: list[KnowledgeHit] = Field(
        ...,
        description="命中的知识片段列表，按混合分数降序排列。",
    )
    answer_summary: KnowledgeAnswerSummary = Field(
        ...,
        description="知识答案摘要层，对命中结果的结构化归纳，为后续 LangChain 提供高质量输入。",
    )
    summary: str = Field(..., description="一句话检索摘要。")
