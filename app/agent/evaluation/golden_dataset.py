"""文件作用摘要：Agent / RAG 质量评测用的 Golden Dataset。

这个文件保存一组人工整理的“标准业务问题”。它不是线上业务逻辑，而是用于
回归测试和离线评测：当 prompt、工具描述、RAG 检索逻辑或模型配置变化后，
可以用这些 case 检查回答是否仍然命中关键事实、是否调用了必要工具、是否出现幻觉。

主要做的事：
1. ``GoldenCase``：定义单个评测样例的数据结构。
2. ``GOLDEN_DATASET``：保存订单、库存、RAG、履约等典型问题。
3. ``expected_tools``：标注至少应该调用哪些工具。
4. ``must_contain``：答案必须包含的订单号、SKU、关键词或数字。
5. ``must_not_hallucinate``：答案不应该凭空出现的错误实体。
6. ``ground_truth_keywords``：给 Ragas / 评测脚本作为参考关键词。
7. ``get_cases_by_tag``：按标签筛选某一类评测问题。

维护原则：
- case 要来自真实或高度仿真的业务场景。
- expected_tools 是最小必要工具集，不要求穷举所有可能工具。
- 新增业务能力时，同步补充对应 GoldenCase，防止后续改代码时退化。

学习时先看：
1. ``GoldenCase`` 字段含义。
2. ``GOLDEN_DATASET`` 里的几个典型问题。
3. ``get_cases_by_tag`` 如何给测试选择子集。
"""

from dataclasses import dataclass, field


# 面试官可能问：Golden Dataset 在大模型项目里有什么用？
# 回答：它让项目不是只靠一次 demo 证明效果，而是有一组固定业务问题做回归。
# 改 prompt、换模型、改 RAG 后，可以检查关键实体、必要工具和幻觉风险有没有退化。
@dataclass
class GoldenCase:
    id: str
    question: str                          # 用户输入
    expected_tools: list[str]              # 至少应调用其中之一
    must_contain: list[str] = field(default_factory=list)      # 答案必须包含
    must_not_hallucinate: list[str] = field(default_factory=list)  # 不能凭空出现
    ground_truth_keywords: list[str] = field(default_factory=list)  # Ragas 参考关键词
    tags: list[str] = field(default_factory=list)              # 场景标签


GOLDEN_DATASET: list[GoldenCase] = [
    # ── 基础查询 ─────────────────────────────────────────────────────────────
    GoldenCase(
        id="gc-001",
        question="订单 SO202502140001 的基本信息是什么？",
        expected_tools=["analyze_order"],
        must_contain=["SO202502140001"],
        must_not_hallucinate=["SO999", "假设"],
        ground_truth_keywords=["订单", "SKU", "数量", "平台"],
        tags=["order", "basic"],
    ),
    GoldenCase(
        id="gc-002",
        question="SO202502140001 当前库存够发货吗？",
        expected_tools=["check_inventory", "analyze_order"],
        must_contain=["SO202502140001"],
        must_not_hallucinate=["数据不可用", "无法确认"],
        ground_truth_keywords=["库存", "充足", "不足", "SKU"],
        tags=["inventory", "fulfillment"],
    ),
    GoldenCase(
        id="gc-003",
        question="SKU-IPHONE-CASE-001 在哪些仓库有货，各有多少？",
        expected_tools=["search_warehouse_inventory"],
        must_contain=["SKU-IPHONE-CASE-001"],
        must_not_hallucinate=["WH-999", "北极仓"],
        ground_truth_keywords=["仓库", "可用", "件"],
        tags=["warehouse", "sku"],
    ),
    # ── 缺货处理 ─────────────────────────────────────────────────────────────
    GoldenCase(
        id="gc-004",
        question="SKU-IPHONE-CASE-001 缺货了，有什么替代方案？",
        expected_tools=["find_substitute_sku"],
        must_contain=["SKU-IPHONE-CASE-001"],
        must_not_hallucinate=["百分之百兼容", "绝对没问题"],
        ground_truth_keywords=["替代", "兼容", "SKU"],
        tags=["substitute", "stockout"],
    ),
    GoldenCase(
        id="gc-005",
        question="订单缺货时，优先级高的客户应该怎么处理？",
        expected_tools=["retrieve_knowledge"],
        must_contain=[],
        must_not_hallucinate=["我猜", "可能是"],
        ground_truth_keywords=["优先级", "规则", "处理", "客户"],
        tags=["knowledge", "policy"],
    ),
    # ── 履约方案 ─────────────────────────────────────────────────────────────
    GoldenCase(
        id="gc-006",
        question="请为订单 SO202502140001 生成完整的履约方案。",
        expected_tools=["generate_fulfillment_plan"],
        must_contain=["SO202502140001"],
        must_not_hallucinate=["方案A", "方案B"],
        ground_truth_keywords=["履约", "仓库", "发货", "天"],
        tags=["fulfillment", "plan"],
    ),
    GoldenCase(
        id="gc-007",
        question="SO202502140002 的库存情况如何？能快速发货吗？",
        expected_tools=["check_inventory", "analyze_order"],
        must_contain=["SO202502140002"],
        must_not_hallucinate=["SO202502140001"],  # 不应混淆不同订单
        ground_truth_keywords=["库存", "发货", "天"],
        tags=["inventory", "fulfillment"],
    ),
    # ── 多步推理 ─────────────────────────────────────────────────────────────
    GoldenCase(
        id="gc-008",
        question="SO202502140001 库存不足时，跨仓调配需要遵守哪些规则？",
        expected_tools=["retrieve_knowledge"],
        must_contain=[],
        must_not_hallucinate=["没有规则", "随意调配"],
        ground_truth_keywords=["跨仓", "规则", "调配"],
        tags=["knowledge", "multi-step"],
    ),
    GoldenCase(
        id="gc-009",
        question="帮我分析订单 SO202502140001 的完整履约可行性，包括库存、仓库分布和备选方案。",
        expected_tools=["analyze_order", "check_inventory"],
        must_contain=["SO202502140001"],
        must_not_hallucinate=["100% 确定", "绝对可以"],
        ground_truth_keywords=["库存", "仓库", "方案", "可行"],
        tags=["multi-tool", "comprehensive"],
    ),
    # ── 边界场景 ─────────────────────────────────────────────────────────────
    GoldenCase(
        id="gc-010",
        question="订单号 SO999999 的信息是什么？",
        expected_tools=["analyze_order"],
        must_contain=[],
        must_not_hallucinate=["SO999999 已发货", "库存充足"],  # 不存在的订单不能被编造
        ground_truth_keywords=["未找到", "不存在", "查询失败"],
        tags=["edge-case", "error-handling"],
    ),
]


# 面试官可能问：为什么按 tag 取 case？
# 回答：不同改动影响范围不同。比如只改库存工具，就只跑 inventory 标签用例；
# 改 RAG 或 prompt 时再跑更大的集合，能提升回归效率。
def get_cases_by_tag(tag: str) -> list[GoldenCase]:
    """按标签筛选测试用例，供分组运行使用。"""
    return [c for c in GOLDEN_DATASET if tag in c.tags]
