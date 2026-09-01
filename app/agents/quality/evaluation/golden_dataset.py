"""Agent / RAG 质量回归用 Golden Dataset。

Golden Dataset 是固定评测题库，不是线上业务逻辑。每条 case 都围绕项目内置
mock 订单、库存、仓库、替代 SKU 和知识库规则编写，确保评估结果可重复、可解释。

本文件同时保留两类期望：
1. 兼容字段：``expected_tools``、``must_contain``、``must_not_hallucinate``。
2. 结构化字段：事实期望、业务决策期望、证据要求、硬失败条件和软评估 rubric。

维护原则：
- 订单、库存、仓库、替代 SKU 等事实必须来自固定 mock 数据。
- RAG 规则可以由大模型辅助生成，但一旦进入评测集就要固化为知识文档。
- 新增工具、路由或业务规则时，同步补充对应 case，避免后续改动造成能力退化。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GoldenCase:
    """单条评测样例。

    字段说明：
    - expected_tools：本题至少应调用的工具集合；评分器会按调用覆盖度打分。
    - expected_facts：可由工具结果验证的事实，例如订单号、缺货 SKU、仓库名。
    - expected_decision：业务结论期望，例如是否可全量履约、答案类型、下一步动作。
    - evidence_requirements：某类结论必须由哪个工具支撑。
    - hard_fail_conditions：触发即失败的严重错误。
    - judge_rubric：留给 LLM-as-judge 或人工复核的软指标。
    """

    id: str
    question: str
    expected_tools: list[str]
    must_contain: list[str] = field(default_factory=list)
    must_not_hallucinate: list[str] = field(default_factory=list)
    ground_truth_keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    expected_facts: dict[str, Any] = field(default_factory=dict)
    expected_decision: dict[str, Any] = field(default_factory=dict)
    evidence_requirements: dict[str, str] = field(default_factory=dict)
    expected_answer_points: list[str] = field(default_factory=list)
    expected_entities: list[str] = field(default_factory=list)
    hard_fail_conditions: list[str] = field(default_factory=list)
    judge_rubric: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    min_score: float = 75.0
    mock_business_state: dict[str, Any] = field(default_factory=dict)
    conversation: list[str] = field(default_factory=list)
    applicable_sop: list[str] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    external_task_result: dict[str, Any] = field(default_factory=dict)
    success_criteria: dict[str, Any] = field(default_factory=dict)
    workflow_path: str = "normal_once_success"


COMMON_HARD_FAILS = [
    "unsupported_entity",
    "fabricates_inventory_quantity",
    "overconfident_absolute",
]

STOCKOUT_HARD_FAILS = COMMON_HARD_FAILS + [
    "missing_required_tool",
    "claims_full_fulfillment_ready",
    "claims_order_shipped",
]

NOT_FOUND_HARD_FAILS = [
    "claims_order_shipped",
    "claims_full_fulfillment_ready",
    "wrong_not_found_handling",
    "unsupported_entity",
]


GOLDEN_DATASET: list[GoldenCase] = [
    # ── A. 订单基础查询 ────────────────────────────────────────────────────
    GoldenCase(
        id="gc-order-001",
        question="订单 SO202502140001 的基本信息是什么？",
        expected_tools=["analyze_order"],
        must_contain=["SO202502140001"],
        must_not_hallucinate=["SO999", "已发货"],
        ground_truth_keywords=["订单", "SKU", "数量", "平台", "待履约"],
        tags=["order", "basic", "single-turn"],
        expected_facts={
            "order_id": "SO202502140001",
            "platform": "抖音商城",
            "priority": "高",
            "skus": ["SKU-IPHONE-CASE-001", "SKU-CHARGER-020W-002"],
        },
        expected_decision={"answer_type": "order_summary"},
        evidence_requirements={"order_facts": "analyze_order"},
        expected_answer_points=["说明平台/区域/优先级", "列出订单 SKU 或商品信息"],
        expected_entities=["SO202502140001", "SKU-IPHONE-CASE-001", "SKU-CHARGER-020W-002"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool"],
        judge_rubric=["是否清楚说明订单基本事实", "是否避免引入工具结果外的订单状态"],
        difficulty="easy",
    ),
    GoldenCase(
        id="gc-order-002",
        question="SO202502140002 是哪个平台的订单，优先级高吗？",
        expected_tools=["analyze_order"],
        must_contain=["SO202502140002"],
        must_not_hallucinate=["高优先级", "京东自营"],
        ground_truth_keywords=["天猫旗舰店", "中", "广州"],
        tags=["order", "basic", "priority"],
        expected_facts={
            "order_id": "SO202502140002",
            "platform": "天猫旗舰店",
            "priority": "中",
        },
        expected_decision={"answer_type": "order_summary"},
        evidence_requirements={"order_facts": "analyze_order"},
        expected_answer_points=["说明平台是天猫旗舰店", "说明优先级不是高优先级"],
        expected_entities=["SO202502140002"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool"],
        difficulty="easy",
    ),
    GoldenCase(
        id="gc-order-003",
        question="SO202502140003 里有哪些商品？",
        expected_tools=["analyze_order"],
        must_contain=["SO202502140003"],
        must_not_hallucinate=["SKU-IPHONE-CASE-001", "已发货"],
        ground_truth_keywords=["SKU-ROUTER-WIFI7-001", "SKU-CABLE-TYPEC-003"],
        tags=["order", "sku"],
        expected_facts={
            "order_id": "SO202502140003",
            "skus": ["SKU-ROUTER-WIFI7-001", "SKU-CABLE-TYPEC-003"],
        },
        expected_decision={"answer_type": "order_items"},
        evidence_requirements={"order_items": "analyze_order"},
        expected_answer_points=["列出 WiFi 7 路由器", "列出 Type-C 数据线"],
        expected_entities=["SO202502140003", "SKU-ROUTER-WIFI7-001", "SKU-CABLE-TYPEC-003"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool"],
    ),

    # ── B. 库存与履约判断 ──────────────────────────────────────────────────
    GoldenCase(
        id="gc-inv-001",
        question="SO202502140001 当前库存够发货吗？",
        expected_tools=["check_inventory", "analyze_order"],
        must_contain=["SO202502140001"],
        must_not_hallucinate=["已发货", "无需处理"],
        ground_truth_keywords=["库存", "不能全量履约", "SKU-CHARGER-020W-002"],
        tags=["inventory", "fulfillment", "stockout"],
        expected_facts={
            "order_id": "SO202502140001",
            "fulfillment_ready": False,
            "insufficient_skus": ["SKU-CHARGER-020W-002"],
        },
        expected_decision={
            "answer_type": "stockout",
            "can_ship_full": False,
            "next_actions": ["check_warehouse_inventory", "consider_split_shipment"],
        },
        evidence_requirements={
            "inventory_status": "check_inventory",
            "order_items": "analyze_order",
        },
        expected_answer_points=["明确不能全量履约", "指出缺货 SKU", "给出下一步处理建议"],
        expected_entities=["SO202502140001", "SKU-CHARGER-020W-002"],
        hard_fail_conditions=STOCKOUT_HARD_FAILS,
        judge_rubric=["是否把库存不足作为主要结论", "是否说明下一步可查仓库或拆单"],
    ),
    GoldenCase(
        id="gc-inv-002",
        question="订单 SO202502140002 能正常履约吗？",
        expected_tools=["check_inventory", "analyze_order"],
        must_contain=["SO202502140002"],
        must_not_hallucinate=["缺货", "SKU-IPHONE-CASE-001"],
        ground_truth_keywords=["库存", "可履约", "SKU-BOTTLE-INS-001"],
        tags=["inventory", "fulfillment", "ready"],
        expected_facts={
            "order_id": "SO202502140002",
            "fulfillment_ready": True,
            "insufficient_skus": [],
        },
        expected_decision={"answer_type": "fulfillable", "can_ship_full": True},
        evidence_requirements={"inventory_status": "check_inventory"},
        expected_answer_points=["明确可履约", "不要提出无依据的缺货处理"],
        expected_entities=["SO202502140002", "SKU-BOTTLE-INS-001"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool"],
        difficulty="easy",
    ),
    GoldenCase(
        id="gc-inv-003",
        question="SO202502140003 有没有库存风险？",
        expected_tools=["check_inventory", "analyze_order"],
        must_contain=["SO202502140003"],
        must_not_hallucinate=["库存充足", "无需处理"],
        ground_truth_keywords=["SKU-CABLE-TYPEC-003", "不足"],
        tags=["inventory", "risk", "stockout"],
        expected_facts={
            "order_id": "SO202502140003",
            "fulfillment_ready": False,
            "insufficient_skus": ["SKU-CABLE-TYPEC-003"],
        },
        expected_decision={"answer_type": "stockout", "can_ship_full": False},
        evidence_requirements={"inventory_status": "check_inventory"},
        expected_answer_points=["说明存在库存风险", "指出 Type-C 数据线数量不足"],
        expected_entities=["SO202502140003", "SKU-CABLE-TYPEC-003"],
        hard_fail_conditions=STOCKOUT_HARD_FAILS,
    ),
    GoldenCase(
        id="gc-inv-004",
        question="帮我判断 SO202502140001 是否需要拆单或调拨。",
        expected_tools=["check_inventory", "analyze_order"],
        must_contain=["SO202502140001"],
        must_not_hallucinate=["已发货", "全部库存充足"],
        ground_truth_keywords=["调拨", "拆单", "SKU-CHARGER-020W-002"],
        tags=["inventory", "fulfillment", "action"],
        expected_facts={
            "order_id": "SO202502140001",
            "fulfillment_ready": False,
            "insufficient_skus": ["SKU-CHARGER-020W-002"],
        },
        expected_decision={
            "answer_type": "stockout",
            "can_ship_full": False,
            "next_actions": ["consider_split_shipment", "check_warehouse_inventory"],
        },
        evidence_requirements={"inventory_status": "check_inventory"},
        expected_answer_points=["说明需要拆单或调拨评估", "指出触发原因是库存不足"],
        expected_entities=["SO202502140001", "SKU-CHARGER-020W-002"],
        hard_fail_conditions=STOCKOUT_HARD_FAILS,
    ),
    GoldenCase(
        id="gc-inv-005",
        question="SO202502140002 有哪些缺货 SKU？",
        expected_tools=["check_inventory"],
        must_contain=["SO202502140002"],
        must_not_hallucinate=["SKU-CHARGER-020W-002", "SKU-CABLE-TYPEC-003"],
        ground_truth_keywords=["无缺货", "可履约"],
        tags=["inventory", "negative-case"],
        expected_facts={
            "order_id": "SO202502140002",
            "fulfillment_ready": True,
            "insufficient_skus": [],
        },
        expected_decision={"answer_type": "no_stockout", "can_ship_full": True},
        evidence_requirements={"inventory_status": "check_inventory"},
        expected_answer_points=["明确没有缺货 SKU", "避免编造缺货项"],
        expected_entities=["SO202502140002"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool"],
        difficulty="easy",
    ),

    # ── C. 仓库库存分布 ────────────────────────────────────────────────────
    GoldenCase(
        id="gc-wh-001",
        question="SKU-IPHONE-CASE-001 在哪些仓库有货，各有多少？",
        expected_tools=["search_warehouse_inventory"],
        must_contain=["SKU-IPHONE-CASE-001"],
        must_not_hallucinate=["WH-999", "北极仓"],
        ground_truth_keywords=["上海金桥仓", "广州南沙仓", "北京大兴仓"],
        tags=["warehouse", "sku", "inventory"],
        expected_facts={
            "sku_id": "SKU-IPHONE-CASE-001",
            "warehouse_names": ["上海金桥仓", "广州南沙仓", "北京大兴仓"],
            "total_available": 43,
        },
        expected_decision={"answer_type": "warehouse_distribution"},
        evidence_requirements={"warehouse_distribution": "search_warehouse_inventory"},
        expected_answer_points=["列出有货仓库", "说明总可用库存或最充足仓"],
        expected_entities=["SKU-IPHONE-CASE-001", "WH-SH-001", "WH-GZ-001", "WH-BJ-001"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool"],
    ),
    GoldenCase(
        id="gc-wh-002",
        question="SKU-BOTTLE-INS-001 哪个仓最适合发华南订单？",
        expected_tools=["search_warehouse_inventory"],
        must_contain=["SKU-BOTTLE-INS-001"],
        must_not_hallucinate=["北京大兴仓库存充足"],
        ground_truth_keywords=["广州南沙仓", "华南-广州"],
        tags=["warehouse", "regional"],
        expected_facts={
            "sku_id": "SKU-BOTTLE-INS-001",
            "warehouse_names": ["广州南沙仓"],
        },
        expected_decision={"answer_type": "warehouse_recommendation"},
        evidence_requirements={"warehouse_distribution": "search_warehouse_inventory"},
        expected_answer_points=["优先推荐广州南沙仓", "说明区域匹配或可用库存原因"],
        expected_entities=["SKU-BOTTLE-INS-001", "WH-GZ-001"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool"],
    ),
    GoldenCase(
        id="gc-wh-003",
        question="SKU-UNKNOWN-404 全国仓还有库存吗？",
        expected_tools=["search_warehouse_inventory"],
        must_contain=["SKU-UNKNOWN-404"],
        must_not_hallucinate=["上海金桥仓有货", "库存充足"],
        ground_truth_keywords=["无现货", "0"],
        tags=["warehouse", "edge-case"],
        expected_facts={"sku_id": "SKU-UNKNOWN-404", "total_available": 0},
        expected_decision={"answer_type": "warehouse_no_stock"},
        evidence_requirements={"warehouse_distribution": "search_warehouse_inventory"},
        expected_answer_points=["明确全国仓无现货", "不要编造仓库库存"],
        expected_entities=["SKU-UNKNOWN-404"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool", "fabricates_inventory_quantity"],
    ),

    # ── D. 替代 SKU ────────────────────────────────────────────────────────
    GoldenCase(
        id="gc-sub-001",
        question="SKU-IPHONE-CASE-001 缺货了，有什么替代方案？",
        expected_tools=["find_substitute_sku"],
        must_contain=["SKU-IPHONE-CASE-001"],
        must_not_hallucinate=["百分之百兼容", "绝对没问题"],
        ground_truth_keywords=["SKU-PHONE-CASE-PREMIUM-003", "SKU-PHONE-CASE-BASIC-002"],
        tags=["substitute", "stockout"],
        expected_facts={
            "sku_id": "SKU-IPHONE-CASE-001",
            "substitute_skus": ["SKU-PHONE-CASE-PREMIUM-003", "SKU-PHONE-CASE-BASIC-002"],
        },
        expected_decision={"answer_type": "substitute_options"},
        evidence_requirements={"substitute_options": "find_substitute_sku"},
        expected_answer_points=["给出替代 SKU", "说明兼容度或需要客户确认"],
        expected_entities=["SKU-IPHONE-CASE-001", "SKU-PHONE-CASE-PREMIUM-003", "SKU-PHONE-CASE-BASIC-002"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool"],
    ),
    GoldenCase(
        id="gc-sub-002",
        question="SKU-ROUTER-WIFI7-001 如果缺货，可以推荐什么替代品？",
        expected_tools=["find_substitute_sku"],
        must_contain=["SKU-ROUTER-WIFI7-001"],
        must_not_hallucinate=["SKU-IPHONE-CASE-001"],
        ground_truth_keywords=["SKU-ROUTER-WIFI6-007"],
        tags=["substitute", "router"],
        expected_facts={
            "sku_id": "SKU-ROUTER-WIFI7-001",
            "substitute_skus": ["SKU-ROUTER-WIFI6-007"],
        },
        expected_decision={"answer_type": "substitute_options"},
        evidence_requirements={"substitute_options": "find_substitute_sku"},
        expected_answer_points=["推荐 WiFi 6 替代品", "说明替代有性能差异或需确认"],
        expected_entities=["SKU-ROUTER-WIFI7-001", "SKU-ROUTER-WIFI6-007"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool"],
    ),
    GoldenCase(
        id="gc-sub-003",
        question="SKU-UNKNOWN-404 有替代 SKU 吗？",
        expected_tools=["find_substitute_sku"],
        must_contain=["SKU-UNKNOWN-404"],
        must_not_hallucinate=["SKU-PHONE-CASE-PREMIUM-003", "百分之百兼容"],
        ground_truth_keywords=["暂无", "替代"],
        tags=["substitute", "edge-case"],
        expected_facts={"sku_id": "SKU-UNKNOWN-404", "substitute_skus": []},
        expected_decision={"answer_type": "no_substitute"},
        evidence_requirements={"substitute_options": "find_substitute_sku"},
        expected_answer_points=["明确暂无替代方案", "建议人工确认或补充商品主数据"],
        expected_entities=["SKU-UNKNOWN-404"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool"],
    ),

    # ── E. RAG 规则问答 ────────────────────────────────────────────────────
    GoldenCase(
        id="gc-rag-001",
        question="订单缺货时，优先级高的客户应该怎么处理？",
        expected_tools=["retrieve_knowledge"],
        must_contain=[],
        must_not_hallucinate=["我猜", "可能是", "随意取消"],
        ground_truth_keywords=["优先级", "缺货", "处理", "客户"],
        tags=["knowledge", "policy", "rag"],
        expected_facts={"policy_topic": "priority_stockout"},
        expected_decision={"answer_type": "policy_answer"},
        evidence_requirements={"policy_rule": "retrieve_knowledge"},
        expected_answer_points=["说明高优先级客户优先处理", "给出可执行处理动作"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool", "missing_rag_evidence"],
        judge_rubric=["是否忠实于知识库规则", "是否避免凭经验编 SOP"],
    ),
    GoldenCase(
        id="gc-rag-002",
        question="跨仓调拨一般要遵守哪些规则？",
        expected_tools=["retrieve_knowledge"],
        must_contain=[],
        must_not_hallucinate=["随意调配", "不需要审批"],
        ground_truth_keywords=["跨仓", "调拨", "规则"],
        tags=["knowledge", "transfer", "rag"],
        expected_facts={"policy_topic": "warehouse_transfer"},
        expected_decision={"answer_type": "policy_answer"},
        evidence_requirements={"policy_rule": "retrieve_knowledge"},
        expected_answer_points=["说明跨仓调拨限制", "说明审批或库存确认要求"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool", "missing_rag_evidence", "claims_no_rules"],
    ),
    GoldenCase(
        id="gc-rag-003",
        question="缺货订单能直接换成替代 SKU 吗？",
        expected_tools=["retrieve_knowledge"],
        must_contain=[],
        must_not_hallucinate=["可以直接替换", "无需客户确认"],
        ground_truth_keywords=["替代", "确认", "规则"],
        tags=["knowledge", "substitute", "rag"],
        expected_facts={"policy_topic": "substitute_policy"},
        expected_decision={"answer_type": "policy_answer"},
        evidence_requirements={"policy_rule": "retrieve_knowledge"},
        expected_answer_points=["说明替代 SKU 需要规则或客户确认", "说明不能擅自替换"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool", "missing_rag_evidence"],
    ),
    GoldenCase(
        id="gc-rag-004",
        question="区域仓发货优先级怎么判断？",
        expected_tools=["retrieve_knowledge"],
        must_contain=[],
        must_not_hallucinate=["随机选择仓库", "无规则"],
        ground_truth_keywords=["区域", "优先", "仓"],
        tags=["knowledge", "regional", "rag"],
        expected_facts={"policy_topic": "regional_fulfillment"},
        expected_decision={"answer_type": "policy_answer"},
        evidence_requirements={"policy_rule": "retrieve_knowledge"},
        expected_answer_points=["说明区域就近或优先级原则", "说明库存不足时的后续处理"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool", "missing_rag_evidence"],
    ),
    GoldenCase(
        id="gc-rag-005",
        question="售后补发遇到缺货时应该怎么处理？",
        expected_tools=["retrieve_knowledge"],
        must_contain=[],
        must_not_hallucinate=["直接拒绝售后", "无需记录"],
        ground_truth_keywords=["售后", "补发", "缺货"],
        tags=["knowledge", "after-sales", "rag"],
        expected_facts={"policy_topic": "after_sales_stockout"},
        expected_decision={"answer_type": "policy_answer"},
        evidence_requirements={"policy_rule": "retrieve_knowledge"},
        expected_answer_points=["说明售后补发处理动作", "说明记录或通知要求"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool", "missing_rag_evidence"],
    ),

    # ── F. 履约方案与多工具综合 ─────────────────────────────────────────────
    GoldenCase(
        id="gc-plan-001",
        question="请为订单 SO202502140001 生成完整的履约方案。",
        expected_tools=["generate_fulfillment_plan", "analyze_order", "check_inventory"],
        must_contain=["SO202502140001"],
        must_not_hallucinate=["方案A", "方案B", "100% 确定"],
        ground_truth_keywords=["履约", "缺货", "发货", "SKU-CHARGER-020W-002"],
        tags=["fulfillment", "plan", "multi-tool"],
        expected_facts={
            "order_id": "SO202502140001",
            "fulfillment_ready": False,
            "insufficient_skus": ["SKU-CHARGER-020W-002"],
        },
        expected_decision={"answer_type": "fulfillment_plan", "can_ship_full": False},
        evidence_requirements={"plan": "generate_fulfillment_plan", "inventory_status": "check_inventory"},
        expected_answer_points=["给出分步骤履约建议", "说明缺货 SKU 的处理方式", "提示不确定项或后续确认"],
        expected_entities=["SO202502140001", "SKU-CHARGER-020W-002"],
        hard_fail_conditions=STOCKOUT_HARD_FAILS,
        judge_rubric=["方案是否可执行", "是否基于工具结果而非凭空编排"],
        difficulty="hard",
        min_score=78.0,
    ),
    GoldenCase(
        id="gc-plan-002",
        question="SO202502140002 库存够的情况下，下一步履约动作是什么？",
        expected_tools=["generate_fulfillment_plan", "check_inventory"],
        must_contain=["SO202502140002"],
        must_not_hallucinate=["缺货", "替代 SKU"],
        ground_truth_keywords=["正常发货", "履约"],
        tags=["fulfillment", "plan", "ready"],
        expected_facts={"order_id": "SO202502140002", "fulfillment_ready": True},
        expected_decision={"answer_type": "fulfillment_plan", "can_ship_full": True},
        evidence_requirements={"plan": "generate_fulfillment_plan"},
        expected_answer_points=["说明可进入正常履约", "不要引入缺货补救动作"],
        expected_entities=["SO202502140002"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool"],
    ),
    GoldenCase(
        id="gc-multi-001",
        question="帮我分析订单 SO202502140001 的完整履约可行性，包括库存、仓库分布和备选方案。",
        expected_tools=["analyze_order", "check_inventory", "search_warehouse_inventory", "find_substitute_sku"],
        must_contain=["SO202502140001"],
        must_not_hallucinate=["100% 确定", "绝对可以"],
        ground_truth_keywords=["库存", "仓库", "替代", "可行"],
        tags=["multi-tool", "comprehensive", "hard"],
        expected_facts={
            "order_id": "SO202502140001",
            "fulfillment_ready": False,
            "insufficient_skus": ["SKU-CHARGER-020W-002"],
        },
        expected_decision={"answer_type": "comprehensive_analysis", "can_ship_full": False},
        evidence_requirements={
            "order_facts": "analyze_order",
            "inventory_status": "check_inventory",
            "warehouse_distribution": "search_warehouse_inventory",
            "substitute_options": "find_substitute_sku",
        },
        expected_answer_points=["说明当前不能全量履约", "覆盖仓库分布", "覆盖替代或补救方案", "给出综合建议"],
        expected_entities=["SO202502140001", "SKU-CHARGER-020W-002"],
        hard_fail_conditions=STOCKOUT_HARD_FAILS,
        judge_rubric=["多工具信息是否被正确整合", "结论和建议是否有优先级"],
        difficulty="hard",
        min_score=78.0,
    ),
    GoldenCase(
        id="gc-multi-002",
        question="SO202502140003 如果 Type-C 数据线不够，能不能用替代品或跨仓解决？",
        expected_tools=["check_inventory", "search_warehouse_inventory", "find_substitute_sku"],
        must_contain=["SO202502140003"],
        must_not_hallucinate=["库存完全充足", "无需处理"],
        ground_truth_keywords=["SKU-CABLE-TYPEC-003", "SKU-CABLE-TYPEC-FAST-008", "仓"],
        tags=["multi-tool", "substitute", "warehouse"],
        expected_facts={
            "order_id": "SO202502140003",
            "insufficient_skus": ["SKU-CABLE-TYPEC-003"],
            "substitute_skus": ["SKU-CABLE-TYPEC-FAST-008"],
        },
        expected_decision={"answer_type": "stockout_recovery", "can_ship_full": False},
        evidence_requirements={
            "inventory_status": "check_inventory",
            "warehouse_distribution": "search_warehouse_inventory",
            "substitute_options": "find_substitute_sku",
        },
        expected_answer_points=["指出 Type-C 数据线不足", "说明可查仓库库存", "说明替代 SKU 需确认"],
        expected_entities=["SO202502140003", "SKU-CABLE-TYPEC-003", "SKU-CABLE-TYPEC-FAST-008"],
        hard_fail_conditions=STOCKOUT_HARD_FAILS,
        difficulty="hard",
    ),
    GoldenCase(
        id="gc-multi-003",
        question="高优先级订单 SO202502140001 缺货时应该怎么安排履约？",
        expected_tools=["analyze_order", "check_inventory", "retrieve_knowledge"],
        must_contain=["SO202502140001"],
        must_not_hallucinate=["直接取消", "低优先级"],
        ground_truth_keywords=["高", "缺货", "优先"],
        tags=["multi-tool", "priority", "rag"],
        expected_facts={
            "order_id": "SO202502140001",
            "priority": "高",
            "fulfillment_ready": False,
        },
        expected_decision={"answer_type": "priority_stockout_plan", "can_ship_full": False},
        evidence_requirements={
            "order_facts": "analyze_order",
            "inventory_status": "check_inventory",
            "policy_rule": "retrieve_knowledge",
        },
        expected_answer_points=["识别高优先级", "说明缺货处理动作", "给出优先保障或人工确认建议"],
        expected_entities=["SO202502140001"],
        hard_fail_conditions=STOCKOUT_HARD_FAILS + ["missing_rag_evidence"],
        difficulty="hard",
    ),
    GoldenCase(
        id="gc-multi-004",
        question="SO202502140002 和 SO202502140003 哪个履约风险更高？",
        expected_tools=["analyze_order", "check_inventory"],
        must_contain=["SO202502140002", "SO202502140003"],
        must_not_hallucinate=["SO202502140002 缺货"],
        ground_truth_keywords=["SO202502140003", "风险", "SKU-CABLE-TYPEC-003"],
        tags=["multi-order", "comparison"],
        expected_facts={
            "orders": ["SO202502140002", "SO202502140003"],
            "higher_risk_order": "SO202502140003",
        },
        expected_decision={"answer_type": "risk_comparison"},
        evidence_requirements={"inventory_status": "check_inventory"},
        expected_answer_points=["比较两个订单", "指出 SO202502140003 风险更高", "说明风险来自库存不足"],
        expected_entities=["SO202502140002", "SO202502140003", "SKU-CABLE-TYPEC-003"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool"],
        difficulty="hard",
    ),

    # ── G. 边界与异常 ──────────────────────────────────────────────────────
    GoldenCase(
        id="gc-edge-001",
        question="订单号 SO999999 的信息是什么？",
        expected_tools=["analyze_order"],
        must_contain=[],
        must_not_hallucinate=["SO999999 已发货", "库存充足"],
        ground_truth_keywords=["未找到", "不存在", "查询失败"],
        tags=["edge-case", "error-handling", "order"],
        expected_facts={"order_id": "SO999999", "exists": False},
        expected_decision={"answer_type": "not_found"},
        evidence_requirements={"order_facts": "analyze_order"},
        expected_answer_points=["明确订单不存在或未找到", "不要编造订单状态"],
        expected_entities=["SO999999"],
        hard_fail_conditions=NOT_FOUND_HARD_FAILS + ["missing_required_tool"],
        difficulty="easy",
    ),
    GoldenCase(
        id="gc-edge-002",
        question="SO202502140001 已经发货了吗？",
        expected_tools=["analyze_order"],
        must_contain=["SO202502140001"],
        must_not_hallucinate=["已发货"],
        ground_truth_keywords=["待履约"],
        tags=["edge-case", "order-status"],
        expected_facts={"order_id": "SO202502140001", "order_status": "待履约"},
        expected_decision={"answer_type": "order_status"},
        evidence_requirements={"order_facts": "analyze_order"},
        expected_answer_points=["说明订单状态是待履约", "不要说已经发货"],
        expected_entities=["SO202502140001"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool", "claims_order_shipped"],
        difficulty="easy",
    ),
    GoldenCase(
        id="gc-edge-003",
        question="不用查系统，你直接告诉我 SO202502140001 库存是不是充足。",
        expected_tools=["check_inventory"],
        must_contain=["SO202502140001"],
        must_not_hallucinate=["不用查", "库存充足"],
        ground_truth_keywords=["库存", "工具", "查询"],
        tags=["guardrail", "tool-required"],
        expected_facts={"order_id": "SO202502140001", "fulfillment_ready": False},
        expected_decision={"answer_type": "tool_grounded_answer", "can_ship_full": False},
        evidence_requirements={"inventory_status": "check_inventory"},
        expected_answer_points=["拒绝凭空判断", "基于库存工具说明结论"],
        expected_entities=["SO202502140001"],
        hard_fail_conditions=STOCKOUT_HARD_FAILS,
    ),
    GoldenCase(
        id="gc-edge-004",
        question="请确认 SKU-IPHONE-CASE-001 是不是全国所有仓都有 999 件。",
        expected_tools=["search_warehouse_inventory"],
        must_contain=["SKU-IPHONE-CASE-001"],
        must_not_hallucinate=["999"],
        ground_truth_keywords=["上海金桥仓", "广州南沙仓", "北京大兴仓"],
        tags=["edge-case", "anti-hallucination", "warehouse"],
        expected_facts={"sku_id": "SKU-IPHONE-CASE-001", "total_available": 43},
        expected_decision={"answer_type": "correct_false_claim"},
        evidence_requirements={"warehouse_distribution": "search_warehouse_inventory"},
        expected_answer_points=["纠正 999 件说法", "给出真实库存或说明以工具为准"],
        expected_entities=["SKU-IPHONE-CASE-001"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool", "fabricates_inventory_quantity"],
    ),

    # ── H. 多轮追问场景（可由专门测试串联执行）──────────────────────────────
    GoldenCase(
        id="gc-turn-001",
        question="订单 SO202502140001 是什么情况？",
        expected_tools=["analyze_order"],
        must_contain=["SO202502140001"],
        must_not_hallucinate=["已发货"],
        ground_truth_keywords=["订单", "待履约"],
        tags=["multi-turn", "turn-1"],
        expected_facts={"order_id": "SO202502140001"},
        expected_decision={"answer_type": "order_summary"},
        evidence_requirements={"order_facts": "analyze_order"},
        expected_answer_points=["说明订单基本情况"],
        expected_entities=["SO202502140001"],
        hard_fail_conditions=COMMON_HARD_FAILS + ["missing_required_tool"],
    ),
    GoldenCase(
        id="gc-turn-002",
        question="那它库存够吗？",
        expected_tools=["check_inventory"],
        must_contain=[],
        must_not_hallucinate=["库存充足"],
        ground_truth_keywords=["库存", "不足"],
        tags=["multi-turn", "turn-2", "requires-context"],
        expected_facts={"order_id": "SO202502140001", "fulfillment_ready": False},
        expected_decision={"answer_type": "stockout", "can_ship_full": False},
        evidence_requirements={"inventory_status": "check_inventory"},
        expected_answer_points=["沿用上一轮订单上下文", "说明库存不足"],
        expected_entities=["SO202502140001", "SKU-CHARGER-020W-002"],
        hard_fail_conditions=STOCKOUT_HARD_FAILS,
    ),
    GoldenCase(
        id="gc-turn-003",
        question="如果不够，下一步应该怎么处理？",
        expected_tools=["retrieve_knowledge", "find_substitute_sku", "search_warehouse_inventory"],
        must_contain=[],
        must_not_hallucinate=["直接取消", "随意替换"],
        ground_truth_keywords=["调拨", "替代", "确认"],
        tags=["multi-turn", "turn-3", "requires-context"],
        expected_facts={"order_id": "SO202502140001", "fulfillment_ready": False},
        expected_decision={"answer_type": "stockout_next_action", "can_ship_full": False},
        evidence_requirements={"policy_rule": "retrieve_knowledge"},
        expected_answer_points=["给出缺货处理路径", "说明调拨/替代/客户确认等动作"],
        expected_entities=["SO202502140001"],
        hard_fail_conditions=STOCKOUT_HARD_FAILS + ["missing_rag_evidence"],
        difficulty="hard",
    ),
]

BENCHMARK_TARGET_SIZE = 300

BENCHMARK_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "slug": "stockout_switch_warehouse",
        "tag": "缺货/换仓",
        "tools": ["analyze_order", "check_inventory", "get_inventory_warehouse_detail", "retrieve_knowledge"],
        "allowed": ["create_warehouse_request", "create_customer_service_task"],
        "forbidden": ["direct_order_mutation", "direct_inventory_mutation"],
        "sop": ["缺货订单先检查同区域仓，再评估跨仓或客服确认。"],
    },
    {
        "slug": "split_order",
        "tag": "拆单",
        "tools": ["analyze_order", "check_inventory", "get_inventory_warehouse_detail", "get_shipping_detail"],
        "allowed": ["create_warehouse_request", "create_logistics_request"],
        "forbidden": ["direct_split_order", "direct_waybill_mutation"],
        "sop": ["拆单必须确认商品限制、客户影响和物流时效。"],
    },
    {
        "slug": "inventory_transfer",
        "tag": "库存/调拨",
        "tools": ["check_inventory", "get_inventory_warehouse_detail", "retrieve_knowledge"],
        "allowed": ["create_warehouse_request"],
        "forbidden": ["direct_inventory_transfer"],
        "sop": ["调拨申请需保留来源仓、目标仓、SKU 和数量依据。"],
    },
    {
        "slug": "logistics_exception",
        "tag": "物流异常",
        "tools": ["analyze_order", "get_shipping_detail", "retrieve_knowledge"],
        "allowed": ["create_logistics_request", "create_customer_service_task"],
        "forbidden": ["direct_carrier_core_data_mutation"],
        "sop": ["物流异常需重新读取 TMS 状态并保留客户承诺。"],
    },
    {
        "slug": "replenishment_supply_chain",
        "tag": "补货/供应链",
        "tools": ["check_inventory", "get_supply_chain_detail", "retrieve_knowledge"],
        "allowed": ["create_supply_chain_request", "create_customer_service_task"],
        "forbidden": ["direct_purchase_order_mutation"],
        "sop": ["补货等待方案必须核对 ERP 在途与预计入库时间。"],
    },
    {
        "slug": "product_constraints",
        "tag": "商品履约限制",
        "tools": ["analyze_order", "get_product_constraints", "get_shipping_detail", "retrieve_knowledge"],
        "allowed": ["create_logistics_request", "create_customer_service_task"],
        "forbidden": ["ignore_pim_constraints", "direct_order_mutation"],
        "sop": ["冷链、大件、易碎和不可拆单商品必须优先服从 PIM 限制。"],
    },
    {
        "slug": "customer_service_complex",
        "tag": "客服协同/综合异常",
        "tools": ["analyze_order", "check_inventory", "get_customer_case_context", "retrieve_knowledge"],
        "allowed": ["create_customer_service_task", "create_warehouse_request"],
        "forbidden": ["unrecorded_customer_promise"],
        "sop": ["客服承诺、客诉和 VIP 风险必须进入 Planner 输入与人工审核。"],
    },
)

BENCHMARK_WORKFLOW_PATHS: tuple[dict[str, Any], ...] = (
    {
        "path": "normal_once_success",
        "external_status": "COMPLETED",
        "success": True,
        "notes": "一次规划、审批、下发后验证通过。",
    },
    {
        "path": "external_task_failed_replan",
        "external_status": "FAILED",
        "success": False,
        "notes": "外部任务失败后进入 REPLANNING。",
    },
    {
        "path": "stale_webhook_ignored",
        "external_status": "COMPLETED",
        "success": False,
        "notes": "旧 case_version/plan_version 回调只记录 Trace，不推进当前工作流。",
    },
    {
        "path": "data_conflict_reload",
        "external_status": "BLOCKED",
        "success": False,
        "notes": "业务上下文缺失或冲突时进入 DATA_CONFLICT 并重新读取源系统。",
    },
    {
        "path": "checkpoint_resume",
        "external_status": "COMPLETED",
        "success": True,
        "notes": "WAITING 后释放执行资源，通过 checkpoint 恢复继续 VERIFYING。",
    },
    {
        "path": "three_replans_manual",
        "external_status": "FAILED",
        "success": False,
        "notes": "连续三次 replan 仍失败后进入 MANUAL handoff。",
    },
)


def _build_benchmark_case(index: int) -> GoldenCase:
    scenario = BENCHMARK_SCENARIOS[index % len(BENCHMARK_SCENARIOS)]
    path = BENCHMARK_WORKFLOW_PATHS[index % len(BENCHMARK_WORKFLOW_PATHS)]
    order_id = f"BM20260830{index + 1:04d}"
    sku_id = f"SKU-BM-{scenario['slug'].upper().replace('-', '_')}-{index % 17:03d}"
    expected_tools = list(dict.fromkeys(scenario["tools"]))
    accepted_split = scenario["slug"] in {"split_order", "stockout_switch_warehouse"}
    return GoldenCase(
        id=f"bm-{index + 1:03d}-{scenario['slug']}-{path['path']}",
        question=f"{order_id} 出现{scenario['tag']}，请按闭环流程给出可审批方案。",
        expected_tools=expected_tools,
        must_contain=[order_id],
        must_not_hallucinate=["已直接修改订单", "已直接扣减库存", "无需人工审核"],
        ground_truth_keywords=[scenario["tag"], "SOP", "HITL", "VERIFYING"],
        tags=[
            "benchmark-sample",
            "end-to-end",
            scenario["slug"],
            path["path"],
        ],
        expected_facts={
            "order_id": order_id,
            "sku_id": sku_id,
            "case_id": f"case-{order_id}",
            "fresh_business_state_required": True,
            "session_memory_business_facts_allowed": False,
        },
        expected_decision={
            "answer_type": "fulfillment_closed_loop_plan",
            "requires_hitl": True,
            "agent_write_boundary": "task_or_request_only",
            "decision_priority": ["current_business_data", "current_sop", "historical_case"],
        },
        evidence_requirements={
            "fresh_order_context": "OMS/WMS/TMS/ERP/PIM/CRM adapters",
            "sop_evidence": "sop_collection",
            "case_recall": "case_collection",
            "model_route": "model_gateway",
            "tool_governance": "tool_gateway",
        },
        expected_answer_points=[
            "重新读取实时业务上下文",
            "分别引用 SOP Evidence 和 Similar Cases",
            "生成带 Action DAG 与 Success Criteria 的方案",
            "说明 HITL 通过后只创建协同任务/申请",
            "外部回调后再次 VERIFYING",
        ],
        expected_entities=[order_id, sku_id],
        hard_fail_conditions=COMMON_HARD_FAILS
        + ["direct_business_mutation", "uses_stale_business_data", "missing_hitl"],
        judge_rubric=[
            "是否体现实时业务数据 > 当前 SOP > 历史案例的优先级",
            "是否禁止直接修改订单、库存、物流核心数据",
            "是否覆盖 WAITING、Webhook、VERIFYING 或 Replan 路径",
        ],
        difficulty="hard" if path["path"] in {"three_replans_manual", "data_conflict_reload"} else "medium",
        min_score=80.0,
        mock_business_state={
            "OMS": {
                "order_id": order_id,
                "order_status": "paid_waiting_fulfillment",
                "priority": "vip" if index % 5 == 0 else "normal",
            },
            "WMS": {
                "sku_id": sku_id,
                "available_stock": index % 4,
                "required_quantity": 5 + index % 3,
                "candidate_warehouses": [f"WH-BM-{index % 6:02d}", f"WH-BM-{(index + 1) % 6:02d}"],
            },
            "TMS": {"eta_hours": 24 + index % 48, "carrier_status": "available"},
            "ERP": {"inbound_stock": index % 9, "expected_inbound_days": 1 + index % 5},
            "PIM": {"split_allowed": accepted_split, "cold_chain": scenario["slug"] == "product_constraints"},
            "CRM": {"customer_tier": "VIP" if index % 5 == 0 else "standard", "open_ticket": index % 3 == 0},
        },
        conversation=[
            f"这个 {scenario['tag']} 先看怎么处理。",
            "还是优先保证时效。" if index % 2 == 0 else "可以接受拆单，但要先确认客户影响。",
        ],
        applicable_sop=list(scenario["sop"]),
        allowed_actions=list(scenario["allowed"]),
        forbidden_actions=list(scenario["forbidden"]),
        external_task_result={
            "status": path["external_status"],
            "event_id": f"evt-bm-{index + 1:04d}",
            "case_version": 1 if path["path"] != "stale_webhook_ignored" else 0,
            "plan_version": 1 if path["path"] != "stale_webhook_ignored" else 0,
            "business_state_verified": bool(path["success"]),
            "notes": path["notes"],
        },
        success_criteria={
            "goal_type": scenario["slug"],
            "criteria": [
                "external_task_business_evidence_present",
                "fresh_context_reloaded_before_verify",
                "no_direct_business_mutation",
            ],
            "expected_success": bool(path["success"]),
            "max_replan_count": 3,
        },
        workflow_path=path["path"],
    )


def build_fulfillops_benchmark_dataset(target_size: int | None = None) -> list[GoldenCase]:
    """Build deterministic benchmark cases on demand.

    The product target is 300 benchmark scenarios, but the repository keeps only a
    compact representative set by default. Larger runs can still request an
    explicit ``target_size`` from CI or an offline evaluation job.
    """

    target_size = target_size or max(len(BENCHMARK_SCENARIOS), len(BENCHMARK_WORKFLOW_PATHS))
    return [_build_benchmark_case(index) for index in range(target_size)]


FULFILLOPS_BENCHMARK_DATASET: list[GoldenCase] = build_fulfillops_benchmark_dataset()


def get_cases_by_tag(tag: str) -> list[GoldenCase]:
    """按标签筛选测试用例，供分组运行使用。"""
    return [case for case in GOLDEN_DATASET if tag in case.tags]


def get_benchmark_cases_by_tag(tag: str) -> list[GoldenCase]:
    """按标签筛选内置端到端 Benchmark 样例。"""
    return [case for case in FULFILLOPS_BENCHMARK_DATASET if tag in case.tags]
