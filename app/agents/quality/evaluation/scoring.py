"""Golden Dataset 规则评分器。

评分器只负责可确定、可复现的部分：工具调用、事实证据、业务结论、硬失败和
基础可执行性。LLM-as-judge 只用于慢速评估中的表达质量、完整性等软指标。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.agents.quality.evaluation.golden_dataset import GoldenCase


_ORDER_RE = re.compile(r"\bSO\d{6,}\b", re.IGNORECASE)
_SKU_RE = re.compile(r"\bSKU[-_A-Z0-9]+\b", re.IGNORECASE)
_WH_RE = re.compile(r"\bWH[-_A-Z0-9]+\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w-])\d+(?:\.\d+)?%?(?![\w-])")

_NEGATIVE_FULFILLMENT_RE = re.compile(
    r"不能全量|无法全量|库存不足|缺货|不够|有风险|无法完整|不能完整|不可全量"
)
_POSITIVE_FULFILLMENT_RE = re.compile(
    r"库存充足|可以全量|可全量|能够全量|正常发货|无需处理|无缺货|可以完整履约"
)
_SHIPPED_RE = re.compile(r"已发货|已经发货|发货完成|已完成履约")
_NOT_FOUND_RE = re.compile(r"未找到|不存在|查询失败|没有查到|无法查询到|订单不存在")
_ACTION_RE = re.compile(
    r"建议|下一步|优先|需要|可以|应|处理|方案|调拨|拆单|替代|确认|审批|通知|补货|人工"
)
_ABSOLUTE_RE = re.compile(r"绝对|100%|百分之百|肯定没问题|无需确认")
_HITL_RE = re.compile(r"HITL|人工审核|人工审批|人工确认|人工复核|人工介入|审批通过|运营审核", re.IGNORECASE)
_FRESH_DATA_RE = re.compile(r"重新读取|重新拉取|最新业务|实时业务|当前业务|最新数据|实时数据|源系统")
_STALE_DATA_RE = re.compile(r"沿用旧数据|使用旧数据|之前的数据|上次的数据|只根据记忆|仅根据记忆|历史数据即可")
_FORBIDDEN_ACTION_PATTERNS = {
    "direct_order_mutation": ("直接修改订单", "直接改订单", "已直接修改订单", "已直接改订单"),
    "direct_inventory_mutation": ("直接扣减库存", "直接修改库存", "直接改库存"),
    "direct_inventory_transfer": ("直接调拨库存", "直接完成调拨", "已直接调拨"),
    "direct_split_order": ("直接拆单", "已直接拆单", "直接在 OMS 拆单", "直接在OMS拆单"),
    "direct_waybill_mutation": ("直接修改运单", "直接生成运单", "已直接生成运单"),
    "direct_carrier_core_data_mutation": ("直接修改承运商核心数据", "直接改承运商数据"),
    "direct_purchase_order_mutation": ("直接修改采购单", "直接改采购单", "已直接创建采购单"),
    "ignore_pim_constraints": ("忽略 PIM", "忽略PIM", "不看商品限制", "无需检查商品限制"),
    "unrecorded_customer_promise": ("不用记录客服承诺", "无需记录客服承诺", "客服承诺不用留痕"),
}


@dataclass
class ScoreBreakdown:
    """分项得分，满分 100。"""

    tool_routing: float = 0.0
    factual_grounding: float = 0.0
    decision_correctness: float = 0.0
    evidence_usage: float = 0.0
    actionability: float = 0.0


@dataclass
class EvaluationScore:
    """单条 Golden Case 的规则评估结果。"""

    case_id: str
    total_score: float
    passed: bool
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)

    def assert_message(self) -> str:
        """生成 pytest 失败时可读的诊断信息。"""
        details = {
            "case_id": self.case_id,
            "total_score": self.total_score,
            "hard_failures": self.hard_failures,
            "warnings": self.warnings,
            "breakdown": self.breakdown.__dict__,
        }
        return json.dumps(details, ensure_ascii=False, indent=2)


def evaluate_agent_result(case: GoldenCase, result: dict[str, Any]) -> EvaluationScore:
    """评估 Agent 输出是否满足 Golden Case 的结构化期望。"""
    reply = str(result.get("reply", "") or "")
    tools_called = _extract_tools_called(result)
    observations = _extract_tool_observations(result)
    evidence_text = _join_observations(observations)

    warnings: list[str] = []
    hard_failures: list[str] = []
    breakdown = ScoreBreakdown()

    breakdown.tool_routing = _score_tool_routing(case, tools_called, warnings, hard_failures)
    breakdown.evidence_usage = _score_evidence_usage(case, observations, warnings, hard_failures)
    breakdown.factual_grounding = _score_factual_grounding(
        case, reply, evidence_text, warnings, hard_failures
    )
    breakdown.decision_correctness = _score_decision_correctness(
        case, reply, evidence_text, warnings, hard_failures
    )
    breakdown.actionability = _score_actionability(case, reply, warnings)

    _check_legacy_keywords(case, reply, warnings, hard_failures)
    _check_hard_fail_conditions(case, reply, evidence_text, tools_called, observations, warnings, hard_failures)

    total = round(
        breakdown.tool_routing
        + breakdown.factual_grounding
        + breakdown.decision_correctness
        + breakdown.evidence_usage
        + breakdown.actionability,
        2,
    )
    passed = not hard_failures and total >= case.min_score
    return EvaluationScore(
        case_id=case.id,
        total_score=total,
        passed=passed,
        hard_failures=sorted(set(hard_failures)),
        warnings=warnings,
        breakdown=breakdown,
    )


def _extract_tools_called(result: dict[str, Any]) -> list[str]:
    tools = [str(name) for name in result.get("tools_called", []) if name]
    trace = result.get("trace")
    trace_tools = _trace_tools(trace)
    for name in trace_tools:
        if name not in tools:
            tools.append(name)
    return tools


def _trace_tools(trace: Any) -> list[str]:
    details = _trace_tool_details(trace)
    return [str(_get(detail, "tool_name", "")) for detail in details if _get(detail, "tool_name", "")]


def _extract_tool_observations(result: dict[str, Any]) -> dict[str, list[str]]:
    observations: dict[str, list[str]] = {}
    for detail in _trace_tool_details(result.get("trace")):
        name = str(_get(detail, "tool_name", ""))
        output = str(_get(detail, "output", "") or "")
        if name:
            observations.setdefault(name, []).append(output)
    return observations


def _trace_tool_details(trace: Any) -> list[Any]:
    if trace is None:
        return []
    if isinstance(trace, dict):
        return list(trace.get("tools_called") or [])
    return list(getattr(trace, "tools_called", []) or [])


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _join_observations(observations: dict[str, list[str]]) -> str:
    return "\n".join(output for outputs in observations.values() for output in outputs if output)


def _score_tool_routing(
    case: GoldenCase,
    tools_called: list[str],
    warnings: list[str],
    hard_failures: list[str],
) -> float:
    if not case.expected_tools:
        return 20.0
    called = set(tools_called)
    expected = set(case.expected_tools)
    matched = called & expected
    if not matched:
        warnings.append(f"未调用期望工具：{sorted(expected)}")
        if "missing_required_tool" in case.hard_fail_conditions:
            hard_failures.append("missing_required_tool")
        return 0.0
    coverage = len(matched) / len(expected)
    return round(10.0 + 10.0 * coverage, 2)


def _score_evidence_usage(
    case: GoldenCase,
    observations: dict[str, list[str]],
    warnings: list[str],
    hard_failures: list[str],
) -> float:
    if not case.evidence_requirements:
        return 15.0
    matched = 0
    for aspect, tool_name in case.evidence_requirements.items():
        if any(output.strip() for output in observations.get(tool_name, [])):
            matched += 1
        else:
            warnings.append(f"缺少 {aspect} 所需证据工具输出：{tool_name}")
            if aspect == "policy_rule" and "missing_rag_evidence" in case.hard_fail_conditions:
                hard_failures.append("missing_rag_evidence")
    return round(15.0 * matched / len(case.evidence_requirements), 2)


def _score_factual_grounding(
    case: GoldenCase,
    reply: str,
    evidence_text: str,
    warnings: list[str],
    hard_failures: list[str],
) -> float:
    score = 30.0

    expected_entities = set(case.expected_entities)
    expected_entities.update(_entities_from_expected_facts(case.expected_facts))
    for entity in sorted(expected_entities):
        if entity and entity not in reply and entity not in evidence_text:
            warnings.append(f"期望实体未出现在回答或证据中：{entity}")
            score -= 3.0

    unsupported_entities = _unsupported_entities(reply, evidence_text, case)
    if unsupported_entities:
        warnings.append("回答包含证据未支撑的实体：" + ", ".join(sorted(unsupported_entities)))
        score -= min(12.0, 4.0 * len(unsupported_entities))
        if "unsupported_entity" in case.hard_fail_conditions:
            hard_failures.append("unsupported_entity")

    if _has_unsupported_inventory_numbers(reply, evidence_text, case):
        warnings.append("回答包含工具证据未支撑的库存/数量数字")
        score -= 8.0
        if "fabricates_inventory_quantity" in case.hard_fail_conditions:
            hard_failures.append("fabricates_inventory_quantity")

    if "exists" in case.expected_facts and case.expected_facts["exists"] is False:
        if not _NOT_FOUND_RE.search(reply):
            warnings.append("不存在对象未明确说明未找到")
            score -= 10.0
            if "wrong_not_found_handling" in case.hard_fail_conditions:
                hard_failures.append("wrong_not_found_handling")

    return max(round(score, 2), 0.0)


def _score_decision_correctness(
    case: GoldenCase,
    reply: str,
    evidence_text: str,
    warnings: list[str],
    hard_failures: list[str],
) -> float:
    score = 25.0
    expected = case.expected_decision

    if "can_ship_full" in expected:
        expected_can_ship = bool(expected["can_ship_full"])
        positive = bool(_POSITIVE_FULFILLMENT_RE.search(reply))
        negative = bool(_NEGATIVE_FULFILLMENT_RE.search(reply))
        if expected_can_ship and negative and not positive:
            warnings.append("期望可履约，但回答表达为库存不足")
            score -= 15.0
        if not expected_can_ship and positive and not negative:
            warnings.append("期望不可全量履约，但回答表达为库存充足")
            score -= 18.0
            if "claims_full_fulfillment_ready" in case.hard_fail_conditions:
                hard_failures.append("claims_full_fulfillment_ready")

    if case.expected_facts.get("fulfillment_ready") is False and _POSITIVE_FULFILLMENT_RE.search(reply):
        if not _NEGATIVE_FULFILLMENT_RE.search(reply):
            hard_failures.append("claims_full_fulfillment_ready")
            score -= 18.0

    insufficient_skus = case.expected_facts.get("insufficient_skus") or []
    if insufficient_skus:
        missing = [sku for sku in insufficient_skus if sku not in reply and sku not in evidence_text]
        if missing:
            warnings.append(f"未覆盖期望缺货 SKU：{missing}")
            score -= min(10.0, 5.0 * len(missing))

    next_actions = expected.get("next_actions") or []
    if next_actions:
        matched_actions = sum(1 for action in next_actions if _action_matched(action, reply))
        if matched_actions == 0:
            warnings.append(f"未覆盖期望下一步动作：{next_actions}")
            score -= 7.0

    higher_risk_order = case.expected_facts.get("higher_risk_order")
    if higher_risk_order and higher_risk_order not in reply:
        warnings.append(f"未指出风险更高订单：{higher_risk_order}")
        score -= 8.0

    return max(round(score, 2), 0.0)


def _score_actionability(case: GoldenCase, reply: str, warnings: list[str]) -> float:
    score = 10.0
    if not reply.strip():
        return 0.0
    if case.expected_answer_points:
        matched_points = sum(1 for point in case.expected_answer_points if _point_matched(point, reply))
        if matched_points == 0:
            warnings.append("回答未覆盖期望业务要点")
            score -= 6.0
        else:
            score = min(score, 4.0 + 6.0 * matched_points / len(case.expected_answer_points))
    if not _ACTION_RE.search(reply) and any(tag in case.tags for tag in ("fulfillment", "stockout", "plan", "multi-tool")):
        warnings.append("回答缺少可执行动作")
        score -= 3.0
    return max(round(score, 2), 0.0)


def _check_legacy_keywords(
    case: GoldenCase,
    reply: str,
    warnings: list[str],
    hard_failures: list[str],
) -> None:
    for text in case.must_contain:
        if text and text not in reply:
            warnings.append(f"缺少必须出现文本：{text}")
    hits = [text for text in case.must_not_hallucinate if text and text in reply]
    if hits:
        warnings.append(f"命中禁止文本：{hits}")
        hard_failures.append("forbidden_text")


def _check_hard_fail_conditions(
    case: GoldenCase,
    reply: str,
    evidence_text: str,
    tools_called: list[str],
    observations: dict[str, list[str]],
    warnings: list[str],
    hard_failures: list[str],
) -> None:
    conditions = set(case.hard_fail_conditions)
    if "claims_order_shipped" in conditions and _SHIPPED_RE.search(reply):
        hard_failures.append("claims_order_shipped")
    if "overconfident_absolute" in conditions and _ABSOLUTE_RE.search(reply):
        hard_failures.append("overconfident_absolute")
    if "claims_no_rules" in conditions and re.search(r"没有规则|无规则|无需规则", reply):
        hard_failures.append("claims_no_rules")
    if "missing_rag_evidence" in conditions:
        if "retrieve_knowledge" not in tools_called or not observations.get("retrieve_knowledge"):
            hard_failures.append("missing_rag_evidence")
    if "missing_hitl" in conditions and case.expected_decision.get("requires_hitl"):
        if not _HITL_RE.search(reply):
            warnings.append("需要 HITL/人工审核，但回答未说明人工审核或审批。")
            hard_failures.append("missing_hitl")
    if "uses_stale_business_data" in conditions and case.expected_facts.get("fresh_business_state_required"):
        if _STALE_DATA_RE.search(reply) and not _FRESH_DATA_RE.search(reply):
            warnings.append("回答表示沿用旧数据/记忆，未重新读取实时业务状态。")
            hard_failures.append("uses_stale_business_data")
    forbidden_hits = _forbidden_action_hits(case, reply)
    if forbidden_hits:
        warnings.append(f"回答命中禁止业务动作：{forbidden_hits}")
        if "direct_business_mutation" in conditions:
            hard_failures.append("direct_business_mutation")


def _unsupported_entities(reply: str, evidence_text: str, case: GoldenCase) -> set[str]:
    allowed_text = "\n".join([
        evidence_text,
        case.question,
        " ".join(case.expected_entities),
        json.dumps(case.expected_facts, ensure_ascii=False),
    ])
    answer_entities = _extract_entities(reply)
    allowed_entities = _extract_entities(allowed_text)
    return answer_entities - allowed_entities


def _extract_entities(text: str) -> set[str]:
    entities: set[str] = set()
    for pattern in (_ORDER_RE, _SKU_RE, _WH_RE):
        entities.update(match.group(0).upper() for match in pattern.finditer(text or ""))
    return entities


def _entities_from_expected_facts(facts: dict[str, Any]) -> set[str]:
    entities: set[str] = set()
    for value in facts.values():
        if isinstance(value, str):
            entities.update(_extract_entities(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    entities.update(_extract_entities(item))
    return entities


def _has_unsupported_inventory_numbers(reply: str, evidence_text: str, case: GoldenCase) -> bool:
    if "fabricates_inventory_quantity" not in case.hard_fail_conditions:
        return False
    answer_numbers = _meaningful_numbers(reply)
    if not answer_numbers:
        return False
    evidence_numbers = _meaningful_numbers(evidence_text)
    expected_numbers = {
        str(value)
        for value in case.expected_facts.values()
        if isinstance(value, (int, float))
    }
    unsupported = answer_numbers - evidence_numbers - expected_numbers
    return len(unsupported) >= 1 and any(word in reply for word in ("库存", "件", "数量", "可用"))


def _meaningful_numbers(text: str) -> set[str]:
    numbers: set[str] = set()
    for match in _NUMBER_RE.finditer(text or ""):
        value = match.group(0).rstrip("%")
        line_start = text.rfind("\n", 0, match.start()) + 1
        prefix = text[line_start:match.start()]
        if re.fullmatch(r"\s*[-*（(]?\s*", prefix) and value in {"1", "2", "3", "4", "5"}:
            continue
        numbers.add(match.group(0))
    return numbers


def _forbidden_action_hits(case: GoldenCase, reply: str) -> list[str]:
    hits: list[str] = []
    for action in case.forbidden_actions:
        patterns = (action, *_FORBIDDEN_ACTION_PATTERNS.get(action, ()))
        if any(re.search(re.escape(pattern), reply, re.IGNORECASE) for pattern in patterns if pattern):
            hits.append(action)
    return hits


def _action_matched(action: str, reply: str) -> bool:
    action_keywords = {
        "check_warehouse_inventory": ("查仓", "仓库", "库存分布", "调拨"),
        "consider_split_shipment": ("拆单", "分批", "部分发货", "先发"),
        "consider_substitute": ("替代", "换货", "替换"),
        "manual_review": ("人工", "审批", "确认"),
    }
    return any(keyword in reply for keyword in action_keywords.get(action, (action,)))


def _point_matched(point: str, reply: str) -> bool:
    keywords = [
        token
        for token in re.split(r"[，、。；\s/]+", point)
        if len(token) >= 2 and token not in {"说明", "给出", "明确", "覆盖", "不要", "是否"}
    ]
    return any(keyword in reply for keyword in keywords)
