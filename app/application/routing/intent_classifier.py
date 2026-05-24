"""意图分类器（学习版注释）— 根据用户问题判断路由目标。

设计理念：
  这是大模型意图识别失败时的规则兜底。
  HybridService 主路径会优先用小模型输出 workflow / agent / rag / multi_agent，
  如果模型不可用或返回格式异常，再回到这里做低成本、可解释的兜底判断。
"""

from dataclasses import dataclass
from enum import Enum


class IntentLevel(Enum):
    """意图复杂度等级。

    Hybrid 路由会根据这个等级选择 Workflow、Agent 或 Multi-Agent。
    """

    SIMPLE = "simple"  # 简单查询 → Workflow
    RAG = "rag"  # 规则/SOP/政策知识问答 → RAG
    MEDIUM = "medium"  # 中等分析 → Agent
    COMPLEX = "complex"  # 复杂决策 → Agent
    MULTI_DOMAIN = "multi_domain"  # 跨领域复杂问题 → Multi-Agent


@dataclass
class IntentClassificationResult:
    """意图分类结果。"""

    # 复杂度等级。
    level: IntentLevel
    # 0.0-1.0，越高表示越确信。
    score: float  # 0.0-1.0，置信度
    # 命中的主要关键词，方便解释路由原因。
    primary_keywords: list[str]  # 匹配到的关键词
    # 人类可读的分类理由。
    reasoning: str  # 分类理由


class IntentClassifier:
    """基于规则的意图分类器。

    这个分类器不用 LLM，优点是快、便宜、可解释；
    缺点是表达能力有限，所以只用于路由前的轻量判断。
    """

    def __init__(self):
        # 简单查询关键词：更适合固定 Workflow。
        self.simple_keywords = {
            "查询": 1.0,
            "是什么": 1.0,
            "怎样": 0.8,
            "多少": 0.8,
            "几个": 0.8,
            "什么": 0.7,
            "哪个": 0.6,
        }

        # 知识规则类问题：更适合 RAG。
        self.rag_keywords = {
            "规则": 1.0,
            "政策": 1.0,
            "制度": 0.9,
            "sop": 0.9,
            "知识库": 0.9,
            "依据": 0.8,
            "售后": 0.8,
            "退货": 0.8,
            "补发": 0.8,
            "高优先级": 0.7,
            "缺货规则": 1.0,
            "跨仓规则": 1.0,
        }

        # 中等问题关键词：可能需要 Agent 自主调用工具分析。
        self.medium_keywords = {
            "可以": 0.9,
            "能否": 0.9,
            "够吗": 0.9,
            "充足": 0.8,
            "足够": 0.8,
            "可否": 0.8,
            "如何": 0.7,
            "怎么办": 0.7,
            "替代": 0.6,
        }

        # 复杂问题关键词：通常需要多步骤分析或生成方案。
        self.complex_keywords = {
            "方案": 1.0,
            "建议": 0.95,
            "生成": 0.9,
            "完整": 0.85,
            "具体": 0.8,
            "详细": 0.75,
            "评估": 0.7,
            "分析": 0.6,
        }

        # 增强信号：出现这些词通常意味着要调用多个工具。
        self.multi_tool_signals = {
            "仓库": 0.5,
            "替代品": 0.6,
            "成本": 0.5,
            "时间": 0.4,
        }

        # 跨领域关键词 — 同时跨库存、风险、履约等多个领域
        # 注意：multi_domain_score 使用原始总分而非归一化（除以 len），
        # 因为每个关键词权重本身就是有意设计的强信号。
        self.multi_domain_keywords = {
            "履约方案": 1.0,
            "风险评估": 1.0,
            "全链路": 0.95,
            "综合分析": 0.9,
            "完整方案": 0.9,
            "端到端": 0.85,
            "全面": 0.7,
            "综合": 0.7,
            "跨仓": 0.65,
            "调配": 0.6,
            "协同": 0.5,
        }

        # 领域标记词 — 用于判断问题跨了几个领域
        self.domain_markers = {
            "库存": {"库存", "缺货", "仓库", "仓位", "现货", "存货", "备货", "跨仓", "调拨", "库容"},
            "履约": {"履约", "发货", "配送", "物流", "快递", "运输", "替代品", "替代", "调配", "方案", "分仓"},
            "风险": {"风险", "规则", "政策", "异常", "合规", "预警", "成本", "时效"},
            "订单": {"订单", "客户", "优先级", "加急", "VIP", "等级"},
        }

    def classify(self, user_input: str) -> IntentClassificationResult:
        """分类用户输入的意图复杂度。

        Args:
            user_input: 用户消息文本

        Returns:
            分类结果，包括等级、置信度、匹配关键词
        """
        if not user_input:
            # 空输入按简单问题处理，避免路由器抛异常。
            return IntentClassificationResult(
                level=IntentLevel.SIMPLE,
                score=0.5,
                primary_keywords=[],
                reasoning="Empty input",
            )

        # 中文不受 lower 影响，但英文关键词可以统一小写匹配。
        text = user_input.lower()

        # 计算各等级的匹配分数。
        simple_score = self._calculate_score(text, self.simple_keywords)
        rag_score = self._calculate_score(text, self.rag_keywords)
        medium_score = self._calculate_score(text, self.medium_keywords)
        complex_score = self._calculate_score(text, self.complex_keywords)
        multi_tool_score = self._calculate_score(text, self.multi_tool_signals)
        # 多域关键词用原始总分（不归一化），因为每个关键词权重本身就是强信号
        multi_domain_score = self._calculate_raw_score(text, self.multi_domain_keywords)

        # 领域覆盖度：问题跨了几个领域。
        domain_count = self._count_domains(text)

        # 综合评分：复杂度不仅看关键词，也看是否需要多工具。
        final_complex = complex_score + multi_tool_score * 0.3
        final_medium = medium_score
        final_simple = simple_score
        final_rag = rag_score
        final_multi = multi_domain_score + multi_tool_score * 0.2 + domain_count * 0.3

        # 规则 1: 跨 3 个以上领域 + 有复杂关键词 → 必定 MULTI_DOMAIN
        if domain_count >= 3 and (complex_score > 0 or multi_domain_score > 0):
            # 这种问题通常需要 Supervisor 调多个专业 Agent 协作。
            level = IntentLevel.MULTI_DOMAIN
            score = min(0.85 + multi_domain_score * 0.1, 1.0)
            primary_keywords = self._get_matched_keywords(text, level)
            return IntentClassificationResult(
                level=level,
                score=score,
                primary_keywords=primary_keywords,
                reasoning=f"MULTI_DOMAIN: 跨 {domain_count} 个领域（{self._get_domain_names(text)}）",
            )

        # 规则 2: 跨 2 个领域 + 多域关键词命中 → MULTI_DOMAIN
        if domain_count >= 2 and multi_domain_score > 0.3:
            # 例如同时问库存、风险、履约方案。
            level = IntentLevel.MULTI_DOMAIN
            score = min(0.7 + multi_domain_score * 0.15, 1.0)
            primary_keywords = self._get_matched_keywords(text, level)
            return IntentClassificationResult(
                level=level,
                score=score,
                primary_keywords=primary_keywords,
                reasoning=f"MULTI_DOMAIN: 跨 {domain_count} 个领域，命中多域关键词",
            )

        # 确定最终等级（非多域路径）。
        scores = {
            IntentLevel.SIMPLE: final_simple,
            IntentLevel.RAG: final_rag,
            IntentLevel.MEDIUM: final_medium,
            IntentLevel.COMPLEX: final_complex,
        }

        level = max(scores, key=scores.get)
        score = scores[level]

        # 规则 3: 如果同时出现简单和复杂关键词，倾向于复杂
        if simple_score > 0 and complex_score > 0:
            # 例如“查询并生成方案”，虽然有查询，但真正需求是方案。
            level = IntentLevel.COMPLEX

        # 规则 3.5: 规则/政策/SOP 问答优先走 RAG，除非明确要求生成方案或综合分析。
        if rag_score > 0 and complex_score == 0 and multi_domain_score == 0:
            level = IntentLevel.RAG
            score = max(score, min(0.75 + rag_score * 0.2, 1.0))

        # 规则 3.6: “库存够不够 / 是否缺货 / 能不能发”属于固定履约判断，优先走 Workflow。
        fixed_workflow_question = (
            any(word in text for word in ["库存", "缺货", "发货", "履约"])
            and any(word in text for word in ["够吗", "充足", "足够", "能否", "可以发", "能不能发"])
        )
        if fixed_workflow_question and level != IntentLevel.RAG and complex_score == 0:
            level = IntentLevel.SIMPLE
            score = max(score, 0.8)

        # 规则 4: 如果提到"替代品"和"仓库"，必定是复杂
        if "替代" in text and ("仓库" in text or "仓" in text):
            # 替代 + 仓库通常要同时查库存、仓库、替代 SKU。
            level = IntentLevel.COMPLEX
            score = min(score + 0.2, 1.0)

        # 规则 5: "方案"、"履约"、"生成" 总是复杂
        if any(word in text for word in ["方案", "履约", "生成完整"]):
            # 方案类问题通常不是单纯查字段。
            level = IntentLevel.COMPLEX
            score = 0.95

        # 收集匹配的关键词，生成解释。
        primary_keywords = self._get_matched_keywords(text, level)
        reasoning = self._generate_reasoning(level, primary_keywords)

        return IntentClassificationResult(
            level=level,
            score=min(score, 1.0),
            primary_keywords=primary_keywords,
            reasoning=reasoning,
        )

    def _calculate_score(self, text: str, keywords: dict) -> float:
        """计算文本与关键词集合的匹配分数（归一化到 0-1）。"""
        if not keywords:
            return 0.0

        total_score = 0.0
        for keyword, weight in keywords.items():
            if keyword in text:
                # 命中关键词就累加对应权重。
                total_score += weight

        # 除以关键词数量做归一化，避免关键词表越大分数越容易爆。
        return min(total_score / len(keywords), 1.0)

    def _calculate_raw_score(self, text: str, keywords: dict) -> float:
        """计算文本与关键词集合的原始匹配分数（不归一化）。

        用于多域关键词等场景，其中每个关键词权重本身就是有意设计的强信号。
        """
        if not keywords:
            return 0.0

        total_score = 0.0
        for keyword, weight in keywords.items():
            if keyword in text:
                total_score += weight

        # 多域关键词保留原始强度，但加硬上限防止极端文本。
        return min(total_score, 2.0)  # 硬上限防止极端情况

    def _count_domains(self, text: str) -> int:
        """计算问题覆盖的领域数。"""
        count = 0
        for _domain_name, markers in self.domain_markers.items():
            if any(m in text for m in markers):
                # 一个领域命中多个词也只算一个领域。
                count += 1
        return count

    def _get_domain_names(self, text: str) -> str:
        """返回问题覆盖的领域名称列表。"""
        names = []
        for domain_name, markers in self.domain_markers.items():
            if any(m in text for m in markers):
                names.append(domain_name)
        return "、".join(names) if names else "无"

    def _get_matched_keywords(self, text: str, level: IntentLevel) -> list[str]:
        """获取匹配到的关键词。"""
        if level == IntentLevel.SIMPLE:
            keyword_dict = self.simple_keywords
        elif level == IntentLevel.RAG:
            keyword_dict = self.rag_keywords
        elif level == IntentLevel.MEDIUM:
            keyword_dict = self.medium_keywords
        elif level == IntentLevel.MULTI_DOMAIN:
            keyword_dict = self.multi_domain_keywords
        else:
            keyword_dict = self.complex_keywords

        matched = [kw for kw in keyword_dict if kw in text]
        # 只返回前 3 个，避免 reasoning 太长。
        return matched[:3]  # 只返回前 3 个

    def _generate_reasoning(self, level: IntentLevel, keywords: list[str]) -> str:
        """生成分类理由。"""
        if not keywords:
            return f"Default {level.value} classification"

        keywords_str = "、".join(keywords)
        return f"{level.value.upper()} intent detected based on keywords: {keywords_str}"


# 单例
_classifier = None


def get_classifier() -> IntentClassifier:
    """获取全局分类器实例。"""
    global _classifier
    if _classifier is None:
        # 单例避免每次请求重复构造关键词表。
        _classifier = IntentClassifier()
    return _classifier
