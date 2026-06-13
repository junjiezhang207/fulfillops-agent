"""Agent 质量回归测试。

测试分层：
  Layer 1 — Golden Dataset 规则评分
    - 工具调用覆盖度
    - 工具证据是否支撑答案事实
    - 业务结论是否符合结构化期望
    - 硬失败条件：错误履约结论、编造实体、编造库存数量等

  Layer 2 — DeepEval LLM-as-judge
    - AnswerRelevancyMetric: 答案与问题相关性 ≥ 0.7
    - GEval: 自定义“答案是否清晰、完整、可执行”

运行方式：
  pytest tests/eval/test_agent_quality.py -m "not slow" -q

  pytest tests/eval/test_agent_quality.py --slow -q

  pytest tests/eval/test_agent_quality.py -k "gc-inv" -q
"""

import pytest

from app.agents.quality.evaluation.golden_dataset import GOLDEN_DATASET, GoldenCase
from app.agents.quality.evaluation.scoring import evaluate_agent_result


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

async def _run_agent(agent_service, case: GoldenCase) -> dict:
    """运行 Agent 并返回结果，超时视为测试失败。"""
    return await agent_service.chat(
        session_id=f"eval-{case.id}",
        message=case.question,
        include_trace=True,
    )


# ── Layer 1：Golden Dataset 规则评分 ────────────────────────────────────────

class TestGoldenRegressionScore:
    """使用结构化 Golden Case 评估 Agent 输出。

    这里不使用 LLM-as-judge，所有判断都来自工具调用、trace 证据和结构化期望。
    """

    @pytest.mark.parametrize("case", GOLDEN_DATASET, ids=[c.id for c in GOLDEN_DATASET])
    @pytest.mark.asyncio
    async def test_agent_result_passes_rule_score(self, case: GoldenCase, agent_service_with_llm):
        result = await _run_agent(agent_service_with_llm, case)
        score = evaluate_agent_result(case, result)

        assert result.get("reply", "").strip(), f"[{case.id}] Agent 返回空回复"
        assert score.passed, score.assert_message()


# ── Layer 2：DeepEval LLM-as-judge（慢速测试）──────────────────────────────

@pytest.mark.slow
class TestDeepEvalQuality:
    """DeepEval 指标测试，需要 LLM 进行评判。

    在 CI 中用 -m "not slow" 跳过；开发者本地评测时加 --slow 运行。
    """

    @pytest.mark.parametrize("case", GOLDEN_DATASET, ids=[c.id for c in GOLDEN_DATASET])
    @pytest.mark.asyncio
    async def test_answer_relevancy(self, case: GoldenCase, agent_service_with_llm):
        """答案与问题相关性 ≥ 0.7（DeepEval AnswerRelevancyMetric）。"""
        try:
            from deepeval import assert_test
            from deepeval.metrics import AnswerRelevancyMetric
            from deepeval.test_case import LLMTestCase
        except ImportError:
            pytest.skip("deepeval 未安装。运行 `uv add deepeval` 安装。")

        result = await _run_agent(agent_service_with_llm, case)
        test_case = LLMTestCase(
            input=case.question,
            actual_output=result.get("reply", ""),
        )
        metric = AnswerRelevancyMetric(threshold=0.7, verbose_mode=False)
        assert_test(test_case, [metric])

    @pytest.mark.parametrize("case", GOLDEN_DATASET, ids=[c.id for c in GOLDEN_DATASET])
    @pytest.mark.asyncio
    async def test_tool_grounding(self, case: GoldenCase, agent_service_with_llm):
        """答案应清晰、完整、可执行（GEval 自定义软指标）。"""
        try:
            from deepeval import assert_test
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCase, SingleTurnParams
        except ImportError:
            pytest.skip("deepeval 未安装。")

        result = await _run_agent(agent_service_with_llm, case)
        tools_called = result.get("tools_called", [])

        test_case = LLMTestCase(
            input=case.question,
            actual_output=result.get("reply", ""),
            context=[
                f"工具调用：{', '.join(tools_called)}" if tools_called else "未调用工具",
                "评估要求：" + "；".join(case.judge_rubric or case.expected_answer_points),
            ],
        )
        metric = GEval(
            name="AnswerUsefulness",
            criteria=(
                "判断回答是否清晰、完整、可执行。事实正确性已经由规则评分器检查，"
                "这里只评估表达是否直接回应问题、是否说明原因、是否给出下一步动作。"
            ),
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                                SingleTurnParams.CONTEXT],
            threshold=0.7,
            verbose_mode=False,
        )
        assert_test(test_case, [metric])
