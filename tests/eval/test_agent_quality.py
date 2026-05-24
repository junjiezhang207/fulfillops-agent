"""Agent 质量回归测试 — 基于 DeepEval + Golden Dataset。

测试分层：
  Layer 1 — 无 LLM 断言（CI 必跑）
    - 工具调用断言：Agent 至少调用了 expected_tools 中的一个工具
    - 幻觉关键词检查：答案不包含 must_not_hallucinate 中的词
    - 非空断言：Agent 返回了有效回复

  Layer 2 — DeepEval LLM-as-judge（加 --slow 参数时运行）
    - AnswerRelevancyMetric: 答案与问题相关性 ≥ 0.7
    - GEval: 自定义"是否基于工具数据而非凭空推断"

运行方式：
  # 仅 Layer 1（CI）：
  pytest tests/eval/test_agent_quality.py -m "not slow"

  # 完整评测（本地，需要 LLM）：
  pytest tests/eval/test_agent_quality.py --slow

  # 只跑某个标签的 case：
  pytest tests/eval/test_agent_quality.py -k "gc-00"
"""

import pytest

from app.agent.evaluation.golden_dataset import GOLDEN_DATASET, GoldenCase

pytestmark = pytest.mark.slow


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

async def _run_agent(agent_service, case: GoldenCase) -> dict:
    """运行 Agent 并返回结果，超时视为测试失败。"""
    return await agent_service.chat(
        session_id=f"eval-{case.id}",
        message=case.question,
        include_trace=True,
    )


# ── Layer 1：无 LLM 断言（CI 常规跑）────────────────────────────────────────

class TestToolCallAssertion:
    """验证 Agent 是否调用了正确的工具。

    不依赖 LLM，快速验证 Agent routing 没有退化。
    """

    @pytest.mark.parametrize("case", GOLDEN_DATASET, ids=[c.id for c in GOLDEN_DATASET])
    @pytest.mark.asyncio
    async def test_expected_tool_called(self, case: GoldenCase, agent_service_with_llm):
        result = await _run_agent(agent_service_with_llm, case)
        called = set(result.get("tools_called", []))

        assert called, (
            f"[{case.id}] Agent 未调用任何工具。\n"
            f"问题：{case.question}\n"
            f"回复：{result.get('reply', '')[:200]}"
        )
        assert any(t in called for t in case.expected_tools), (
            f"[{case.id}] 期望工具 {case.expected_tools} 之一被调用，实际调用了 {called}。\n"
            f"问题：{case.question}"
        )

    @pytest.mark.parametrize("case", GOLDEN_DATASET, ids=[c.id for c in GOLDEN_DATASET])
    @pytest.mark.asyncio
    async def test_reply_not_empty(self, case: GoldenCase, agent_service_with_llm):
        result = await _run_agent(agent_service_with_llm, case)
        reply = result.get("reply", "").strip()

        assert reply, (
            f"[{case.id}] Agent 返回了空回复。\n"
            f"问题：{case.question}"
        )
        assert len(reply) >= 20, (
            f"[{case.id}] 回复过短（{len(reply)} 字符），疑似未正常作答。\n"
            f"回复：{reply}"
        )


class TestHallucinationKeyword:
    """黑名单关键词检查 — 答案中不应出现幻觉词。

    检查逻辑：逐字符匹配 must_not_hallucinate 中的词，
    任意命中则测试失败并报告具体词语。
    """

    @pytest.mark.parametrize(
        "case",
        [c for c in GOLDEN_DATASET if c.must_not_hallucinate],
        ids=[c.id for c in GOLDEN_DATASET if c.must_not_hallucinate],
    )
    @pytest.mark.asyncio
    async def test_no_hallucination_keywords(self, case: GoldenCase, agent_service_with_llm):
        result = await _run_agent(agent_service_with_llm, case)
        reply = result.get("reply", "")

        hits = [kw for kw in case.must_not_hallucinate if kw in reply]
        assert not hits, (
            f"[{case.id}] 答案包含幻觉关键词 {hits}。\n"
            f"问题：{case.question}\n"
            f"回复：{reply[:300]}"
        )


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
        """答案必须基于工具数据，不能凭空推断（GEval 自定义标准）。"""
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
            context=[f"工具调用：{', '.join(tools_called)}"] if tools_called else ["未调用工具"],
        )
        metric = GEval(
            name="ToolGrounding",
            criteria=(
                "判断 Agent 的回答是否基于工具调用的实际数据，"
                "而非凭空推断或使用训练知识。"
                "如果回答中包含具体数字、订单号、SKU 或仓库名，且与工具调用结果一致，则得高分。"
            ),
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                                SingleTurnParams.CONTEXT],
            threshold=0.6,
            verbose_mode=False,
        )
        assert_test(test_case, [metric])


# ── 注册自定义 marker（避免 pytest 警告）────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: 需要 LLM 的慢速评测，CI 跳过，本地加 --slow 运行"
    )
