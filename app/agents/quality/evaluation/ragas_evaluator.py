"""Ragas RAG 质量评估封装。

本模块专门服务于 RAG 质量评测。它不负责检索文档，也不负责生成业务答案；
它拿到 question、answer、contexts 后，调用 Ragas 指标判断回答是否忠实于上下文、
是否真正回答问题、检索上下文是否有用。

主要做的事：
1. ``RagasScore``：保存单次 Ragas 评测结果。
2. ``RagasEvaluator.evaluate_single``：评估一条 question / answer / contexts。
3. ``evaluate_batch``：批量评估多条样例，适合离线回归。
4. 无可用 LLM / embeddings 时降级返回错误信息，不影响主流程。
5. 和 Golden Dataset、线上业务 Trace 抽样配合，观察 RAG 质量变化。

核心指标：
- ``faithfulness``：答案是否忠实于检索上下文，主要防幻觉。
- ``answer_relevancy``：答案是否切题，主要防答非所问。
- ``context_precision``：检索回来的 chunk 有多少是有用的，通常需要人工标注。

典型使用场景：
1. Prompt 修改前后对比 RAG 质量。
2. Reranker 或 Embedding 切换后做回归。
3. 从线上 trace 抽样，定期评估 faithfulness 和 relevancy。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Ragas 依赖 LLM / Embedding，本地或 CI 未配置模型时分数字段允许为空。
@dataclass
class RagasScore:
    faithfulness: float | None = None       # 0-1，越高越不幻觉
    answer_relevancy: float | None = None   # 0-1，越高越切题
    context_precision: float | None = None  # 0-1，需 ground_truth
    error: str = ""


# RAG 评估重点关注答案是否忠实于检索上下文，以及是否真正回答问题。
class RagasEvaluator:
    """封装 Ragas 评测逻辑，支持单条和批量评测。

    Args:
        llm: LangChain LLM 实例（用于 Ragas 的 LLM-as-judge 判断）。
             传 None 时尝试使用项目已配置的 LLM，仍为 None 则跳过 LLM 指标。
        embeddings: LangChain Embeddings 实例（用于 answer_relevancy 计算）。
                    传 None 时尝试使用默认 embeddings。
    """

    def __init__(self, llm=None, embeddings=None) -> None:
        self._llm = llm
        self._embeddings = embeddings

    # ── 单条评测 ─────────────────────────────────────────────────────────────

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RagasScore:
        """评测单条 RAG 输出。

        Args:
            question:     用户原始问题
            answer:       Agent 最终回答
            contexts:     RAG 检索到的文本片段列表（即工具 retrieve_knowledge 的原始 chunk）
            ground_truth: 标准答案（可选，有则额外计算 context_precision）

        Returns:
            RagasScore，各维度 0-1 分值
        """
        record = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }
        if ground_truth is not None:
            record["ground_truth"] = ground_truth

        result = self.evaluate_batch([record])
        if result.get("error"):
            return RagasScore(error=result["error"])
        details = result.get("details") or []
        return details[0] if details else RagasScore(error="empty ragas result")

    # ── 批量评测 ─────────────────────────────────────────────────────────────

    def evaluate_batch(self, records: list[dict]) -> dict:
        """批量评测，返回每条的分数及总体均值。

        Args:
            records: list of {
                "question": str,
                "answer": str,
                "contexts": list[str],
                "ground_truth": str (optional),
            }

        Returns:
            {
              "mean_faithfulness": float,
              "mean_answer_relevancy": float,
              "details": list[RagasScore],
            }
        """
        if not records:
            return {"mean_faithfulness": 0.0, "mean_answer_relevancy": 0.0, "details": []}

        try:
            from ragas import evaluate
            from ragas.metrics.collections import answer_relevancy, faithfulness
            from datasets import Dataset
        except ImportError:
            logger.warning("ragas/datasets 未安装，跳过批量评测。")
            return {"error": "ragas not installed", "details": []}

        try:
            has_ground_truth = all("ground_truth" in r for r in records)
            data: dict = {
                "question": [r["question"] for r in records],
                "answer": [r["answer"] for r in records],
                "contexts": [[c for c in r.get("contexts", []) if c] for r in records],
            }
            metrics = [faithfulness, answer_relevancy]

            if has_ground_truth:
                data["ground_truth"] = [r["ground_truth"] for r in records]
                from ragas.metrics import context_precision
                metrics.append(context_precision)

            dataset = Dataset.from_dict(data)
            kwargs: dict = {}
            if self._llm is not None:
                kwargs["llm"] = self._llm
            if self._embeddings is not None:
                kwargs["embeddings"] = self._embeddings

            result = evaluate(dataset, metrics=metrics, **kwargs)
            df = result.to_pandas()

            details = [
                RagasScore(
                    faithfulness=round(float(row.get("faithfulness", 0)), 3),
                    answer_relevancy=round(float(row.get("answer_relevancy", 0)), 3),
                    context_precision=round(float(row.get("context_precision", 0)), 3)
                    if has_ground_truth else None,
                )
                for _, row in df.iterrows()
            ]

            return {
                "mean_faithfulness": round(df["faithfulness"].mean(), 3),
                "mean_answer_relevancy": round(df["answer_relevancy"].mean(), 3),
                "details": details,
            }
        except Exception as exc:
            logger.error("Ragas 批量评测失败：%s", exc)
            return {"error": str(exc), "details": []}
