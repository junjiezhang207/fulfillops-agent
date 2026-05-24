"""Ragas 批量离线评测脚本。

用途：
  - 从历史对话中抽样，批量评估 RAG 检索质量
  - 在每次发布前运行，检测 faithfulness / answer_relevancy 是否下降
  - 结果写入 storage/eval_results/ 留存，便于纵向对比

运行方式：
  uv run python scripts/eval_ragas.py
  uv run python scripts/eval_ragas.py --limit 20  # 只跑前 20 条

输出示例：
  === Ragas 批量评测结果 ===
  样本数        : 10
  faithfulness  : 0.873
  answer_relevancy: 0.912
  详细结果已写入: storage/eval_results/ragas_2026-05-03T14:30:00.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── 内置评测样本（来自 golden_dataset，模拟历史对话数据）───────────────────

def _build_eval_records() -> list[dict]:
    """构造评测记录。

    生产中替换为：从 Langfuse 导出历史 trace → 解析 question/answer/contexts。
    当前阶段使用 golden_dataset 中的问题 + mock 答案演示流程。
    """
    from app.agents.quality.evaluation.golden_dataset import GOLDEN_DATASET

    records = []
    for case in GOLDEN_DATASET:
        # 使用 mock 答案（生产中替换为 Langfuse 导出的真实 answer）
        mock_answer = (
            f"根据系统数据分析：{case.question} "
            f"涉及以下关键实体：{'、'.join(case.must_contain or ['订单'])}。"
            f"建议操作：{', '.join(case.ground_truth_keywords[:2]) if case.ground_truth_keywords else '请查看详细报告'}。"
        )
        # contexts：RAG 检索到的文本片段（生产中从 Langfuse trace 解析）
        mock_contexts = [
            f"知识库条目：{kw} 相关规则" for kw in case.ground_truth_keywords[:3]
        ] or ["暂无相关规则"]

        records.append({
            "case_id": case.id,
            "question": case.question,
            "answer": mock_answer,
            "contexts": mock_contexts,
            "tags": case.tags,
        })
    return records


# ── 主评测流程 ────────────────────────────────────────────────────────────────

def run_eval(limit: int | None = None, output_dir: str = "storage/eval_results") -> dict:
    from app.agents.quality.evaluation.ragas_evaluator import RagasEvaluator
    from app.core.config import get_settings
    from app.infrastructure.llm.chat_adapter import LLMFactory

    settings = get_settings()
    llm = LLMFactory.create_chat_model(settings, use_case="judge")  # 评测优先使用 judge 模型

    records = _build_eval_records()
    if limit:
        records = records[:limit]

    print(f"开始 Ragas 评测，样本数: {len(records)} ...")
    ev = RagasEvaluator(llm=llm)
    result = ev.evaluate_batch(records)

    if "error" in result:
        print(f"评测失败：{result['error']}")
        return result

    # ── 打印摘要 ──────────────────────────────────────────────────────────────
    print("\n=== Ragas 批量评测结果 ===")
    print(f"样本数          : {len(records)}")
    print(f"faithfulness    : {result.get('mean_faithfulness', 'N/A')}")
    print(f"answer_relevancy: {result.get('mean_answer_relevancy', 'N/A')}")

    # ── 写入结果文件 ──────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_file = out_dir / f"ragas_{timestamp}.json"

    output = {
        "timestamp": timestamp,
        "sample_count": len(records),
        "mean_faithfulness": result.get("mean_faithfulness"),
        "mean_answer_relevancy": result.get("mean_answer_relevancy"),
        "details": [
            {
                "case_id": records[i]["case_id"],
                "faithfulness": s.faithfulness,
                "answer_relevancy": s.answer_relevancy,
            }
            for i, s in enumerate(result.get("details", []))
        ],
    }
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"详细结果已写入: {out_file}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ragas 批量离线评测")
    parser.add_argument("--limit", type=int, default=None, help="最多评测 N 条样本")
    parser.add_argument(
        "--output-dir", default="storage/eval_results", help="结果输出目录"
    )
    args = parser.parse_args()

    result = run_eval(limit=args.limit, output_dir=args.output_dir)
    sys.exit(0 if "error" not in result else 1)
