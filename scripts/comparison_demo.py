#!/usr/bin/env python3
"""对比演示脚本：Agent vs 固定工作流。

运行方式：
  cd e:/fulfillops-agent
  python -m scripts.comparison_demo
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.comparison.executor import ComparisonExecutor
from app.comparison.framework import ComparisonScenario


def create_test_scenarios() -> list[ComparisonScenario]:
    """创建 5 个测试场景，从简单到复杂。"""
    return [
        ComparisonScenario(
            name="Scenario 1: Simple Order Query",
            description="用户只是想查询订单的基本信息",
            user_input="订单 SO202502140001 是什么？",
            expected_answer_hints=["SO202502140001", "抖音", "磁吸手机壳"],
            complexity_level="simple",
        ),
        ComparisonScenario(
            name="Scenario 2: Basic Inventory Check",
            description="用户询问库存是否充足",
            user_input="订单 SO202502140001 的库存充足吗？",
            expected_answer_hints=["充足", "库存", "发货"],
            complexity_level="simple",
        ),
        ComparisonScenario(
            name="Scenario 3: Multi-step Analysis",
            description="用户需要多步推理：查看订单、检查库存、考虑替代方案",
            user_input="订单 SO202502140002 的库存够吗？如果不够有替代方案吗？",
            expected_answer_hints=["库存", "替代", "建议"],
            complexity_level="medium",
        ),
        ComparisonScenario(
            name="Scenario 4: Complex Decision",
            description="用户需要完整的履约方案，涉及多个信息源",
            user_input="请为订单 SO202502140003 生成一个完整的履约方案，"
                      "包括仓库选择、可能的替代品和时间预估。",
            expected_answer_hints=["方案", "履约", "仓", "时间"],
            complexity_level="complex",
        ),
        ComparisonScenario(
            name="Scenario 5: Exception Handling",
            description="用户询问一个不存在的订单",
            user_input="订单 SO999999999 的情况怎么样？",
            expected_answer_hints=["不存在", "找不到", "错误"],
            complexity_level="simple",
        ),
    ]


def main():
    """运行对比演示。"""
    print("=" * 80)
    print("Agent vs Fixed Workflow - Comparison Demo")
    print("=" * 80)
    print()

    # 初始化执行器
    print("[*] Initializing executor...")
    try:
        executor = ComparisonExecutor()
        print("[OK] Initialization successful")
    except Exception as exc:
        print(f"[FAIL] Initialization failed: {exc}")
        print("  Hint: Make sure LLM is configured in .env")
        return

    # 创建测试场景
    print("\n[*] Creating test scenarios...")
    scenarios = create_test_scenarios()
    print(f"[OK] Created {len(scenarios)} test scenarios")
    for scenario in scenarios:
        print(f"  - {scenario.name} ({scenario.complexity_level})")

    # 运行对比
    print("\n[*] Running comparison tests...")
    print("    (This may take a few minutes due to LLM calls)")
    print()

    try:
        framework = executor.run_comparison(scenarios)
    except Exception as exc:
        print(f"\n[FAIL] Comparison failed: {exc}")
        import traceback
        traceback.print_exc()
        return

    # 生成报告
    print("\n" + "=" * 80)
    report = framework.generate_report()
    print(report)

    # 保存报告
    report_path = "docs/stage-04-core-module-round-06-agent-tools-memory/COMPARISON-REPORT.md"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[OK] Report saved to: {report_path}")
    except Exception as exc:
        print(f"\n[WARN] Could not save report: {exc}")

    # 返回摘要数据（便于进一步分析）
    return framework.get_summary()


if __name__ == "__main__":
    main()
