"""Agent vs 固定工作流对比框架。

设计目标：
  1. 同一个问题，用两条路径分别处理
  2. 记录关键指标：成功状态、响应时间、工具/步骤调用次数
  3. 生成对比报告，帮助决定何时用 Agent、何时用固定流程
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from time import time


class PathType(Enum):
    """执行路径类型。"""

    AGENT = "agent"
    WORKFLOW = "workflow"


@dataclass
class ExecutionMetrics:
    """单次执行的性能指标。"""

    path_type: PathType
    scenario_name: str
    user_input: str

    # 时间指标
    start_time: float = field(default_factory=time)
    end_time: float = 0.0
    execution_time_ms: float = 0.0

    # 调用指标
    tools_called: list[str] = field(default_factory=list)
    tool_call_count: int = 0

    # 结果指标
    final_reply: str = ""
    final_reply_length: int = 0

    # 其他
    error_message: str = ""
    success: bool = True

    def complete(self):
        """标记执行完成，计算执行时间。"""
        self.end_time = time()
        self.execution_time_ms = (self.end_time - self.start_time) * 1000
        self.tool_call_count = len(self.tools_called)
        self.final_reply_length = len(self.final_reply)


@dataclass
class ComparisonResult:
    """两条路径的对比结果。"""

    scenario_name: str
    user_input: str

    agent_metrics: ExecutionMetrics
    workflow_metrics: ExecutionMetrics

    # 对比分析
    time_advantage: str = ""  # "Agent faster" / "Workflow faster" / "Similar"
    complexity_advantage: str = ""  # 哪条路径的工具调用更简洁
    reliability_advantage: str = ""  # 哪条路径执行更稳定
    recommendation: str = ""  # 推荐用哪条路径

    def analyze(self):
        """自动分析两条路径的差异。"""
        agent_time = self.agent_metrics.execution_time_ms
        workflow_time = self.workflow_metrics.execution_time_ms
        time_diff = abs(agent_time - workflow_time)

        # 时间对比
        if agent_time < workflow_time * 0.9:
            self.time_advantage = f"Agent faster ({agent_time:.0f}ms vs {workflow_time:.0f}ms)"
        elif workflow_time < agent_time * 0.9:
            self.time_advantage = f"Workflow faster ({workflow_time:.0f}ms vs {agent_time:.0f}ms)"
        else:
            self.time_advantage = f"Similar ({agent_time:.0f}ms vs {workflow_time:.0f}ms)"

        # 复杂度对比
        agent_calls = self.agent_metrics.tool_call_count
        workflow_calls = self.workflow_metrics.tool_call_count
        if agent_calls < workflow_calls:
            self.complexity_advantage = f"Agent simpler ({agent_calls} tools vs {workflow_calls})"
        elif workflow_calls < agent_calls:
            self.complexity_advantage = f"Workflow simpler ({workflow_calls} tools vs {agent_calls})"
        else:
            self.complexity_advantage = f"Same complexity ({agent_calls} tools)"

        # 稳定性对比
        if self.agent_metrics.success and not self.workflow_metrics.success:
            self.reliability_advantage = "Agent succeeded, Workflow failed"
        elif self.workflow_metrics.success and not self.agent_metrics.success:
            self.reliability_advantage = "Workflow succeeded, Agent failed"
        elif self.agent_metrics.success and self.workflow_metrics.success:
            self.reliability_advantage = "Both succeeded"
        else:
            self.reliability_advantage = "Both failed"

        # 推荐
        self._generate_recommendation()

    def _generate_recommendation(self):
        """根据对比结果生成推荐。"""
        # 评分：Agent 得分越高越推荐用 Agent
        agent_score = 0
        workflow_score = 0

        # 时间：快的得 1 分
        if "Agent faster" in self.time_advantage:
            agent_score += 1
        elif "Workflow faster" in self.time_advantage:
            workflow_score += 1

        # 复杂度：简单的得 1 分
        if "Agent simpler" in self.complexity_advantage:
            agent_score += 1
        elif "Workflow simpler" in self.complexity_advantage:
            workflow_score += 1

        # 稳定性：成功路径优先
        if self.agent_metrics.success and not self.workflow_metrics.success:
            agent_score += 2
        elif self.workflow_metrics.success and not self.agent_metrics.success:
            workflow_score += 2

        if agent_score > workflow_score:
            self.recommendation = f"[YES] Use Agent (score: {agent_score}-{workflow_score})"
        elif workflow_score > agent_score:
            self.recommendation = f"[YES] Use Workflow (score: {workflow_score}-{agent_score})"
        else:
            self.recommendation = "[TIE] Both viable (score: tied)"


@dataclass
class ComparisonScenario:
    """测试场景定义。"""

    name: str
    description: str
    user_input: str
    expected_answer_hints: list[str]  # 保留兼容旧脚本，不参与自动打分
    complexity_level: str  # simple / medium / complex


class ComparisonFramework:
    """对比框架：协调 Agent 和固定工作流的执行、指标收集、报告生成。"""

    def __init__(self):
        self.scenarios: list[ComparisonScenario] = []
        self.results: list[ComparisonResult] = []

    def add_scenario(self, scenario: ComparisonScenario):
        """添加测试场景。"""
        self.scenarios.append(scenario)

    def record_result(self, result: ComparisonResult):
        """记录对比结果。"""
        result.analyze()
        self.results.append(result)

    def generate_report(self) -> str:
        """生成对比报告。"""
        if not self.results:
            return "No results to report."

        lines = [
            "=" * 80,
            "Agent vs 固定工作流 对比报告",
            f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
        ]

        # 汇总统计
        total_scenarios = len(self.results)
        agent_wins = sum(1 for r in self.results if "Agent" in r.recommendation)
        workflow_wins = sum(1 for r in self.results if "Workflow" in r.recommendation)
        tied = total_scenarios - agent_wins - workflow_wins

        lines.extend([
            "[SUMMARY] Overall Statistics",
            f"  Total Scenarios: {total_scenarios}",
            f"  Agent Recommended: {agent_wins}",
            f"  Workflow Recommended: {workflow_wins}",
            f"  Tied: {tied}",
            "",
        ])

        # 详细对比
        lines.append("[DETAILS] Detailed Comparison")
        for i, result in enumerate(self.results, 1):
            lines.extend([
                f"\n{i}. [{result.scenario_name}]",
                f"   Input: {result.user_input}",
                f"   Time: {result.time_advantage}",
                f"   Complexity: {result.complexity_advantage}",
                f"   Reliability: {result.reliability_advantage}",
                f"   Agent reply length: {result.agent_metrics.final_reply_length}",
                f"   Workflow reply length: {result.workflow_metrics.final_reply_length}",
                f"   Recommendation: {result.recommendation}",
            ])

        # 决策建议
        lines.extend([
            "",
            "=" * 80,
            "[DECISION] Recommendation",
            "",
        ])

        if agent_wins > workflow_wins:
            lines.append(
                f"[YES] Overall Recommendation: Prefer Agent\n"
                f"   Reason: Better in {agent_wins}/{total_scenarios} scenarios\n"
                f"   Use Cases: Open-ended Q&A, multi-step reasoning, diverse user inputs"
            )
        elif workflow_wins > agent_wins:
            lines.append(
                f"[YES] Overall Recommendation: Prefer Fixed Workflow\n"
                f"   Reason: Better in {workflow_wins}/{total_scenarios} scenarios\n"
                f"   Use Cases: Fixed process, performance-critical, high predictability"
            )
        else:
            lines.append(
                f"[YES] Overall Recommendation: Hybrid Strategy\n"
                f"   Strategy: Choose per scenario - use Workflow for simple cases, Agent for complex\n"
                f"   Boundary: See detailed comparison above"
            )

        lines.extend([
            "",
            "=" * 80,
        ])

        return "\n".join(lines)

    def get_summary(self) -> dict:
        """获取结果摘要（便于进一步分析）。"""
        return {
            "total_scenarios": len(self.results),
            "agent_wins": sum(1 for r in self.results if "Agent" in r.recommendation),
            "workflow_wins": sum(1 for r in self.results if "Workflow" in r.recommendation),
            "results": self.results,
        }
