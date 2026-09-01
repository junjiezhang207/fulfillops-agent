"""Aggregate FulfillOps benchmark scores into version-level reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.agents.quality.evaluation.golden_dataset import GoldenCase
from app.agents.quality.evaluation.scoring import evaluate_agent_result


TRACE_MODULES = (
    "memory",
    "context",
    "rag",
    "model_gateway",
    "planner",
    "tool_gateway",
    "verification",
)


@dataclass
class CaseEvaluationRecord:
    """Version report row for a single benchmark case."""

    case_id: str
    workflow_path: str
    score: float
    passed: bool
    hard_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TraceEvaluationSummary:
    """Trace metrics required by the FulfillOps benchmark contract."""

    trace_count: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    rag_result_count: int = 0
    memory_result_count: int = 0
    loaded_field_groups: list[str] = field(default_factory=list)
    planner_result_count: int = 0
    tool_result_count: int = 0
    replan_count: int = 0
    workflow_result_counts: dict[str, int] = field(default_factory=dict)
    module_coverage: dict[str, int] = field(default_factory=lambda: {module: 0 for module in TRACE_MODULES})


@dataclass
class BenchmarkEvaluationReport:
    """Aggregated benchmark report for one agent/workflow version."""

    version: str
    case_count: int
    passed_count: int
    pass_rate: float
    average_score: float
    case_results: list[CaseEvaluationRecord] = field(default_factory=list)
    hard_case_set: list[CaseEvaluationRecord] = field(default_factory=list)
    workflow_path_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    trace_summary: TraceEvaluationSummary = field(default_factory=TraceEvaluationSummary)


@dataclass
class BenchmarkRegressionReport:
    """Compare two reports produced from the same benchmark ids."""

    previous_version: str
    current_version: str
    pass_rate_delta: float
    new_failures: list[str] = field(default_factory=list)
    fixed_cases: list[str] = field(default_factory=list)
    score_drops: dict[str, float] = field(default_factory=dict)


def build_benchmark_evaluation_report(
    cases: Sequence[GoldenCase],
    results_by_case_id: Mapping[str, dict[str, Any]],
    *,
    version: str = "dev",
) -> BenchmarkEvaluationReport:
    """Build a version-level report without expanding the benchmark dataset."""

    records: list[CaseEvaluationRecord] = []
    trace_summary = TraceEvaluationSummary()
    for case in cases:
        result = results_by_case_id.get(case.id)
        if result is None:
            records.append(
                CaseEvaluationRecord(
                    case_id=case.id,
                    workflow_path=case.workflow_path,
                    score=0.0,
                    passed=False,
                    hard_failures=["missing_result"],
                    warnings=["未提供该 benchmark case 的执行结果。"],
                )
            )
            continue
        score = evaluate_agent_result(case, result)
        records.append(
            CaseEvaluationRecord(
                case_id=case.id,
                workflow_path=case.workflow_path,
                score=score.total_score,
                passed=score.passed,
                hard_failures=score.hard_failures,
                warnings=score.warnings,
            )
        )
        _merge_trace_summary(trace_summary, _trace_from_result(result))

    passed_count = sum(1 for record in records if record.passed)
    average_score = round(sum(record.score for record in records) / len(records), 2) if records else 0.0
    pass_rate = round(passed_count / len(records), 4) if records else 0.0
    return BenchmarkEvaluationReport(
        version=version,
        case_count=len(records),
        passed_count=passed_count,
        pass_rate=pass_rate,
        average_score=average_score,
        case_results=records,
        hard_case_set=[record for record in records if not record.passed],
        workflow_path_summary=_workflow_path_summary(records),
        trace_summary=trace_summary,
    )


def compare_benchmark_reports(
    previous: BenchmarkEvaluationReport,
    current: BenchmarkEvaluationReport,
    *,
    score_drop_threshold: float = 5.0,
) -> BenchmarkRegressionReport:
    """Identify regression cases between two benchmark report versions."""

    previous_by_id = {record.case_id: record for record in previous.case_results}
    current_by_id = {record.case_id: record for record in current.case_results}
    shared_ids = sorted(previous_by_id.keys() & current_by_id.keys())
    new_failures = [
        case_id for case_id in shared_ids
        if previous_by_id[case_id].passed and not current_by_id[case_id].passed
    ]
    fixed_cases = [
        case_id for case_id in shared_ids
        if not previous_by_id[case_id].passed and current_by_id[case_id].passed
    ]
    score_drops = {
        case_id: round(previous_by_id[case_id].score - current_by_id[case_id].score, 2)
        for case_id in shared_ids
        if previous_by_id[case_id].score - current_by_id[case_id].score >= score_drop_threshold
    }
    return BenchmarkRegressionReport(
        previous_version=previous.version,
        current_version=current.version,
        pass_rate_delta=round(current.pass_rate - previous.pass_rate, 4),
        new_failures=new_failures,
        fixed_cases=fixed_cases,
        score_drops=score_drops,
    )


def _trace_from_result(result: Mapping[str, Any]) -> Any:
    return (
        result.get("fulfillops_trace")
        or result.get("business_trace")
        or result.get("trace")
    )


def _merge_trace_summary(summary: TraceEvaluationSummary, trace: Any) -> None:
    if trace is None:
        return
    if hasattr(trace, "model_dump"):
        trace = trace.model_dump(mode="json")
    if not isinstance(trace, dict):
        return
    if "modules" in trace or "summary" in trace:
        _merge_fulfillops_trace(summary, trace)
    else:
        _merge_agent_trace(summary, trace)


def _merge_fulfillops_trace(summary: TraceEvaluationSummary, trace: dict[str, Any]) -> None:
    summary.trace_count += 1
    trace_summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    summary.total_latency_ms += _number(trace_summary, "duration_ms")
    summary.total_tokens += int(_number(trace_summary, "total_tokens"))
    summary.total_cost_usd = round(summary.total_cost_usd + _number(trace_summary, "total_cost_usd"), 6)
    trace_tool_count = int(_number(trace_summary, "tool_call_count"))

    modules = trace.get("modules") if isinstance(trace.get("modules"), dict) else {}
    for module in TRACE_MODULES:
        module_steps = modules.get(module) or []
        if module_steps:
            summary.module_coverage[module] = summary.module_coverage.get(module, 0) + 1
    summary.rag_result_count += len(modules.get("rag") or [])
    summary.memory_result_count += len(modules.get("memory") or [])
    summary.planner_result_count += len(modules.get("planner") or [])
    summary.tool_result_count += trace_tool_count or len(modules.get("tool_gateway") or [])

    for step in _iter_module_steps(modules):
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        summary.loaded_field_groups = _merge_unique(summary.loaded_field_groups, _field_groups(metadata))
        if "replan" in " ".join(str(step.get(key) or "").lower() for key in ("name", "summary")):
            summary.replan_count += 1

    events = trace.get("events") if isinstance(trace.get("events"), list) else []
    summary.replan_count += sum(
        1 for event in events
        if "replan" in str(event.get("event_type") or event.get("name") or "").lower()
    )
    trace_meta = trace.get("trace_meta") if isinstance(trace.get("trace_meta"), dict) else {}
    status = str(trace_summary.get("status") or trace_meta.get("status") or "unknown")
    summary.workflow_result_counts[status] = summary.workflow_result_counts.get(status, 0) + 1


def _merge_agent_trace(summary: TraceEvaluationSummary, trace: dict[str, Any]) -> None:
    summary.trace_count += 1
    summary.total_latency_ms += _number(trace, "duration_ms", "latency_ms")
    summary.total_tokens += int(_number(trace, "total_tokens", "tokens"))
    summary.total_cost_usd = round(summary.total_cost_usd + _number(trace, "total_cost_usd", "cost_usd"), 6)
    tools = trace.get("tools_called") if isinstance(trace.get("tools_called"), list) else []
    summary.tool_result_count += len(tools)
    if tools:
        summary.module_coverage["tool_gateway"] = summary.module_coverage.get("tool_gateway", 0) + 1
    status = str(trace.get("status") or "success")
    summary.workflow_result_counts[status] = summary.workflow_result_counts.get(status, 0) + 1


def _workflow_path_summary(records: Sequence[CaseEvaluationRecord]) -> dict[str, dict[str, float]]:
    paths: dict[str, dict[str, float]] = {}
    for record in records:
        bucket = paths.setdefault(record.workflow_path, {"total": 0.0, "passed": 0.0, "pass_rate": 0.0})
        bucket["total"] += 1.0
        bucket["passed"] += 1.0 if record.passed else 0.0
    for bucket in paths.values():
        bucket["pass_rate"] = round(bucket["passed"] / bucket["total"], 4) if bucket["total"] else 0.0
    return paths


def _iter_module_steps(modules: Mapping[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for raw_steps in modules.values():
        if not isinstance(raw_steps, list):
            continue
        steps.extend(step for step in raw_steps if isinstance(step, dict))
    return steps


def _number(source: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = source.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _field_groups(metadata: Mapping[str, Any]) -> list[str]:
    for key in ("loaded_field_groups", "field_groups"):
        value = metadata.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if item]
    value = metadata.get("field_group")
    return [str(value)] if value else []


def _merge_unique(existing: list[str], new_values: list[str]) -> list[str]:
    seen = set(existing)
    merged = list(existing)
    for value in new_values:
        if value not in seen:
            seen.add(value)
            merged.append(value)
    return merged
