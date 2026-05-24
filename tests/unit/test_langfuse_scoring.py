from app.agent.evaluation.langfuse_tracer import score_after_run


class FakeTracer:
    def __init__(self):
        self.scores = []
        self.flushed = False

    def score(self, trace_id: str, name: str, value: float, comment: str = "") -> None:
        self.scores.append({
            "trace_id": trace_id,
            "name": name,
            "value": value,
            "comment": comment,
        })

    def flush(self) -> None:
        self.flushed = True


def test_langfuse_scoring_without_reflection_only_writes_objective_signals():
    tracer = FakeTracer()

    score_after_run(
        tracer=tracer,  # type: ignore[arg-type]
        trace_id="trace-1",
        reply="订单可以发货。",
        tools_called=["check_inventory"],
    )

    names = {item["name"] for item in tracer.scores}
    assert names == {"tool_grounding", "answer_completeness"}
    assert "overall_quality" not in names
    assert tracer.flushed is True


def test_langfuse_scoring_with_reflection_writes_quality_gate_scores():
    tracer = FakeTracer()

    score_after_run(
        tracer=tracer,  # type: ignore[arg-type]
        trace_id="trace-1",
        reply="订单 SO202502140001 缺货，需要查询替代 SKU。",
        tools_called=["check_inventory"],
        reflection_score=0.82,
        reflection_passed=True,
        reflection_reason="质量达标",
    )

    by_name = {item["name"]: item for item in tracer.scores}
    assert by_name["overall_quality"]["value"] == 0.82
    assert by_name["reflection_passed"]["value"] == 1.0
    assert by_name["tool_grounding"]["value"] == 1.0
    assert tracer.flushed is True


def test_langfuse_scoring_flushes_even_without_trace_id():
    tracer = FakeTracer()

    score_after_run(
        tracer=tracer,  # type: ignore[arg-type]
        trace_id=None,
        reply="订单可以发货。",
        tools_called=["check_inventory"],
    )

    assert tracer.scores == []
    assert tracer.flushed is True
