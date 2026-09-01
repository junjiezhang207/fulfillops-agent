import asyncio

from app.application.workflow.hitl_store import PostgreSQLWorkflowIdempotencyStore
from app.application.workflow.workflow_service import WorkflowService
from app.schemas.workflow import WorkflowRunRequest


class _CapturingAsyncGraph:
    def __init__(self):
        self.initial_state = None

    async def ainvoke(self, initial_state, _config):
        self.initial_state = initial_state
        return {
            **initial_state,
            "order_result": None,
            "inventory_result": None,
            "knowledge_result": None,
            "final_answer": None,
            "execution_proposal": None,
            "preflight_validation": None,
        }


def test_workflow_run_with_timeout_passes_session_memory_to_graph():
    graph = _CapturingAsyncGraph()
    service = WorkflowService.__new__(WorkflowService)
    service._graph = graph
    request = WorkflowRunRequest(
        order_id="SO-WF-MEM-001",
        question="还是优先保证时效",
        session_memory={
            "structured": {
                "user_preferences": {"priority": "speed"},
                "confirmed_constraints": {"split_order_accepted": True},
            }
        },
    )

    result = asyncio.run(service.run_with_timeout(request, timeout=1))

    assert graph.initial_state["session_memory"] == request.session_memory
    assert result.session_memory == request.session_memory


def test_workflow_idempotency_key_includes_session_memory():
    speed_request = WorkflowRunRequest(
        order_id="SO-WF-IDEM-001",
        question="这个订单怎么处理？",
        session_memory={"structured": {"user_preferences": {"priority": "speed"}}},
    )
    cost_request = WorkflowRunRequest(
        order_id="SO-WF-IDEM-001",
        question="这个订单怎么处理？",
        session_memory={"structured": {"user_preferences": {"priority": "cost"}}},
    )

    assert PostgreSQLWorkflowIdempotencyStore.make_key(speed_request) != PostgreSQLWorkflowIdempotencyStore.make_key(cost_request)
