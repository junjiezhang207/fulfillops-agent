"""Fulfillment case lifecycle after HITL approval."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.application.routing.action_router import ACTION_ROUTE_MAP, ActionRouter
from app.application.routing.order_context_service import OrderContextService
from app.application.routing.tool_gateway import ToolGateway, ToolGatewayCircuitOpen
from app.domain.inventory.analysis import InventoryAnalysisService
from app.domain.orders.analysis import OrderAnalysisService
from app.infrastructure.llm.model_gateway import get_model_gateway
from app.observability.business_trace import add_trace_step, update_current_trace
from app.schemas.fulfillment_case import (
    CaseIngestionJob,
    ExcellentCaseRecord,
    ExternalTaskStatusPollRequest,
    ExternalTaskWebhookRequest,
    FulfillmentCase,
    FulfillmentCaseVerifyResult,
    RoutedExternalTask,
)
from app.schemas.workflow import ExecutionProposal, PreflightValidation


TERMINAL_TASK_STATUS = {"COMPLETED", "FAILED", "REJECTED"}
WAITING_TASK_STATUS = {"CREATED", "READY", "BLOCKED", "RUNNING"}
FORBIDDEN_DIRECT_MUTATION_ACTION_TYPES = {
    "direct_inventory_mutation",
    "direct_order_update",
    "direct_shipment_create",
    "direct_inventory_transfer",
    "update_order_status",
    "reserve_inventory",
    "create_shipment",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _json_load(payload: str | bytes | None) -> Any:
    if not payload:
        return {}
    if isinstance(payload, (dict, list)):
        return payload
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)


def _proposal_expiry_failure(proposal: ExecutionProposal) -> str | None:
    if not proposal.expires_at:
        return None
    raw = proposal.expires_at.strip().replace("Z", "+00:00")
    try:
        expires_at = datetime.fromisoformat(raw)
    except ValueError:
        return f"执行提案 expires_at 格式无效：{proposal.expires_at}。"
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= _utc_now():
        return "执行提案已过期，禁止直接执行。"
    return None


def _preflight_snapshot_failure(
    proposal: ExecutionProposal,
    preflight_validation: PreflightValidation,
) -> str | None:
    old_fingerprint = preflight_validation.old_fingerprint
    new_fingerprint = preflight_validation.new_fingerprint
    if old_fingerprint != proposal.data_fingerprint:
        return (
            "二次校验旧快照指纹与当前执行提案不一致，"
            "禁止下发外部任务。"
        )
    if new_fingerprint != proposal.data_fingerprint:
        return (
            "二次校验发现实时业务上下文已变化，"
            "禁止下发外部任务，请基于最新上下文重新规划。"
        )
    return None


def _case_closed_for_knowledge(case: FulfillmentCase) -> bool:
    """优秀案例只能从已关闭业务案件沉淀，兼容旧的 COMPLETED+phase=CLOSED 表示法。"""

    return case.case_status == "CLOSED" or (
        case.case_status == "COMPLETED" and case.current_state.get("phase") == "CLOSED"
    )


def _dispatch_policy_failures(proposal: ExecutionProposal) -> list[str]:
    failures: list[str] = []
    actions = proposal.actions
    action_ids = [action.action_id for action in actions]
    action_id_set = set(action_ids)
    if not actions:
        failures.append("执行提案至少需要包含一个可路由动作。")
    if len(action_ids) != len(action_id_set):
        failures.append("动作 action_id 必须唯一。")
    for index, action in enumerate(actions, start=1):
        label = action.action_id or f"#{index}"
        if not action.action_id.strip():
            failures.append(f"动作 {label} 的 action_id 不能为空。")
        if action.action_type in FORBIDDEN_DIRECT_MUTATION_ACTION_TYPES:
            failures.append(
                f"动作 {label} 使用了直接业务突变类型 {action.action_type}，只能创建外部协同请求/任务。"
            )
        elif action.action_type not in ACTION_ROUTE_MAP:
            failures.append(
                f"动作 {label} 的 action_type={action.action_type} 不在受控协同动作白名单中。"
            )
        else:
            parameter_errors = ToolGateway._validate_action_payload(
                action.action_type,
                action.model_dump(mode="json"),
            )
            for field in parameter_errors:
                failures.append(f"动作 {label} 缺少或非法参数：{field}。")
        dangling_dependencies = [item for item in action.depends_on if item not in action_id_set]
        if dangling_dependencies:
            failures.append(
                f"动作 {label} 依赖不存在的前置动作：{', '.join(dangling_dependencies)}。"
            )
    return failures


class FulfillmentCaseStore(Protocol):
    def save_case(self, case: FulfillmentCase) -> None: ...
    def get_case(self, case_id: str) -> FulfillmentCase | None: ...
    def list_cases(self, order_id: str | None = None, limit: int = 50) -> list[FulfillmentCase]: ...
    def save_task(self, task: RoutedExternalTask) -> None: ...
    def get_task(self, task_id: str) -> RoutedExternalTask | None: ...
    def find_case_by_task(self, task_id: str) -> FulfillmentCase | None: ...
    def save_excellent_case(self, record: ExcellentCaseRecord) -> None: ...


class ExcellentCaseExtractor(Protocol):
    """Safely post-process a desensitized excellent-case draft."""

    def extract(self, *, fallback_record: ExcellentCaseRecord) -> ExcellentCaseRecord: ...


class LocalExcellentCaseExtractor:
    """Default extractor: deterministic, local and Pydantic-validated."""

    def extract(self, *, fallback_record: ExcellentCaseRecord) -> ExcellentCaseRecord:
        return ExcellentCaseRecord.model_validate(fallback_record.model_dump(mode="json"))


class InMemoryFulfillmentCaseStore:
    """Small test/dev store. Production wiring should use PostgreSQLFulfillmentCaseStore."""

    def __init__(self) -> None:
        self.cases: dict[str, FulfillmentCase] = {}
        self.tasks: dict[str, RoutedExternalTask] = {}
        self.excellent_cases: dict[str, ExcellentCaseRecord] = {}

    def save_case(self, case: FulfillmentCase) -> None:
        self.cases[case.case_id] = case
        for task in case.tasks:
            self.tasks[task.task_id] = task

    def get_case(self, case_id: str) -> FulfillmentCase | None:
        return self.cases.get(case_id)

    def list_cases(self, order_id: str | None = None, limit: int = 50) -> list[FulfillmentCase]:
        cases = list(self.cases.values())
        if order_id:
            cases = [case for case in cases if case.order_id == order_id]
        return sorted(cases, key=lambda case: case.updated_at, reverse=True)[:limit]

    def save_task(self, task: RoutedExternalTask) -> None:
        self.tasks[task.task_id] = task
        case = self.cases.get(task.case_id)
        if case:
            updated_tasks = [task if item.task_id == task.task_id else item for item in case.tasks]
            self.cases[case.case_id] = case.model_copy(update={"tasks": updated_tasks, "updated_at": _utc_now()})

    def get_task(self, task_id: str) -> RoutedExternalTask | None:
        return self.tasks.get(task_id)

    def find_case_by_task(self, task_id: str) -> FulfillmentCase | None:
        task = self.tasks.get(task_id)
        return self.cases.get(task.case_id) if task else None

    def save_excellent_case(self, record: ExcellentCaseRecord) -> None:
        self.excellent_cases[record.excellent_case_id] = record


class PostgreSQLFulfillmentCaseStore:
    """PostgreSQL-backed case/task store used by the API layer."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._init_schema()

    def save_case(self, case: FulfillmentCase) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO fulfillment_cases (
                        case_id, order_id, proposal_id, case_status, case_json, created_at, updated_at
                    ) VALUES (
                        :case_id, :order_id, :proposal_id, :case_status, :case_json, :created_at, :updated_at
                    )
                    ON CONFLICT (case_id) DO UPDATE SET
                        case_status=EXCLUDED.case_status,
                        case_json=EXCLUDED.case_json,
                        updated_at=EXCLUDED.updated_at
                    """
                ),
                {
                    "case_id": case.case_id,
                    "order_id": case.order_id,
                    "proposal_id": case.proposal_id,
                    "case_status": case.case_status,
                    "case_json": _json_dump(case.model_dump(mode="json")),
                    "created_at": case.created_at,
                    "updated_at": case.updated_at,
                },
            )
            for task in case.tasks:
                self._save_task_with_conn(conn, task)

    def get_case(self, case_id: str) -> FulfillmentCase | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT case_json FROM fulfillment_cases WHERE case_id=:case_id"),
                {"case_id": case_id},
            ).mappings().first()
        if row is None:
            return None
        return FulfillmentCase.model_validate(_json_load(row["case_json"]))

    def list_cases(self, order_id: str | None = None, limit: int = 50) -> list[FulfillmentCase]:
        sql = "SELECT case_json FROM fulfillment_cases"
        params: dict[str, Any] = {"limit": int(limit)}
        if order_id:
            sql += " WHERE order_id=:order_id"
            params["order_id"] = order_id
        sql += " ORDER BY updated_at DESC LIMIT :limit"
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [FulfillmentCase.model_validate(_json_load(row["case_json"])) for row in rows]

    def save_task(self, task: RoutedExternalTask) -> None:
        with self._engine.begin() as conn:
            self._save_task_with_conn(conn, task)
        case = self.get_case(task.case_id)
        if case:
            updated_tasks = [task if item.task_id == task.task_id else item for item in case.tasks]
            self.save_case(case.model_copy(update={"tasks": updated_tasks, "updated_at": _utc_now()}))

    def get_task(self, task_id: str) -> RoutedExternalTask | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT task_json FROM fulfillment_external_tasks WHERE task_id=:task_id"),
                {"task_id": task_id},
            ).mappings().first()
        if row is None:
            return None
        return RoutedExternalTask.model_validate(_json_load(row["task_json"]))

    def find_case_by_task(self, task_id: str) -> FulfillmentCase | None:
        task = self.get_task(task_id)
        return self.get_case(task.case_id) if task else None

    def save_excellent_case(self, record: ExcellentCaseRecord) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO fulfillment_excellent_cases (
                        excellent_case_id, case_id, order_id_masked, record_json, created_at
                    ) VALUES (
                        :excellent_case_id, :case_id, :order_id_masked, :record_json, :created_at
                    )
                    ON CONFLICT (excellent_case_id) DO UPDATE SET
                        record_json=EXCLUDED.record_json
                    """
                ),
                {
                    "excellent_case_id": record.excellent_case_id,
                    "case_id": record.case_id,
                    "order_id_masked": record.order_id_masked,
                    "record_json": _json_dump(record.model_dump(mode="json")),
                    "created_at": record.created_at,
                },
            )

    def _save_task_with_conn(self, conn, task: RoutedExternalTask) -> None:
        conn.execute(
            text(
                """
                INSERT INTO fulfillment_external_tasks (
                    task_id, case_id, target_system, status, task_json, created_at, updated_at
                ) VALUES (
                    :task_id, :case_id, :target_system, :status, :task_json, :created_at, :updated_at
                )
                ON CONFLICT (task_id) DO UPDATE SET
                    status=EXCLUDED.status,
                    task_json=EXCLUDED.task_json,
                    updated_at=EXCLUDED.updated_at
                """
            ),
            {
                "task_id": task.task_id,
                "case_id": task.case_id,
                "target_system": task.target_system,
                "status": task.status,
                "task_json": _json_dump(task.model_dump(mode="json")),
                "created_at": task.created_at,
                "updated_at": task.updated_at,
            },
        )

    def _init_schema(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fulfillment_cases (
                    case_id VARCHAR(128) PRIMARY KEY,
                    order_id VARCHAR(128) NOT NULL,
                    proposal_id VARCHAR(128) NOT NULL,
                    case_status VARCHAR(64) NOT NULL,
                    case_json JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fulfillment_cases_order ON fulfillment_cases (order_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fulfillment_cases_status ON fulfillment_cases (case_status)"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fulfillment_external_tasks (
                    task_id VARCHAR(128) PRIMARY KEY,
                    case_id VARCHAR(128) NOT NULL,
                    target_system VARCHAR(32) NOT NULL,
                    status VARCHAR(64) NOT NULL,
                    task_json JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fulfillment_tasks_case ON fulfillment_external_tasks (case_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fulfillment_tasks_status ON fulfillment_external_tasks (status)"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS fulfillment_excellent_cases (
                    excellent_case_id VARCHAR(128) PRIMARY KEY,
                    case_id VARCHAR(128) NOT NULL,
                    order_id_masked VARCHAR(128) NOT NULL,
                    record_json JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_excellent_cases_case ON fulfillment_excellent_cases (case_id)"))




class FulfillmentCaseService:
    """State machine for approved fulfillment action cards."""

    def __init__(
        self,
        *,
        order_service: OrderAnalysisService,
        inventory_service: InventoryAnalysisService,
        store: FulfillmentCaseStore,
        action_router: ActionRouter | None = None,
        tool_gateway: ToolGateway | None = None,
        context_service: OrderContextService | None = None,
        proposal_builder: Any | None = None,
        case_extractor: ExcellentCaseExtractor | None = None,
    ) -> None:
        self.order_service = order_service
        self.inventory_service = inventory_service
        self.store = store
        self.action_router = action_router or ActionRouter()
        self.tool_gateway = tool_gateway or ToolGateway()
        self.context_service = context_service or OrderContextService(
            order_service=order_service,
            inventory_service=inventory_service,
        )
        self.proposal_builder = proposal_builder
        self.case_extractor = case_extractor or LocalExcellentCaseExtractor()

    def confirm(
        self,
        *,
        proposal: ExecutionProposal,
        preflight_validation: PreflightValidation,
        approver_id: str,
        notes: str,
        human_changes: list[str] | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> FulfillmentCase:
        if proposal.status != "pending_approval":
            raise ValueError("只有待审批且未失效的提案才能创建外部任务。")
        expiry_failure = _proposal_expiry_failure(proposal)
        if expiry_failure is not None:
            raise ValueError(f"{expiry_failure} 请重新读取最新业务上下文并重新规划。")
        if preflight_validation.status != "valid":
            raise ValueError("只有二次校验通过的提案才能创建外部任务。")
        snapshot_failure = _preflight_snapshot_failure(proposal, preflight_validation)
        if snapshot_failure is not None:
            raise ValueError(snapshot_failure)
        policy_failures = _dispatch_policy_failures(proposal)
        if policy_failures:
            raise ValueError("Policy/Schema Validator 未通过：" + "; ".join(policy_failures))

        human_changes = [item for item in (human_changes or []) if str(item).strip()]
        case_id = f"case-{proposal.order_id}-{proposal.data_fingerprint[:10]}"
        routed_tasks = [
            task.model_copy(
                update={
                    "payload": {
                        **task.payload,
                        "hitl": "APPROVED",
                        "approved_by": approver_id,
                    }
                }
            )
            for task in self.action_router.route(case_id=case_id, proposal=proposal)
        ]
        tasks = self._dispatch_schedulable_tasks(routed_tasks)
        case_status = self._case_status_from_tasks(tasks)
        now = _utc_now()
        case = FulfillmentCase(
            case_id=case_id,
            order_id=proposal.order_id,
            proposal_id=proposal.proposal_id,
            case_status=case_status,
            case_version=1,
            plan=proposal,
            plan_version=proposal.plan_version,
            context_version=proposal.context_version,
            action_dag=proposal.action_dag,
            success_criteria=proposal.success_criteria,
            replan_count=0,
            checkpoint={
                    **(checkpoint or {}),
                    "approved_by": approver_id,
                    "approval_notes": notes,
                    "approved_at": now.isoformat(),
                    "preflight": preflight_validation.model_dump(mode="json"),
                    "hitl": "APPROVED",
                    "human_changes": human_changes,
                },
            current_state={
                "phase": "WAITING" if case_status == "WAITING_EXTERNAL_TASK" else case_status,
                "workflow_released": True,
                "agent_running": False,
                "dispatch_status": "scheduled",
                "scheduler": self._scheduler_state(tasks),
                "approved_by": approver_id,
                "approval_notes": notes,
                "human_changes": human_changes,
                "hitl_decision": "MODIFIED_APPROVED" if human_changes else "APPROVED",
            },
            tasks=tasks,
            created_at=now,
            updated_at=now,
        )
        self.store.save_case(case)
        self._trace_case_event(
            "HITL_APPROVED",
            case=case,
            status="success",
            summary="人工审批通过，外部协同任务已进入调度。",
            metadata={"approved_by": approver_id, "approval_notes": notes},
        )
        if human_changes:
            self._trace_case_event(
                "HITL_MODIFIED",
                case=case,
                status="success",
                summary="人工修改后的最终方案已通过策略和结构校验。",
                metadata={
                    "approved_by": approver_id,
                    "human_changes": human_changes,
                    "preflight_status": preflight_validation.status,
                },
            )
        if checkpoint:
            self._trace_case_event(
                "CHECKPOINT_SAVED",
                case=case,
                status="success",
                summary="工作流 checkpoint 已保存，等待外部任务期间释放执行资源。",
                metadata={"checkpoint": checkpoint},
            )
        if case.case_status == "WAITING_EXTERNAL_TASK":
            self._trace_case_event(
                "ENTER_WAITING",
                case=case,
                status="pending_human",
                summary="案件进入 WAITING_EXTERNAL_TASK，Agent 暂停并等待外部系统回调。",
                metadata={"scheduler": case.current_state.get("scheduler")},
            )
        if case.case_status in {"FAILED", "REJECTED"}:
            return self._advance_after_dispatch_failure(case)
        return case

    def confirm_replan(
        self,
        *,
        case_id: str,
        proposal: ExecutionProposal,
        preflight_validation: PreflightValidation,
        approver_id: str,
        notes: str,
        human_changes: list[str] | None = None,
        checkpoint: dict[str, Any] | None = None,
    ) -> FulfillmentCase:
        """Approve and dispatch a regenerated plan for an existing case."""

        case = self.store.get_case(case_id)
        if case is None:
            raise ValueError(f"案件不存在：{case_id}")
        if case.case_status != "REPLAN_REQUIRED":
            raise ValueError("只有 REPLAN_REQUIRED 案件才能确认重新规划方案。")
        if proposal.status != "pending_approval":
            raise ValueError("只有待审批且未失效的重新规划方案才能创建外部任务。")
        expiry_failure = _proposal_expiry_failure(proposal)
        if expiry_failure is not None:
            raise ValueError(f"{expiry_failure} 请重新读取最新业务上下文并重新规划。")
        if preflight_validation.status != "valid":
            raise ValueError("只有二次校验通过的重新规划方案才能创建外部任务。")
        snapshot_failure = _preflight_snapshot_failure(proposal, preflight_validation)
        if snapshot_failure is not None:
            raise ValueError(snapshot_failure)
        policy_failures = _dispatch_policy_failures(proposal)
        if policy_failures:
            raise ValueError("Policy/Schema Validator 未通过：" + "; ".join(policy_failures))
        if proposal.order_id != case.order_id:
            raise ValueError("重新规划方案的 order_id 与案件不一致。")
        expected_plan_version = case.plan_version + 1
        if proposal.plan_version != expected_plan_version:
            raise ValueError(f"重新规划方案 plan_version 必须为 {expected_plan_version}。")

        human_changes = [item for item in (human_changes or []) if str(item).strip()]
        routed_tasks = [
            task.model_copy(
                update={
                    "payload": {
                        **task.payload,
                        "hitl": "APPROVED",
                        "approved_by": approver_id,
                        "replan_count": case.replan_count,
                    }
                }
            )
            for task in self.action_router.route(
                case_id=case.case_id,
                proposal=proposal,
                case_version=case.case_version + 1,
            )
        ]
        tasks = self._dispatch_schedulable_tasks(routed_tasks)
        case_status = self._case_status_from_tasks(tasks)
        now = _utc_now()
        prior_state = dict(case.current_state)
        prior_state.pop("pending_replan_proposal", None)
        attempted_plans = [
            *self._attempted_plans_from_state(case.current_state),
            self._plan_attempt_record(case=case),
        ]
        failure_reasons = [
            *(case.current_state.get("failure_reasons") or []),
            case.current_state.get("latest_replan_reason"),
        ]
        failure_reasons = [str(item) for item in failure_reasons if item]
        updated = case.model_copy(
            update={
                "proposal_id": proposal.proposal_id,
                "case_status": case_status,
                "case_version": case.case_version + 1,
                "plan": proposal,
                "plan_version": proposal.plan_version,
                "context_version": proposal.context_version,
                "action_dag": proposal.action_dag,
                "success_criteria": proposal.success_criteria,
                "tasks": tasks,
                "checkpoint": {
                    **case.checkpoint,
                    **(checkpoint or {}),
                    "approved_by": approver_id,
                    "approval_notes": notes,
                    "approved_at": now.isoformat(),
                    "preflight": preflight_validation.model_dump(mode="json"),
                    "hitl": "REPLAN_APPROVED",
                    "human_changes": human_changes,
                    "previous_plan_version": case.plan_version,
                },
                "current_state": {
                    **prior_state,
                    "phase": "WAITING" if case_status == "WAITING_EXTERNAL_TASK" else case_status,
                    "workflow_released": True,
                    "agent_running": False,
                    "dispatch_status": "replan_scheduled",
                    "scheduler": self._scheduler_state(tasks),
                    "approved_by": approver_id,
                    "approval_notes": notes,
                    "human_changes": human_changes,
                    "hitl_decision": "REPLAN_MODIFIED_APPROVED" if human_changes else "REPLAN_APPROVED",
                    "attempted_plans": attempted_plans,
                    "failure_reasons": failure_reasons,
                    "previous_tasks": [task.model_dump(mode="json") for task in case.tasks],
                },
                "updated_at": now,
            }
        )
        self.store.save_case(updated)
        self._trace_case_event(
            "HITL_APPROVED",
            case=updated,
            status="success",
            summary="重新规划方案已由人工审批通过，外部协同任务重新下发。",
            metadata={
                "approved_by": approver_id,
                "approval_notes": notes,
                "previous_plan_version": case.plan_version,
                "new_plan_version": proposal.plan_version,
            },
        )
        if human_changes:
            self._trace_case_event(
                "HITL_MODIFIED",
                case=updated,
                status="success",
                summary="人工修改后的重新规划方案已通过策略和结构校验。",
                metadata={"approved_by": approver_id, "human_changes": human_changes},
            )
        if checkpoint:
            self._trace_case_event(
                "CHECKPOINT_SAVED",
                case=updated,
                status="success",
                summary="重新规划确认后的 checkpoint 已保存。",
                metadata={"checkpoint": checkpoint},
            )
        if updated.case_status == "WAITING_EXTERNAL_TASK":
            self._trace_case_event(
                "ENTER_WAITING",
                case=updated,
                status="pending_human",
                summary="重新规划任务已下发，案件再次进入 WAITING_EXTERNAL_TASK。",
                metadata={"scheduler": updated.current_state.get("scheduler")},
            )
        if updated.case_status in {"FAILED", "REJECTED"}:
            return self._advance_after_dispatch_failure(updated)
        return updated

    def review_proposal(
        self,
        *,
        proposal: ExecutionProposal,
        decision: str,
        approver_id: str,
        notes: str = "",
        checkpoint: dict[str, Any] | None = None,
    ) -> FulfillmentCase:
        """Persist a HITL review decision before dispatching external tasks."""

        if decision not in {"rejected", "ask_followup"}:
            raise ValueError("HITL 审核决策必须是 rejected 或 ask_followup。")
        if proposal.status != "pending_approval":
            raise ValueError("只有待审批提案才能进入 HITL 拒绝或追问分支。")

        now = _utc_now()
        case_id = f"case-{proposal.order_id}-{proposal.data_fingerprint[:10]}"
        reviewed_plan = proposal.model_copy(
            update={
                "status": "rejected" if decision == "rejected" else proposal.status,
                "invalidation_reason": notes or ("人工拒绝当前方案。" if decision == "rejected" else None),
            }
        )
        replacement = (
            self._build_replacement_proposal_for_review(
                order_id=proposal.order_id,
                next_plan_version=proposal.plan_version + 1,
            )
            if decision == "rejected"
            else None
        )
        case_status = "REPLAN_REQUIRED" if decision == "rejected" else "REVIEWING"
        phase = "REPLANNING" if decision == "rejected" else "REVIEWING"
        case = FulfillmentCase(
            case_id=case_id,
            order_id=proposal.order_id,
            proposal_id=proposal.proposal_id,
            case_status=case_status,
            case_version=1,
            plan=reviewed_plan,
            plan_version=proposal.plan_version,
            context_version=proposal.context_version,
            action_dag=proposal.action_dag,
            success_criteria=proposal.success_criteria,
            replan_count=0,
            checkpoint={
                **(checkpoint or {}),
                "reviewed_by": approver_id,
                "review_notes": notes,
                "reviewed_at": now.isoformat(),
                "hitl": decision.upper(),
            },
            current_state={
                "phase": phase,
                "workflow_released": decision == "ask_followup",
                "agent_running": decision == "rejected",
                "dispatch_status": "not_dispatched",
                "scheduler": self._scheduler_state([]),
                "reviewed_by": approver_id,
                "review_notes": notes,
                "hitl_decision": decision.upper(),
                "rejected_plan": reviewed_plan.model_dump(mode="json") if decision == "rejected" else None,
                "pending_replan_proposal": (
                    replacement.model_dump(mode="json")
                    if replacement is not None
                    else None
                ),
                "failure_reasons": [notes or "人工拒绝当前方案。"] if decision == "rejected" else [],
            },
            tasks=[],
            created_at=now,
            updated_at=now,
        )
        self.store.save_case(case)
        self._trace_case_event(
            "HITL_REJECTED" if decision == "rejected" else "HITL_FOLLOWUP_REQUESTED",
            case=case,
            status="pending_human" if decision == "ask_followup" else "success",
            summary=(
                "人工拒绝当前执行提案，案件进入 REPLAN_REQUIRED 等待 Planner 修改。"
                if decision == "rejected"
                else "人工要求继续追问，案件保持 REVIEWING 且未创建外部任务。"
            ),
            metadata={
                "reviewed_by": approver_id,
                "review_notes": notes,
                "has_replacement_proposal": replacement is not None,
                "dispatch_status": "not_dispatched",
            },
        )
        if decision == "rejected" and replacement is not None:
            self._trace_case_event(
                "PLAN_VERSION_CHANGED",
                case=case,
                status="success",
                summary="HITL 拒绝后已生成下一版候选方案，等待重新审核。",
                metadata={
                    "previous_plan_version": proposal.plan_version,
                    "next_plan_version": replacement.plan_version,
                    "next_proposal_id": replacement.proposal_id,
                },
            )
        return case

    def update_task_from_webhook(
        self,
        *,
        task_id: str,
        webhook: ExternalTaskWebhookRequest,
    ) -> FulfillmentCase:
        return self._apply_external_task_update(
            task_id=task_id,
            event=webhook,
            event_type="WEBHOOK_RECEIVED",
            source_label="Webhook",
            event_id=webhook.event_id,
        )

    def sync_task_from_status_api(
        self,
        *,
        task_id: str,
        status_update: ExternalTaskStatusPollRequest,
    ) -> FulfillmentCase:
        """Apply a status snapshot polled from an external Status API.

        This is the polling counterpart to ``update_task_from_webhook``. Both
        paths intentionally share the same validation and scheduler semantics so
        WAITING cases can resume from either integration style.
        """

        event_id = status_update.poll_id or status_update.event_id
        return self._apply_external_task_update(
            task_id=task_id,
            event=status_update,
            event_type="STATUS_API_POLLED",
            source_label="Status API",
            event_id=event_id,
        )

    def start_manual_processing(
        self,
        *,
        case_id: str,
        operator_id: str,
        notes: str = "",
    ) -> FulfillmentCase:
        """Move a MANUAL handoff case into human processing."""

        case = self.store.get_case(case_id)
        if case is None:
            raise ValueError(f"案件不存在：{case_id}")
        if case.case_status != "MANUAL":
            raise ValueError("只有 MANUAL 案件才能进入人工处理。")
        updated = case.model_copy(
            update={
                "current_state": {
                    **case.current_state,
                    "phase": "MANUAL_PROCESSING",
                    "manual_operator_id": operator_id,
                    "manual_processing_notes": notes,
                    "manual_processing_started_at": _utc_now().isoformat(),
                    "workflow_released": True,
                    "agent_running": False,
                },
                "updated_at": _utc_now(),
            }
        )
        self.store.save_case(updated)
        self._trace_case_event(
            "MANUAL_PROCESSING",
            case=updated,
            status="pending_human",
            summary="人工接管已开始处理，Agent 继续释放执行资源。",
            metadata={"operator_id": operator_id, "notes": notes},
        )
        return updated

    def complete_manual_resolution(
        self,
        *,
        case_id: str,
        operator_id: str,
        notes: str = "",
        business_result: dict[str, Any] | None = None,
    ) -> FulfillmentCase:
        """Record human completion and resume the workflow at VERIFYING."""

        case = self.store.get_case(case_id)
        if case is None:
            raise ValueError(f"案件不存在：{case_id}")
        if case.case_status != "MANUAL":
            raise ValueError("只有 MANUAL 案件才能记录人工完成。")
        if case.current_state.get("phase") not in {"MANUAL_WAITING", "MANUAL_PROCESSING"}:
            raise ValueError("当前案件不处于人工接管阶段。")

        now = _utc_now()
        evidence = {
            "business_state_verified": True,
            "manual_resolution": {
                "operator_id": operator_id,
                "notes": notes,
                "completed_at": now.isoformat(),
            },
            **(business_result or {}),
        }
        next_case_version = case.case_version + 1
        tasks = [
            task.model_copy(
                update={
                    "status": "COMPLETED",
                    "case_version": next_case_version,
                    "updated_at": now,
                    "result": {
                        **task.result,
                        **evidence,
                        "status_source": "Manual",
                    },
                }
            )
            if task.status != "COMPLETED"
            else task.model_copy(
                update={
                    "case_version": next_case_version,
                    "updated_at": now,
                    "result": {
                        **task.result,
                        "manual_resolution_confirmed": True,
                        "status_source": task.result.get("status_source") or "Manual",
                    },
                }
            )
            for task in case.tasks
        ]
        updated = case.model_copy(
            update={
                "case_status": "VERIFYING",
                "case_version": next_case_version,
                "tasks": tasks,
                "current_state": {
                    **case.current_state,
                    "phase": "VERIFYING",
                    "manual_completed_by": operator_id,
                    "manual_completion_notes": notes,
                    "manual_completed_at": now.isoformat(),
                    "manual_business_result": business_result or {},
                    "scheduler": self._scheduler_state(tasks),
                    "workflow_released": False,
                    "agent_running": True,
                },
                "updated_at": now,
            }
        )
        self.store.save_case(updated)
        self._trace_case_event(
            "MANUAL_COMPLETED",
            case=updated,
            status="success",
            summary="人工已完成真实业务处理，工作流恢复并进入 VERIFYING。",
            metadata={"operator_id": operator_id, "business_result": business_result or {}},
        )
        self._trace_case_event(
            "WORKFLOW_RESUMED",
            case=updated,
            status="success",
            summary="人工接管完成后恢复工作流，准备重新读取最新业务状态校验。",
            metadata={"resume_from": "MANUAL", "operator_id": operator_id},
        )
        return updated

    def _apply_external_task_update(
        self,
        *,
        task_id: str,
        event: ExternalTaskWebhookRequest | ExternalTaskStatusPollRequest,
        event_type: str,
        source_label: str,
        event_id: str | None,
    ) -> FulfillmentCase:
        task = self.store.get_task(task_id)
        if task is None:
            raise ValueError(f"外部任务不存在：{task_id}")
        status = event.status.upper()
        if status not in {"RUNNING", "COMPLETED", "FAILED", "REJECTED"}:
            raise ValueError("任务状态必须是 RUNNING / COMPLETED / FAILED / REJECTED")
        case = self.store.get_case(task.case_id)
        if case is None:
            raise ValueError(f"任务所属案件不存在：{task.case_id}")
        missing_fields = self._missing_external_event_contract_fields(
            event=event,
            event_id=event_id,
        )
        if missing_fields:
            ignored_case = case.model_copy(
                update={
                    "current_state": {
                        **case.current_state,
                        "last_ignored_event": {
                            "event_id": event_id,
                            "task_id": task.task_id,
                            "reason": "missing_event_contract_fields",
                            "missing_fields": missing_fields,
                            "source": source_label,
                        },
                    },
                    "updated_at": _utc_now(),
                }
            )
            self.store.save_case(ignored_case)
            self._trace_case_event(
                event_type,
                case=ignored_case,
                status="ignored",
                summary=f"{source_label} 缺少必要事件合同字段，案件状态不推进。",
                metadata=ignored_case.current_state.get("last_ignored_event"),
            )
            return ignored_case
        if task.case_version != case.case_version or task.plan_version != case.plan_version:
            stale_case = case.model_copy(
                update={
                    "current_state": {
                        **case.current_state,
                        "last_ignored_event": {
                            "event_id": event_id,
                            "task_id": task.task_id,
                            "reason": "stale_task_version",
                            "task_case_version": task.case_version,
                            "current_case_version": case.case_version,
                            "task_plan_version": task.plan_version,
                            "current_plan_version": case.plan_version,
                            "source": source_label,
                        },
                    },
                    "updated_at": _utc_now(),
                }
            )
            self.store.save_case(stale_case)
            self._trace_case_event(
                event_type,
                case=stale_case,
                status="ignored",
                summary=f"旧任务版本的 {source_label} 事件被忽略，当前案件状态不推进。",
                metadata=stale_case.current_state.get("last_ignored_event"),
            )
            return stale_case
        if event.idempotency_key and event.idempotency_key != task.idempotency_key:
            stale_case = case.model_copy(
                update={
                    "current_state": {
                        **case.current_state,
                        "last_ignored_event": {
                            "event_id": event_id,
                            "task_id": task.task_id,
                            "reason": "idempotency_key_mismatch",
                            "idempotency_key": event.idempotency_key,
                            "source": source_label,
                        },
                    },
                    "updated_at": _utc_now(),
                }
            )
            self.store.save_case(stale_case)
            self._trace_case_event(
                event_type,
                case=stale_case,
                status="ignored",
                summary=f"{source_label} 因幂等键不匹配被忽略，案件状态不推进。",
                metadata=stale_case.current_state.get("last_ignored_event"),
            )
            return stale_case
        if event_id and event_id in (case.current_state.get("processed_event_ids") or []):
            duplicate_case = case.model_copy(
                update={
                    "current_state": {
                        **case.current_state,
                        "last_ignored_event": {
                            "event_id": event_id,
                            "task_id": task.task_id,
                            "reason": "duplicate_event",
                            "source": source_label,
                        },
                    },
                    "updated_at": _utc_now(),
                }
            )
            self.store.save_case(duplicate_case)
            self._trace_case_event(
                event_type,
                case=duplicate_case,
                status="ignored",
                summary=f"重复 {source_label} 事件被忽略，案件状态不推进。",
                metadata=duplicate_case.current_state.get("last_ignored_event"),
            )
            return duplicate_case
        if (
            event.case_version is not None
            and event.case_version != case.case_version
            or event.plan_version is not None
            and event.plan_version != task.plan_version
        ):
            stale_case = case.model_copy(
                update={
                    "current_state": {
                        **case.current_state,
                        "last_ignored_event": {
                            "event_id": event_id,
                            "task_id": task.task_id,
                            "reason": "stale_case_or_plan_version",
                            "case_version": event.case_version,
                            "plan_version": event.plan_version,
                            "source": source_label,
                        },
                    },
                    "updated_at": _utc_now(),
                }
            )
            self.store.save_case(stale_case)
            self._trace_case_event(
                event_type,
                case=stale_case,
                status="ignored",
                summary=f"旧 case_version 或 plan_version 的 {source_label} 事件被忽略。",
                metadata=stale_case.current_state.get("last_ignored_event"),
            )
            return stale_case
        updated_task = task.model_copy(
            update={
                "status": status,
                "updated_at": _utc_now(),
                "result": {
                    **task.result,
                    "event_id": event_id,
                    "idempotency_key": event.idempotency_key,
                    "external_ref": event.external_ref,
                    "operator": event.operator,
                    "message": event.message,
                    "status_source": source_label,
                    **event.result,
                },
            }
        )
        self.store.save_task(updated_task)
        case = self.store.get_case(task.case_id)
        if case is None:
            raise ValueError(f"任务所属案件不存在：{task.case_id}")
        scheduled_tasks = self._dispatch_schedulable_tasks(case.tasks)
        next_status = self._case_status_from_tasks(scheduled_tasks)
        processed_event_ids = list(case.current_state.get("processed_event_ids") or [])
        if event_id and event_id not in processed_event_ids:
            processed_event_ids.append(event_id)
        updated_case = case.model_copy(update={
            "case_status": next_status,
            "tasks": scheduled_tasks,
            "current_state": {
                **case.current_state,
                "phase": "VERIFYING" if next_status == "VERIFYING" else "WAITING",
                "last_event_id": event_id,
                "last_status_source": source_label,
                "processed_event_ids": processed_event_ids[-100:],
                "scheduler": self._scheduler_state(scheduled_tasks),
            },
            "updated_at": _utc_now(),
        })
        self.store.save_case(updated_case)
        self._trace_case_event(
            event_type,
            case=updated_case,
            status="success",
            summary=f"收到外部任务{source_label}状态，任务状态更新为 {status}。",
            metadata={
                "task_id": updated_task.task_id,
                "action_id": updated_task.action_id,
                "task_status": updated_task.status,
                "event_id": event_id,
                "external_ref": event.external_ref,
                "source": source_label,
                "case_status": updated_case.case_status,
            },
        )
        if next_status == "VERIFYING":
            self._trace_case_event(
                "WORKFLOW_RESUMED",
                case=updated_case,
                status="success",
                summary="所有外部协同任务已完成，工作流恢复并进入 VERIFYING。",
                metadata={"last_event_id": event_id, "source": source_label},
            )
        if next_status in {"FAILED", "REJECTED"}:
            return self._advance_after_dispatch_failure(updated_case)
        return updated_case

    @staticmethod
    def _missing_external_event_contract_fields(
        *,
        event: ExternalTaskWebhookRequest | ExternalTaskStatusPollRequest,
        event_id: str | None,
    ) -> list[str]:
        missing: list[str] = []
        if not event_id:
            missing.append("event_id")
        if event.case_version is None:
            missing.append("case_version")
        if event.plan_version is None:
            missing.append("plan_version")
        if not event.idempotency_key:
            missing.append("idempotency_key")
        return missing

    def verify(self, case_id: str) -> FulfillmentCaseVerifyResult:
        case = self.store.get_case(case_id)
        if case is None:
            raise ValueError(f"案件不存在：{case_id}")

        if any(task.status in {"FAILED", "REJECTED"} for task in case.tasks):
            self._trace_case_event(
                "CASE_STATUS_CHANGED",
                case=case,
                status="failed",
                summary="VERIFYING 前发现外部任务失败或拒绝，准备进入 Replan。",
                metadata={"failed_tasks": [task.task_id for task in case.tasks if task.status in {"FAILED", "REJECTED"}]},
            )
            return self._mark_replan(case, "外部任务失败或被拒绝，必须基于最新业务事实重新规划。")
        if not all(task.status == "COMPLETED" for task in case.tasks):
            self._trace_case_event(
                "ENTER_WAITING",
                case=case,
                status="pending_human",
                summary="仍有外部任务未完成，保持 WAITING_EXTERNAL_TASK。",
                metadata={"scheduler": case.current_state.get("scheduler")},
            )
            return FulfillmentCaseVerifyResult(
                case_id=case.case_id,
                order_id=case.order_id,
                status="WAITING_EXTERNAL_TASK",
                message="仍有外部任务未完成，Agent 不继续运行。",
                checks=[{"name": task.task_id, "status": task.status} for task in case.tasks],
            )

        latest_order = self.order_service.analyze_order(case.order_id)
        latest_inventory = self.inventory_service.analyze_inventory(case.order_id)
        latest_context = self.context_service.build_context(
            order_id=case.order_id,
            intent="fulfillment_action",
            question=case.plan.summary,
        )
        checks = self._business_state_checks(case, latest_order, latest_inventory, latest_context)
        success = all(check["status"] == "pass" for check in checks)
        self._trace_case_event(
            "VERIFYING",
            case=case,
            status="success" if success else "failed",
            summary="VERIFYING 阶段已重新读取 OMS/WMS/TMS/ERP/PIM/CRM 最新业务状态。",
            metadata={
                "checks": checks,
                "latest_order_status": getattr(latest_order, "order_status", None),
                "insufficient_skus": getattr(latest_inventory, "insufficient_skus", []),
                "context_completeness": latest_context.completeness,
                "source_systems": list((latest_context.full_context.get("source_systems") or {}).keys()),
            },
        )
        if success:
            updated = case.model_copy(
                update={
                    "case_status": "COMPLETED",
                    "updated_at": _utc_now(),
                    "verification": {"checks": checks, "verified_at": _utc_now().isoformat()},
                    "current_state": {
                        **case.current_state,
                        "phase": "CLOSED",
                        "workflow_released": True,
                        "agent_running": False,
                    },
                }
            )
            self.store.save_case(updated)
            self._trace_case_event(
                "CASE_STATUS_CHANGED",
                case=updated,
                status="success",
                summary="外部任务与最新业务状态验证通过，案件关闭。",
                metadata={"new_status": "COMPLETED"},
            )
            return FulfillmentCaseVerifyResult(
                case_id=case.case_id,
                order_id=case.order_id,
                status="COMPLETED",
            message="外部任务完成，且 OMS/WMS/TMS/ERP/PIM/CRM 最新业务状态已验证通过，案件可关闭。",
            checks=checks,
        )
        return self._mark_replan(
            case,
            "外部任务完成，但真实业务状态未达到目标，需要重新规划。",
            checks=checks,
            latest_context=latest_context,
        )

    def persist_excellent_case(self, *, case_id: str, operator_id: str, notes: str = "") -> ExcellentCaseRecord:
        case = self.store.get_case(case_id)
        if case is None:
            raise ValueError(f"案件不存在：{case_id}")
        if not _case_closed_for_knowledge(case):
            raise ValueError("只有已关闭案件才能沉淀为优秀案例。")
        extraction_metadata = self._case_extraction_metadata()
        fallback = self._build_deterministic_excellent_case_record(
            case=case,
            operator_id=operator_id,
            notes=notes,
            extraction_metadata=extraction_metadata,
        )
        record = self._extract_excellent_case_record(
            case=case,
            fallback=fallback,
            extraction_metadata=extraction_metadata,
        )
        self.store.save_excellent_case(record)
        return record

    def _build_deterministic_excellent_case_record(
        self,
        *,
        case: FulfillmentCase,
        operator_id: str,
        notes: str,
        extraction_metadata: dict[str, str | None],
    ) -> ExcellentCaseRecord:
        human_changes = self._human_changes(case=case, operator_id=operator_id, notes=notes)
        final_result = {
            "case_status": case.case_status,
            "verification": case.verification,
            "completed_task_ids": [task.task_id for task in case.tasks if task.status == "COMPLETED"],
            "prompt_id": extraction_metadata.get("prompt_id"),
            "prompt_source": extraction_metadata.get("prompt_source"),
        }
        return ExcellentCaseRecord(
            excellent_case_id=f"excellent-{case.case_id}",
            case_id=case.case_id,
            order_id_masked=self._mask_order_id(case.order_id),
            scenario=case.plan.title,
            key_conditions=[
                f"capabilities={','.join(case.plan.capabilities)}",
                f"eta={case.plan.eta.get('eta_label') or case.plan.eta.get('eta_hours')}",
                f"operator={operator_id}",
                f"notes={notes}" if notes else "notes=",
            ],
            final_plan=[action.model_dump(mode="json") for action in case.plan.actions],
            human_changes=human_changes,
            final_result=final_result,
            sop_evidence=case.plan.rule_citations,
            execution_result=case.verification,
            vector_backend="pgvector",
            extraction_use_case="case_extraction",
            extraction_mode="deterministic_local",
            prompt_version=extraction_metadata.get("prompt_version"),
        )

    def _extract_excellent_case_record(
        self,
        *,
        case: FulfillmentCase,
        fallback: ExcellentCaseRecord,
        extraction_metadata: dict[str, str | None],
    ) -> ExcellentCaseRecord:
        try:
            extractor_input = self._redacted_extractor_input(fallback)
            extracted = self.case_extractor.extract(fallback_record=extractor_input)
            record = ExcellentCaseRecord.model_validate(
                extracted.model_dump(mode="json") if hasattr(extracted, "model_dump") else extracted
            )
            record = self._merge_excellent_case_deterministic_fields(
                fallback=fallback,
                extracted=record,
                extraction_metadata=extraction_metadata,
            )
            self._trace_case_event(
                "CASE_EXTRACTION_SUCCEEDED",
                case=case,
                status="success",
                summary="优秀案例抽取已完成并通过 Pydantic 校验。",
                metadata={
                    "extraction_mode": record.extraction_mode,
                    "prompt_version": record.prompt_version,
                    "warnings": record.extraction_warnings,
                },
            )
            return record
        except Exception as exc:
            record = fallback.model_copy(
                update={
                    "extraction_mode": "deterministic_fallback",
                    "extraction_warnings": [
                        *fallback.extraction_warnings,
                        f"case_extractor_failed:{exc.__class__.__name__}",
                    ],
                }
            )
            self._trace_case_event(
                "CASE_EXTRACTION_FAILED",
                case=case,
                status="failed",
                summary="优秀案例抽取器失败，已回退本地确定性记录。",
                metadata={
                    "error_code": exc.__class__.__name__,
                    "error_message": str(exc),
                    "fallback_mode": record.extraction_mode,
                },
            )
            return ExcellentCaseRecord.model_validate(record.model_dump(mode="json"))

    @staticmethod
    def _redacted_extractor_input(fallback: ExcellentCaseRecord) -> ExcellentCaseRecord:
        safe_case_ref = fallback.order_id_masked.replace("*", "x").replace("/", "-")
        return fallback.model_copy(
            update={
                "excellent_case_id": f"excellent-{safe_case_ref}",
                "case_id": f"case-{safe_case_ref}",
            }
        )

    @staticmethod
    def _merge_excellent_case_deterministic_fields(
        *,
        fallback: ExcellentCaseRecord,
        extracted: ExcellentCaseRecord,
        extraction_metadata: dict[str, str | None],
    ) -> ExcellentCaseRecord:
        return extracted.model_copy(
            update={
                "excellent_case_id": fallback.excellent_case_id,
                "case_id": fallback.case_id,
                "order_id_masked": fallback.order_id_masked,
                "vector_backend": fallback.vector_backend,
                "extraction_use_case": "case_extraction",
                "prompt_version": extraction_metadata.get("prompt_version"),
                "created_at": fallback.created_at,
            }
        )

    def start_case_ingestion(self, *, case_id: str, operator_id: str, notes: str = "") -> CaseIngestionJob:
        case = self.store.get_case(case_id)
        if case is None:
            raise ValueError(f"案件不存在：{case_id}")
        if not _case_closed_for_knowledge(case):
            raise ValueError("只有已关闭案件才能沉淀为优秀案例。")
        now = _utc_now()
        existing = case.current_state.get("case_ingestion") or {}
        if existing:
            existing_job = CaseIngestionJob.model_validate(existing)
            if existing_job.status in {"PENDING", "PROCESSING", "SUCCESS"} or existing_job.dead_letter:
                return existing_job
            if existing_job.status == "FAILED":
                job = existing_job.model_copy(
                    update={
                        "status": "PENDING",
                        "error": None,
                        "updated_at": now,
                    }
                )
                self._save_case_ingestion_job(case, job)
                self._trace_case_event(
                    "CASE_INGESTION_STARTED",
                    case=case,
                    status="success",
                    summary="优秀案例沉淀失败后重试任务已提交，等待后台写入案例知识库。",
                    metadata={"case_ingestion": job.model_dump(mode="json")},
                )
                return job
        job = CaseIngestionJob(
            job_id=f"case-ingestion-{case.case_id}",
            case_id=case.case_id,
            status="PENDING",
            operator_id=operator_id,
            notes=notes,
            created_at=now,
            updated_at=now,
        )
        self._save_case_ingestion_job(case, job)
        self._trace_case_event(
            "CASE_INGESTION_STARTED",
            case=case,
            status="success",
            summary="优秀案例沉淀任务已创建，等待后台写入案例知识库。",
            metadata={"case_ingestion": job.model_dump(mode="json")},
        )
        return job

    def mark_case_ingestion_processing(self, case_id: str) -> CaseIngestionJob:
        case = self.store.get_case(case_id)
        if case is None:
            raise ValueError(f"案件不存在：{case_id}")
        job = CaseIngestionJob.model_validate(case.current_state.get("case_ingestion") or {})
        job = job.model_copy(update={"status": "PROCESSING", "updated_at": _utc_now()})
        self._save_case_ingestion_job(case, job)
        self._trace_case_event(
            "CASE_INGESTION_PROCESSING",
            case=case,
            status="success",
            summary="优秀案例沉淀任务开始处理。",
            metadata={"case_ingestion": job.model_dump(mode="json")},
        )
        return job

    def mark_case_ingestion_success(
        self,
        *,
        case_id: str,
        excellent_case_id: str,
        knowledge_path: str,
    ) -> CaseIngestionJob:
        case = self.store.get_case(case_id)
        if case is None:
            raise ValueError(f"案件不存在：{case_id}")
        job = CaseIngestionJob.model_validate(case.current_state.get("case_ingestion") or {})
        job = job.model_copy(
            update={
                "status": "SUCCESS",
                "excellent_case_id": excellent_case_id,
                "knowledge_path": knowledge_path,
                "error": None,
                "updated_at": _utc_now(),
            }
        )
        self._save_case_ingestion_job(case, job)
        self._trace_case_event(
            "CASE_INGESTION_SUCCEEDED",
            case=case,
            status="success",
            summary="优秀案例已写入案例知识库并等待后续 RAG 检索。",
            metadata={"case_ingestion": job.model_dump(mode="json")},
        )
        return job

    def mark_case_ingestion_failed(self, *, case_id: str, error: str) -> CaseIngestionJob:
        case = self.store.get_case(case_id)
        if case is None:
            raise ValueError(f"案件不存在：{case_id}")
        job = CaseIngestionJob.model_validate(case.current_state.get("case_ingestion") or {})
        job = job.model_copy(
            update={
                "status": "FAILED",
                "retry_count": job.retry_count + 1,
                "error": error,
                "dead_letter": job.retry_count + 1 >= 3,
                "manual_queue": "CASE_INGESTION_FAILED" if job.retry_count + 1 >= 3 else None,
                "updated_at": _utc_now(),
            }
        )
        self._save_case_ingestion_job(case, job)
        self._trace_case_event(
            "CASE_INGESTION_FAILED",
            case=case,
            status="failed",
            summary="优秀案例沉淀失败，达到重试上限后进入人工处理队列。",
            metadata={"case_ingestion": job.model_dump(mode="json")},
        )
        return job

    def _build_replacement_proposal_for_review(
        self,
        *,
        order_id: str,
        next_plan_version: int,
    ) -> ExecutionProposal | None:
        if self.proposal_builder is None:
            return None
        latest_order = self.order_service.analyze_order(order_id)
        latest_inventory = self.inventory_service.analyze_inventory(order_id)
        try:
            return self.proposal_builder(
                latest_order,
                latest_inventory,
                plan_version=next_plan_version,
            )
        except TypeError:
            proposal = self.proposal_builder(latest_order, latest_inventory)
            if hasattr(proposal, "model_copy"):
                return proposal.model_copy(update={"plan_version": next_plan_version})
            return proposal

    def get_case(self, case_id: str) -> FulfillmentCase | None:
        return self.store.get_case(case_id)

    def list_cases(self, order_id: str | None = None, limit: int = 50) -> list[FulfillmentCase]:
        return self.store.list_cases(order_id=order_id, limit=limit)

    def get_task(self, task_id: str) -> RoutedExternalTask | None:
        return self.store.get_task(task_id)

    def _save_case_ingestion_job(self, case: FulfillmentCase, job: CaseIngestionJob) -> None:
        updated = case.model_copy(
            update={
                "current_state": {
                    **case.current_state,
                    "case_ingestion": job.model_dump(mode="json"),
                },
                "updated_at": _utc_now(),
            }
        )
        self.store.save_case(updated)

    @staticmethod
    def _case_extraction_metadata() -> dict[str, str | None]:
        try:
            prompt = get_model_gateway().load_prompt(use_case="case_extraction")
            return {
                "prompt_id": prompt.id,
                "prompt_version": prompt.version,
                "prompt_source": prompt.source,
            }
        except Exception:
            return {
                "prompt_id": "case_extraction",
                "prompt_version": None,
                "prompt_source": "unavailable",
            }

    @staticmethod
    def _human_changes(*, case: FulfillmentCase, operator_id: str, notes: str) -> list[str]:
        changes: list[str] = []
        approval_notes = str(case.current_state.get("approval_notes") or "").strip()
        approved_by = str(case.current_state.get("approved_by") or operator_id or "").strip()
        for change in case.current_state.get("human_changes") or []:
            text = str(change).strip()
            if text:
                changes.append(f"human_change={text}")
        if approval_notes:
            changes.append(f"approval_notes={approval_notes}")
        if notes:
            changes.append(f"closure_notes={notes}")
        if approved_by:
            changes.append(f"approved_by={approved_by}")
        completed_actions = [
            task.action_id
            for task in case.tasks
            if task.action_id and task.status == "COMPLETED"
        ]
        if completed_actions:
            changes.append(f"completed_actions={','.join(completed_actions)}")
        return changes

    def _dispatch_schedulable_tasks(self, tasks: list[RoutedExternalTask]) -> list[RoutedExternalTask]:
        scheduled: list[RoutedExternalTask] = []
        current = list(tasks)
        for task in current:
            if task.status not in WAITING_TASK_STATUS:
                scheduled.append(task)
                continue
            if not self._dependencies_satisfied(task, current):
                scheduled.append(
                    task.model_copy(
                        update={
                            "status": "BLOCKED",
                            "updated_at": _utc_now(),
                            "result": {
                                **task.result,
                                "scheduler_state": "BLOCKED",
                                "waiting_for": list(task.dependency_ids),
                            },
                        }
                    )
                )
                continue
            if task.status == "RUNNING":
                scheduled.append(task)
                continue
            ready_task = task.model_copy(
                update={
                    "status": "READY",
                    "updated_at": _utc_now(),
                    "result": {
                        **task.result,
                        "scheduler_state": "READY",
                    },
                }
            )
            try:
                scheduled.append(self.tool_gateway.submit(ready_task))
            except (ToolGatewayCircuitOpen, TimeoutError, RuntimeError) as exc:
                scheduled.append(
                    ready_task.model_copy(
                        update={
                            "status": "FAILED",
                            "updated_at": _utc_now(),
                            "result": {
                                **ready_task.result,
                                "scheduler_state": "FAILED",
                                "tool_gateway_failure": {
                                    "error_code": exc.__class__.__name__,
                                    "error_message": str(exc),
                                    "retryable": True,
                                },
                            },
                        }
                    )
                )
        return scheduled

    def _advance_after_dispatch_failure(self, case: FulfillmentCase) -> FulfillmentCase:
        failed_tasks = [
            task for task in case.tasks
            if task.status in {"FAILED", "REJECTED"} and task.result.get("tool_gateway_failure")
        ]
        if not failed_tasks:
            return case
        message = "外部协同任务创建失败，调度重试耗尽后进入重新规划或人工兜底。"
        failure_summary = [
            {
                "task_id": task.task_id,
                "action_id": task.action_id,
                "status": task.status,
                "tool_name": task.collaborative_tool_name,
                "external_system": task.target_system,
                "failure": task.result.get("tool_gateway_failure") or task.result,
            }
            for task in failed_tasks
        ]
        dispatch_failed_case = case.model_copy(
            update={
                "current_state": {
                    **case.current_state,
                    "dispatch_status": "dispatch_failed",
                    "dispatch_failure": {
                        "message": message,
                        "failed_tasks": failure_summary,
                    },
                }
            }
        )
        self._trace_case_event(
            "TASK_DISPATCH_FAILED",
            case=dispatch_failed_case,
            status="failed",
            summary=message,
            metadata={"failed_tasks": failure_summary},
        )
        self._mark_replan(
            dispatch_failed_case,
            message,
            checks=[
                {
                    "name": "external_task_creation",
                    "status": "fail",
                    "value": failure_summary,
                    "message": message,
                }
            ],
        )
        advanced = self.store.get_case(case.case_id)
        return advanced or dispatch_failed_case

    @staticmethod
    def _dependencies_satisfied(task: RoutedExternalTask, tasks: list[RoutedExternalTask]) -> bool:
        if not task.dependency_ids:
            return True
        by_action_id = {item.action_id: item for item in tasks}
        by_task_id = {item.task_id: item for item in tasks}
        for dependency_id in task.dependency_ids:
            dependency = by_action_id.get(dependency_id) or by_task_id.get(dependency_id)
            if dependency is None or dependency.status != "COMPLETED":
                return False
        return True

    @staticmethod
    def _scheduler_state(tasks: list[RoutedExternalTask]) -> dict[str, Any]:
        return {
            "ready": [task.task_id for task in tasks if task.status == "READY"],
            "blocked": [task.task_id for task in tasks if task.status == "BLOCKED"],
            "running": [task.task_id for task in tasks if task.status == "RUNNING"],
            "completed": [task.task_id for task in tasks if task.status == "COMPLETED"],
            "failed": [task.task_id for task in tasks if task.status in {"FAILED", "REJECTED"}],
        }

    def _mark_replan(
        self,
        case: FulfillmentCase,
        message: str,
        *,
        checks: list[dict[str, Any]] | None = None,
        latest_context: Any | None = None,
    ) -> FulfillmentCaseVerifyResult:
        latest_order = self.order_service.analyze_order(case.order_id)
        latest_inventory = self.inventory_service.analyze_inventory(case.order_id)
        context_error: str | None = None
        if latest_context is None:
            try:
                latest_context = self.context_service.build_context(
                    order_id=case.order_id,
                    intent="fulfillment_action",
                    question=case.plan.summary,
                )
            except Exception as exc:
                context_error = f"{exc.__class__.__name__}: {exc}"
        replacement = (
            self.proposal_builder(latest_order, latest_inventory)
            if self.proposal_builder is not None
            else None
        )
        if replacement is not None and hasattr(replacement, "model_copy"):
            replacement = replacement.model_copy(update={"plan_version": case.plan_version + 1})
        next_replan_count = case.replan_count + 1
        next_case_status = "REPLAN_REQUIRED" if next_replan_count <= 3 else "MANUAL"
        manual_handoff_package = (
            self._manual_handoff_package(
                case=case,
                latest_order=latest_order,
                latest_inventory=latest_inventory,
                latest_context=latest_context,
                context_error=context_error,
                message=message,
                checks=checks or [],
            )
            if next_case_status == "MANUAL"
            else None
        )
        updated = case.model_copy(
            update={
                "case_status": next_case_status,
                "case_version": case.case_version + 1,
                "replan_count": next_replan_count,
                "updated_at": _utc_now(),
                "verification": {"checks": checks or [], "message": message, "verified_at": _utc_now().isoformat()},
                "current_state": {
                    **case.current_state,
                    "phase": "REPLANNING" if next_replan_count <= 3 else "MANUAL_WAITING",
                    "agent_running": False,
                    "latest_replan_reason": message,
                    "pending_replan_proposal": (
                        replacement.model_dump(mode="json")
                        if replacement is not None and next_case_status == "REPLAN_REQUIRED"
                        else None
                    ),
                    "failure_reasons": [
                        *(case.current_state.get("failure_reasons") or []),
                        message,
                    ],
                    "manual_handoff_package": manual_handoff_package,
                },
            }
        )
        self.store.save_case(updated)
        self._trace_case_event(
            "PLAN_VERSION_CHANGED",
            case=updated,
            status="failed" if next_case_status == "MANUAL" else "success",
            summary=message,
            metadata={
                "previous_plan_version": case.plan_version,
                "next_plan_version": case.plan_version + 1,
                "replan_count": next_replan_count,
                "next_case_status": next_case_status,
                "has_replacement_proposal": replacement is not None,
            },
        )
        if next_case_status == "MANUAL":
            self._trace_case_event(
                "MANUAL_HANDOFF",
                case=updated,
                status="failed",
                summary="连续重规划失败，生成 MANUAL handoff package 等待人工接管。",
                metadata={"manual_handoff_package": manual_handoff_package},
            )
        return FulfillmentCaseVerifyResult(
            case_id=case.case_id,
            order_id=case.order_id,
            status=next_case_status,
            message=message,
            checks=checks or [],
            replan_required=next_case_status == "REPLAN_REQUIRED",
            replacement_proposal=replacement if next_case_status == "REPLAN_REQUIRED" else None,
        )

    @staticmethod
    def _manual_handoff_package(
        *,
        case: FulfillmentCase,
        latest_order: Any,
        latest_inventory: Any,
        latest_context: Any | None,
        context_error: str | None,
        message: str,
        checks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        full_context = getattr(latest_context, "full_context", {}) or {}
        selected_context = getattr(latest_context, "selected_context", {}) or {}
        context_snapshot = {
            "field_groups_loaded": getattr(latest_context, "field_groups_loaded", []) or [],
            "completeness": getattr(latest_context, "completeness", {}) or {},
            "source_systems": full_context.get("source_systems") or {},
            "selected_context": selected_context,
            "context_error": context_error,
        }
        return {
            "case_id": case.case_id,
            "order_id": case.order_id,
            "original_exception": case.plan.summary,
            "latest_business_state": {
                "order_status": getattr(latest_order, "order_status", None),
                "fulfillment_status_flags": getattr(latest_order, "fulfillment_status_flags", {}),
                "insufficient_skus": getattr(latest_inventory, "insufficient_skus", []),
                "order_context": context_snapshot,
            },
            "attempted_plans": [
                *FulfillmentCaseService._attempted_plans_from_state(case.current_state),
                FulfillmentCaseService._plan_attempt_record(case=case),
            ],
            "failure_reasons": [
                *(case.current_state.get("failure_reasons") or []),
                message,
            ],
            "completed_business_actions": [
                task.model_dump(mode="json")
                for task in case.tasks
                if task.status == "COMPLETED"
            ],
            "unfinished_tasks": [
                task.model_dump(mode="json")
                for task in case.tasks
                if task.status != "COMPLETED"
            ],
            "sop_evidence": case.plan.rule_citations,
            "case_evidence": (
                (case.plan.decision_context.get("rag_context") or {})
                .get("similar_cases", {})
                .get("evidence", [])
            ),
            "unresolved_checks": checks,
            "current_unresolved_problem": message,
        }

    @staticmethod
    def _attempted_plans_from_state(current_state: dict[str, Any]) -> list[dict[str, Any]]:
        attempted = current_state.get("attempted_plans") or []
        return [item for item in attempted if isinstance(item, dict)]

    @staticmethod
    def _plan_attempt_record(*, case: FulfillmentCase) -> dict[str, Any]:
        return {
            "proposal_id": case.plan.proposal_id,
            "plan_version": case.plan_version,
            "goal_type": case.plan.goal_type,
            "context_version": case.context_version,
            "summary": case.plan.summary,
            "actions": [action.model_dump(mode="json") for action in case.plan.actions],
            "task_statuses": [
                {
                    "task_id": task.task_id,
                    "action_id": task.action_id,
                    "target_system": task.target_system,
                    "status": task.status,
                }
                for task in case.tasks
            ],
        }

    @staticmethod
    def _trace_case_event(
        event_type: str,
        *,
        case: FulfillmentCase,
        status: str,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        trace_metadata = {
            "case_id": case.case_id,
            "order_id": case.order_id,
            "case_status": case.case_status,
            "case_version": case.case_version,
            "plan_version": case.plan_version,
            "context_version": case.context_version,
        }
        update_current_trace(order_id=case.order_id, metadata=trace_metadata)
        add_trace_step(
            step_type="workflow" if event_type != "VERIFYING" else "verification",
            name=f"fulfillment_case.{event_type.lower()}",
            status=status,
            summary=summary,
            metadata={
                "event_type": event_type,
                **trace_metadata,
                **(metadata or {}),
            },
        )

    @staticmethod
    def _case_status_from_tasks(tasks: list[RoutedExternalTask]) -> str:
        if any(task.status == "REJECTED" for task in tasks):
            return "REJECTED"
        if any(task.status == "FAILED" for task in tasks):
            return "FAILED"
        if tasks and all(task.status in TERMINAL_TASK_STATUS for task in tasks):
            return "VERIFYING"
        return "WAITING_EXTERNAL_TASK"

    @staticmethod
    def _business_state_checks(
        case: FulfillmentCase,
        latest_order: Any,
        latest_inventory: Any,
        latest_context: Any | None = None,
    ) -> list[dict[str, Any]]:
        fulfillment_state = getattr(latest_order, "fulfillment_status_flags", {}) or {}
        direct_actions = [
            action for action in case.plan.actions
            if action.action_type in {"ship_from_warehouse", "split_order", "merge_order"}
        ]
        active_tasks = fulfillment_state.get("active_fulfillment_tasks") or []
        checks: list[dict[str, Any]] = FulfillmentCaseService._latest_context_checks(latest_context)
        checks.extend([
            {
                "name": "no_active_fulfillment_tasks",
                "status": "pass" if not active_tasks else "fail",
                "value": active_tasks,
            }
        ])
        if direct_actions:
            materialized = any(
                bool(fulfillment_state.get(field))
                for field in ("inventory_reserved", "package_created", "waybill_created", "outbound_completed")
            )
            checks.append({
                "name": "direct_fulfillment_materialized",
                "status": "pass" if materialized else "fail",
                "value": {
                    "inventory_reserved": fulfillment_state.get("inventory_reserved", False),
                    "package_created": fulfillment_state.get("package_created", False),
                    "waybill_created": fulfillment_state.get("waybill_created", False),
                    "outbound_completed": fulfillment_state.get("outbound_completed", False),
                },
            })
        for action in direct_actions:
            check = next((item for item in latest_inventory.sku_checks if item.sku_id == action.sku_id), None)
            record = None
            if check:
                record = next(
                    (
                        warehouse for warehouse in check.warehouse_records
                        if warehouse.warehouse_id == action.from_warehouse
                    ),
                    None,
                )
            checks.append({
                "name": f"latest_inventory_visible:{action.sku_id}:{action.from_warehouse}",
                "status": "pass" if record is not None else "fail",
                "available_stock": record.available_stock if record else 0,
                "inventory_version": getattr(record, "inventory_version", None) if record else None,
            })
        checks.extend(FulfillmentCaseService._external_task_business_checks(case))
        return checks

    @staticmethod
    def _latest_context_checks(latest_context: Any | None) -> list[dict[str, Any]]:
        required_sources = {"OMS", "WMS", "TMS", "ERP", "PIM", "CRM"}
        if latest_context is None:
            return [
                {
                    "name": "fresh_order_context_reloaded",
                    "status": "fail",
                    "required_source_systems": sorted(required_sources),
                    "loaded_source_systems": [],
                    "value": None,
                }
            ]
        full_context = getattr(latest_context, "full_context", {}) or {}
        completeness = getattr(latest_context, "completeness", {}) or {}
        source_systems = set((full_context.get("source_systems") or {}).keys())
        return [
            {
                "name": "fresh_order_context_reloaded",
                "status": "pass" if required_sources <= source_systems else "fail",
                "required_source_systems": sorted(required_sources),
                "loaded_source_systems": sorted(source_systems),
            },
            {
                "name": "context_completeness",
                "status": "pass" if completeness.get("status") == "complete" else "fail",
                "value": completeness,
            },
        ]

    @staticmethod
    def _external_task_business_checks(case: FulfillmentCase) -> list[dict[str, Any]]:
        return [
            FulfillmentCaseService._external_task_business_check(task)
            for task in case.tasks
        ]

    @staticmethod
    def _external_task_business_check(task: RoutedExternalTask) -> dict[str, Any]:
        result = task.result or {}
        proof = FulfillmentCaseService._business_proof_for_task(task, result)
        return {
            "name": f"external_business_state:{task.target_system}:{task.action_id}",
            "status": "pass" if proof else "fail",
            "target_system": task.target_system,
            "action_type": task.action_type,
            "external_task_type": task.external_task_type,
            "external_ref": result.get("external_ref"),
            "proof": proof,
            "required_evidence": FulfillmentCaseService._required_business_evidence(task),
        }

    @staticmethod
    def _business_proof_for_task(task: RoutedExternalTask, result: dict[str, Any]) -> dict[str, Any] | None:
        if result.get("business_state_verified") is True:
            return {"business_state_verified": True}

        if task.target_system == "WMS":
            if task.action_type == "inventory_transfer":
                transfer_status = str(result.get("transfer_status") or "").upper()
                transfer_id = result.get("transfer_order_id") or result.get("inventory_transfer_id")
                if transfer_id or transfer_status in {"CREATED", "ACCEPTED", "CONFIRMED", "COMPLETED"}:
                    return {
                        "transfer_order_id": transfer_id,
                        "transfer_status": transfer_status or None,
                    }
            else:
                materialized = {
                    key: result.get(key)
                    for key in ("fulfillment_task_id", "reservation_id", "package_id", "waybill_no")
                    if result.get(key)
                }
                if materialized or result.get("fulfillment_task_created") is True:
                    return materialized or {"fulfillment_task_created": True}

        if task.target_system == "TMS":
            channel_status = str(result.get("channel_status") or "").lower()
            if (
                result.get("quote_confirmed") is True
                or result.get("serviceable") is True
                or channel_status in {"available", "confirmed", "changed", "active"}
            ):
                return {
                    "channel_status": channel_status or None,
                    "quote_confirmed": result.get("quote_confirmed"),
                    "serviceable": result.get("serviceable"),
                }

        if task.target_system == "ERP":
            replenishment_status = str(result.get("replenishment_status") or "").upper()
            replenishment_id = result.get("replenishment_request_id") or result.get("purchase_request_id")
            if (
                replenishment_id
                or result.get("inbound_stock_visible") is True
                or replenishment_status in {"CREATED", "ACCEPTED", "CONFIRMED", "COMPLETED"}
            ):
                return {
                    "replenishment_request_id": replenishment_id,
                    "replenishment_status": replenishment_status or None,
                    "inbound_stock_visible": result.get("inbound_stock_visible"),
                }

        if task.target_system == "CRM":
            confirmation_status = str(result.get("customer_confirmation_status") or "").lower()
            crm_case_status = str(result.get("crm_case_status") or "").lower()
            crm_ref = result.get("crm_case_id") or result.get("customer_contact_task_id")
            if (
                crm_ref
                or confirmation_status in {"contacted", "confirmed", "accepted", "declined"}
                or crm_case_status in {"created", "open", "completed", "closed"}
            ):
                return {
                    "crm_ref": crm_ref,
                    "customer_confirmation_status": confirmation_status or None,
                    "crm_case_status": crm_case_status or None,
                }

        return None

    @staticmethod
    def _required_business_evidence(task: RoutedExternalTask) -> list[str]:
        if task.target_system == "WMS" and task.action_type == "inventory_transfer":
            return ["transfer_order_id", "transfer_status", "business_state_verified"]
        if task.target_system == "WMS":
            return ["fulfillment_task_id", "reservation_id", "package_id", "waybill_no", "business_state_verified"]
        if task.target_system == "TMS":
            return ["quote_confirmed", "serviceable", "channel_status", "business_state_verified"]
        if task.target_system == "ERP":
            return ["replenishment_request_id", "purchase_request_id", "inbound_stock_visible", "business_state_verified"]
        if task.target_system == "CRM":
            return ["crm_case_id", "customer_contact_task_id", "customer_confirmation_status", "business_state_verified"]
        return ["business_state_verified"]

    @staticmethod
    def _mask_order_id(order_id: str) -> str:
        if len(order_id) <= 6:
            return "***"
        return f"{order_id[:3]}***{order_id[-3:]}"
