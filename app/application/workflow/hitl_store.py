"""Workflow 生产状态存储。

包含两类生产状态：
- Workflow 幂等结果：写 PostgreSQL，保证多实例重复请求能共享状态。
- HITL 人工审核任务：写 PostgreSQL，保证待审、决策和审计可查询。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import create_engine, text

from app.schemas.workflow import ApprovalAuditEntry, InterruptEvent, WorkflowRunRequest, WorkflowRunResult


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


class PostgreSQLWorkflowIdempotencyStore:
    """Workflow 幂等存储。

    PostgreSQL 中保存完整 WorkflowRunResult JSON。结果体如果后续变大，可以拆为
    状态表和结果表；对外接口保持不变。
    """

    def __init__(self, database_url: str, ttl_seconds: int = 300) -> None:
        if not database_url:
            raise RuntimeError("Workflow 幂等必须配置 DATABASE_URL/POSTGRES_URL，生产模式不允许使用内存缓存。")
        self.ttl_seconds = int(ttl_seconds)
        self._engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=1800, future=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS workflow_idempotency_results (
                    idempotency_key VARCHAR(128) PRIMARY KEY,
                    result_json JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_workflow_idempotency_expires
                ON workflow_idempotency_results (expires_at)
            """))

    def _delete_expired(self, conn) -> None:
        conn.execute(
            text("DELETE FROM workflow_idempotency_results WHERE expires_at <= :now"),
            {"now": _utc_now().replace(tzinfo=None)},
        )

    @staticmethod
    def make_key(request: WorkflowRunRequest) -> str:
        import hashlib

        content = _json_dump(
            {
                "order_id": request.order_id,
                "question": request.question or "",
                "filter_categories": sorted(request.filter_categories or []),
                "session_memory": request.session_memory or {},
            }
        )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return f"fulfillops:workflow:idempotency:{digest}"

    def get(self, key: str) -> WorkflowRunResult | None:
        with self._engine.begin() as conn:
            self._delete_expired(conn)
            row = conn.execute(
                text("""
                    SELECT result_json
                    FROM workflow_idempotency_results
                    WHERE idempotency_key=:idempotency_key AND expires_at > :now
                """),
                {"idempotency_key": key, "now": _utc_now().replace(tzinfo=None)},
            ).mappings().first()
        if row is None:
            return None
        return WorkflowRunResult.model_validate(_json_load(row["result_json"]))

    def set(self, key: str, result: WorkflowRunResult) -> None:
        now = _utc_now().replace(tzinfo=None)
        expires_at = now + timedelta(seconds=self.ttl_seconds)
        with self._engine.begin() as conn:
            self._delete_expired(conn)
            conn.execute(
                text("""
                    INSERT INTO workflow_idempotency_results (
                        idempotency_key, result_json, created_at, expires_at
                    ) VALUES (
                        :idempotency_key, :result_json, :created_at, :expires_at
                    )
                    ON CONFLICT (idempotency_key) DO UPDATE SET
                        result_json=EXCLUDED.result_json,
                        created_at=EXCLUDED.created_at,
                        expires_at=EXCLUDED.expires_at
                """),
                {
                    "idempotency_key": key,
                    "result_json": result.model_dump_json(),
                    "created_at": now,
                    "expires_at": expires_at,
                },
            )


class PostgreSQLHitlStore:
    """HITL 审批状态 PostgreSQL 存储。"""

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise RuntimeError("HITL 状态必须配置 DATABASE_URL/POSTGRES_URL，生产模式不允许使用进程内存。")
        self._engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=1800, future=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS hitl_tasks (
                    thread_id VARCHAR(160) PRIMARY KEY,
                    order_id VARCHAR(128) NOT NULL,
                    interrupt_type VARCHAR(64) NOT NULL,
                    node VARCHAR(128) NOT NULL,
                    risk_level VARCHAR(32) NOT NULL,
                    risk_signals_json JSONB NOT NULL,
                    prompt TEXT NOT NULL,
                    context_json JSONB NOT NULL,
                    options_json JSONB NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    requested_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NULL,
                    decided_at TIMESTAMP NULL,
                    decision VARCHAR(32) NULL,
                    reason TEXT NULL,
                    approver_id VARCHAR(128) NULL
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_hitl_tasks_order_id ON hitl_tasks (order_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_hitl_tasks_status ON hitl_tasks (status)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_hitl_tasks_requested_at ON hitl_tasks (requested_at)"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS hitl_decisions (
                    id BIGSERIAL PRIMARY KEY,
                    thread_id VARCHAR(160) NOT NULL,
                    order_id VARCHAR(128) NOT NULL,
                    decision VARCHAR(32) NOT NULL,
                    reason TEXT NOT NULL,
                    approver_id VARCHAR(128) NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    CONSTRAINT fk_hitl_decisions_task
                        FOREIGN KEY (thread_id) REFERENCES hitl_tasks(thread_id)
                        ON DELETE CASCADE
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_hitl_decisions_thread_id ON hitl_decisions (thread_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_hitl_decisions_order_id ON hitl_decisions (order_id)"))

    def upsert_task(self, *, order_id: str, interrupt: InterruptEvent) -> None:
        requested_at = _utc_now()
        expires_at = requested_at + timedelta(seconds=int(interrupt.timeout_seconds or 1800))
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO hitl_tasks (
                        thread_id, order_id, interrupt_type, node, risk_level,
                        risk_signals_json, prompt, context_json, options_json,
                        status, requested_at, expires_at
                    ) VALUES (
                        :thread_id, :order_id, :interrupt_type, :node, :risk_level,
                        :risk_signals_json, :prompt, :context_json, :options_json,
                        'pending', :requested_at, :expires_at
                    )
                    ON CONFLICT (thread_id) DO UPDATE SET
                        order_id=EXCLUDED.order_id,
                        interrupt_type=EXCLUDED.interrupt_type,
                        node=EXCLUDED.node,
                        risk_level=EXCLUDED.risk_level,
                        risk_signals_json=EXCLUDED.risk_signals_json,
                        prompt=EXCLUDED.prompt,
                        context_json=EXCLUDED.context_json,
                        options_json=EXCLUDED.options_json,
                        status='pending',
                        requested_at=EXCLUDED.requested_at,
                        expires_at=EXCLUDED.expires_at,
                        decided_at=NULL,
                        decision=NULL,
                        reason=NULL,
                        approver_id=NULL
                """),
                {
                    "thread_id": interrupt.thread_id,
                    "order_id": order_id,
                    "interrupt_type": interrupt.type,
                    "node": interrupt.node,
                    "risk_level": interrupt.risk_level,
                    "risk_signals_json": _json_dump(interrupt.risk_signals),
                    "prompt": interrupt.prompt,
                    "context_json": _json_dump(interrupt.context),
                    "options_json": _json_dump(interrupt.options),
                    "requested_at": requested_at,
                    "expires_at": expires_at,
                },
            )

    def record_decision(
        self,
        *,
        thread_id: str,
        decision: str,
        reason: str,
        approver_id: str,
    ) -> None:
        decided_at = _utc_now()
        with self._engine.begin() as conn:
            task = conn.execute(
                text("SELECT order_id FROM hitl_tasks WHERE thread_id=:thread_id"),
                {"thread_id": thread_id},
            ).mappings().first()
            if task is None:
                raise ValueError(f"HITL 任务不存在：{thread_id}")
            conn.execute(
                text("""
                    UPDATE hitl_tasks
                    SET status=:status, decision=:decision, reason=:reason,
                        approver_id=:approver_id, decided_at=:decided_at
                    WHERE thread_id=:thread_id
                """),
                {
                    "thread_id": thread_id,
                    "status": "approved" if decision == "approved" else "rejected",
                    "decision": decision,
                    "reason": reason,
                    "approver_id": approver_id,
                    "decided_at": decided_at,
                },
            )
            conn.execute(
                text("""
                    INSERT INTO hitl_decisions (
                        thread_id, order_id, decision, reason, approver_id, created_at
                    ) VALUES (
                        :thread_id, :order_id, :decision, :reason, :approver_id, :created_at
                    )
                """),
                {
                    "thread_id": thread_id,
                    "order_id": task["order_id"],
                    "decision": decision,
                    "reason": reason,
                    "approver_id": approver_id,
                    "created_at": decided_at,
                },
            )

    def list_pending(self, risk_level: str | None = None) -> list[ApprovalAuditEntry]:
        clauses = ["status='pending'"]
        params: dict[str, Any] = {}
        if risk_level:
            clauses.append("risk_level=:risk_level")
            params["risk_level"] = risk_level
        where_sql = " AND ".join(clauses)
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM hitl_tasks WHERE {where_sql} ORDER BY requested_at DESC"),
                params,
            ).mappings().all()
        return [self._row_to_audit_entry(row) for row in rows]

    def history(self, order_id: str) -> list[ApprovalAuditEntry]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("SELECT * FROM hitl_tasks WHERE order_id=:order_id ORDER BY requested_at DESC"),
                {"order_id": order_id},
            ).mappings().all()
        return [self._row_to_audit_entry(row) for row in rows]

    def stats(self) -> dict:
        with self._engine.connect() as conn:
            pending = conn.execute(text("SELECT COUNT(*) FROM hitl_tasks WHERE status='pending'")).scalar_one()
            total = conn.execute(text("SELECT COUNT(*) FROM hitl_decisions")).scalar_one()
            approved = conn.execute(text("SELECT COUNT(*) FROM hitl_decisions WHERE decision='approved'")).scalar_one()
        total_int = int(total)
        return {
            "pending_count": int(pending),
            "total_decisions": total_int,
            "approval_rate": round(int(approved) / total_int, 3) if total_int else 0.0,
        }

    def _row_to_audit_entry(self, row) -> ApprovalAuditEntry:
        decided_at = row["decided_at"] or row["requested_at"]
        return ApprovalAuditEntry(
            thread_id=row["thread_id"],
            order_id=row["order_id"],
            interrupt_type=row["interrupt_type"],
            risk_level=row["risk_level"],
            risk_signals=list(_json_load(row["risk_signals_json"]) or []),
            decision=row["decision"] or "",
            reason=row["reason"] or "",
            approver_id=row["approver_id"] or "",
            requested_at=str(row["requested_at"]),
            decided_at=str(decided_at),
        )
