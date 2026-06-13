"""Workflow 生产状态存储。

包含两类生产状态：
- Workflow 幂等结果：写 Redis，保证多实例重复请求能共享状态。
- HITL 人工审核任务：写 MySQL，保证待审、决策和审计可查询。
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
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(payload)


class RedisWorkflowIdempotencyStore:
    """Workflow 幂等存储。

    Redis key 中保存完整 WorkflowRunResult JSON。结果体如果后续变大，可以改成
    Redis 保存状态与 result_id，MySQL 保存完整结果；对外接口保持不变。
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 300) -> None:
        if not redis_url:
            raise RuntimeError("Workflow 幂等必须配置 REDIS_URL，生产模式不允许使用内存缓存。")
        import redis

        self.ttl_seconds = int(ttl_seconds)
        self._client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        self._client.ping()

    @staticmethod
    def make_key(request: WorkflowRunRequest) -> str:
        import hashlib

        content = (
            f"{request.order_id}:{request.question or ''}:"
            f"{sorted(request.filter_categories or [])}"
        )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return f"multiship:workflow:idempotency:{digest}"

    def get(self, key: str) -> WorkflowRunResult | None:
        payload = self._client.get(key)
        if not payload:
            return None
        return WorkflowRunResult.model_validate_json(payload)

    def set(self, key: str, result: WorkflowRunResult) -> None:
        self._client.setex(key, self.ttl_seconds, result.model_dump_json())


class MySQLHitlStore:
    """HITL 审批状态 MySQL 存储。"""

    def __init__(self, mysql_url: str) -> None:
        if not mysql_url:
            raise RuntimeError("HITL 状态必须配置 MYSQL_URL，生产模式不允许使用进程内存。")
        self._engine = create_engine(mysql_url, pool_pre_ping=True, pool_recycle=1800, future=True)
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
                    risk_signals_json LONGTEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    context_json LONGTEXT NOT NULL,
                    options_json LONGTEXT NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    requested_at DATETIME(6) NOT NULL,
                    expires_at DATETIME(6) NULL,
                    decided_at DATETIME(6) NULL,
                    decision VARCHAR(32) NULL,
                    reason TEXT NULL,
                    approver_id VARCHAR(128) NULL,
                    KEY idx_hitl_tasks_order_id (order_id),
                    KEY idx_hitl_tasks_status (status),
                    KEY idx_hitl_tasks_requested_at (requested_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS hitl_decisions (
                    id BIGINT NOT NULL AUTO_INCREMENT,
                    thread_id VARCHAR(160) NOT NULL,
                    order_id VARCHAR(128) NOT NULL,
                    decision VARCHAR(32) NOT NULL,
                    reason TEXT NOT NULL,
                    approver_id VARCHAR(128) NOT NULL,
                    created_at DATETIME(6) NOT NULL,
                    PRIMARY KEY (id),
                    KEY idx_hitl_decisions_thread_id (thread_id),
                    KEY idx_hitl_decisions_order_id (order_id),
                    CONSTRAINT fk_hitl_decisions_task
                        FOREIGN KEY (thread_id) REFERENCES hitl_tasks(thread_id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """))

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
                    ON DUPLICATE KEY UPDATE
                        order_id=VALUES(order_id),
                        interrupt_type=VALUES(interrupt_type),
                        node=VALUES(node),
                        risk_level=VALUES(risk_level),
                        risk_signals_json=VALUES(risk_signals_json),
                        prompt=VALUES(prompt),
                        context_json=VALUES(context_json),
                        options_json=VALUES(options_json),
                        status='pending',
                        requested_at=VALUES(requested_at),
                        expires_at=VALUES(expires_at),
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
