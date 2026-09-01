"""业务 Trace 与审计事件的 PostgreSQL 持久化层。

生产策略：
- Trace、Step、Audit Event 统一写 PostgreSQL。
- 不再使用 SQLite 本地文件，避免多实例部署时链路数据分散。
- 这里只保存摘要、脱敏 metadata 和 evidence，不保存完整 prompt 或客户隐私原文。
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from sqlalchemy import create_engine, text

from app.core.config import get_settings


INTERNAL_ROUTE_PREFIXES = (
    "/api/v1/observability",
    "/api/v1/health",
    "/api/v1/metrics",
)


def _json(data: Any) -> str:
    if is_dataclass(data):
        data = asdict(data)
    return json.dumps({} if data is None else data, ensure_ascii=False, default=str)


def _decode(raw: str | bytes | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


class PostgreSQLTraceStore:
    """给中间件和 Trace API 使用的 PostgreSQL 存储。"""

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise RuntimeError("Business Trace 必须配置 DATABASE_URL/POSTGRES_URL，生产模式不允许使用 SQLite。")
        self._engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=1800, future=True)
        self._initialized = False
        self.init()

    def init(self) -> None:
        if self._initialized:
            return
        with self._engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trace_runs (
                    trace_id VARCHAR(128) PRIMARY KEY,
                    request_id VARCHAR(128),
                    session_id VARCHAR(160),
                    tenant_id VARCHAR(128),
                    user_id VARCHAR(128),
                    order_id VARCHAR(128),
                    route VARCHAR(255),
                    status VARCHAR(64),
                    error_code VARCHAR(128),
                    error_message TEXT,
                    started_at VARCHAR(64),
                    ended_at VARCHAR(64),
                    duration_ms DOUBLE PRECISION,
                    metadata_json JSONB
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trace_runs_order_id ON trace_runs (order_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trace_runs_route ON trace_runs (route)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trace_runs_started_at ON trace_runs (started_at)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trace_runs_status ON trace_runs (status)"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trace_steps (
                    id VARCHAR(128) PRIMARY KEY,
                    trace_id VARCHAR(128) NOT NULL,
                    order_index INT NOT NULL,
                    parent_id VARCHAR(128),
                    type VARCHAR(64),
                    name VARCHAR(255),
                    status VARCHAR(64),
                    started_at VARCHAR(64),
                    ended_at VARCHAR(64),
                    duration_ms DOUBLE PRECISION,
                    summary TEXT,
                    error_code VARCHAR(128),
                    error_message TEXT,
                    input_summary TEXT,
                    output_summary TEXT,
                    metadata_json JSONB,
                    evidence_json JSONB,
                    CONSTRAINT fk_trace_steps_trace
                        FOREIGN KEY (trace_id) REFERENCES trace_runs(trace_id)
                        ON DELETE CASCADE
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trace_steps_trace_id ON trace_steps (trace_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trace_steps_parent_id ON trace_steps (parent_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trace_steps_type ON trace_steps (type)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trace_steps_name ON trace_steps (name)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_trace_steps_status ON trace_steps (status)"))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id VARCHAR(128) PRIMARY KEY,
                    trace_id VARCHAR(128),
                    request_id VARCHAR(128),
                    tenant_id VARCHAR(128),
                    user_id VARCHAR(128),
                    order_id VARCHAR(128),
                    event_type VARCHAR(128) NOT NULL,
                    action VARCHAR(128),
                    actor_type VARCHAR(64),
                    status VARCHAR(64),
                    created_at VARCHAR(64),
                    summary TEXT,
                    metadata_json JSONB
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_events_order_id ON audit_events (order_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_audit_events_trace_id ON audit_events (trace_id)"))
        self._initialized = True

    def save_trace(self, trace: dict[str, Any]) -> None:
        self.init()
        trace_id = trace.get("trace_id")
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO trace_runs (
                        trace_id, request_id, session_id, tenant_id, user_id, order_id,
                        route, status, error_code, error_message, started_at, ended_at,
                        duration_ms, metadata_json
                    ) VALUES (
                        :trace_id, :request_id, :session_id, :tenant_id, :user_id, :order_id,
                        :route, :status, :error_code, :error_message, :started_at, :ended_at,
                        :duration_ms, :metadata_json
                    )
                    ON CONFLICT (trace_id) DO UPDATE SET
                        request_id=EXCLUDED.request_id,
                        session_id=EXCLUDED.session_id,
                        tenant_id=EXCLUDED.tenant_id,
                        user_id=EXCLUDED.user_id,
                        order_id=EXCLUDED.order_id,
                        route=EXCLUDED.route,
                        status=EXCLUDED.status,
                        error_code=EXCLUDED.error_code,
                        error_message=EXCLUDED.error_message,
                        started_at=EXCLUDED.started_at,
                        ended_at=EXCLUDED.ended_at,
                        duration_ms=EXCLUDED.duration_ms,
                        metadata_json=EXCLUDED.metadata_json
                """),
                {
                    "trace_id": trace_id,
                    "request_id": trace.get("request_id"),
                    "session_id": trace.get("session_id"),
                    "tenant_id": trace.get("tenant_id"),
                    "user_id": trace.get("user_id"),
                    "order_id": trace.get("order_id"),
                    "route": trace.get("route"),
                    "status": trace.get("status"),
                    "error_code": trace.get("error_code"),
                    "error_message": trace.get("error_message"),
                    "started_at": trace.get("started_at"),
                    "ended_at": trace.get("ended_at"),
                    "duration_ms": trace.get("duration_ms"),
                    "metadata_json": _json(trace.get("metadata")),
                },
            )
            conn.execute(text("DELETE FROM trace_steps WHERE trace_id=:trace_id"), {"trace_id": trace_id})
            for index, step in enumerate(trace.get("steps") or []):
                conn.execute(
                    text("""
                        INSERT INTO trace_steps (
                            id, trace_id, order_index, parent_id, type, name, status,
                            started_at, ended_at, duration_ms, summary, error_code,
                            error_message, input_summary, output_summary, metadata_json,
                            evidence_json
                        ) VALUES (
                            :id, :trace_id, :order_index, :parent_id, :type, :name, :status,
                            :started_at, :ended_at, :duration_ms, :summary, :error_code,
                            :error_message, :input_summary, :output_summary, :metadata_json,
                            :evidence_json
                        )
                    """),
                    {
                        "id": step.get("id"),
                        "trace_id": trace_id,
                        "order_index": index,
                        "parent_id": step.get("parent_id"),
                        "type": step.get("type"),
                        "name": step.get("name"),
                        "status": step.get("status"),
                        "started_at": step.get("started_at"),
                        "ended_at": step.get("ended_at"),
                        "duration_ms": step.get("duration_ms"),
                        "summary": step.get("summary"),
                        "error_code": step.get("error_code"),
                        "error_message": step.get("error_message"),
                        "input_summary": step.get("input_summary"),
                        "output_summary": step.get("output_summary"),
                        "metadata_json": _json(step.get("metadata")),
                        "evidence_json": _json(step.get("evidence")),
                    },
                )

    def save_audit_event(self, event: dict[str, Any]) -> None:
        self.init()
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO audit_events (
                        id, trace_id, request_id, tenant_id, user_id, order_id,
                        event_type, action, actor_type, status, created_at, summary,
                        metadata_json
                    ) VALUES (
                        :id, :trace_id, :request_id, :tenant_id, :user_id, :order_id,
                        :event_type, :action, :actor_type, :status, :created_at, :summary,
                        :metadata_json
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        status=EXCLUDED.status,
                        summary=EXCLUDED.summary,
                        metadata_json=EXCLUDED.metadata_json
                """),
                {
                    "id": event.get("id"),
                    "trace_id": event.get("trace_id"),
                    "request_id": event.get("request_id"),
                    "tenant_id": event.get("tenant_id"),
                    "user_id": event.get("user_id"),
                    "order_id": event.get("order_id"),
                    "event_type": event.get("event_type"),
                    "action": event.get("action"),
                    "actor_type": event.get("actor_type"),
                    "status": event.get("status"),
                    "created_at": event.get("created_at"),
                    "summary": event.get("summary"),
                    "metadata_json": _json(event.get("metadata")),
                },
            )

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        self.init()
        with self._engine.connect() as conn:
            run = conn.execute(
                text("SELECT * FROM trace_runs WHERE trace_id=:trace_id"),
                {"trace_id": trace_id},
            ).mappings().first()
            if run is None:
                return None
            steps = conn.execute(
                text("SELECT * FROM trace_steps WHERE trace_id=:trace_id ORDER BY order_index ASC"),
                {"trace_id": trace_id},
            ).mappings().all()
        return self._hydrate_trace(run, steps)

    def list_traces(
        self,
        *,
        order_id: str | None = None,
        status: str | None = None,
        tool_name: str | None = None,
        route: str | None = None,
        step_type: str | None = None,
        model_id: str | None = None,
        prompt_id: str | None = None,
        min_duration_ms: float | None = None,
        started_from: str | None = None,
        started_to: str | None = None,
        error: str | None = None,
        limit: int = 50,
        include_internal: bool = False,
    ) -> list[dict[str, Any]]:
        self.init()
        clauses: list[str] = []
        params: dict[str, Any] = {"limit": max(1, min(limit, 200))}
        if not include_internal:
            for index, prefix in enumerate(INTERNAL_ROUTE_PREFIXES):
                key = f"internal_prefix_{index}"
                clauses.append(f"(r.route IS NULL OR r.route NOT LIKE :{key})")
                params[key] = f"{prefix}%"
            clauses.append(
                "(r.order_id IS NOT NULL OR EXISTS (SELECT 1 FROM trace_steps bs WHERE bs.trace_id = r.trace_id))"
            )
        if order_id:
            clauses.append("r.order_id=:order_id")
            params["order_id"] = order_id
        if status:
            clauses.append("r.status=:status")
            params["status"] = status
        if route:
            clauses.append("r.route LIKE :route")
            params["route"] = f"%{route}%"
        if tool_name:
            clauses.append(
                "EXISTS (SELECT 1 FROM trace_steps s WHERE s.trace_id = r.trace_id AND s.name=:tool_name)"
            )
            params["tool_name"] = tool_name
        if step_type:
            clauses.append(
                "EXISTS (SELECT 1 FROM trace_steps s WHERE s.trace_id = r.trace_id AND s.type=:step_type)"
            )
            params["step_type"] = step_type
        if model_id:
            clauses.append(
                "EXISTS (SELECT 1 FROM trace_steps s WHERE s.trace_id = r.trace_id AND s.metadata_json::text LIKE :model_id)"
            )
            params["model_id"] = f"%{model_id}%"
        if prompt_id:
            clauses.append(
                "EXISTS (SELECT 1 FROM trace_steps s WHERE s.trace_id = r.trace_id AND s.metadata_json::text LIKE :prompt_id)"
            )
            params["prompt_id"] = f"%{prompt_id}%"
        if min_duration_ms is not None:
            clauses.append("r.duration_ms >= :min_duration_ms")
            params["min_duration_ms"] = float(min_duration_ms)
        if started_from:
            clauses.append("r.started_at >= :started_from")
            params["started_from"] = started_from
        if started_to:
            clauses.append("r.started_at <= :started_to")
            params["started_to"] = started_to
        if error:
            clauses.append("(r.error_code LIKE :error OR r.error_message LIKE :error)")
            params["error"] = f"%{error}%"
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                    SELECT r.*, COUNT(s.id) AS step_count
                    FROM trace_runs r
                    LEFT JOIN trace_steps s ON s.trace_id = r.trace_id
                    {where_sql}
                    GROUP BY r.trace_id
                    ORDER BY r.started_at DESC
                    LIMIT :limit
                """),
                params,
            ).mappings().all()
        return [self._row_to_summary(row) for row in rows]

    def list_audit_events(
        self,
        *,
        trace_id: str | None = None,
        order_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        self.init()
        clauses: list[str] = []
        params: dict[str, Any] = {"limit": max(1, min(limit, 200))}
        if trace_id:
            clauses.append("trace_id=:trace_id")
            params["trace_id"] = trace_id
        if order_id:
            clauses.append("order_id=:order_id")
            params["order_id"] = order_id
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT * FROM audit_events {where_sql} ORDER BY created_at DESC LIMIT :limit"),
                params,
            ).mappings().all()
        return [self._audit_row_to_dict(row) for row in rows]

    def _hydrate_trace(self, run, steps) -> dict[str, Any]:
        data = self._row_to_summary(run)
        data["metadata"] = _decode(run["metadata_json"], {})
        data["steps"] = [
            {
                "id": step["id"],
                "parent_id": step["parent_id"],
                "type": step["type"],
                "name": step["name"],
                "status": step["status"],
                "started_at": step["started_at"],
                "ended_at": step["ended_at"],
                "duration_ms": step["duration_ms"],
                "summary": step["summary"],
                "error_code": step["error_code"],
                "error_message": step["error_message"],
                "input_summary": step["input_summary"],
                "output_summary": step["output_summary"],
                "metadata": _decode(step["metadata_json"], {}),
                "evidence": _decode(step["evidence_json"], []),
            }
            for step in steps
        ]
        return data

    def _row_to_summary(self, row) -> dict[str, Any]:
        return {
            "trace_id": row["trace_id"],
            "request_id": row["request_id"],
            "session_id": row["session_id"],
            "tenant_id": row["tenant_id"],
            "user_id": row["user_id"],
            "order_id": row["order_id"],
            "route": row["route"],
            "status": row["status"],
            "error_code": row["error_code"],
            "error_message": row["error_message"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "duration_ms": row["duration_ms"],
            "step_count": row.get("step_count") if hasattr(row, "get") else None,
        }

    def _audit_row_to_dict(self, row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "trace_id": row["trace_id"],
            "request_id": row["request_id"],
            "tenant_id": row["tenant_id"],
            "user_id": row["user_id"],
            "order_id": row["order_id"],
            "event_type": row["event_type"],
            "action": row["action"],
            "actor_type": row["actor_type"],
            "status": row["status"],
            "created_at": row["created_at"],
            "summary": row["summary"],
            "metadata": _decode(row["metadata_json"], {}),
        }


_STORE: PostgreSQLTraceStore | None = None


def get_trace_store() -> PostgreSQLTraceStore:
    global _STORE
    if _STORE is None:
        _STORE = PostgreSQLTraceStore(get_settings().effective_database_url)
    return _STORE

