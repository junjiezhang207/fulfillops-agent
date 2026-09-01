"""Hybrid/Workflow 使用的短期结构化 Session Memory。

设计边界：
  - 保存最近 1-2 轮有效对话和运营明确表达的偏好、约束、反馈、指代。
  - 不保存库存数量、订单状态、物流 ETA、仓库实时状态等业务事实。
  - 这些实时业务数据必须每轮由 OMS/WMS/TMS/ERP/PIM/CRM 适配器重新读取。

生产环境把 Store 写入 PostgreSQL；当前进程仍保留线程安全热缓存，保持本地可跑。
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Protocol
from threading import RLock

from app.schemas.session_memory import (
    RecentMessage,
    SessionMemoryPatch,
    SessionMemorySnapshot,
    StructuredSessionMemory,
)

logger = logging.getLogger(__name__)

RECENT_MESSAGE_LIMIT = 4
MEMORY_TOKEN_BUDGET = 800
FORBIDDEN_BUSINESS_MEMORY_KEYS = {
    "current_inventory",
    "inventory",
    "stock",
    "available_stock",
    "current_order_status",
    "order_status",
    "shipping_eta",
    "eta",
    "warehouse_status",
    "inbound_stock",
    "logistics_status",
    "当前库存",
    "库存数量",
    "订单状态",
    "物流eta",
    "物流 eta",
    "仓库状态",
    "在途库存",
}
REALTIME_FACT_PLACEHOLDER = "[实时业务事实已过滤]"
REALTIME_FACT_PATTERNS = (
    re.compile(r"(?:当前)?库存(?:只有|为|是|剩余|不足)?\s*[:：=]?\s*\d+\s*(?:件|个|pcs|units)?", re.IGNORECASE),
    re.compile(r"\b(?:available_stock|current_inventory|inventory|stock)\s*[:=]\s*\d+\b", re.IGNORECASE),
    re.compile(r"(?:订单状态|current_order_status|order_status)\s*[:：=]\s*[\w\u4e00-\u9fff_-]+", re.IGNORECASE),
    re.compile(r"(?:物流\s*ETA|物流eta|shipping_eta|\beta\b)\s*[:：=]\s*[\w\u4e00-\u9fff:：\-年月日小时天 ]+", re.IGNORECASE),
    re.compile(r"(?:仓库状态|warehouse_status|logistics_status)\s*[:：=]\s*[\w\u4e00-\u9fff_-]+", re.IGNORECASE),
    re.compile(r"(?:在途库存|inbound_stock)\s*[:：=]?\s*\d+\s*(?:件|个|pcs|units)?", re.IGNORECASE),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact_message(text: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


class MemoryPatchExtractor(Protocol):
    """Extract an incremental session-memory patch from the latest user text."""

    def extract(self, message: str) -> SessionMemoryPatch: ...


class GatewayMemoryPatchExtractor:
    """Model Gateway backed structured-output extractor.

    The LLM is only allowed to return a ``SessionMemoryPatch``. Realtime business
    facts are filtered again by ``SessionMemoryService`` before storage.
    """

    def __init__(self, chat_model: Any | None = None) -> None:
        self._chat_model = chat_model
        self._chain: Any | None = None
        self._initialized = False

    def extract(self, message: str) -> SessionMemoryPatch:
        chain = self._get_chain()
        if chain is None:
            return SessionMemoryPatch()
        result = chain.invoke({"message": message})
        return self._coerce_patch(result)

    def _get_chain(self) -> Any | None:
        if self._initialized:
            return self._chain
        self._initialized = True
        try:
            from langchain_core.prompts import ChatPromptTemplate

            from app.infrastructure.llm.model_gateway import get_model_gateway

            gateway = get_model_gateway()
            chat_model = self._chat_model or gateway.create_chat_model(use_case="memory_extraction")
            if chat_model is None:
                return None
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", gateway.prompt_system(use_case="memory_extraction")),
                    (
                        "human",
                        "只从下面用户最新消息中抽取增量 SessionMemoryPatch。"
                        "不要抽取库存数量、订单状态、物流 ETA、仓库状态或在途库存。\n"
                        "用户消息：{message}",
                    ),
                ]
            )
            self._chain = prompt | chat_model.with_structured_output(SessionMemoryPatch)
        except Exception as exc:
            logger.warning("Session Memory 模型抽取器不可用，回退规则抽取：%s", exc)
            self._chain = None
        return self._chain

    @staticmethod
    def _coerce_patch(result: Any) -> SessionMemoryPatch:
        if isinstance(result, SessionMemoryPatch):
            return result
        if isinstance(result, dict):
            return SessionMemoryPatch.model_validate(result)
        if hasattr(result, "model_dump"):
            return SessionMemoryPatch.model_validate(result.model_dump())
        return SessionMemoryPatch.model_validate(result)

@dataclass
class SessionContext:
    """会话上下文。

    order_analysis_cache 保留旧 Hybrid 路由缓存语义；structured_memory 是本次
    设计文档要求的短期结构化记忆。
    """

    # 会话 ID，通常来自 thread_id。
    thread_id: str  # 会话 ID
    # 当前会话正在处理的订单。
    order_id: str  # 当前订单 ID
    # 可放订单分析结果，减少同一轮路由重复分析。
    order_analysis_cache: dict = field(default_factory=dict)  # 订单分析缓存
    # 用户临时偏好，例如更关注时效/成本。保留旧字段兼容调用方。
    user_preferences: dict = field(default_factory=dict)  # 用户偏好
    recent_messages: list[RecentMessage] = field(default_factory=list)
    structured_memory: StructuredSessionMemory = field(default_factory=StructuredSessionMemory)
    last_memory_patch: dict = field(default_factory=dict)
    memory_updated_at: str | None = None
    # Hybrid 会话轮次计数。
    conversation_turns: int = 0  # 对话轮数
    # 创建时间和最后访问时间，用于过期判断。
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed_at: datetime = field(default_factory=datetime.now)
    # 预估过期时间，便于前端/调试展示。
    expires_at: Optional[datetime] = None

    def update_last_accessed(self):
        """更新最后访问时间。"""
        self.last_accessed_at = datetime.now()

    def is_expired(self, ttl: timedelta) -> bool:
        """检查会话是否已过期。"""
        return datetime.now() > (self.last_accessed_at + ttl)


class SessionMemoryService:
    """短期 Session Memory 服务。

    特点：
      - 完全内存存储，无外部依赖
      - 支持自动过期清理
      - 线程安全（使用 RLock）
      - 只做增量更新，不让模型每轮重写完整 Memory
    """

    def __init__(
        self,
        ttl_hours: int = 2,
        database_url: str | None = None,
        memory_extractor: MemoryPatchExtractor | None = None,
        enable_model_extractor: bool = False,
    ):
        """初始化会话缓存。

        Args:
            ttl_hours: 会话生命周期（小时）
        """
        # thread_id/case_id -> SessionContext。本地热缓存，PostgreSQL 是持久化后端。
        self.sessions = {}  # thread_id → SessionContext
        # 会话多久不访问就过期。
        self.ttl = timedelta(hours=ttl_hours)
        # 多请求并发读写内存 dict 时用锁保护。
        self.lock = RLock()  # 线程锁
        self._database_url = database_url or ""
        self._engine = self._connect_postgres(database_url)
        self.memory_extractor = memory_extractor
        self.enable_model_extractor = enable_model_extractor

    def get_session(self, thread_id: str) -> Optional[SessionContext]:
        """获取会话。

        Args:
            thread_id: 会话 ID

        Returns:
            会话上下文，如果不存在或已过期则返回 None
        """
        with self.lock:
            session = self.sessions.get(thread_id)
            if session is None:
                session = self._load_from_postgres(thread_id)
                if session is not None:
                    self.sessions[thread_id] = session
            if session is None:
                return None

            # 检查是否过期；过期会话直接删除，避免内存积累。
            if session.is_expired(self.ttl):
                del self.sessions[thread_id]
                return None

            # 更新最后访问时间，相当于滑动过期时间。
            session.update_last_accessed()
            self._save_to_postgres(session)
            return session

    def get_or_create_session(self, thread_id: str, order_id: str = "") -> SessionContext:
        with self.lock:
            session = self.get_session(thread_id)
            if session is None:
                session = SessionContext(thread_id=thread_id, order_id=order_id)
            elif order_id:
                session.order_id = order_id
            self.set_session(session)
            return session

    def set_session(self, session: SessionContext) -> None:
        """保存或更新会话。

        Args:
            session: 会话上下文
        """
        with self.lock:
            # 保存时刷新最后访问时间，并计算 expires_at 供展示。
            session.update_last_accessed()
            session.expires_at = datetime.now() + self.ttl
            self.sessions[session.thread_id] = session
            self._save_to_postgres(session)

    def update_from_user_message(
        self,
        thread_id: str,
        order_id: str,
        message: str,
    ) -> SessionMemorySnapshot:
        """从用户当前输入中抽取增量记忆并返回快照。

        当前实现是确定性抽取 fallback；模型网关 use_case 保存在快照中，后续接
        低成本模型时可替换 `_extract_patch`，其余存储和校验逻辑无需变化。
        """
        patch = self.extract_memory_patch(message)
        with self.lock:
            session = self.sessions.get(thread_id)
            if session is None:
                session = self._load_from_postgres(thread_id)
            if session is None or session.is_expired(self.ttl):
                session = SessionContext(thread_id=thread_id, order_id=order_id)
            elif order_id:
                session.order_id = order_id

            self._append_recent_message(session, role="user", content=message)
            self._apply_patch(session, patch)
            session.last_memory_patch = patch.model_dump()
            session.memory_updated_at = _utc_now()
            self.set_session(session)
            return self.snapshot(thread_id)

    def extract_memory_patch(self, message: str) -> SessionMemoryPatch:
        """Extract, validate and sanitize an incremental memory patch.

        The preferred path is Model Gateway ``use_case=memory_extraction`` with
        Pydantic structured output. A deterministic extractor remains the local
        fallback so memory governance does not depend on model availability.
        """

        model_patch = SessionMemoryPatch()
        extractor = self.memory_extractor
        if extractor is None and self.enable_model_extractor:
            extractor = GatewayMemoryPatchExtractor()
            self.memory_extractor = extractor
        if extractor is not None:
            try:
                model_patch = extractor.extract(message)
            except Exception as exc:
                logger.warning("Session Memory 模型抽取失败，回退规则抽取：%s", exc)
        rule_patch = self._extract_rule_patch(message)
        patch = model_patch if self._has_patch_content(model_patch) else rule_patch
        return self._sanitize_patch(patch)

    def record_assistant_summary(self, thread_id: str, summary: str) -> None:
        with self.lock:
            session = self.sessions.get(thread_id)
            if session is None:
                session = self._load_from_postgres(thread_id)
                if session is None:
                    return
                self.sessions[thread_id] = session
            self._append_recent_message(session, role="assistant", content=summary)
            session.memory_updated_at = _utc_now()
            self.set_session(session)

    def snapshot(self, thread_id: str) -> SessionMemorySnapshot:
        with self.lock:
            session = self.sessions.get(thread_id)
            if session is None:
                session = self._load_from_postgres(thread_id)
                if session is not None:
                    self.sessions[thread_id] = session
            if session is None:
                return SessionMemorySnapshot(thread_id=thread_id)
            return SessionMemorySnapshot(
                thread_id=session.thread_id,
                order_id=session.order_id,
                recent_messages=list(session.recent_messages[-RECENT_MESSAGE_LIMIT:]),
                structured=session.structured_memory,
                token_budget=MEMORY_TOKEN_BUDGET,
                updated_at=session.memory_updated_at,
            )

    @staticmethod
    def _append_recent_message(session: SessionContext, *, role: str, content: str) -> None:
        cleaned = _compact_message(SessionMemoryService._sanitize_recent_message(content))
        if not cleaned:
            return
        session.recent_messages.append(
            RecentMessage(role=role, content=cleaned, created_at=_utc_now())
        )
        session.recent_messages = session.recent_messages[-RECENT_MESSAGE_LIMIT:]

    @staticmethod
    def _apply_patch(session: SessionContext, patch: SessionMemoryPatch) -> None:
        structured = session.structured_memory
        if patch.current_topic:
            structured.current_topic = patch.current_topic
        structured.user_preferences.update(patch.user_preferences)
        structured.confirmed_constraints.update(patch.confirmed_constraints)
        structured.plan_feedback.update(patch.plan_feedback)
        structured.references.update(patch.references)
        session.user_preferences.update(patch.user_preferences)

    @staticmethod
    def _extract_rule_patch(message: str) -> SessionMemoryPatch:
        text = (message or "").strip()
        lowered = text.lower()
        patch = SessionMemoryPatch()

        topic_map = [
            ("库存调拨/跨仓调货", ("调拨", "跨仓调货", "跨仓")),
            ("拆单/合单履约", ("拆单", "合单")),
            ("物流渠道变更", ("物流渠道", "渠道变更", "快递", "承运商")),
            ("缺货处置", ("缺货", "库存不足")),
            ("换仓履约", ("换仓", "换到", "改仓")),
        ]
        for topic, markers in topic_map:
            if any(marker in text for marker in markers):
                patch.current_topic = topic
                patch.extracted_signals.append(f"topic:{topic}")
                break

        if any(marker in text for marker in ("优先保证时效", "时效优先", "最快", "加急")):
            patch.user_preferences["priority"] = "speed"
            patch.extracted_signals.append("preference:priority=speed")
        if any(marker in text for marker in ("成本优先", "降低成本", "省运费", "便宜")):
            patch.user_preferences["priority"] = "cost"
            patch.extracted_signals.append("preference:priority=cost")
        if any(marker in text for marker in ("少包裹", "减少包裹", "尽量合单")):
            patch.user_preferences["package_preference"] = "fewer_packages"
            patch.extracted_signals.append("preference:package=fewer_packages")

        if any(marker in text for marker in ("可以接受拆单", "接受拆单", "允许拆单")):
            patch.confirmed_constraints["split_order_accepted"] = True
            patch.extracted_signals.append("constraint:split_order_accepted=true")
        if any(marker in text for marker in ("不接受拆单", "不要拆单", "不能拆单")):
            patch.confirmed_constraints["split_order_accepted"] = False
            patch.extracted_signals.append("constraint:split_order_accepted=false")

        rejected_option = re.search(r"第([一二三四五六七八九十\d]+)个方案不要", text)
        if rejected_option:
            option = rejected_option.group(1)
            patch.plan_feedback["latest_feedback"] = f"拒绝第{option}个方案"
            patch.plan_feedback["rejected_option"] = option
            patch.references["last_referenced_plan_option"] = option
            patch.extracted_signals.append(f"feedback:reject_option={option}")
        elif any(marker in text for marker in ("方案不要", "拒绝这个方案", "不采纳")):
            patch.plan_feedback["latest_feedback"] = "拒绝当前方案"
            patch.extracted_signals.append("feedback:reject_current")

        referenced_option = re.search(r"第([一二三四五六七八九十\d]+)个方案", text)
        if referenced_option:
            patch.references["last_referenced_plan_option"] = referenced_option.group(1)
        if "那个仓" in text or "这个仓" in text:
            patch.references["warehouse_reference"] = "last_discussed_warehouse"
        if "modify" in lowered or "修改" in text or "调整" in text:
            patch.plan_feedback["requested_modification"] = _compact_message(text, 120)

        return patch

    @staticmethod
    def _has_patch_content(patch: SessionMemoryPatch) -> bool:
        return bool(
            patch.current_topic
            or patch.user_preferences
            or patch.confirmed_constraints
            or patch.plan_feedback
            or patch.references
            or patch.extracted_signals
        )

    @staticmethod
    def _sanitize_patch(patch: SessionMemoryPatch) -> SessionMemoryPatch:
        """Drop realtime business facts before they can enter Session Memory."""

        return SessionMemoryPatch(
            current_topic=patch.current_topic,
            user_preferences=SessionMemoryService._sanitize_memory_dict(patch.user_preferences),
            confirmed_constraints=SessionMemoryService._sanitize_memory_dict(patch.confirmed_constraints),
            plan_feedback=SessionMemoryService._sanitize_memory_dict(patch.plan_feedback),
            references=SessionMemoryService._sanitize_memory_dict(patch.references),
            extracted_signals=[
                signal
                for signal in patch.extracted_signals
                if not SessionMemoryService._contains_forbidden_business_key(signal)
            ][:20],
        )

    @staticmethod
    def _sanitize_recent_message(content: str) -> str:
        """Keep conversational continuity without storing realtime business facts."""

        text = content or ""
        for pattern in REALTIME_FACT_PATTERNS:
            text = pattern.sub(REALTIME_FACT_PLACEHOLDER, text)
        text = re.sub(rf"(?:{re.escape(REALTIME_FACT_PLACEHOLDER)}[\s,，。；;]*)+", f"{REALTIME_FACT_PLACEHOLDER}，", text)
        return text.strip(" ，,;；")

    @staticmethod
    def _sanitize_memory_dict(payload: dict[str, object]) -> dict[str, object]:
        clean: dict[str, object] = {}
        for key, value in payload.items():
            if SessionMemoryService._contains_forbidden_business_key(str(key)):
                continue
            if isinstance(value, dict):
                nested = SessionMemoryService._sanitize_memory_dict(value)  # type: ignore[arg-type]
                if nested:
                    clean[str(key)] = nested
                continue
            if isinstance(value, list):
                clean[str(key)] = [
                    item
                    for item in value[:20]
                    if not SessionMemoryService._contains_forbidden_business_key(str(item))
                ]
                continue
            if SessionMemoryService._contains_forbidden_business_key(str(value)):
                continue
            clean[str(key)] = value
        return clean

    @staticmethod
    def _contains_forbidden_business_key(text: str) -> bool:
        normalized = text.strip().lower()
        return any(key in normalized for key in FORBIDDEN_BUSINESS_MEMORY_KEYS)

    def delete_session(self, thread_id: str) -> None:
        """删除会话。

        Args:
            thread_id: 会话 ID
        """
        with self.lock:
            # 删除不存在的会话不报错，方便调用方幂等清理。
            if thread_id in self.sessions:
                del self.sessions[thread_id]
            self._delete_from_postgres(thread_id)

    def clear_expired(self) -> int:
        """清理所有过期会话。

        Returns:
            被清理的会话数量
        """
        with self.lock:
            # 先收集过期 ID，避免遍历 dict 时删除导致错误。
            expired_ids = [
                tid
                for tid, session in self.sessions.items()
                if session.is_expired(self.ttl)
            ]
            for tid in expired_ids:
                del self.sessions[tid]
            return len(expired_ids)

    def get_stats(self) -> dict:
        """获取会话统计信息。

        Returns:
            统计信息字典
        """
        with self.lock:
            self.clear_expired()  # 先清理过期
            # 返回简化统计，不暴露完整缓存内容。
            return {
                "active_sessions": len(self.sessions),
                "ttl_hours": self.ttl.total_seconds() / 3600,
                "postgres_enabled": self._engine is not None,
                "sessions": [
                    {
                        "thread_id": sid,
                        "order_id": s.order_id,
                        "turns": s.conversation_turns,
                        "current_topic": s.structured_memory.current_topic,
                        "created_at": s.created_at.isoformat(),
                        "last_accessed_at": s.last_accessed_at.isoformat(),
                    }
                    for sid, s in self.sessions.items()
                ],
            }

    @staticmethod
    def _connect_postgres(database_url: str | None):
        if not database_url:
            return None
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=1800, future=True)
            with engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS session_memory_snapshots (
                        thread_id VARCHAR(160) PRIMARY KEY,
                        payload_json JSONB NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        expires_at TIMESTAMP NOT NULL
                    )
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_session_memory_expires
                    ON session_memory_snapshots (expires_at)
                """))
            return engine
        except Exception as exc:
            logger.warning("Session Memory PostgreSQL 不可用，使用进程内热缓存：%s", exc)
            return None

    def _save_to_postgres(self, session: SessionContext) -> None:
        if self._engine is None:
            return
        try:
            from sqlalchemy import text

            payload = self._session_to_dict(session)
            with self._engine.begin() as conn:
                conn.execute(
                    text("DELETE FROM session_memory_snapshots WHERE expires_at <= CURRENT_TIMESTAMP")
                )
                conn.execute(
                    text("""
                        INSERT INTO session_memory_snapshots (
                            thread_id, payload_json, updated_at, expires_at
                        ) VALUES (
                            :thread_id, :payload_json, :updated_at, :expires_at
                        )
                        ON CONFLICT (thread_id) DO UPDATE SET
                            payload_json=EXCLUDED.payload_json,
                            updated_at=EXCLUDED.updated_at,
                            expires_at=EXCLUDED.expires_at
                    """),
                    {
                        "thread_id": session.thread_id,
                        "payload_json": json.dumps(payload, ensure_ascii=False),
                        "updated_at": datetime.now(),
                        "expires_at": session.expires_at or (datetime.now() + self.ttl),
                    },
                )
        except Exception as exc:
            logger.warning("Session Memory 写 PostgreSQL 失败：%s", exc)

    def _load_from_postgres(self, thread_id: str) -> SessionContext | None:
        if self._engine is None:
            return None
        try:
            from sqlalchemy import text

            with self._engine.begin() as conn:
                conn.execute(text("DELETE FROM session_memory_snapshots WHERE expires_at <= CURRENT_TIMESTAMP"))
                row = conn.execute(
                    text("""
                        SELECT payload_json
                        FROM session_memory_snapshots
                        WHERE thread_id=:thread_id AND expires_at > CURRENT_TIMESTAMP
                    """),
                    {"thread_id": thread_id},
                ).mappings().first()
            if row is None:
                return None
            payload = row["payload_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            return self._session_from_dict(payload)
        except Exception as exc:
            logger.warning("Session Memory 读 PostgreSQL 失败：%s", exc)
            return None

    def _delete_from_postgres(self, thread_id: str) -> None:
        if self._engine is None:
            return
        try:
            from sqlalchemy import text

            with self._engine.begin() as conn:
                conn.execute(
                    text("DELETE FROM session_memory_snapshots WHERE thread_id=:thread_id"),
                    {"thread_id": thread_id},
                )
        except Exception:
            return

    @staticmethod
    def _session_to_dict(session: SessionContext) -> dict:
        return {
            "thread_id": session.thread_id,
            "order_id": session.order_id,
            "order_analysis_cache": session.order_analysis_cache,
            "user_preferences": session.user_preferences,
            "conversation_turns": session.conversation_turns,
            "created_at": session.created_at.isoformat(),
            "last_accessed_at": session.last_accessed_at.isoformat(),
            "expires_at": session.expires_at.isoformat() if session.expires_at else None,
            "recent_messages": [item.model_dump(mode="json") for item in session.recent_messages],
            "structured_memory": session.structured_memory.model_dump(mode="json"),
            "last_memory_patch": session.last_memory_patch,
            "memory_updated_at": session.memory_updated_at,
        }

    @staticmethod
    def _session_from_dict(payload: dict) -> SessionContext:
        session = SessionContext(
            thread_id=str(payload.get("thread_id") or ""),
            order_id=str(payload.get("order_id") or ""),
            order_analysis_cache=dict(payload.get("order_analysis_cache") or {}),
            user_preferences=dict(payload.get("user_preferences") or {}),
            conversation_turns=int(payload.get("conversation_turns") or 0),
            created_at=datetime.fromisoformat(payload["created_at"]) if payload.get("created_at") else datetime.now(),
            last_accessed_at=datetime.fromisoformat(payload["last_accessed_at"]) if payload.get("last_accessed_at") else datetime.now(),
            expires_at=datetime.fromisoformat(payload["expires_at"]) if payload.get("expires_at") else None,
            recent_messages=[
                RecentMessage(**item)
                for item in list(payload.get("recent_messages") or [])[-RECENT_MESSAGE_LIMIT:]
                if isinstance(item, dict)
            ],
            structured_memory=StructuredSessionMemory(**dict(payload.get("structured_memory") or {})),
            last_memory_patch=dict(payload.get("last_memory_patch") or {}),
            memory_updated_at=payload.get("memory_updated_at"),
        )
        return session


# 全局单例
_session_service: Optional[SessionMemoryService] = None


def get_session_service() -> SessionMemoryService:
    """获取全局会话服务实例。"""
    global _session_service
    if _session_service is None:
        from app.core.config import get_settings

        settings = get_settings()
        ttl_hours = max(1, int(getattr(settings, "short_term_memory_ttl_seconds", 7200)) // 3600)
        _session_service = SessionMemoryService(
            ttl_hours=ttl_hours,
            database_url=getattr(settings, "effective_database_url", None),
            enable_model_extractor=bool(
                getattr(settings, "short_term_memory_model_extraction_enabled", False)
            ),
        )
    return _session_service
