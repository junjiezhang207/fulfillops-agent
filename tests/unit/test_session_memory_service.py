"""短期结构化 Session Memory 回归测试。"""

from app.application.memory.session_memory_service import SessionMemoryService
from app.schemas.session_memory import SessionMemoryPatch


class FakePostgresResult:
    def __init__(self, row=None):
        self.row = row

    def mappings(self):
        return self

    def first(self):
        return self.row


class FakePostgresConn:
    def __init__(self, store):
        self.store = store

    def execute(self, statement, params=None):
        sql = str(statement)
        params = params or {}
        if "INSERT INTO session_memory_snapshots" in sql:
            self.store[params["thread_id"]] = params["payload_json"]
        if "SELECT payload_json" in sql:
            payload = self.store.get(params["thread_id"])
            return FakePostgresResult({"payload_json": payload} if payload else None)
        if "DELETE FROM session_memory_snapshots WHERE thread_id" in sql:
            self.store.pop(params["thread_id"], None)
        return FakePostgresResult()


class FakePostgresBegin:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        return False


class FakePostgresEngine:
    def __init__(self):
        self.values = {}
        self.conn = FakePostgresConn(self.values)

    def begin(self):
        return FakePostgresBegin(self.conn)


class FakeModelExtractor:
    def __init__(self, patch):
        self.patch = patch
        self.calls = []

    def extract(self, message: str):
        self.calls.append(message)
        return self.patch


class FailingModelExtractor:
    def extract(self, message: str):
        raise RuntimeError("model gateway timeout")


def test_session_memory_extracts_incremental_user_preferences_and_constraints():
    service = SessionMemoryService(ttl_hours=1)

    first = service.update_from_user_message(
        thread_id="case-1",
        order_id="SO-1",
        message="还是优先保证时效，可以接受拆单。",
    )

    assert first.structured.current_topic == "拆单/合单履约"
    assert first.structured.user_preferences["priority"] == "speed"
    assert first.structured.confirmed_constraints["split_order_accepted"] is True
    assert first.memory_use_case == "memory_extraction"

    second = service.update_from_user_message(
        thread_id="case-1",
        order_id="SO-1",
        message="第二个方案不要，现在不接受拆单。",
    )

    assert second.structured.confirmed_constraints["split_order_accepted"] is False
    assert second.structured.plan_feedback["rejected_option"] == "二"
    assert second.structured.references["last_referenced_plan_option"] == "二"
    assert len(second.recent_messages) == 2


def test_session_memory_uses_model_gateway_patch_when_configured():
    extractor = FakeModelExtractor(
        SessionMemoryPatch(
            current_topic="物流渠道变更",
            user_preferences={"priority": "speed"},
            confirmed_constraints={"split_order_accepted": True},
            plan_feedback={"latest_feedback": "第二个方案不要"},
            references={"last_referenced_plan_option": "二"},
            extracted_signals=["model_gateway:memory_extraction"],
        )
    )
    service = SessionMemoryService(ttl_hours=1, memory_extractor=extractor)

    snapshot = service.update_from_user_message(
        thread_id="case-model",
        order_id="SO-MODEL",
        message="还是时效优先，第二个方案不要。",
    )

    assert extractor.calls == ["还是时效优先，第二个方案不要。"]
    assert snapshot.structured.current_topic == "物流渠道变更"
    assert snapshot.structured.user_preferences["priority"] == "speed"
    assert snapshot.structured.confirmed_constraints["split_order_accepted"] is True
    assert snapshot.structured.references["last_referenced_plan_option"] == "二"
    assert snapshot.memory_use_case == "memory_extraction"


def test_session_memory_filters_realtime_business_facts_from_model_patch():
    service = SessionMemoryService(
        ttl_hours=1,
        memory_extractor=FakeModelExtractor(
            SessionMemoryPatch(
                user_preferences={
                    "priority": "speed",
                    "available_stock": 3,
                    "note": "当前库存 3 件",
                },
                confirmed_constraints={"shipping_eta": "2026-08-31"},
                references={"warehouse_reference": "last_discussed_warehouse"},
                extracted_signals=["preference:priority=speed", "inventory:available_stock=3"],
            )
        ),
    )

    snapshot = service.update_from_user_message(
        thread_id="case-filter",
        order_id="SO-FILTER",
        message="库存只有 3 件，但我的偏好还是时效优先。",
    )

    assert snapshot.structured.user_preferences == {"priority": "speed"}
    assert snapshot.structured.confirmed_constraints == {}
    assert snapshot.structured.references == {"warehouse_reference": "last_discussed_warehouse"}
    assert "库存只有 3 件" not in snapshot.recent_messages[0].content
    assert "实时业务事实已过滤" in snapshot.recent_messages[0].content
    assert "时效优先" in snapshot.recent_messages[0].content


def test_session_memory_falls_back_to_rule_patch_when_model_extraction_fails(caplog):
    service = SessionMemoryService(ttl_hours=1, memory_extractor=FailingModelExtractor())

    with caplog.at_level("WARNING"):
        snapshot = service.update_from_user_message(
            thread_id="case-fallback",
            order_id="SO-FALLBACK",
            message="可以接受拆单，还是优先保证时效。",
        )

    assert snapshot.structured.user_preferences["priority"] == "speed"
    assert snapshot.structured.confirmed_constraints["split_order_accepted"] is True
    assert any("模型抽取失败" in record.message for record in caplog.records)


def test_session_memory_keeps_recent_messages_bounded():
    service = SessionMemoryService(ttl_hours=1)

    for index in range(6):
        service.update_from_user_message(
            thread_id="case-2",
            order_id="SO-2",
            message=f"第{index}轮，优先保证时效。",
        )

    snapshot = service.snapshot("case-2")

    assert len(snapshot.recent_messages) == 4
    assert snapshot.recent_messages[0].content.startswith("第2轮")


def test_session_memory_can_restore_from_postgres_backend(monkeypatch):
    fake_postgres = FakePostgresEngine()
    monkeypatch.setattr(
        SessionMemoryService,
        "_connect_postgres",
        staticmethod(lambda _database_url: fake_postgres),
    )
    writer = SessionMemoryService(ttl_hours=1, database_url="postgresql+psycopg://unit-test")
    writer.update_from_user_message(
        thread_id="case-postgres",
        order_id="SO-POSTGRES",
        message="优先保证时效，可以接受拆单。",
    )

    reader = SessionMemoryService(ttl_hours=1, database_url="postgresql+psycopg://unit-test")
    snapshot = reader.snapshot("case-postgres")

    assert snapshot.order_id == "SO-POSTGRES"
    assert snapshot.structured.user_preferences["priority"] == "speed"
    restored = reader.get_session("case-postgres")
    assert restored is not None
    assert restored.order_id == "SO-POSTGRES"
    assert restored.structured_memory.user_preferences["priority"] == "speed"


def test_session_memory_records_assistant_summary_after_postgres_restore(monkeypatch):
    fake_postgres = FakePostgresEngine()
    monkeypatch.setattr(
        SessionMemoryService,
        "_connect_postgres",
        staticmethod(lambda _database_url: fake_postgres),
    )
    writer = SessionMemoryService(ttl_hours=1, database_url="postgresql+psycopg://unit-test")
    writer.update_from_user_message(
        thread_id="case-postgres-summary",
        order_id="SO-POSTGRES-SUMMARY",
        message="还是优先保证时效。",
    )

    reader = SessionMemoryService(ttl_hours=1, database_url="postgresql+psycopg://unit-test")
    reader.record_assistant_summary("case-postgres-summary", "已生成时效优先的履约方案。")
    snapshot = reader.snapshot("case-postgres-summary")

    assert [item.role for item in snapshot.recent_messages] == ["user", "assistant"]
    assert snapshot.recent_messages[-1].content == "已生成时效优先的履约方案。"


def test_session_memory_sanitizes_assistant_summary_realtime_facts():
    service = SessionMemoryService(ttl_hours=1)
    service.update_from_user_message(
        thread_id="case-assistant-filter",
        order_id="SO-ASSISTANT-FILTER",
        message="时效优先。",
    )

    service.record_assistant_summary(
        "case-assistant-filter",
        "库存=3，ETA:24h，建议继续时效优先方案。",
    )
    snapshot = service.snapshot("case-assistant-filter")

    assistant_message = snapshot.recent_messages[-1].content
    assert "库存=3" not in assistant_message
    assert "ETA:24h" not in assistant_message
    assert "实时业务事实已过滤" in assistant_message
    assert "时效优先方案" in assistant_message
