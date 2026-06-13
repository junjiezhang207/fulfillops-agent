from datetime import datetime, timezone

from langgraph.store.base import Item, SearchItem

from app.memory import MemoryGovernanceService


class InMemoryLongTermMemoryStore:
    def __init__(self):
        self._items = {}

    def put(self, namespace, key, value, index=None, *, ttl=None):
        self._items[(tuple(namespace), key)] = {
            "namespace": tuple(namespace),
            "key": key,
            "value": dict(value),
            "created_at": datetime.now(tz=timezone.utc),
            "updated_at": datetime.now(tz=timezone.utc),
        }

    def get(self, namespace, key, *, refresh_ttl=None):
        raw = self._items.get((tuple(namespace), key))
        if raw is None:
            return None
        return Item(
            namespace=raw["namespace"],
            key=raw["key"],
            value=dict(raw["value"]),
            created_at=raw["created_at"],
            updated_at=raw["updated_at"],
        )

    def search(self, namespace_prefix, /, *, query=None, filter=None, limit=10, offset=0, refresh_ttl=None):
        prefix = tuple(namespace_prefix)
        results = []
        for raw in self._items.values():
            if raw["namespace"][: len(prefix)] != prefix:
                continue
            value = dict(raw["value"])
            if filter and any(value.get(key) != expected for key, expected in filter.items()):
                continue
            results.append(
                SearchItem(
                    namespace=raw["namespace"],
                    key=raw["key"],
                    value=value,
                    created_at=raw["created_at"],
                    updated_at=raw["updated_at"],
                    score=1.0,
                )
            )
        return results[offset : offset + limit]


def test_govern_is_side_effect_free_when_conflict_detected():
    store = InMemoryLongTermMemoryStore()
    governance = MemoryGovernanceService()
    namespace = ("customers", "C-VIP-001", "preferences")

    governance.govern_and_write(
        store,
        namespace=namespace,
        key="pref-old",
        value={
            "memory_type": "user_preference",
            "preference": "客户接受替代 SKU，缺货时可以优先推荐替代方案。",
        },
    )

    decision = governance.govern(
        store,
        namespace=namespace,
        key="pref-new",
        value={
            "memory_type": "user_preference",
            "preference": "客户不接受替代 SKU，缺货时先人工确认。",
        },
    )

    old_item = store.get(namespace, "pref-old")
    assert decision.action == "supersede_old"
    assert decision.operations
    assert old_item is not None
    assert old_item.value["memory_status"] == "active"
    assert store.get(namespace, "pref-new") is None


def test_memory_governance_supersedes_conflicting_preference():
    store = InMemoryLongTermMemoryStore()
    governance = MemoryGovernanceService()
    namespace = ("customers", "C-VIP-001", "preferences")

    first = governance.govern_and_write(
        store,
        namespace=namespace,
        key="pref-old",
        value={
            "memory_type": "user_preference",
            "preference": "客户接受替代 SKU，缺货时可以优先推荐替代方案。",
        },
    )
    second = governance.govern_and_write(
        store,
        namespace=namespace,
        key="pref-new",
        value={
            "memory_type": "user_preference",
            "preference": "客户不接受替代 SKU，缺货时先人工确认。",
        },
    )

    old_item = store.get(namespace, "pref-old")
    new_item = store.get(namespace, "pref-new")

    assert first.action == "write_new"
    assert second.action == "supersede_old"
    assert second.conflicts
    assert old_item is not None
    assert old_item.value["memory_status"] == "superseded"
    assert old_item.value["preference"] == "客户接受替代 SKU，缺货时可以优先推荐替代方案。"
    assert new_item is not None
    assert new_item.value["memory_status"] == "active"
    assert new_item.value["_governance"]["conflict_detected"] is True
    assert new_item.value["_governance"]["resolution"] == "newer_or_higher_priority_memory_supersedes_old"


def test_memory_governance_merges_duplicate_preference_evidence():
    store = InMemoryLongTermMemoryStore()
    governance = MemoryGovernanceService()
    namespace = ("sessions", "s1", "preferences")

    governance.govern_and_write(
        store,
        namespace=namespace,
        key="pref-1",
        value={
            "memory_type": "user_preference",
            "preference": "用户优先时效，尽快发货。",
        },
    )
    decision = governance.govern_and_write(
        store,
        namespace=namespace,
        key="pref-2",
        value={
            "memory_type": "user_preference",
            "preference": "用户优先时效，尽快发货。",
        },
    )

    merged = store.get(namespace, "pref-1")
    assert decision.action == "merge_evidence"
    assert decision.reason == "duplicate_memory_confirmed"
    assert store.get(namespace, "pref-2") is None
    assert merged is not None
    assert merged.value["confirmation_count"] == 2
    assert "last_confirmed_at" in merged.value


def test_higher_priority_memory_conflict_requires_review():
    store = InMemoryLongTermMemoryStore()
    governance = MemoryGovernanceService()
    namespace = ("customers", "C-VIP-001", "preferences")

    governance.govern_and_write(
        store,
        namespace=namespace,
        key="rule-like-pref",
        value={
            "memory_type": "user_preference",
            "preference": "客户不接受替代 SKU，缺货时先人工确认。",
            "source_priority": 95,
        },
    )
    decision = governance.govern_and_write(
        store,
        namespace=namespace,
        key="lower-priority-pref",
        value={
            "memory_type": "user_preference",
            "preference": "客户接受替代 SKU，缺货时可以推荐替代方案。",
            "source_priority": 80,
        },
    )

    assert decision.action == "requires_review"
    assert decision.reason == "higher_priority_memory_conflict"
    assert store.get(namespace, "lower-priority-pref") is None
    assert store.get(namespace, "rule-like-pref").value["memory_status"] == "active"


def test_memory_governance_redacts_sensitive_values():
    store = InMemoryLongTermMemoryStore()
    governance = MemoryGovernanceService()
    namespace = ("orders", "SO1")

    decision = governance.govern_and_write(
        store,
        namespace=namespace,
        key="decision-1",
        value={
            "memory_type": "order_decision",
            "summary": "客户手机号 13812345678，邮箱 ops@example.com，密钥 sk-abcdefghijklmnopqrstuvwxyz。",
            "authorization": "Bearer secret-token",
        },
    )
    item = store.get(namespace, "decision-1")

    assert decision.action == "write_new"
    assert item is not None
    text = str(item.value)
    assert "13812345678" not in text
    assert "ops@example.com" not in text
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in text
    assert "authorization" not in item.value
    assert "[PHONE]" in text
    assert "[EMAIL]" in text
    assert "[SECRET_KEY]" in text
    assert set(item.value["_governance"]["safety_flags"]) >= {"phone", "email", "secret_key", "dropped_authorization"}
