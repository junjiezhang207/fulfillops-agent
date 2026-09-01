"""长期记忆治理层（企业级学习版）。

这个模块解决的是 Agent 项目里很容易被忽略的问题：
“记忆不是写进去就完了，记忆之间会冲突，也可能带有敏感信息。”

企业级设计里，记忆治理至少要分成两步：

1. 决策阶段：只判断，不修改数据库。
   例如判断这条新记忆是否安全、是否重复、是否和旧记忆冲突。

2. 应用阶段：根据决策统一写数据库。
   例如写入新记忆、合并重复记忆的证据、把旧记忆标记为 superseded。

为什么要这样拆？
如果 govern() 判断冲突时直接改旧记忆，而后面新记忆写入失败，就会出现：
旧记忆已经失效，新记忆却没写进去。生产系统不能接受这种半成功状态。

当前实现仍然保持轻量，但结构上按企业级方向设计：
- 安全脱敏
- 结构化事实抽取
- 作用域与优先级
- 冲突检测
- 重复记忆合并
- 写入操作计划
- 审计元数据
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol


MemoryAction = Literal[
    "write_new",
    "supersede_old",
    "merge_evidence",
    "skip_duplicate",
    "requires_review",
    "reject_unsafe",
]

MemoryOperationType = Literal["put_new", "supersede", "merge_evidence"]


_SENSITIVE_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]", "email"),
    (re.compile(r"\b1[3-9]\d{9}\b"), "[PHONE]", "phone"),
    (re.compile(r"\b\d{17}[\dXx]\b"), "[ID_CARD]", "id_card"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "[SECRET_KEY]", "secret_key"),
    (re.compile(r"\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+", re.I), "[SECRET]", "secret"),
)

_DROPPED_FIELDS = {"raw_prompt", "raw_messages", "authorization", "cookie"}


class MemoryStoreLike(Protocol):
    """长期记忆 Store 的最小接口。

    PostgreSQLPGVectorLongTermMemoryStore 符合这个接口。
    治理层只依赖 search/put，可以避免和具体存储实现强绑定。
    """

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[Any]:
        ...

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Any = None,
        *,
        ttl: Any = None,
    ) -> None:
        ...


@dataclass(frozen=True)
class MemoryConflict:
    """一条新记忆和旧记忆之间的冲突记录。"""

    conflict_type: str
    namespace: tuple[str, ...]
    key: str
    field: str
    old_value: Any
    new_value: Any
    old_priority: int
    new_priority: int
    resolution: str
    old_record: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.conflict_type,
            "namespace": self.namespace,
            "key": self.key,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "old_priority": self.old_priority,
            "new_priority": self.new_priority,
            "resolution": self.resolution,
        }


@dataclass(frozen=True)
class MemoryWriteOperation:
    """治理决策产生的数据库写操作。

    govern() 只返回这些操作，不执行它们。
    govern_and_write() 才会按顺序应用。
    """

    operation_type: MemoryOperationType
    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any]


@dataclass(frozen=True)
class GovernanceDecision:
    """一次治理判断的结果。"""

    action: MemoryAction
    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any]
    reason: str
    conflicts: list[dict[str, Any]]
    safety_flags: list[str]
    operations: list[MemoryWriteOperation] = field(default_factory=list)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _redact_text(text: str) -> tuple[str, list[str]]:
    """对单段文本做敏感信息脱敏。"""
    flags: list[str] = []
    redacted = text
    for pattern, replacement, flag in _SENSITIVE_PATTERNS:
        if pattern.search(redacted):
            flags.append(flag)
            redacted = pattern.sub(replacement, redacted)
    return redacted, flags


def _sanitize(value: Any, flags: list[str]) -> Any:
    """递归清洗即将进入长期记忆的 value。

    长期记忆会跨会话复用，清洗策略比短期上下文更严格。
    """
    if isinstance(value, str):
        redacted, found = _redact_text(value)
        flags.extend(found)
        return redacted[:2000]
    if isinstance(value, list):
        return [_sanitize(item, flags) for item in value[:50]]
    if isinstance(value, tuple):
        return [_sanitize(item, flags) for item in value[:50]]
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).lower()
            if normalized_key in _DROPPED_FIELDS:
                flags.append(f"dropped_{normalized_key}")
                continue
            clean[str(key)] = _sanitize(item, flags)
        return clean
    return value


def _active(value: dict[str, Any]) -> bool:
    """只有 active 记忆参与召回和冲突判断。"""
    return value.get("memory_status", "active") not in {"superseded", "deleted", "rejected"}


def _preference_facts(text: str) -> dict[str, Any]:
    """把自然语言偏好转成结构化事实。

    真实生产里可以用规则 + 小模型结构化抽取。
    这里先用可解释规则，方便学习和测试。
    """
    facts: dict[str, Any] = {}
    normalized = text.lower()

    if any(token in text for token in ("不接受替代", "不要替代", "不允许替代", "不接受兼容替代")):
        facts["substitute_sku"] = "reject"
    elif any(token in text for token in ("接受替代", "可以替代", "允许替代", "兼容替代")):
        facts["substitute_sku"] = "accept"

    if any(token in text for token in ("优先时效", "尽快发货", "最快", "加急")):
        facts["fulfillment_priority"] = "speed"
    elif any(token in text for token in ("优先成本", "降低成本", "最便宜", "省运费")):
        facts["fulfillment_priority"] = "cost"

    if any(token in text for token in ("先人工确认", "人工复核", "人工确认", "不要自动放行")):
        facts["review_policy"] = "manual_first"
    elif any(token in text for token in ("自动放行", "无需人工", "不用人工")):
        facts["review_policy"] = "auto_release"

    if "vip" in normalized or "高价值" in text:
        facts["customer_segment"] = "high_value"

    return facts


def _facts_for(value: dict[str, Any]) -> dict[str, Any]:
    """获取用于治理的结构化事实。"""
    if isinstance(value.get("memory_facts"), dict):
        return dict(value["memory_facts"])
    if value.get("memory_type") == "user_preference":
        text = " ".join(
            str(value.get(name, ""))
            for name in ("preference", "summary", "source_message", "assistant_reply")
        )
        return _preference_facts(text)
    return {}


def _scope_type(namespace: tuple[str, ...]) -> str:
    if len(namespace) >= 3 and namespace[0] == "sessions":
        return "session"
    if len(namespace) >= 3 and namespace[0] == "customers":
        return "customer"
    if namespace[:2] == ("global", "preferences"):
        return "global"
    if namespace and namespace[0] == "rules":
        return "rule"
    return "unknown"


def _priority_for(namespace: tuple[str, ...], value: dict[str, Any]) -> int:
    """计算记忆优先级。

    优先级不是重要性分数。它表示冲突时谁更有权威。

    推荐理解：
    硬规则 > 当前会话明确指令 > 客户长期偏好 > 全局默认偏好 > 历史经验。
    """
    explicit = value.get("source_priority")
    if isinstance(explicit, int):
        return explicit
    if value.get("memory_type") == "business_rule":
        return 100
    scope = _scope_type(namespace)
    if scope == "session":
        return 90
    if scope == "customer":
        return 80
    if scope == "global":
        return 60
    return 40


def _related_namespaces(namespace: tuple[str, ...], value: dict[str, Any]) -> list[tuple[str, ...]]:
    """返回需要一起检查冲突的作用域。

    例子：
    - 写 session 偏好时，如果 value 里带 customer_id，也要检查客户级偏好。
    - 写客户偏好时，至少检查同一个客户 namespace。
    """
    namespaces = [namespace]
    customer_id = value.get("customer_id")
    if isinstance(customer_id, str) and customer_id:
        customer_ns = ("customers", customer_id, "preferences")
        if customer_ns not in namespaces:
            namespaces.append(customer_ns)
    global_ns = ("global", "preferences")
    if global_ns not in namespaces:
        namespaces.append(global_ns)
    return namespaces


class MemoryGovernanceService:
    """长期记忆治理服务。

    企业级关键点：
    - govern() 无副作用，只生成决策和操作计划。
    - govern_and_write() 才应用操作。
    - 冲突不是简单“新覆盖旧”，而是看优先级。
    """

    def govern(
        self,
        store: MemoryStoreLike,
        *,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> GovernanceDecision:
        safety_flags: list[str] = []
        safe_value = _sanitize(copy.deepcopy(value), safety_flags)
        if not isinstance(safe_value, dict):
            safe_value = {"memory_type": "unknown", "value": safe_value}

        now = _now_iso()
        safe_value.setdefault("memory_status", "active")
        safe_value.setdefault("governed_at", now)
        safe_value.setdefault("scope_type", _scope_type(namespace))
        safe_value["memory_facts"] = _facts_for(safe_value)
        safe_value["source_priority"] = _priority_for(namespace, safe_value)

        if "secret" in safety_flags and not safe_value.get("memory_facts") and safe_value.get("memory_type") == "unknown":
            return GovernanceDecision(
                action="reject_unsafe",
                namespace=namespace,
                key=key,
                value=safe_value,
                reason="unsafe_memory_without_business_value",
                conflicts=[],
                safety_flags=sorted(set(safety_flags)),
            )

        conflicts, duplicates = self._find_conflicts_and_duplicates(
            store,
            namespace=namespace,
            key=key,
            value=safe_value,
        )

        if duplicates and not conflicts:
            operations = [
                MemoryWriteOperation(
                    operation_type="merge_evidence",
                    namespace=tuple(item["namespace"]),
                    key=str(item["key"]),
                    value=self._merged_duplicate_value(item["value"], safe_value),
                )
                for item in duplicates[:1]
            ]
            return GovernanceDecision(
                action="merge_evidence",
                namespace=namespace,
                key=key,
                value=safe_value,
                reason="duplicate_memory_confirmed",
                conflicts=[],
                safety_flags=sorted(set(safety_flags)),
                operations=operations,
            )

        blocking_conflicts = [item for item in conflicts if item.resolution == "requires_review"]
        if blocking_conflicts:
            safe_value["_governance"] = {
                "conflict_detected": True,
                "resolution": "requires_review",
                "conflicts": [item.as_dict() for item in blocking_conflicts],
                "safety_flags": sorted(set(safety_flags)),
            }
            return GovernanceDecision(
                action="requires_review",
                namespace=namespace,
                key=key,
                value=safe_value,
                reason="higher_priority_memory_conflict",
                conflicts=[item.as_dict() for item in blocking_conflicts],
                safety_flags=sorted(set(safety_flags)),
            )

        operations: list[MemoryWriteOperation] = []
        if conflicts:
            for conflict in conflicts:
                old_value = self._superseded_value(conflict)
                operations.append(MemoryWriteOperation(
                    operation_type="supersede",
                    namespace=conflict.namespace,
                    key=conflict.key,
                    value=old_value,
                ))
            safe_value["_governance"] = {
                "conflict_detected": True,
                "resolution": "newer_or_higher_priority_memory_supersedes_old",
                "conflicts": [item.as_dict() for item in conflicts],
                "safety_flags": sorted(set(safety_flags)),
            }
            action: MemoryAction = "supersede_old"
            reason = "conflict_resolved_by_priority"
        else:
            if safety_flags:
                safe_value["_governance"] = {
                    "conflict_detected": False,
                    "resolution": "write_after_safety_redaction",
                    "safety_flags": sorted(set(safety_flags)),
                }
            action = "write_new"
            reason = "safe_to_write"

        operations.append(MemoryWriteOperation(
            operation_type="put_new",
            namespace=namespace,
            key=key,
            value=safe_value,
        ))
        return GovernanceDecision(
            action=action,
            namespace=namespace,
            key=key,
            value=safe_value,
            reason=reason,
            conflicts=[item.as_dict() for item in conflicts],
            safety_flags=sorted(set(safety_flags)),
            operations=operations,
        )

    def govern_and_write(
        self,
        store: MemoryStoreLike,
        *,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> GovernanceDecision:
        """治理并应用操作计划。

        注意：这里仍然不是完整数据库事务。真正上线时，PostgreSQL 版本应把 supersede + put_new
        放进一个事务；PGVector 作为可重建索引异步同步。
        """
        decision = self.govern(store, namespace=namespace, key=key, value=value)
        if decision.action in {"reject_unsafe", "requires_review", "skip_duplicate"}:
            return decision
        for operation in decision.operations:
            store.put(operation.namespace, operation.key, operation.value)
        return decision

    def _find_conflicts_and_duplicates(
        self,
        store: MemoryStoreLike,
        *,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> tuple[list[MemoryConflict], list[dict[str, Any]]]:
        if value.get("memory_type") != "user_preference":
            return [], []
        new_facts = _facts_for(value)
        if not new_facts:
            return [], []

        new_priority = _priority_for(namespace, value)
        conflicts: list[MemoryConflict] = []
        duplicates: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, ...], str]] = set()

        for candidate_namespace in _related_namespaces(namespace, value):
            try:
                existing = store.search(
                    candidate_namespace,
                    query=None,
                    filter={"memory_type": "user_preference"},
                    limit=30,
                )
            except Exception:
                continue
            for item in existing:
                old_key = str(getattr(item, "key", ""))
                old_namespace = tuple(getattr(item, "namespace", candidate_namespace))
                identity = (old_namespace, old_key)
                if identity in seen or identity == (namespace, key):
                    continue
                seen.add(identity)

                old_value = dict(getattr(item, "value", {}) or {})
                if not _active(old_value):
                    continue
                old_facts = _facts_for(old_value)
                if not old_facts:
                    continue

                if old_facts == new_facts:
                    duplicates.append({
                        "namespace": old_namespace,
                        "key": old_key,
                        "value": old_value,
                        "memory_facts": old_facts,
                    })
                    continue

                old_priority = _priority_for(old_namespace, old_value)
                for field_name, new_fact_value in new_facts.items():
                    old_fact_value = old_facts.get(field_name)
                    if old_fact_value is None or old_fact_value == new_fact_value:
                        continue
                    resolution = (
                        "requires_review"
                        if old_priority > new_priority
                        else "supersede_old"
                    )
                    conflicts.append(MemoryConflict(
                        conflict_type="preference_conflict",
                        namespace=old_namespace,
                        key=old_key,
                        field=field_name,
                        old_value=old_fact_value,
                        new_value=new_fact_value,
                        old_priority=old_priority,
                        new_priority=new_priority,
                        resolution=resolution,
                        old_record=old_value,
                    ))
        return conflicts, duplicates

    def _merged_duplicate_value(self, old_value: dict[str, Any], new_value: dict[str, Any]) -> dict[str, Any]:
        """重复记忆不直接丢弃，而是合并“再次确认”的证据。"""
        merged = copy.deepcopy(old_value)
        merged["memory_status"] = "active"
        merged["last_confirmed_at"] = _now_iso()
        merged["confirmation_count"] = int(merged.get("confirmation_count") or 1) + 1
        merged.setdefault("_governance", {})
        merged["_governance"]["last_duplicate_resolution"] = {
            "resolution": "merge_evidence",
            "merged_from_source": new_value.get("source", "unknown"),
            "merged_at": merged["last_confirmed_at"],
        }
        return merged

    def _superseded_value(self, conflict: MemoryConflict) -> dict[str, Any]:
        """生成旧记忆失效后的 value。

        注意：不能把旧记忆直接覆盖成一条空记录。
        企业系统需要保留原偏好内容，只是在状态上标记为 superseded，
        这样以后排查时能知道旧偏好原来是什么、为什么失效。
        """
        value = copy.deepcopy(conflict.old_record)
        value["memory_status"] = "superseded"
        value["superseded_at"] = _now_iso()
        value.setdefault("memory_facts", {conflict.field: conflict.old_value})
        value.setdefault("_governance", {})
        value["_governance"]["superseded_by_conflict"] = {
            "type": conflict.conflict_type,
            "field": conflict.field,
            "old_value": conflict.old_value,
            "new_value": conflict.new_value,
            "resolution": conflict.resolution,
        }
        return value
