"""Hybrid 响应缓存（学习版注释）。

这不是模型服务端的 Prompt Caching，而是接口层的最终响应缓存：
同一个订单、同一个问题、同一份业务数据版本下，短时间内重复请求可以直接返回上次结果。

设计边界：
  - 只缓存 completed 响应，不缓存中断、错误或人工决策中的结果。
  - 默认按订单隔离，不做跨订单共享，避免把 A 订单答案返回给 B 订单。
  - 高风险/会改变业务状态的问题不缓存，避免返回过期决策。
  - 缓存统计只记录命中次数和估算节省延迟，不伪装成服务端 token 复用。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Optional


HIGH_RISK_CACHE_KEYWORDS = (
    "缺货",
    "替代",
    "调拨",
    "延期",
    "取消",
    "修改",
    "审批",
    "人工",
    "退款",
    "赔付",
    "风险",
    "锁定",
    "解锁",
    "紧急",
)


@dataclass
class ResponseCacheEntry:
    """响应缓存项。

    缓存项保存的是一次 Hybrid 最终回答，不保存中间 Agent 状态。
    """

    # 哈希后的缓存键，避免长问题直接做 dict key。
    cache_key: str
    # 默认按订单隔离缓存。
    order_id: str
    # 用户原始问题，保留用于统计展示。
    question: str
    # 最终回答文本。
    answer: str
    # 使用了 workflow/agent/multi_agent 哪条路径。
    path_used: str
    # 本次回答调用过哪些工具。
    tools_called: list[str]
    # 创建时间。
    created_at: datetime
    # 最近访问时间，用于 TTL 和 LRU。
    last_accessed_at: datetime
    # 原请求耗时，用于估算缓存节省了多少延迟。
    execution_time_ms: float = 0.0
    # 业务数据版本上下文，比如知识库版本/模型 ID/订单更新时间。
    cache_context: str = ""
    # 命中次数。
    access_count: int = 0
    # 跳过原因预留字段。
    skip_reason: str = ""

    # 兼容旧字段名，避免旧调用方读 question_hash 时出错。
    @property
    def question_hash(self) -> str:
        return self.cache_key

    def is_expired(self, ttl: timedelta) -> bool:
        """检查是否已过期。"""
        return datetime.now() > (self.last_accessed_at + ttl)

    def record_access(self) -> None:
        """记录一次访问。"""
        # 命中缓存时更新时间，LRU 淘汰会用到。
        self.access_count += 1
        self.last_accessed_at = datetime.now()


class ResponseCacheService:
    """Hybrid 响应缓存。

    它缓存的是最终业务回答，不是模型 provider 的 token 前缀缓存。
    因此缓存 key 必须绑定订单和业务数据上下文，TTL 也应该短一些。
    """

    def __init__(
        self,
        ttl_hours: int = 1,
        max_entries: int = 5000,
        enable_cross_order_cache: bool = False,
    ) -> None:
        # TTL 控制缓存有效期，Hybrid 结果不建议长时间缓存。
        self.ttl = timedelta(hours=ttl_hours)
        # 防止内存无限增长。
        self.max_entries = max_entries
        # 默认禁止跨订单复用，避免订单 A 的回答给到订单 B。
        self.enable_cross_order_cache = enable_cross_order_cache

        # 内存缓存字典：cache_key -> ResponseCacheEntry。
        self.cache: dict[str, ResponseCacheEntry] = {}
        # RLock 保证多线程 API 请求下读写安全。
        self.lock = RLock()

        # 简单统计，用于系统状态页面展示缓存效果。
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_writes": 0,
            "cache_skips": 0,
            "estimated_latency_saved_ms": 0.0,
        }

    def get_cache_key(
        self,
        order_id: str,
        question: str,
        *,
        cache_context: str = "",
    ) -> str:
        """生成缓存键。

        cache_context 用来放业务数据版本，例如订单更新时间、库存快照版本、
        知识库版本、模型 ID 等。当前调用方没有真实版本时可留空。
        """
        # 先归一化问题，减少空格/大小写导致的缓存 miss。
        normalized_question = self._normalize_question(question)
        if self.enable_cross_order_cache:
            # 跨订单缓存只适合完全不依赖订单数据的问题。
            key_str = f"{normalized_question}||{cache_context}"
        else:
            # 默认把 order_id 加入 key，确保订单隔离。
            key_str = f"{order_id}||{normalized_question}||{cache_context}"
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    def get(
        self,
        order_id: str,
        question: str,
        *,
        cache_context: str = "",
    ) -> Optional[ResponseCacheEntry]:
        """获取缓存项，不存在或过期则返回 None。"""
        with self.lock:
            # 所有 get 都算一次请求，用于命中率统计。
            self.stats["total_requests"] += 1
            cache_key = self.get_cache_key(order_id, question, cache_context=cache_context)
            entry = self.cache.get(cache_key)

            if entry is None:
                # 没找到缓存。
                self.stats["cache_misses"] += 1
                return None

            if entry.is_expired(self.ttl):
                # 找到了但过期，删除后按 miss 处理。
                del self.cache[cache_key]
                self.stats["cache_misses"] += 1
                return None

            # 命中缓存，更新访问统计。
            entry.record_access()
            self.stats["cache_hits"] += 1
            self.stats["estimated_latency_saved_ms"] += max(entry.execution_time_ms, 0.0)
            return entry

    def set(
        self,
        order_id: str,
        question: str,
        answer: str,
        path_used: str,
        tools_called: list[str] | None = None,
        *,
        status: str = "completed",
        execution_time_ms: float = 0.0,
        cache_context: str = "",
    ) -> bool:
        """保存缓存项。

        Returns:
            True 表示写入缓存，False 表示因为策略原因跳过。
        """
        tools_called = tools_called or []
        # 写入前先走策略判断，高风险/错误/空答案都不缓存。
        if not self.should_cache(
            question=question,
            answer=answer,
            path_used=path_used,
            tools_called=tools_called,
            status=status,
        ):
            with self.lock:
                self.stats["cache_skips"] += 1
            return False

        with self.lock:
            if len(self.cache) >= self.max_entries:
                # 容量满了，先淘汰一个最不常用的缓存项。
                self._evict_lru()

            cache_key = self.get_cache_key(order_id, question, cache_context=cache_context)
            now = datetime.now()
            # 写入新的缓存项；相同 key 会覆盖旧值。
            self.cache[cache_key] = ResponseCacheEntry(
                cache_key=cache_key,
                order_id=order_id,
                question=question,
                answer=answer,
                path_used=path_used,
                tools_called=tools_called,
                created_at=now,
                last_accessed_at=now,
                execution_time_ms=execution_time_ms,
                cache_context=cache_context,
            )
            self.stats["cache_writes"] += 1
            return True

    @staticmethod
    def should_cache(
        *,
        question: str,
        answer: str,
        path_used: str,
        tools_called: list[str],
        status: str,
    ) -> bool:
        """判断一次 Hybrid 结果是否适合缓存。"""
        # 人工审批中断、错误结果、未完成结果都不能缓存。
        if status != "completed":
            return False
        # 空答案不能缓存。
        if not answer.strip():
            return False
        # 只缓存正式路径，避免调试/未知路径污染缓存。
        if path_used not in {"workflow", "agent", "multi_agent"}:
            return False
        # 高风险关键词通常意味着答案依赖实时状态或需要人工确认。
        combined = f"{question}\n{answer}"
        if any(keyword in combined for keyword in HIGH_RISK_CACHE_KEYWORDS):
            return False
        return True

    def _evict_lru(self) -> None:
        """驱逐访问最少且最久未使用的项。"""
        if not self.cache:
            return
        lru_key = min(
            self.cache.keys(),
            key=lambda k: (
                self.cache[k].access_count,
                self.cache[k].last_accessed_at,
            ),
        )
        del self.cache[lru_key]

    def clear_expired(self) -> int:
        """清理所有过期项。"""
        with self.lock:
            # 先找出 key，再删除，避免遍历 dict 时修改 dict。
            expired_keys = [
                key for key, value in self.cache.items() if value.is_expired(self.ttl)
            ]
            for key in expired_keys:
                del self.cache[key]
            return len(expired_keys)

    def get_stats(self) -> dict:
        """获取缓存统计。"""
        with self.lock:
            # 查询统计前顺手清理过期项，让 cache_size 更真实。
            self.clear_expired()
            total_requests = self.stats["total_requests"]
            hit_rate = self.stats["cache_hits"] / total_requests if total_requests else 0.0
            latency_saved_ms = float(self.stats["estimated_latency_saved_ms"])

            return {
                "cache_type": "hybrid_response_cache",
                "cache_size": len(self.cache),
                "max_entries": self.max_entries,
                "total_requests": total_requests,
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "cache_writes": self.stats["cache_writes"],
                "cache_skips": self.stats["cache_skips"],
                "hit_rate": f"{hit_rate:.1%}",
                "estimated_latency_saved_ms": round(latency_saved_ms, 1),
                "estimated_latency_saved_seconds": round(latency_saved_ms / 1000, 2),
                "ttl_hours": self.ttl.total_seconds() / 3600,
                "cross_order_cache_enabled": self.enable_cross_order_cache,
                "entries": [
                    {
                        "order_id": entry.order_id,
                        "question": entry.question[:50],
                        "path": entry.path_used,
                        "tools_called": entry.tools_called,
                        "access_count": entry.access_count,
                        "cache_context": entry.cache_context,
                        "created_at": entry.created_at.isoformat(),
                    }
                    for entry in list(self.cache.values())[:20]
                ],
            }

    def clear_all(self) -> int:
        """清空所有缓存。"""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            return count

    @staticmethod
    def _normalize_question(question: str) -> str:
        """归一化问题文本，降低无意义差异造成的缓存 miss。"""
        return re.sub(r"\s+", " ", question.strip().lower())


# 兼容旧导入名。后续代码建议使用 ResponseCacheService。
ClientCacheService = ResponseCacheService
CacheEntry = ResponseCacheEntry

_cache_service: Optional[ResponseCacheService] = None


def get_response_cache() -> ResponseCacheService:
    """获取全局响应缓存服务实例。"""
    global _cache_service
    if _cache_service is None:
        # 使用模块级单例，保证所有 Hybrid 请求共享同一个内存缓存。
        _cache_service = ResponseCacheService(
            ttl_hours=1,
            max_entries=5000,
            enable_cross_order_cache=False,
        )
    return _cache_service


def get_client_cache() -> ResponseCacheService:
    """兼容旧入口：实际返回 Hybrid 响应缓存。"""
    return get_response_cache()
