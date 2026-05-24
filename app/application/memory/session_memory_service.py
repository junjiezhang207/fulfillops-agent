"""Hybrid 路由会话缓存（学习版注释）。

这个文件不是 Agent 主链路的短期记忆。

当前项目的正式短期记忆由 LangGraph checkpointer + Redis/MemorySaver 负责；
这里保留一个轻量内存缓存，只服务 `/hybrid/*` 演示路由，用来记录路由轮次、
当前订单和少量缓存字段。
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from threading import RLock


@dataclass
class SessionContext:
    """会话上下文 — 存储用户的会话信息。

    这是 Hybrid 演示路由用的轻量会话，不等于 Agent 的长期/短期记忆。
    """

    # 会话 ID，通常来自 thread_id。
    thread_id: str  # 会话 ID
    # 当前会话正在处理的订单。
    order_id: str  # 当前订单 ID
    # 可放订单分析结果，减少同一轮路由重复分析。
    order_analysis_cache: dict = field(default_factory=dict)  # 订单分析缓存
    # 用户临时偏好，例如更关注时效/成本。
    user_preferences: dict = field(default_factory=dict)  # 用户偏好
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
    """Hybrid 路由内存会话缓存。

    特点：
      - 完全内存存储，无外部依赖
      - 支持自动过期清理
      - 线程安全（使用 RLock）
      - 不参与 Agent / Workflow 的 LangGraph 对话历史
    """

    def __init__(self, ttl_hours: int = 2):
        """初始化会话缓存。

        Args:
            ttl_hours: 会话生命周期（小时）
        """
        # thread_id -> SessionContext。
        self.sessions = {}  # thread_id → SessionContext
        # 会话多久不访问就过期。
        self.ttl = timedelta(hours=ttl_hours)
        # 多请求并发读写内存 dict 时用锁保护。
        self.lock = RLock()  # 线程锁

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
                return None

            # 检查是否过期；过期会话直接删除，避免内存积累。
            if session.is_expired(self.ttl):
                del self.sessions[thread_id]
                return None

            # 更新最后访问时间，相当于滑动过期时间。
            session.update_last_accessed()
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

    def delete_session(self, thread_id: str) -> None:
        """删除会话。

        Args:
            thread_id: 会话 ID
        """
        with self.lock:
            # 删除不存在的会话不报错，方便调用方幂等清理。
            if thread_id in self.sessions:
                del self.sessions[thread_id]

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
                "sessions": [
                    {
                        "thread_id": sid,
                        "order_id": s.order_id,
                        "turns": s.conversation_turns,
                        "created_at": s.created_at.isoformat(),
                        "last_accessed_at": s.last_accessed_at.isoformat(),
                    }
                    for sid, s in self.sessions.items()
                ],
            }


# 全局单例
_session_service: Optional[SessionMemoryService] = None


def get_session_service() -> SessionMemoryService:
    """获取全局会话服务实例。"""
    global _session_service
    if _session_service is None:
        # 模块级单例，保证同一进程内 Hybrid 路由共享会话缓存。
        _session_service = SessionMemoryService()
    return _session_service
