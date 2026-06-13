"""轻量级工具调用观测。

项目已经在 API 层暴露 Prometheus 指标。这个模块在本地维护一份低依赖的
工具级观测数据源，方便测试、管理接口和指标采集读取同一套计数器。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass
class ToolMetric:
    calls: int = 0
    success: int = 0
    error: int = 0
    timeout: int = 0
    circuit_open: int = 0
    permission_denied: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    last_error_code: str = ""
    last_error_message: str = ""

    def as_dict(self) -> dict[str, Any]:
        avg = self.total_latency_ms / self.calls if self.calls else 0.0
        cache_total = self.cache_hits + self.cache_misses
        return {
            "calls": self.calls,
            "success": self.success,
            "error": self.error,
            "timeout": self.timeout,
            "circuit_open": self.circuit_open,
            "permission_denied": self.permission_denied,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hits / cache_total, 3) if cache_total else 0.0,
            "avg_latency_ms": round(avg, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "last_error_code": self.last_error_code,
            "last_error_message": self.last_error_message,
        }


@dataclass
class ToolTelemetry:
    _metrics: dict[str, ToolMetric] = field(default_factory=lambda: defaultdict(ToolMetric))
    _lock: RLock = field(default_factory=RLock)

    def record_cache_hit(self, tool_name: str) -> None:
        with self._lock:
            self._metrics[tool_name].cache_hits += 1

    def record_cache_miss(self, tool_name: str) -> None:
        with self._lock:
            self._metrics[tool_name].cache_misses += 1

    def record_call(
        self,
        tool_name: str,
        status: str,
        latency_ms: float,
        *,
        error_code: str = "",
        error_message: str = "",
    ) -> None:
        with self._lock:
            metric = self._metrics[tool_name]
            metric.calls += 1
            metric.total_latency_ms += max(latency_ms, 0.0)
            metric.max_latency_ms = max(metric.max_latency_ms, latency_ms)
            if status == "success":
                metric.success += 1
            elif status == "timeout":
                metric.timeout += 1
                metric.error += 1
            elif status == "circuit_open":
                metric.circuit_open += 1
                metric.error += 1
            elif status == "permission_denied":
                metric.permission_denied += 1
                metric.error += 1
            else:
                metric.error += 1
            if error_code or error_message:
                metric.last_error_code = error_code
                metric.last_error_message = error_message[:300]

    @property
    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {name: metric.as_dict() for name, metric in self._metrics.items()}

    def reset(self) -> None:
        with self._lock:
            self._metrics.clear()


_TELEMETRY = ToolTelemetry()


def get_tool_telemetry() -> ToolTelemetry:
    return _TELEMETRY
