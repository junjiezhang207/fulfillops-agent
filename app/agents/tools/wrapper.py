"""Agent 工具弹性包装。

本模块不定义业务工具本身，而是在工具外层统一处理缓存、熔断、重试、
超时、权限检查、Telemetry 和输出净化。工具访问数据库、知识库、外部 API
或内部服务时，失败会被转换成可读的工具结果，避免直接中断整轮 Agent 调用。

主要组成：
1. ``CircuitBreaker``：连续失败后短时间熔断，避免反复打爆下游服务。
2. ``wrap_tool_with_resilience``：给单个工具加缓存、熔断、重试、超时和输出净化。
3. ``wrap_all_tools``：批量包装工具列表，供 ``AgentService`` 装配时使用。

包装后的执行顺序：
cache.get -> 命中直接返回
          -> 未命中 -> 熔断检查 -> 重试 -> 超时控制 -> 调真实工具
          -> 工具输出净化 -> 成功结果写入缓存。
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool

from app.agents.tools.contracts import (
    authorize_tool_call,
    error_envelope,
    get_tool_runtime_context,
    is_tool_success,
    parse_tool_error,
)
from app.agents.tools.guardrails import get_tool_sanitizer
from app.agents.tools.cache import GLOBAL_SCOPE, get_tool_cache
from app.agents.tools.telemetry import get_tool_telemetry
from app.observability.business_trace import add_trace_step

_TOOL_TIMEOUT_EXECUTOR = ThreadPoolExecutor(
    max_workers=16,
    thread_name_prefix="agent-tool-timeout",
)


# ============================================================================
# 熔断器（LangChain 无内置实现，保留 Python）
# ============================================================================

# 熔断器用于在下游连续失败时快速失败，避免 Agent 重复调用放大故障。
class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_seconds: float = 30.0
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)

    def is_open(self) -> bool:
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_seconds:
                self._state = CircuitState.HALF_OPEN
                return False
            return True
        return False

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        if self._consecutive_failures >= self.failure_threshold:
            self._state = CircuitState.OPEN

    @property
    def state(self) -> CircuitState:
        return self._state


# ============================================================================
# 弹性工具包装
# ============================================================================

# 工具包装层集中处理横切能力，业务工具只保留“查什么数据”的核心逻辑。
def wrap_tool_with_resilience(
    tool: BaseTool,
    max_retries: int = 2,
    timeout_seconds: float = 10.0,
    enable_circuit_breaker: bool = True,
    failure_threshold: int = 3,
    enable_cache: bool = True,
    cache_scope: str = GLOBAL_SCOPE,
) -> BaseTool:
    """为工具增加缓存、重试、超时、熔断能力。

    执行顺序（快路径优先）：
      1. 缓存命中 → 立即返回，跳过所有下游开销           [新增]
      2. 熔断检查 → OPEN 状态直接返回错误
      3. 重试 + 超时控制 → 真实调用下游服务
      4. 成功结果写回缓存（错误结果不缓存）              [新增]

    LangChain API 使用：

      1. RunnableLambda.with_retry()
         ─────────────────────────
         用 LangChain 的 Runnable 重试机制代替手写循环：
           retrying = RunnableLambda(orig_func).with_retry(
               retry_if_exception_type=(Exception,),
               stop_after_attempt=max_retries + 1,
           )
         内部使用 tenacity（LangChain 的依赖），与手写循环效果相同，但更语义化。

      2. StructuredTool(handle_tool_error=True)
         ───────────────────────────────────────
         用 LangChain 内置异常处理代替手写 try/except：
           当 _run() 抛出异常时，LangChain 自动捕获并返回
           "Tool execution failed: {error}" 字符串给 Agent，
           而不是让异常向上传播导致 Agent 崩溃。

      3. StructuredTool(handle_validation_error=True)
         ──────────────────────────────────────────────
         当 Agent 传入的参数不符合 args_schema 时，LangChain 自动返回
         参数格式错误提示，而不是抛出 Pydantic ValidationError。

    参数：
        enable_cache:  是否启用 TTL 缓存（默认 True）
        cache_scope:   缓存 scope（默认 GLOBAL_SCOPE 跨会话共享；
                       传入 session_id 则按会话隔离）
    """
    breaker = CircuitBreaker(failure_threshold=failure_threshold) if enable_circuit_breaker else None
    _cache = get_tool_cache() if enable_cache else None
    _sanitizer = get_tool_sanitizer()  # 间接 Prompt Injection 净化器
    _telemetry = get_tool_telemetry()

    # ── LangChain: BaseTool.invoke() + RunnableLambda.with_retry() ────────
    # 直接调用原始 BaseTool，而不是拆 tool.func 自己执行。这样保留了 LangChain
    # 工具自身的 args_schema、回调和未来扩展能力，包装层只负责横切能力。
    async def _ainvoke_tool(kwargs):
        return await tool.ainvoke(kwargs)

    retrying_runnable = RunnableLambda(
        lambda kwargs: tool.invoke(kwargs),
        afunc=_ainvoke_tool,
    ).with_retry(
        retry_if_exception_type=(Exception,),
        stop_after_attempt=max_retries + 1,
        wait_exponential_jitter=False,
    )

    def _cache_get(kwargs: dict) -> str | None:
        if _cache is None:
            return None
        effective_scope = _effective_cache_scope()
        cached = _cache.get(effective_scope, tool.name, kwargs)
        if cached is None:
            _telemetry.record_cache_miss(tool.name)
            return None
        _telemetry.record_cache_hit(tool.name)
        return cached

    def _cache_set(kwargs: dict, result: str) -> None:
        if _cache is not None and is_tool_success(result):
            _cache.set(_effective_cache_scope(), tool.name, kwargs, result)

    def _effective_cache_scope() -> str:
        context = get_tool_runtime_context()
        if cache_scope == GLOBAL_SCOPE and context.tenant_id:
            return f"{GLOBAL_SCOPE}:tenant:{context.tenant_id}"
        return cache_scope

    def _circuit_open_message() -> str | None:
        if not (breaker and breaker.is_open()):
            return None
        return error_envelope(
            "circuit_open",
            (
                f"{tool.name} 熔断中（连续失败 {breaker._consecutive_failures} 次），"
                f"{breaker.recovery_seconds:.0f}s 后自动恢复。"
            ),
            retryable=True,
            details={"tool_name": tool.name, "state": breaker.state.value},
        )

    def _sanitize_and_record_success(kwargs: dict, result: object) -> str:
        result_text = _sanitizer.sanitize(str(result), source=tool.name)
        if breaker:
            if is_tool_success(result_text):
                breaker.record_success()
            else:
                breaker.record_failure()
        _cache_set(kwargs, result_text)
        return result_text

    def _record_result(
        started_at: float,
        result: str,
        *,
        cache_hit: bool | None = None,
        metadata: dict | None = None,
    ) -> str:
        latency_ms = (time.monotonic() - started_at) * 1000
        error = parse_tool_error(result)
        if error is None:
            _telemetry.record_call(tool.name, "success", latency_ms)
            add_trace_step(
                step_type="tool",
                name=tool.name,
                status="success",
                duration_ms=latency_ms,
                summary=f"工具 {tool.name} 调用成功",
                output_summary=result,
                metadata={
                    "source": "agent",
                    "cache_hit": cache_hit,
                    "retry_count": max_retries,
                    "timeout_seconds": timeout_seconds,
                    **(metadata or {}),
                },
            )
        else:
            status = error.code if error.code in {"timeout", "circuit_open", "permission_denied"} else "error"
            _telemetry.record_call(
                tool.name,
                status,
                latency_ms,
                error_code=error.code,
                error_message=error.message,
            )
            add_trace_step(
                step_type="tool",
                name=tool.name,
                status=status,
                duration_ms=latency_ms,
                summary=f"工具 {tool.name} 调用失败：{error.code}",
                error_code=error.code,
                error_message=error.message,
                output_summary=result,
                metadata={
                    "source": "agent",
                    "cache_hit": cache_hit,
                    "retry_count": max_retries,
                    "timeout_seconds": timeout_seconds,
                    **(metadata or {}),
                },
            )
        return result

    def _permission_error_message(exc: PermissionError) -> str:
        return error_envelope(
            "permission_denied",
            str(exc),
            retryable=False,
            details={"tool_name": tool.name},
        )

    def resilient_func(**kwargs) -> str:
        started_at = time.monotonic()
        try:
            authorize_tool_call(tool.name)
        except PermissionError as exc:
            return _record_result(started_at, _permission_error_message(exc), metadata={"permission_denied": True})

        # ── 1. 缓存命中快路径 ────────────────────────────────────────────────
        cached = _cache_get(kwargs)
        if cached is not None:
            return _record_result(started_at, cached, cache_hit=True)

        # ── 2. 熔断检查（LangChain 无内置熔断器，保留）────────────────────
        circuit_message = _circuit_open_message()
        if circuit_message is not None:
            return _record_result(started_at, circuit_message, cache_hit=False, metadata={"circuit_open": True})

        # ── 3. 真实调用（含重试 + 超时）────────────────────────────────────
        try:
            # 同步路径仍保留 timeout 兜底，但使用共享 executor，避免每次工具调用都创建线程池。
            future = _TOOL_TIMEOUT_EXECUTOR.submit(retrying_runnable.invoke, kwargs)
            try:
                result = future.result(timeout=timeout_seconds)
                return _record_result(started_at, _sanitize_and_record_success(kwargs, result), cache_hit=False)
            except FutureTimeoutError:
                future.cancel()
                if breaker:
                    breaker.record_failure()
                return _record_result(
                    started_at,
                    error_envelope(
                        "timeout",
                        f"{tool.name}: 超过 {timeout_seconds}s",
                        retryable=True,
                        details={"tool_name": tool.name, "timeout_seconds": timeout_seconds},
                    ),
                )
        except Exception as e:
            if breaker:
                breaker.record_failure()
            return _record_result(
                started_at,
                error_envelope(
                    "tool_execution_failed",
                    f"{tool.name} 执行失败：{e}",
                    retryable=True,
                    details={"tool_name": tool.name},
                ),
            )

    async def resilient_afunc(**kwargs) -> str:
        # 异步 Agent / LangGraph 路径会走这里，优先使用 BaseTool.ainvoke 和 Runnable.ainvoke。
        started_at = time.monotonic()
        try:
            authorize_tool_call(tool.name)
        except PermissionError as exc:
            return _record_result(started_at, _permission_error_message(exc), metadata={"permission_denied": True})

        cached = _cache_get(kwargs)
        if cached is not None:
            return _record_result(started_at, cached, cache_hit=True)

        circuit_message = _circuit_open_message()
        if circuit_message is not None:
            return _record_result(started_at, circuit_message, cache_hit=False, metadata={"circuit_open": True})

        try:
            try:
                result = await asyncio.wait_for(
                    retrying_runnable.ainvoke(kwargs),
                    timeout=timeout_seconds,
                )
                return _record_result(started_at, _sanitize_and_record_success(kwargs, result), cache_hit=False)
            except asyncio.TimeoutError:
                if breaker:
                    breaker.record_failure()
                return _record_result(
                    started_at,
                    error_envelope(
                        "timeout",
                        f"{tool.name}: 超过 {timeout_seconds}s",
                        retryable=True,
                        details={"tool_name": tool.name, "timeout_seconds": timeout_seconds},
                    ),
                )
        except Exception as e:
            if breaker:
                breaker.record_failure()
            return _record_result(
                started_at,
                error_envelope(
                    "tool_execution_failed",
                    f"{tool.name} 执行失败：{e}",
                    retryable=True,
                    details={"tool_name": tool.name},
                ),
            )

    # ── LangChain: StructuredTool + handle_tool_error 代替手写 try/except ──
    return StructuredTool.from_function(
        func=resilient_func,
        coroutine=resilient_afunc,
        name=tool.name,
        description=tool.description or "",
        args_schema=getattr(tool, "args_schema", None),
        handle_tool_error=lambda exc: error_envelope("tool_exception", str(exc), retryable=True),
        handle_validation_error=lambda exc: error_envelope("validation_error", f"工具参数校验失败：{exc}"),
        infer_schema=getattr(tool, "args_schema", None) is None,
    )


# 实时数据工具应关闭缓存，目录和规则类工具可短时间缓存以减少重复调用。
def wrap_all_tools(
    tools: list[BaseTool],
    max_retries: int = 2,
    timeout_seconds: float = 10.0,
    enable_circuit_breaker: bool = True,
    enable_cache: bool = True,
    cache_scope: str = GLOBAL_SCOPE,
) -> list[BaseTool]:
    """批量包装工具列表。

    参数：
        enable_cache: 是否启用 TTL 缓存（默认 True）
        cache_scope:  缓存 scope（默认全局共享；传入 session_id 可按会话隔离）
    """
    return [
        wrap_tool_with_resilience(
            tool,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            enable_circuit_breaker=enable_circuit_breaker,
            enable_cache=enable_cache,
            cache_scope=cache_scope,
        )
        for tool in tools
    ]
