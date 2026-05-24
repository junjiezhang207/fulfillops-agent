"""文件作用摘要：给 Agent 工具加工程弹性，避免工具失败拖垮整轮对话。

这个文件不定义业务工具本身，而是包装 ``tools.py`` 创建出来的工具。
在真实 Agent 系统里，工具可能访问数据库、知识库、外部 API 或内部服务，
这些下游可能超时、偶发失败、参数错误或被重复调用。本文件负责把这些问题
收口在工具外层，让 Agent 收到可读的失败结果，而不是直接抛异常中断。

主要做的事：
1. ``CircuitBreaker``：连续失败后短时间熔断，避免反复打爆下游服务。
2. ``wrap_tool_with_resilience``：给单个工具加缓存、熔断、重试、超时和输出净化。
3. ``wrap_all_tools``：批量包装工具列表，供 ``AgentService`` 装配时使用。
4. 和 ``tool_cache.py`` 配合：目录/规则类工具可以缓存，实时数据工具可禁用缓存。
5. 和 ``guardrails.py`` 配合：工具输出进入 LLM 前做间接注入净化。

包装后的执行顺序：
cache.get -> 命中直接返回
          -> 未命中 -> 熔断检查 -> 重试 -> 超时控制 -> 调真实工具
          -> 工具输出净化 -> 成功结果写入缓存。

学习时先看：
1. ``CircuitBreaker``：理解熔断状态如何从 closed/open/half_open 切换。
2. ``wrap_tool_with_resilience``：这是工具弹性层主流程。
3. ``wrap_all_tools``：看 AgentService 如何一次性包装所有工具。
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, StructuredTool

from app.agents.tools.guardrails import get_tool_sanitizer
from app.agents.tools.cache import GLOBAL_SCOPE, get_tool_cache

_TOOL_TIMEOUT_EXECUTOR = ThreadPoolExecutor(
    max_workers=16,
    thread_name_prefix="agent-tool-timeout",
)


# ============================================================================
# 熔断器（LangChain 无内置实现，保留 Python）
# ============================================================================

# 面试官可能问：为什么工具层要做熔断？
# 回答：Agent 可能反复调用同一个失败工具，如果下游数据库/API 已经异常，
# 不熔断会放大故障、浪费时间和成本。熔断让系统短时间快速失败，保护下游服务。
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

# 面试官可能问：这个包装函数解决了哪些生产问题？
# 回答：它把缓存、熔断、重试、超时、工具输出净化都收口到工具外层。
# 业务工具只关心“查什么数据”，工程问题在这里统一处理，避免每个工具重复写。
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

    Args:
        enable_cache:  是否启用 TTL 缓存（默认 True）
        cache_scope:   缓存 scope（默认 GLOBAL_SCOPE 跨会话共享；
                       传入 session_id 则按会话隔离）
    """
    breaker = CircuitBreaker(failure_threshold=failure_threshold) if enable_circuit_breaker else None
    _cache = get_tool_cache() if enable_cache else None
    _sanitizer = get_tool_sanitizer()  # 间接 Prompt Injection 净化器

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
        return _cache.get(cache_scope, tool.name, kwargs)

    def _cache_set(kwargs: dict, result: str) -> None:
        if _cache is not None:
            _cache.set(cache_scope, tool.name, kwargs, result)

    def _circuit_open_message() -> str | None:
        if not (breaker and breaker.is_open()):
            return None
        return (
            f"[CIRCUIT_OPEN] {tool.name} 熔断中"
            f"（连续失败 {breaker._consecutive_failures} 次），"
            f"{breaker.recovery_seconds:.0f}s 后自动恢复。"
        )

    def _sanitize_and_record_success(kwargs: dict, result: object) -> str:
        if breaker:
            breaker.record_success()
        result_text = _sanitizer.sanitize(str(result), source=tool.name)
        _cache_set(kwargs, result_text)
        return result_text

    def resilient_func(**kwargs) -> str:
        # ── 1. 缓存命中快路径 ────────────────────────────────────────────────
        cached = _cache_get(kwargs)
        if cached is not None:
            return cached

        # ── 2. 熔断检查（LangChain 无内置熔断器，保留）────────────────────
        circuit_message = _circuit_open_message()
        if circuit_message is not None:
            return circuit_message

        # ── 3. 真实调用（含重试 + 超时）────────────────────────────────────
        try:
            # 同步路径仍保留 timeout 兜底，但使用共享 executor，避免每次工具调用都创建线程池。
            future = _TOOL_TIMEOUT_EXECUTOR.submit(retrying_runnable.invoke, kwargs)
            try:
                result = future.result(timeout=timeout_seconds)
                return _sanitize_and_record_success(kwargs, result)
            except FutureTimeoutError:
                future.cancel()
                if breaker:
                    breaker.record_failure()
                return f"[TIMEOUT] {tool.name}: 超过 {timeout_seconds}s"
        except Exception as e:
            if breaker:
                breaker.record_failure()
            # 抛出异常，交给 StructuredTool 的 handle_tool_error 处理
            raise e

    async def resilient_afunc(**kwargs) -> str:
        # async Agent / LangGraph 路径会走这里，优先使用 BaseTool.ainvoke 和 Runnable.ainvoke。
        cached = _cache_get(kwargs)
        if cached is not None:
            return cached

        circuit_message = _circuit_open_message()
        if circuit_message is not None:
            return circuit_message

        try:
            try:
                result = await asyncio.wait_for(
                    retrying_runnable.ainvoke(kwargs),
                    timeout=timeout_seconds,
                )
                return _sanitize_and_record_success(kwargs, result)
            except TimeoutError:
                if breaker:
                    breaker.record_failure()
                return f"[TIMEOUT] {tool.name}: 超过 {timeout_seconds}s"
        except Exception as e:
            if breaker:
                breaker.record_failure()
            raise e

    # ── LangChain: StructuredTool + handle_tool_error 代替手写 try/except ──
    return StructuredTool.from_function(
        func=resilient_func,
        coroutine=resilient_afunc,
        name=tool.name,
        description=tool.description or "",
        args_schema=getattr(tool, "args_schema", None),
        handle_tool_error=True,          # LangChain 捕获异常 → 返回错误字符串
        handle_validation_error=True,    # LangChain 捕获参数校验错误
        infer_schema=getattr(tool, "args_schema", None) is None,
    )


# 面试官可能问：为什么有些工具 enable_cache=False？
# 回答：订单、库存、履约方案是强实时数据，缓存可能导致错误决策；
# 知识规则、替代 SKU 属于稳定目录数据，短时间缓存可以减少重复调用。
def wrap_all_tools(
    tools: list[BaseTool],
    max_retries: int = 2,
    timeout_seconds: float = 10.0,
    enable_circuit_breaker: bool = True,
    enable_cache: bool = True,
    cache_scope: str = GLOBAL_SCOPE,
) -> list[BaseTool]:
    """批量包装工具列表。

    Args:
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
