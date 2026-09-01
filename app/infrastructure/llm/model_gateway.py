"""后端模型网关（学习版注释）。

企业级项目里，前端不应该直接持有各家大模型 API Key。
本模块把“有哪些模型、哪个模型用于哪个场景、密钥从哪里读取”收口到后端：

- 模型清单来自 YAML 配置，后续可以替换成数据库/配置中心。
- API Key 只从环境变量读取，不出现在前端响应里。
- 调用方只按 use_case / model_id 获取 LangChain ChatModel。
"""

from __future__ import annotations

import os
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import ConfigDict, Field as PydanticField

from app.core.config import get_settings

logger = logging.getLogger(__name__)


KNOWN_USE_CASES = {
    "agent",
    "workflow",
    "plan_execute",
    "supervisor",
    "multi_agent",
    "mcp",
    "hybrid_router",
    "casual_chat",
    "judge",
    "workflow_finalize",
    "structured_extract",
    "memory_extract",
    "memory_extraction",
    "intent_classification",
    "query_rewrite",
    "planner",
    "replan",
    "replanner",
    "case_extraction",
    "rag_rewrite",
    "rag_compress",
    "input_guardrail",
    "embedding",
    "reranker",
}

PROMPT_REQUIRED_USE_CASES = {
    "agent",
    "supervisor",
    "multi_agent",
    "plan_execute",
    "hybrid_router",
    "casual_chat",
    "workflow_finalize",
    "structured_extract",
    "memory_extract",
    "memory_extraction",
    "intent_classification",
    "query_rewrite",
    "planner",
    "replan",
    "replanner",
    "case_extraction",
    "rag_rewrite",
    "rag_compress",
    "input_guardrail",
}

PROMPT_ONLY_USE_CASES = {
    "hybrid_router",
    "casual_chat",
    "memory_extract",
    "memory_extraction",
    "input_guardrail",
}

_PROMPT_PLACEHOLDER_RE = re.compile(r"(?<!{){([a-zA-Z_][a-zA-Z0-9_]*)}(?!})")

_BUILTIN_PROMPTS: dict[str, str] = {
    "fulfillment_agent": (
        "你是高级供应链履约决策助手。请基于订单、库存、知识库和工具结果，"
        "给出简洁、准确、可执行的履约建议；不确定时说明假设和风险。"
    ),
    "supervisor_agent": "你是多 Agent Supervisor，请根据上下文选择下一步专家或汇总器。",
    "synthesizer_agent": "你是多 Agent 汇总器，请基于专家报告生成简洁、准确、可操作的最终答案。",
    "multi_inventory_agent": "你是库存专家 Agent，请检查库存、缺货 SKU 和跨仓可用量。",
    "multi_fulfillment_agent": "你是履约规划专家 Agent，请生成履约方案、替代品和执行建议。",
    "multi_risk_agent": "你是风险评估专家 Agent，请分析订单风险并检索关键规则。",
    "hybrid_router": "你是电商履约系统的意图路由器，只输出路由 JSON。",
    "casual_chat": "你是电商履约运营智能协同 Agent，请友好简短地回答日常对话，不要编造业务数据。",
    "rag_rewrite": "你是电商履约知识库检索 query 改写器，只输出改写查询。",
    "rag_compress": "你是 RAG 上下文压缩器，只保留与问题直接相关的证据。",
    "structured_extract": "你是履约决策结构化抽取器，只抽取文本中明确出现的信息。",
    "memory_extract": "你是长期记忆抽取器，只抽取后续确实可能复用的非敏感记忆。",
    "memory_extraction": "你是短期会话记忆抽取器，只输出用户明确表达的偏好、约束、反馈和指代，不抽取实时业务数据。",
    "intent_classification": "你是电商履约系统的意图识别器，只输出结构化路由 JSON。",
    "query_rewrite": "你是电商履约检索 query 改写器，只输出用于 SOP 和案例召回的查询。",
    "planner": "你是供应链履约 Planner，请基于当前业务事实、SOP 和案例参考生成可审批方案。",
    "replan": "你是供应链履约 Replanner，请在旧方案失效或执行未达标时生成新方案。",
    "replanner": "你是供应链履约 Replanner，请在旧方案失效或执行未达标时生成新方案。",
    "case_extraction": "你是优秀案例抽取器，只从已关闭案件中抽取可复用经验并脱敏输出。",
    "workflow_finalize": "你是 Workflow 结果说明器，请把结构化结果转成运营人员可执行的话术。",
    "input_guardrail": "你是输入安全审查器，请判断越权、注入、敏感信息或危险操作风险。",
    "plan_execute_planner": "你是供应链履约 Planner，请把复杂问题拆成有依赖顺序的工具执行步骤。",
    "plan_execute_replanner": "你是供应链履约 Replanner，请根据已完成结果决定是否继续执行。",
    "plan_execute_synthesizer": "你是供应链履约 Synthesizer，请基于所有步骤结果生成最终答复。",
    "plan_execute_executor": "你是供应链履约执行助手，只完成当前任务并返回具体数据。",
}


class ModelErrorType(str, Enum):
    """模型调用错误分类，用于判断是否应该切换到 fallback。"""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    CONNECTION_ERROR = "connection_error"
    AUTH_ERROR = "auth_error"
    INPUT_TOO_LONG = "input_too_long"
    SCHEMA_ERROR = "schema_error"
    BAD_REQUEST = "bad_request"
    UNKNOWN = "unknown"


RETRYABLE_MODEL_ERRORS = {
    ModelErrorType.TIMEOUT,
    ModelErrorType.RATE_LIMIT,
    ModelErrorType.SERVER_ERROR,
    ModelErrorType.CONNECTION_ERROR,
}


@dataclass(frozen=True)
class FallbackPolicy:
    """模型降级策略。

    默认只对明确可恢复的问题降级；未知错误是否降级交给配置决定。
    """

    fallback_on_unknown_error: bool = False
    retryable_error_types: set[ModelErrorType] = field(default_factory=lambda: set(RETRYABLE_MODEL_ERRORS))


def _classify_model_error(exc: Exception) -> ModelErrorType:
    """把不同 SDK 抛出的异常粗分类，避免所有错误都盲目降级。"""
    text = f"{exc.__class__.__name__}: {exc}".lower()
    if any(marker in text for marker in ("timeout", "timed out", "readtimeout")):
        return ModelErrorType.TIMEOUT
    if any(marker in text for marker in ("rate limit", "ratelimit", "429", "too many requests")):
        return ModelErrorType.RATE_LIMIT
    if any(marker in text for marker in ("500", "502", "503", "504", "server error", "service unavailable", "primary down", "temporarily unavailable")):
        return ModelErrorType.SERVER_ERROR
    if any(marker in text for marker in ("connection", "connecterror", "network", "dns", "remote protocol")):
        return ModelErrorType.CONNECTION_ERROR
    if any(marker in text for marker in ("401", "403", "unauthorized", "forbidden", "api key", "authentication", "permission denied")):
        return ModelErrorType.AUTH_ERROR
    if any(marker in text for marker in ("context length", "maximum context", "too many tokens", "token limit", "input too long")):
        return ModelErrorType.INPUT_TOO_LONG
    if any(marker in text for marker in ("schema", "validation", "pydantic", "json schema")):
        return ModelErrorType.SCHEMA_ERROR
    if any(marker in text for marker in ("400", "bad request", "invalid request", "invalid parameter")):
        return ModelErrorType.BAD_REQUEST
    return ModelErrorType.UNKNOWN


def _should_fallback(exc: Exception, policy: FallbackPolicy | None = None) -> bool:
    """判断一次模型错误是否适合自动降级。"""
    policy = policy or FallbackPolicy()
    error_type = _classify_model_error(exc)
    if error_type == ModelErrorType.UNKNOWN:
        return policy.fallback_on_unknown_error
    return error_type in policy.retryable_error_types


def _extract_prompt_variables(system_prompt: str) -> set[str]:
    """抽取 prompt 文本中的 {变量名}，用于配置体检。"""
    return set(_PROMPT_PLACEHOLDER_RE.findall(system_prompt))


def _safe_usage_metadata(result: Any) -> dict[str, Any]:
    generations = getattr(result, "generations", None)
    if generations:
        try:
            message = generations[0][0].message
            usage = getattr(message, "usage_metadata", None)
            if usage:
                return dict(usage)
            response_metadata = getattr(message, "response_metadata", None) or {}
            token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
            if token_usage:
                return dict(token_usage)
        except Exception:
            return {}
    return {}


def _record_llm_trace(
    *,
    use_case: str,
    model_id: str,
    status: str,
    duration_ms: float,
    fallback_chain: list[str],
    fallback_from: str | None = None,
    error: Exception | None = None,
    prompt_metadata: dict[str, Any] | None = None,
    usage_metadata: dict[str, Any] | None = None,
) -> None:
    try:
        from app.observability.business_trace import add_trace_step

        metadata = {
            "use_case": use_case,
            "model_id": model_id,
            "fallback_chain": fallback_chain,
            "fallback_from": fallback_from,
            "error_type": _classify_model_error(error).value if error else None,
            "usage": usage_metadata or {},
            **(prompt_metadata or {}),
        }
        add_trace_step(
            step_type="llm",
            name=f"{use_case}:{model_id}",
            status=status,
            duration_ms=duration_ms,
            summary=(
                f"LLM 调用成功：{model_id}"
                if status == "success"
                else f"LLM 调用失败：{model_id}"
            ),
            error_code=error.__class__.__name__ if error else None,
            error_message=str(error) if error else None,
            metadata=metadata,
        )
    except Exception:
        return


class FallbackRunnable(Runnable[Any, Any]):
    """给工具绑定/结构化输出后的 Runnable 复用同一套降级策略。"""

    def __init__(
        self,
        runnables: list[Runnable[Any, Any]],
        model_ids: list[str],
        use_case: str,
        prompt_metadata: dict[str, Any] | None = None,
        fallback_on_unknown_error: bool = False,
        retryable_error_types: list[str] | None = None,
    ):
        self.runnables = runnables
        self.model_ids = model_ids
        self.use_case = use_case
        self.prompt_metadata = prompt_metadata or {}
        self.fallback_on_unknown_error = fallback_on_unknown_error
        self.retryable_error_types = retryable_error_types or [item.value for item in RETRYABLE_MODEL_ERRORS]

    @property
    def fallback_model_ids(self) -> list[str]:
        return self.model_ids[1:]

    def _iter_candidates(self) -> Iterator[tuple[str, Runnable[Any, Any]]]:
        return iter(zip(self.model_ids, self.runnables))

    def _fallback_policy(self) -> FallbackPolicy:
        retryable: set[ModelErrorType] = set()
        for raw_name in self.retryable_error_types:
            try:
                retryable.add(ModelErrorType(str(raw_name)))
            except Exception:
                continue
        return FallbackPolicy(
            fallback_on_unknown_error=self.fallback_on_unknown_error,
            retryable_error_types=retryable or set(RETRYABLE_MODEL_ERRORS),
        )

    def _raise_if_not_fallbackable(self, model_id: str, exc: Exception) -> None:
        error_type = _classify_model_error(exc)
        if not _should_fallback(exc, self._fallback_policy()):
            logger.warning(
                "Runnable 调用失败且不满足降级条件：use_case=%s，model=%s，error_type=%s，error=%s",
                self.use_case,
                model_id,
                error_type.value,
                exc,
            )
            raise exc

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for model_id, runnable in self._iter_candidates():
            started = time.monotonic()
            try:
                if last_exc:
                    logger.warning("Runnable 降级：use_case=%s，切换到 %s", self.use_case, model_id)
                result = runnable.invoke(input, config=config, **kwargs)
                _record_llm_trace(
                    use_case=self.use_case,
                    model_id=model_id,
                    status="success",
                    duration_ms=(time.monotonic() - started) * 1000,
                    fallback_chain=self.model_ids,
                    fallback_from=self.model_ids[0] if last_exc and self.model_ids else None,
                    prompt_metadata=self.prompt_metadata,
                )
                return result
            except Exception as exc:
                _record_llm_trace(
                    use_case=self.use_case,
                    model_id=model_id,
                    status="error",
                    duration_ms=(time.monotonic() - started) * 1000,
                    fallback_chain=self.model_ids,
                    error=exc,
                    prompt_metadata=self.prompt_metadata,
                )
                self._raise_if_not_fallbackable(model_id, exc)
                last_exc = exc
                logger.warning(
                    "Runnable 调用失败，准备尝试降级：use_case=%s，model=%s，error_type=%s，error=%s",
                    self.use_case,
                    model_id,
                    _classify_model_error(exc).value,
                    exc,
                )
        if last_exc:
            raise last_exc
        raise RuntimeError("Runnable 降级链为空")

    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        for model_id, runnable in self._iter_candidates():
            started = time.monotonic()
            try:
                if last_exc:
                    logger.warning("Runnable 异步降级：use_case=%s，切换到 %s", self.use_case, model_id)
                result = await runnable.ainvoke(input, config=config, **kwargs)
                _record_llm_trace(
                    use_case=self.use_case,
                    model_id=model_id,
                    status="success",
                    duration_ms=(time.monotonic() - started) * 1000,
                    fallback_chain=self.model_ids,
                    fallback_from=self.model_ids[0] if last_exc and self.model_ids else None,
                    prompt_metadata=self.prompt_metadata,
                )
                return result
            except Exception as exc:
                _record_llm_trace(
                    use_case=self.use_case,
                    model_id=model_id,
                    status="error",
                    duration_ms=(time.monotonic() - started) * 1000,
                    fallback_chain=self.model_ids,
                    error=exc,
                    prompt_metadata=self.prompt_metadata,
                )
                self._raise_if_not_fallbackable(model_id, exc)
                last_exc = exc
                logger.warning(
                    "Runnable 异步调用失败，准备尝试降级：use_case=%s，model=%s，error_type=%s，error=%s",
                    self.use_case,
                    model_id,
                    _classify_model_error(exc).value,
                    exc,
                )
        if last_exc:
            raise last_exc
        raise RuntimeError("Runnable 降级链为空")

    def stream(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Iterator[Any]:
        last_exc: Exception | None = None
        for model_id, runnable in self._iter_candidates():
            emitted = False
            try:
                if last_exc:
                    logger.warning("Runnable 流式降级：use_case=%s，切换到 %s", self.use_case, model_id)
                for chunk in runnable.stream(input, config=config, **kwargs):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                if emitted:
                    logger.warning(
                        "Runnable 流式已输出内容后失败，不再自动降级：use_case=%s，model=%s，error=%s",
                        self.use_case,
                        model_id,
                        exc,
                    )
                    raise exc
                self._raise_if_not_fallbackable(model_id, exc)
                last_exc = exc
                logger.warning(
                    "Runnable 流式调用首 token 前失败，准备尝试降级：use_case=%s，model=%s，error_type=%s，error=%s",
                    self.use_case,
                    model_id,
                    _classify_model_error(exc).value,
                    exc,
                )
        if last_exc:
            raise last_exc
        raise RuntimeError("Runnable 降级链为空")

    async def astream(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> AsyncIterator[Any]:
        last_exc: Exception | None = None
        for model_id, runnable in self._iter_candidates():
            emitted = False
            try:
                if last_exc:
                    logger.warning("Runnable 异步流式降级：use_case=%s，切换到 %s", self.use_case, model_id)
                async for chunk in runnable.astream(input, config=config, **kwargs):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                if emitted:
                    logger.warning(
                        "Runnable 异步流式已输出内容后失败，不再自动降级：use_case=%s，model=%s，error=%s",
                        self.use_case,
                        model_id,
                        exc,
                    )
                    raise exc
                self._raise_if_not_fallbackable(model_id, exc)
                last_exc = exc
                logger.warning(
                    "Runnable 异步流式调用首 token 前失败，准备尝试降级：use_case=%s，model=%s，error_type=%s，error=%s",
                    self.use_case,
                    model_id,
                    _classify_model_error(exc).value,
                    exc,
                )
        if last_exc:
            raise last_exc
        raise RuntimeError("Runnable 降级链为空")


@dataclass(frozen=True)
class ModelProfile:
    """模型网关中的单个模型配置。

    一条 ModelProfile 对应 config/model_gateway.yaml 里的一个模型。
    frozen=True 表示配置对象不可变，避免运行中被随手改掉。
    """

    # 模型在项目内部的唯一 ID，前端选择模型时传这个。
    id: str
    # 展示名，给前端下拉框使用。
    display_name: str
    # provider 决定用哪个 LangChain ChatModel 类。
    provider: str
    # provider 内部真实模型名，例如 gpt-4o-mini、qwen-plus。
    model: str
    # 是否启用。禁用后不会被自动选择。
    enabled: bool = True
    # 支持哪些场景，例如 agent/workflow/rag_rewrite/judge。
    use_cases: list[str] = field(default_factory=list)
    # 优先级越小越靠前，用于自动选择默认模型。
    priority: int = 100
    # OpenAI-compatible 或私有部署模型常需要 base_url。
    base_url: str = ""
    # API Key 的环境变量名，只存变量名，不存密钥。
    api_key_env: str = "LLM_API_KEY"
    # 本地模型或内网无鉴权服务可设为 False。
    requires_api_key: bool = True
    # 大/小模型分层，例如 large/small/standard。
    tier: str = "standard"
    # 部署位置，例如 cloud/local/private。
    deployment: str = "cloud"
    # 模型类型：chat/embedding/reranker。
    model_type: str = "chat"
    # embedding 维度，embedding profile 才需要。
    dimension: int | None = None
    # reranker 默认 top_k，reranker profile 才需要。
    top_k: int | None = None
    # 透传给 OpenAI-compatible 接口的扩展请求体。
    # 例如 DeepSeek V4 关闭 thinking：{"thinking": {"type": "disabled"}}。
    extra_body: dict[str, Any] = field(default_factory=dict)
    # 是否启用模型原生流式输出。Agent / Chat 页面开启后首字响应更快。
    streaming: bool = False
    # 计费用输入 token 单价，单位：美元 / 100 万 token。
    # 不同厂商、合同、缓存命中会变化，所以放在配置里，不写死到代码。
    input_price_per_1m_tokens: float | None = None
    # 计费用输出 token 单价，单位：美元 / 100 万 token。
    output_price_per_1m_tokens: float | None = None
    # 输入缓存命中 token 单价，单位：美元 / 100 万 token。
    # DeepSeek 等 provider 会把缓存命中和未命中的输入 token 分开计费。
    input_cache_hit_price_per_1m_tokens: float | None = None
    # 输入缓存未命中 token 单价，单位：美元 / 100 万 token。
    input_cache_miss_price_per_1m_tokens: float | None = None

    def supports(self, use_case: str) -> bool:
        """判断模型是否支持某个业务场景。

        use_cases 为空表示通用模型，支持所有场景。
        """
        return not self.use_cases or use_case in self.use_cases

    def public_dict(self, api_key_configured: bool = False) -> dict[str, Any]:
        """返回给前端的安全字段，不包含 api_key_env 的实际值。"""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "provider": self.provider,
            "model": self.model,
            "enabled": self.enabled,
            "use_cases": self.use_cases,
            "priority": self.priority,
            "model_type": self.model_type,
            "tier": self.tier,
            "deployment": self.deployment,
            "requires_api_key": self.requires_api_key,
            "dimension": self.dimension,
            "top_k": self.top_k,
            "base_url_configured": bool(self.base_url),
            "api_key_configured": api_key_configured or not self.requires_api_key,
            "streaming": self.streaming,
            "input_price_per_1m_tokens": self.input_price_per_1m_tokens,
            "output_price_per_1m_tokens": self.output_price_per_1m_tokens,
            "input_cache_hit_price_per_1m_tokens": self.input_cache_hit_price_per_1m_tokens,
            "input_cache_miss_price_per_1m_tokens": self.input_cache_miss_price_per_1m_tokens,
        }


@dataclass(frozen=True)
class PromptProfile:
    """模型网关中的 Prompt 配置。

    它描述“哪个 use_case 使用哪个 prompt 文件”，不直接保存完整 prompt 正文。
    """

    id: str
    use_case: str
    path: str
    version: str = "latest"
    fallback_builtin: bool = True
    input_variables: list[str] = field(default_factory=list)
    output_contract: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    base_dir: str = ""


@dataclass(frozen=True)
class GatewayPromptEntry:
    """运行时加载后的 Prompt 内容。"""

    id: str
    name: str
    use_case: str
    system: str
    version: str = "unknown"
    source: str = "yaml"
    description: str = ""
    changelog: str = ""
    path: str = ""
    input_variables: list[str] = field(default_factory=list)
    output_contract: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelGatewayConfig:
    """模型网关完整配置。"""

    # 老版默认模型字段。
    default_model_id: str = ""
    # 新版按 use_case 指定默认模型，例如 agent -> cloud-large。
    default_models: dict[str, str] = field(default_factory=dict)
    # 按 use_case 配置降级模型链。例如 agent: [qwen3.7-max, kimi-chat, local-qwen-small]。
    fallback_models: dict[str, list[str]] = field(default_factory=dict)
    # 降级策略：控制未知错误是否降级、哪些错误类型允许自动降级。
    fallback_policy: FallbackPolicy = field(default_factory=FallbackPolicy)
    # 按 use_case 指定默认 Prompt，例如 agent -> fulfillment_agent。
    default_prompts: dict[str, str] = field(default_factory=dict)
    # 所有 Prompt profile，key 是 prompt_id。
    prompt_profiles: dict[str, PromptProfile] = field(default_factory=dict)
    # 所有模型 profile。
    models: list[ModelProfile] = field(default_factory=list)


def _load_gateway_config(path: str) -> ModelGatewayConfig:
    """从 YAML 加载模型网关配置。

    这个函数只做“配置解析”，不创建真实模型，也不校验 API Key。
    """
    config_path = Path(path)
    if not config_path.exists():
        return ModelGatewayConfig()

    # yaml.safe_load 避免执行 YAML 里的任意对象构造。
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    # 把 YAML dict 转成强类型 dataclass，后面使用更安全。
    models = [
        ModelProfile(
            id=str(item.get("id", "")).strip(),
            display_name=str(item.get("display_name") or item.get("id") or "").strip(),
            provider=str(item.get("provider", "")).strip().lower(),
            model=str(item.get("model", "")).strip(),
            enabled=bool(item.get("enabled", True)),
            use_cases=list(item.get("use_cases") or []),
            priority=int(item.get("priority", 100)),
            base_url=str(item.get("base_url", "") or "").strip(),
            api_key_env=str(item.get("api_key_env", "LLM_API_KEY")).strip(),
            requires_api_key=bool(item.get("requires_api_key", True)),
            tier=str(item.get("tier", "standard") or "standard").strip(),
            deployment=str(item.get("deployment", "cloud") or "cloud").strip(),
            model_type=str(item.get("model_type", "chat") or "chat").strip(),
            dimension=int(item["dimension"]) if item.get("dimension") is not None else None,
            top_k=int(item["top_k"]) if item.get("top_k") is not None else None,
            extra_body=dict(item.get("extra_body") or {}),
            streaming=bool(item.get("streaming", False)),
            input_price_per_1m_tokens=(
                float(item["input_price_per_1m_tokens"])
                if item.get("input_price_per_1m_tokens") is not None
                else None
            ),
            output_price_per_1m_tokens=(
                float(item["output_price_per_1m_tokens"])
                if item.get("output_price_per_1m_tokens") is not None
                else None
            ),
            input_cache_hit_price_per_1m_tokens=(
                float(item["input_cache_hit_price_per_1m_tokens"])
                if item.get("input_cache_hit_price_per_1m_tokens") is not None
                else None
            ),
            input_cache_miss_price_per_1m_tokens=(
                float(item["input_cache_miss_price_per_1m_tokens"])
                if item.get("input_cache_miss_price_per_1m_tokens") is not None
                else None
            ),
        )
        for item in list(data.get("models") or [])
        if item.get("id") and item.get("provider") and item.get("model")
    ]
    policy_data = dict(data.get("fallback_policy") or {})
    retryable_types: set[ModelErrorType] = set()
    for raw_name in list(policy_data.get("retryable_error_types") or []):
        try:
            retryable_types.add(ModelErrorType(str(raw_name).strip()))
        except Exception:
            logger.warning("忽略未知模型降级错误类型：%s", raw_name)
    prompt_profiles: dict[str, PromptProfile] = {}
    for prompt_id, item in dict(data.get("prompt_profiles") or {}).items():
        if not isinstance(item, dict):
            continue
        normalized_id = str(item.get("id") or prompt_id).strip()
        if not normalized_id:
            continue
        prompt_profiles[normalized_id] = PromptProfile(
            id=normalized_id,
            use_case=str(item.get("use_case", "") or "").strip(),
            path=str(item.get("path", "") or "").strip(),
            version=str(item.get("version", "latest") or "latest").strip(),
            fallback_builtin=bool(item.get("fallback_builtin", True)),
            input_variables=[
                str(value).strip()
                for value in list(item.get("input_variables") or item.get("variables") or [])
                if str(value).strip()
            ],
            output_contract=dict(item.get("output_contract") or {}),
            description=str(item.get("description", "") or "").strip(),
            base_dir=str(config_path.parent),
        )

    return ModelGatewayConfig(
        default_model_id=str(data.get("default_model_id", "") or "").strip(),
        default_models={
            str(key).strip(): str(value).strip()
            for key, value in dict(data.get("default_models") or {}).items()
            if str(key).strip() and str(value).strip()
        },
        fallback_models={
            str(key).strip(): [
                str(item).strip()
                for item in list(value or [])
                if str(item).strip()
            ]
            for key, value in dict(data.get("fallback_models") or {}).items()
            if str(key).strip()
        },
        fallback_policy=FallbackPolicy(
            fallback_on_unknown_error=bool(policy_data.get("fallback_on_unknown_error", False)),
            retryable_error_types=retryable_types or set(RETRYABLE_MODEL_ERRORS),
        ),
        default_prompts={
            str(key).strip(): str(value).strip()
            for key, value in dict(data.get("default_prompts") or {}).items()
            if str(key).strip() and str(value).strip()
        },
        prompt_profiles=prompt_profiles,
        models=models,
    )


class FallbackChatModel(BaseChatModel):
    """ChatModel wrapper that tries fallback models when the primary fails.

    LangChain models are Runnables, but Agent creation expects a chat model with
    ``bind_tools`` and ``with_structured_output``. This wrapper preserves those
    entry points and applies the same fallback chain to normal chat, tool calls
    and structured extraction.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    models: list[BaseChatModel] = PydanticField(default_factory=list)
    model_ids: list[str] = PydanticField(default_factory=list)
    use_case: str = ""
    prompt_metadata: dict[str, Any] = PydanticField(default_factory=dict)
    fallback_on_unknown_error: bool = False
    retryable_error_types: list[str] = PydanticField(
        default_factory=lambda: [item.value for item in RETRYABLE_MODEL_ERRORS]
    )

    @property
    def _llm_type(self) -> str:
        return "fallback-chat-model"

    @property
    def fallback_model_ids(self) -> list[str]:
        return self.model_ids[1:]

    @property
    def extra_body(self) -> Any:
        return getattr(self.models[0], "extra_body", None) if self.models else None

    @property
    def streaming(self) -> Any:
        return getattr(self.models[0], "streaming", None) if self.models else None

    def _iter_candidates(self) -> Iterator[tuple[str, BaseChatModel]]:
        return iter(zip(self.model_ids, self.models))

    def _fallback_policy(self) -> FallbackPolicy:
        retryable: set[ModelErrorType] = set()
        for raw_name in self.retryable_error_types:
            try:
                retryable.add(ModelErrorType(str(raw_name)))
            except Exception:
                continue
        return FallbackPolicy(
            fallback_on_unknown_error=self.fallback_on_unknown_error,
            retryable_error_types=retryable or set(RETRYABLE_MODEL_ERRORS),
        )

    def _raise_if_not_fallbackable(self, model_id: str, exc: Exception) -> None:
        error_type = _classify_model_error(exc)
        if not _should_fallback(exc, self._fallback_policy()):
            logger.warning(
                "模型调用失败且不满足降级条件：use_case=%s，model=%s，error_type=%s，error=%s",
                self.use_case,
                model_id,
                error_type.value,
                exc,
            )
            raise exc

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager=None, **kwargs: Any) -> ChatResult:
        last_exc: Exception | None = None
        for model_id, model in self._iter_candidates():
            started = time.monotonic()
            try:
                if last_exc:
                    logger.warning("模型降级：use_case=%s，切换到 %s", self.use_case, model_id)
                result = model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                _record_llm_trace(
                    use_case=self.use_case,
                    model_id=model_id,
                    status="success",
                    duration_ms=(time.monotonic() - started) * 1000,
                    fallback_chain=self.model_ids,
                    fallback_from=self.model_ids[0] if last_exc and self.model_ids else None,
                    prompt_metadata=self.prompt_metadata,
                    usage_metadata=_safe_usage_metadata(result),
                )
                return result
            except Exception as exc:
                _record_llm_trace(
                    use_case=self.use_case,
                    model_id=model_id,
                    status="error",
                    duration_ms=(time.monotonic() - started) * 1000,
                    fallback_chain=self.model_ids,
                    error=exc,
                    prompt_metadata=self.prompt_metadata,
                )
                self._raise_if_not_fallbackable(model_id, exc)
                last_exc = exc
                logger.warning(
                    "模型调用失败，准备尝试降级：use_case=%s，model=%s，error_type=%s，error=%s",
                    self.use_case,
                    model_id,
                    _classify_model_error(exc).value,
                    exc,
                )
        if last_exc:
            raise last_exc
        raise RuntimeError("模型降级链为空")

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager=None, **kwargs: Any) -> ChatResult:
        last_exc: Exception | None = None
        for model_id, model in self._iter_candidates():
            started = time.monotonic()
            try:
                if last_exc:
                    logger.warning("模型降级：use_case=%s，切换到 %s", self.use_case, model_id)
                result = await model._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                _record_llm_trace(
                    use_case=self.use_case,
                    model_id=model_id,
                    status="success",
                    duration_ms=(time.monotonic() - started) * 1000,
                    fallback_chain=self.model_ids,
                    fallback_from=self.model_ids[0] if last_exc and self.model_ids else None,
                    prompt_metadata=self.prompt_metadata,
                    usage_metadata=_safe_usage_metadata(result),
                )
                return result
            except Exception as exc:
                _record_llm_trace(
                    use_case=self.use_case,
                    model_id=model_id,
                    status="error",
                    duration_ms=(time.monotonic() - started) * 1000,
                    fallback_chain=self.model_ids,
                    error=exc,
                    prompt_metadata=self.prompt_metadata,
                )
                self._raise_if_not_fallbackable(model_id, exc)
                last_exc = exc
                logger.warning(
                    "模型异步调用失败，准备尝试降级：use_case=%s，model=%s，error_type=%s，error=%s",
                    self.use_case,
                    model_id,
                    _classify_model_error(exc).value,
                    exc,
                )
        if last_exc:
            raise last_exc
        raise RuntimeError("模型降级链为空")

    def _stream(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        last_exc: Exception | None = None
        for model_id, model in self._iter_candidates():
            emitted = False
            try:
                if last_exc:
                    logger.warning("流式模型降级：use_case=%s，切换到 %s", self.use_case, model_id)
                for chunk in model._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                if emitted:
                    logger.warning(
                        "流式模型已输出内容后失败，不再自动降级：use_case=%s，model=%s，error=%s",
                        self.use_case,
                        model_id,
                        exc,
                    )
                    raise exc
                self._raise_if_not_fallbackable(model_id, exc)
                last_exc = exc
                logger.warning(
                    "模型流式调用首 token 前失败，准备尝试降级：use_case=%s，model=%s，error_type=%s，error=%s",
                    self.use_case,
                    model_id,
                    _classify_model_error(exc).value,
                    exc,
                )
        if last_exc:
            raise last_exc
        raise RuntimeError("模型降级链为空")

    async def _astream(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager=None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        last_exc: Exception | None = None
        for model_id, model in self._iter_candidates():
            emitted = False
            try:
                if last_exc:
                    logger.warning("异步流式模型降级：use_case=%s，切换到 %s", self.use_case, model_id)
                async for chunk in model._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
                    emitted = True
                    yield chunk
                return
            except Exception as exc:
                if emitted:
                    logger.warning(
                        "异步流式模型已输出内容后失败，不再自动降级：use_case=%s，model=%s，error=%s",
                        self.use_case,
                        model_id,
                        exc,
                    )
                    raise exc
                self._raise_if_not_fallbackable(model_id, exc)
                last_exc = exc
                logger.warning(
                    "模型异步流式调用首 token 前失败，准备尝试降级：use_case=%s，model=%s，error_type=%s，error=%s",
                    self.use_case,
                    model_id,
                    _classify_model_error(exc).value,
                    exc,
                )
        if last_exc:
            raise last_exc
        raise RuntimeError("模型降级链为空")

    def bind_tools(self, tools, *, tool_choice: str | None = None, **kwargs: Any):
        bound = []
        bound_ids: list[str] = []
        last_exc: Exception | None = None
        for model_id, model in self._iter_candidates():
            try:
                bound.append(model.bind_tools(tools, tool_choice=tool_choice, **kwargs))
                bound_ids.append(model_id)
            except Exception as exc:
                # 有些降级模型不支持工具调用，绑定阶段就剔除，避免运行中才爆掉。
                last_exc = exc
                logger.warning("模型不支持工具绑定，已从工具调用降级链跳过：model=%s，error=%s", model_id, exc)
        if not bound:
            if last_exc:
                raise last_exc
            raise RuntimeError("模型降级链为空，无法绑定工具")
        return FallbackRunnable(
            runnables=bound,
            model_ids=bound_ids,
            use_case=self.use_case,
            prompt_metadata=self.prompt_metadata,
            fallback_on_unknown_error=self.fallback_on_unknown_error,
            retryable_error_types=self.retryable_error_types,
        )

    def with_structured_output(self, schema, *, include_raw: bool = False, **kwargs: Any):
        structured = []
        structured_ids: list[str] = []
        last_exc: Exception | None = None
        for model_id, model in self._iter_candidates():
            try:
                structured.append(model.with_structured_output(schema, include_raw=include_raw, **kwargs))
                structured_ids.append(model_id)
            except Exception as exc:
                # 结构化输出依赖 provider 能力，不能把不支持的模型放进降级链。
                last_exc = exc
                logger.warning("模型不支持结构化输出，已从结构化降级链跳过：model=%s，error=%s", model_id, exc)
        if not structured:
            if last_exc:
                raise last_exc
            raise RuntimeError("模型降级链为空，无法创建结构化输出模型")
        return FallbackRunnable(
            runnables=structured,
            model_ids=structured_ids,
            use_case=self.use_case,
            prompt_metadata=self.prompt_metadata,
            fallback_on_unknown_error=self.fallback_on_unknown_error,
            retryable_error_types=self.retryable_error_types,
        )


class ModelGateway:
    """后端模型网关，负责模型选择和 LangChain ChatModel 构造。

    调用方不需要知道 API Key 在哪里，也不需要知道 provider 对应哪个类。
    调用方只说“我要 agent 场景的 chat 模型”，网关负责解析。
    """

    def __init__(self, settings: object | None = None):
        # settings 可注入，测试时可以传 fake settings。
        self.settings = settings or get_settings()
        # 模型清单来自 YAML，路径由 settings.model_gateway_config_path 指定。
        self.config = _load_gateway_config(getattr(self.settings, "model_gateway_config_path", ""))

    def _warn_unknown_use_case(self, use_case: str | None) -> None:
        if use_case and use_case not in KNOWN_USE_CASES:
            logger.warning("未知模型 use_case：%s，请补充 KNOWN_USE_CASES 与 model_gateway.yaml", use_case)

    def list_models(
        self,
        use_case: str | None = None,
        include_disabled: bool = False,
        model_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出可用模型，主要给前端模型网关面板使用。"""
        self._warn_unknown_use_case(use_case)
        profiles = self.config.models
        # 按业务场景过滤，例如只看 agent 可用模型。
        if use_case:
            profiles = [profile for profile in profiles if profile.supports(use_case)]
        # 按模型类型过滤，例如 chat/embedding/reranker。
        if model_type:
            profiles = [profile for profile in profiles if profile.model_type == model_type]
        # 默认不展示禁用模型。
        if not include_disabled:
            profiles = [profile for profile in profiles if profile.enabled]
        # 返回 public_dict，确保不会把真实 API Key 泄露给前端。
        return [
            profile.public_dict(api_key_configured=bool(self._api_key_for(profile)))
            for profile in sorted(profiles, key=lambda p: p.priority)
        ]

    def active_model(
        self,
        use_case: str = "agent",
        model_id: str | None = None,
        model_type: str | None = None,
    ) -> dict[str, Any] | None:
        """返回当前场景实际会使用的模型信息。"""
        profile = self.resolve_profile(use_case=use_case, model_id=model_id, model_type=model_type)
        if profile:
            data = profile.public_dict(api_key_configured=bool(self._api_key_for(profile)))
            data["fallback_model_ids"] = [
                item.id
                for item in self.resolve_profile_chain(use_case=use_case, model_id=model_id, model_type=model_type)[1:]
            ]
            return data
        return self._legacy_public_dict()

    def resolve_prompt_profile(
        self,
        use_case: str = "agent",
        prompt_id: str | None = None,
    ) -> PromptProfile | None:
        """按 prompt_id 或 use_case 解析 Prompt profile。"""
        self._warn_unknown_use_case(use_case)
        if prompt_id:
            return self.config.prompt_profiles.get(prompt_id)
        default_id = self.config.default_prompts.get(use_case, "")
        if default_id:
            return self.config.prompt_profiles.get(default_id)
        return next(
            (profile for profile in self.config.prompt_profiles.values() if profile.use_case == use_case),
            None,
        )

    def load_prompt(
        self,
        use_case: str = "agent",
        prompt_id: str | None = None,
    ) -> GatewayPromptEntry:
        """从模型网关加载某个 use_case 对应的 Prompt。

        调用方只关心 use_case，不需要知道 prompt 文件路径；文件不可用时按
        profile.fallback_builtin 决定是否回退到内置 prompt。
        """
        profile = self.resolve_prompt_profile(use_case=use_case, prompt_id=prompt_id)
        if profile is None:
            builtin_id = prompt_id or self.config.default_prompts.get(use_case, "") or use_case
            return self._builtin_prompt_entry(builtin_id, use_case)

        path = self._prompt_path(profile)
        if path.exists():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if not isinstance(data, dict) or not str(data.get("system", "")).strip():
                    raise ValueError("Prompt YAML 缺少 system 字段")
                input_variables = [
                    str(item).strip()
                    for item in list(data.get("input_variables") or profile.input_variables)
                    if str(item).strip()
                ]
                output_contract = dict(data.get("output_contract") or profile.output_contract)
                return GatewayPromptEntry(
                    id=profile.id,
                    name=str(data.get("name") or profile.id),
                    use_case=str(profile.use_case or data.get("use_case") or use_case),
                    system=str(data["system"]),
                    version=str(data.get("version") or profile.version),
                    source="yaml",
                    description=str(data.get("description") or profile.description),
                    changelog=str(data.get("changelog") or ""),
                    path=str(path),
                    input_variables=input_variables,
                    output_contract=output_contract,
                )
            except Exception as exc:
                logger.warning("Prompt %s 加载失败，将尝试兜底：%s", profile.id, exc)

        if profile.fallback_builtin:
            return self._builtin_prompt_entry(profile.id, profile.use_case or use_case, profile=profile)
        raise FileNotFoundError(f"Prompt {profile.id} 文件不存在或不可用：{path}")

    def prompt_system(
        self,
        use_case: str = "agent",
        prompt_id: str | None = None,
    ) -> str:
        """便捷方法：只返回 system prompt 文本。"""
        return self.load_prompt(use_case=use_case, prompt_id=prompt_id).system

    def _prompt_trace_metadata(self, use_case: str) -> dict[str, Any]:
        """为 LLM trace 生成 Prompt 版本信息。"""
        try:
            prompt = self.load_prompt(use_case=use_case)
        except Exception:
            return {}
        digest = hashlib.sha256(prompt.system.encode("utf-8")).hexdigest()[:16]
        return {
            "prompt_id": prompt.id,
            "prompt_version": prompt.version,
            "prompt_source": prompt.source,
            "prompt_hash": digest,
            "prompt_path": prompt.path,
            "output_contract": prompt.output_contract,
        }

    def create_chat_model(self, use_case: str = "agent", model_id: str | None = None) -> BaseChatModel | None:
        """创建 LangChain ChatModel。

        失败时返回 None，让上层可以降级到规则模板，而不是启动失败。
        """
        profiles = self.resolve_profile_chain(use_case=use_case, model_id=model_id, model_type="chat")
        prompt_metadata = self._prompt_trace_metadata(use_case)
        try:
            if profiles:
                built_models: list[BaseChatModel] = []
                built_ids: list[str] = []
                for profile in profiles:
                    try:
                        built_models.append(self._build_from_profile(profile))
                        built_ids.append(profile.id)
                    except Exception as exc:
                        logger.warning(
                            "构造模型 %s 失败，跳过并尝试降级候选：%s",
                            profile.id,
                            exc,
                        )
                if not built_models:
                    return None
                return FallbackChatModel(
                    models=built_models,
                    model_ids=built_ids,
                    use_case=use_case,
                    prompt_metadata=prompt_metadata,
                    fallback_on_unknown_error=self.config.fallback_policy.fallback_on_unknown_error,
                    retryable_error_types=[
                        item.value for item in self.config.fallback_policy.retryable_error_types
                    ],
                )
            return self._build_legacy_model()
        except Exception as exc:
            target = model_id or getattr(self.settings, "default_llm_model_id", "") or "legacy-env"
            logger.warning("构造模型 %s 失败，将使用规则模板：%s", target, exc)
            return None

    def resolve_profile_chain(
        self,
        use_case: str = "agent",
        model_id: str | None = None,
        model_type: str | None = "chat",
    ) -> list[ModelProfile]:
        """解析主模型 + 降级模型链。

        主模型仍按 ``resolve_profile`` 的规则确定；降级模型来自 YAML
        ``fallback_models[use_case]``，如果没有配置，则按 priority 追加同场景候选。
        """
        primary = self.resolve_profile(use_case=use_case, model_id=model_id, model_type=model_type)
        if primary is None:
            return []

        enabled = [profile for profile in self.config.models if profile.enabled]
        if model_type:
            enabled = [profile for profile in enabled if profile.model_type == model_type]
        by_id = {profile.id: profile for profile in enabled if profile.supports(use_case)}

        chain: list[ModelProfile] = [primary]
        seen = {primary.id}
        configured_fallbacks = self.config.fallback_models.get(use_case, [])
        for fallback_id in configured_fallbacks:
            profile = by_id.get(fallback_id)
            if profile and profile.id not in seen:
                chain.append(profile)
                seen.add(profile.id)

        if configured_fallbacks:
            return chain

        # 未配置显式 fallback 时，保守地按 priority 追加同 use_case 候选，
        # 这样只要用户启用多个模型，就天然具备降级能力。
        for profile in sorted(by_id.values(), key=lambda item: item.priority):
            if profile.id not in seen:
                chain.append(profile)
                seen.add(profile.id)
        return chain

    def validate_use_case_coverage(self, required_use_cases: set[str] | None = None) -> dict[str, list[str]]:
        """检查模型网关配置是否覆盖项目约定的 use_case。

        这个方法不阻断线上调用，主要给测试、启动自检或运维面板使用。
        """
        required = set(required_use_cases or KNOWN_USE_CASES)
        model_required = required - PROMPT_ONLY_USE_CASES
        profiles_by_id = {profile.id: profile for profile in self.config.models}
        all_supported = {
            use_case
            for profile in self.config.models
            for use_case in profile.use_cases
        }

        unknown_default_use_cases = sorted(set(self.config.default_models) - KNOWN_USE_CASES)
        unknown_fallback_use_cases = sorted(set(self.config.fallback_models) - KNOWN_USE_CASES)
        unknown_profile_use_cases = sorted(
            use_case
            for profile in self.config.models
            for use_case in profile.use_cases
            if use_case not in KNOWN_USE_CASES
        )

        missing_default_models = sorted(
            use_case
            for use_case in model_required
            if use_case not in self.config.default_models
        )
        missing_supported_profiles = sorted(model_required - all_supported)

        unknown_model_refs: list[str] = []
        default_model_use_case_mismatches: list[str] = []
        disabled_default_models: list[str] = []
        for use_case, model_id in self.config.default_models.items():
            profile = profiles_by_id.get(model_id)
            if profile is None:
                unknown_model_refs.append(f"default_models.{use_case}:{model_id}")
                continue
            if not profile.supports(use_case):
                default_model_use_case_mismatches.append(f"{use_case}:{model_id}")
            if not profile.enabled:
                disabled_default_models.append(f"{use_case}:{model_id}")

        fallback_model_use_case_mismatches: list[str] = []
        disabled_fallback_models: list[str] = []
        for use_case, model_ids in self.config.fallback_models.items():
            for model_id in model_ids:
                profile = profiles_by_id.get(model_id)
                if profile is None:
                    unknown_model_refs.append(f"fallback_models.{use_case}:{model_id}")
                    continue
                if not profile.supports(use_case):
                    fallback_model_use_case_mismatches.append(f"{use_case}:{model_id}")
                if not profile.enabled:
                    disabled_fallback_models.append(f"{use_case}:{model_id}")

        prompt_required = PROMPT_REQUIRED_USE_CASES & required
        prompt_supported = {
            profile.use_case
            for profile in self.config.prompt_profiles.values()
            if profile.use_case
        }
        unknown_default_prompt_use_cases = sorted(set(self.config.default_prompts) - KNOWN_USE_CASES)
        unknown_prompt_profile_use_cases = sorted(
            profile.use_case
            for profile in self.config.prompt_profiles.values()
            if profile.use_case and profile.use_case not in KNOWN_USE_CASES
        )
        missing_default_prompts = sorted(
            use_case
            for use_case in prompt_required
            if use_case not in self.config.default_prompts
        )
        missing_prompt_profiles = sorted(prompt_required - prompt_supported)
        unknown_prompt_refs: list[str] = []
        prompt_use_case_mismatches: list[str] = []
        prompt_file_missing: list[str] = []
        prompt_variable_mismatches: list[str] = []

        for use_case, prompt_id in self.config.default_prompts.items():
            profile = self.config.prompt_profiles.get(prompt_id)
            if profile is None:
                unknown_prompt_refs.append(f"default_prompts.{use_case}:{prompt_id}")
                continue
            if profile.use_case != use_case:
                prompt_use_case_mismatches.append(f"{use_case}:{prompt_id}->{profile.use_case}")

        for profile in self.config.prompt_profiles.values():
            path = self._prompt_path(profile)
            data = self._read_prompt_yaml(path)
            if data is None:
                if not profile.fallback_builtin:
                    prompt_file_missing.append(profile.id)
                continue
            declared = set(profile.input_variables or data.get("input_variables") or [])
            actual = _extract_prompt_variables(str(data.get("system", "")))
            missing_declared = sorted(actual - declared)
            stale_declared = sorted(declared - actual)
            if missing_declared or stale_declared:
                prompt_variable_mismatches.append(
                    f"{profile.id}:missing={missing_declared};unused={stale_declared}"
                )

        return {
            "missing_default_models": missing_default_models,
            "missing_supported_profiles": missing_supported_profiles,
            "unknown_default_use_cases": unknown_default_use_cases,
            "unknown_fallback_use_cases": unknown_fallback_use_cases,
            "unknown_profile_use_cases": unknown_profile_use_cases,
            "unknown_model_refs": sorted(unknown_model_refs),
            "default_model_use_case_mismatches": sorted(default_model_use_case_mismatches),
            "fallback_model_use_case_mismatches": sorted(fallback_model_use_case_mismatches),
            "disabled_default_models": sorted(disabled_default_models),
            "disabled_fallback_models": sorted(disabled_fallback_models),
            "missing_default_prompts": missing_default_prompts,
            "missing_prompt_profiles": missing_prompt_profiles,
            "unknown_default_prompt_use_cases": unknown_default_prompt_use_cases,
            "unknown_prompt_profile_use_cases": unknown_prompt_profile_use_cases,
            "unknown_prompt_refs": sorted(unknown_prompt_refs),
            "prompt_use_case_mismatches": sorted(prompt_use_case_mismatches),
            "prompt_file_missing": sorted(prompt_file_missing),
            "prompt_variable_mismatches": sorted(prompt_variable_mismatches),
        }

    def _prompt_path(self, profile: PromptProfile) -> Path:
        path = Path(profile.path)
        if path.is_absolute():
            return path
        base = Path(profile.base_dir or ".")
        return (base / path).resolve()

    def _read_prompt_yaml(self, path: Path) -> dict[str, Any] | None:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return None
        return data if isinstance(data, dict) and str(data.get("system", "")).strip() else None

    def _builtin_prompt_entry(
        self,
        prompt_id: str,
        use_case: str,
        profile: PromptProfile | None = None,
    ) -> GatewayPromptEntry:
        system = _BUILTIN_PROMPTS.get(prompt_id) or _BUILTIN_PROMPTS.get(use_case) or ""
        if not system:
            logger.warning("Prompt %s 没有可用 YAML，也没有内置兜底。", prompt_id)
        return GatewayPromptEntry(
            id=prompt_id,
            name=prompt_id,
            use_case=use_case,
            system=system,
            version="builtin",
            source="builtin",
            description=profile.description if profile else "",
            path=profile.path if profile else "",
            input_variables=profile.input_variables if profile else [],
            output_contract=profile.output_contract if profile else {},
        )

    def resolve_profile(
        self,
        use_case: str = "agent",
        model_id: str | None = None,
        model_type: str | None = "chat",
    ) -> ModelProfile | None:
        """解析某个场景最终应该用哪个 profile。

        优先级：
        1. 用户指定 model_id。
        2. YAML default_models[use_case]。
        3. .env default_llm_model_id。
        4. YAML default_model_id。
        5. 按 priority 选第一个支持该 use_case 的模型。
        """
        self._warn_unknown_use_case(use_case)
        enabled = [profile for profile in self.config.models if profile.enabled]
        if model_type:
            enabled = [profile for profile in enabled if profile.model_type == model_type]
        if model_id:
            # 用户指定模型时，仍要检查 enabled/use_case/model_type。
            return next(
                (profile for profile in enabled if profile.id == model_id and profile.supports(use_case)),
                None,
            )

        for default_id in [
            self.config.default_models.get(use_case, ""),
            getattr(self.settings, "default_llm_model_id", ""),
            self.config.default_model_id,
        ]:
            if not default_id:
                continue
            # 依次尝试不同来源的默认模型。
            default = next(
                (profile for profile in enabled if profile.id == default_id and profile.supports(use_case)),
                None,
            )
            if default:
                return default

        # 没有显式默认模型时，按 priority 自动选择。
        candidates = [profile for profile in enabled if profile.supports(use_case)]
        return sorted(candidates, key=lambda p: p.priority)[0] if candidates else None

    def _build_from_profile(self, profile: ModelProfile) -> BaseChatModel:
        """根据 profile 创建真实 ChatModel。"""
        api_key = self._api_key_for(profile)
        if not api_key and profile.requires_api_key:
            raise ValueError(f"模型 {profile.id} 需要环境变量 {profile.api_key_env}。")
        return _build_chat_model(
            provider=profile.provider,
            model=profile.model,
            api_key=api_key or "EMPTY",
            base_url=profile.base_url,
            extra_body=profile.extra_body,
            streaming=profile.streaming,
        )

    def _api_key_for(self, profile: ModelProfile) -> str:
        """读取模型密钥。

        生产环境优先使用真实环境变量；本地开发时兼容 pydantic-settings 从
        `.env` 读到的字段，避免必须手动 export。
        """
        value = os.getenv(profile.api_key_env, "")
        if value:
            return value
        env_name = profile.api_key_env.upper()
        settings_field = env_name.lower()
        return getattr(self.settings, settings_field, "")

    def api_key_for(self, profile: ModelProfile) -> str:
        """给 Embedding / Reranker 工厂读取密钥；调用方不应把返回值暴露给前端。"""
        return self._api_key_for(profile)

    def _build_legacy_model(self) -> BaseChatModel | None:
        """兼容旧 .env 配置。

        有些代码或用户可能还没迁移到 model_gateway.yaml，
        所以这里保留 llm_provider/llm_model/llm_api_key 的旧入口。
        """
        provider = getattr(self.settings, "llm_provider", "").strip().lower()
        if not provider:
            return None
        api_key = (
            getattr(self.settings, "anthropic_api_key", "")
            if provider == "anthropic"
            else getattr(self.settings, "llm_api_key", "")
        )
        if not api_key:
            return None
        model = getattr(self.settings, "llm_model", "") or ""
        base_url = getattr(self.settings, "llm_base_url", "") or ""
        return _build_chat_model(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            extra_body=_thinking_disabled_extra_body(provider, model, base_url),
        )

    def _legacy_public_dict(self) -> dict[str, Any] | None:
        """把旧 .env 模型配置转成前端可展示结构。"""
        provider = getattr(self.settings, "llm_provider", "").strip().lower()
        model = getattr(self.settings, "llm_model", "").strip()
        if not provider:
            return None
        return {
            "id": "legacy-env",
            "display_name": "Legacy .env Model",
            "provider": provider,
            "model": model,
            "enabled": True,
            "use_cases": ["agent", "workflow", "rag_rewrite", "judge"],
            "priority": 999,
            "model_type": "chat",
            "tier": "standard",
            "deployment": "env",
            "requires_api_key": True,
            "dimension": None,
            "top_k": None,
            "base_url_configured": bool(getattr(self.settings, "llm_base_url", "")),
            "api_key_configured": bool(
                getattr(self.settings, "anthropic_api_key", "")
                if provider == "anthropic"
                else getattr(self.settings, "llm_api_key", "")
            ),
        }


def _build_chat_model(
    provider: str,
    model: str,
    api_key: str,
    base_url: str = "",
    extra_body: dict[str, Any] | None = None,
    streaming: bool = False,
) -> BaseChatModel:
    """按 provider 创建对应的 LangChain ChatModel。

    新增模型厂商时，通常只需要在这里加一个 provider 分支。
    """
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model or "claude-3-5-sonnet-20241022", api_key=api_key)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model or "gpt-4o", api_key=api_key)

    if provider == "openai_compatible":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "deepseek-chat",
            api_key=api_key,
            base_url=base_url or None,
            extra_body=extra_body or None,
            streaming=streaming,
        )

    if provider == "tongyi":
        from langchain_community.chat_models import ChatTongyi
        return ChatTongyi(model=model or "qwen-max", dashscope_api_key=api_key)

    if provider == "zhipuai":
        from langchain_community.chat_models import ChatZhipuAI
        return ChatZhipuAI(model=model or "glm-4", api_key=api_key)

    if provider == "baidu":
        from langchain_community.chat_models import QianfanChatEndpoint
        return QianfanChatEndpoint(model=model or "ERNIE-4.0-8K", api_key=api_key)

    raise ValueError(f"不支持的 provider：{provider}")


def _thinking_disabled_extra_body(provider: str, model: str, base_url: str = "") -> dict[str, Any]:
    """为支持的 OpenAI-compatible 模型关闭原生 thinking/reasoning 模式。

    DeepSeek V4 这类模型如果不显式关闭 thinking，首字响应会明显变慢。
    YAML profile 可以自己配置 extra_body；这个函数主要兜底旧 `.env` 入口。
    """
    provider_name = provider.strip().lower()
    model_name = model.strip().lower()
    endpoint = base_url.strip().lower()
    if provider_name != "openai_compatible":
        return {}
    if "deepseek" in model_name or "deepseek" in endpoint:
        return {"thinking": {"type": "disabled"}}
    return {}


@lru_cache
def get_model_gateway() -> ModelGateway:
    """获取全局模型网关实例。

    lru_cache 让配置只加载一次，避免每次请求都读 YAML。
    """
    return ModelGateway(get_settings())
