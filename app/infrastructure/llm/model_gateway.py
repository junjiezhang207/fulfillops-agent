"""后端模型网关（学习版注释）。

企业级项目里，前端不应该直接持有各家大模型 API Key。
本模块把“有哪些模型、哪个模型用于哪个场景、密钥从哪里读取”收口到后端：

- 模型清单来自 YAML 配置，后续可以替换成数据库/配置中心。
- API Key 只从环境变量读取，不出现在前端响应里。
- 调用方只按 use_case / model_id 获取 LangChain ChatModel。
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel

from app.core.config import get_settings

logger = logging.getLogger(__name__)


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
class ModelGatewayConfig:
    """模型网关完整配置。"""

    # 老版默认模型字段。
    default_model_id: str = ""
    # 新版按 use_case 指定默认模型，例如 agent -> cloud-large。
    default_models: dict[str, str] = field(default_factory=dict)
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
    return ModelGatewayConfig(
        default_model_id=str(data.get("default_model_id", "") or "").strip(),
        default_models={
            str(key).strip(): str(value).strip()
            for key, value in dict(data.get("default_models") or {}).items()
            if str(key).strip() and str(value).strip()
        },
        models=models,
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

    def list_models(
        self,
        use_case: str | None = None,
        include_disabled: bool = False,
        model_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """列出可用模型，主要给前端模型网关面板使用。"""
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
            return profile.public_dict(api_key_configured=bool(self._api_key_for(profile)))
        return self._legacy_public_dict()

    def create_chat_model(self, use_case: str = "agent", model_id: str | None = None) -> BaseChatModel | None:
        """创建 LangChain ChatModel。

        失败时返回 None，让上层可以降级到规则模板，而不是启动失败。
        """
        profile = self.resolve_profile(use_case=use_case, model_id=model_id, model_type="chat")
        try:
            if profile:
                return self._build_from_profile(profile)
            return self._build_legacy_model()
        except Exception as exc:
            target = model_id or getattr(self.settings, "default_llm_model_id", "") or "legacy-env"
            logger.warning("构造模型 %s 失败，将使用规则模板：%s", target, exc)
            return None

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
