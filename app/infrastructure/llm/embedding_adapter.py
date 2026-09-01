"""Embedding 工厂 — 根据模型网关构造 LlamaIndex Embedding 实例。

优先级：
  1. model_gateway.yaml 中 default_models.embedding 指定的模型。
  2. 如果没有显式默认模型，按 enabled + priority 选择支持 embedding 的 profile。
  3. 模型网关不可用时，回退旧版 embed_provider / embed_model_name 配置。
  4. 构造失败 → 明确抛错，不再使用伪向量兜底。

当前默认模型：
  config/model_gateway.yaml 里 default_models.embedding 指向
  aliyun-text-embedding-v4，维度为 1024。BGE small 中文模型仍保留为
  本地可选 profile / 旧配置 fallback，维度为 512。

为什么不再保留伪向量兜底？
  伪向量不具备真实语义检索能力，维度也容易和 PGVector 表不一致。
  对这个项目来说，静默写入无语义向量比直接失败更危险：RAG 看起来能跑，
  但召回质量和评测结果都不可信。因此 embedding 配置失败时应显式暴露问题。

这个文件同时服务两条链路：
  - RAG 文档检索：KnowledgeRetrievalService 构建 VectorStoreIndex 时使用。
  - 长期记忆检索：PostgreSQL + PGVector 后端需要把记忆文本转成向量。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbedding(BaseEmbedding):
    """通用 OpenAI-compatible Embedding 适配器。

    LlamaIndex 官方 ``OpenAIEmbedding`` 会校验模型名枚举，阿里云百炼
    ``text-embedding-v4`` 这类兼容接口模型名可能被拒绝。
    这个适配器直接调用 ``{base_url}/embeddings``，只依赖通用协议字段：
    ``model``、``input``、``dimensions``。
    """

    api_key: str = Field(default="", exclude=True)
    api_base: str = Field(default="")
    dimensions: int | None = Field(default=None)
    timeout: float = Field(default=60.0)
    batch_size: int = Field(default=8)
    retry_attempts: int = Field(default=3)
    retry_backoff_seconds: float = Field(default=0.8)

    def _endpoint(self) -> str:
        return f"{self.api_base.rstrip('/')}/embeddings"

    def _payload(self, inputs: list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "input": inputs,
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        return payload

    def _embed_batch(self, inputs: list[str]) -> list[list[float]]:
        import httpx

        response = None
        for attempt in range(1, max(1, self.retry_attempts) + 1):
            try:
                response = httpx.post(
                    self._endpoint(),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=self._payload(inputs),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                break
            except Exception as exc:
                if not self._should_retry_embedding_error(exc, attempt):
                    raise
                time.sleep(self.retry_backoff_seconds * attempt)
        if response is None:
            raise RuntimeError("Embedding request did not return a response.")
        response.raise_for_status()
        payload = response.json()
        items = sorted(payload.get("data") or [], key=lambda item: int(item.get("index", 0)))
        embeddings = [item.get("embedding") for item in items]
        if len(embeddings) != len(inputs) or not all(isinstance(item, list) for item in embeddings):
            raise ValueError("OpenAI-compatible Embedding 返回格式异常。")
        return embeddings

    async def _aembed_batch(self, inputs: list[str]) -> list[list[float]]:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = None
            for attempt in range(1, max(1, self.retry_attempts) + 1):
                try:
                    response = await client.post(
                        self._endpoint(),
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=self._payload(inputs),
                    )
                    response.raise_for_status()
                    break
                except Exception as exc:
                    if not self._should_retry_embedding_error(exc, attempt):
                        raise
                    await asyncio.sleep(self.retry_backoff_seconds * attempt)
        if response is None:
            raise RuntimeError("Embedding request did not return a response.")
        response.raise_for_status()
        payload = response.json()
        items = sorted(payload.get("data") or [], key=lambda item: int(item.get("index", 0)))
        embeddings = [item.get("embedding") for item in items]
        if len(embeddings) != len(inputs) or not all(isinstance(item, list) for item in embeddings):
            raise ValueError("OpenAI-compatible Embedding 返回格式异常。")
        return embeddings

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed_batch([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return (await self._aembed_batch([query]))[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return (await self._aembed_batch([text]))[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), max(1, self.batch_size)):
            embeddings.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return embeddings

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), max(1, self.batch_size)):
            embeddings.extend(await self._aembed_batch(texts[start : start + self.batch_size]))
        return embeddings

    def _should_retry_embedding_error(self, exc: Exception, attempt: int) -> bool:
        if attempt >= max(1, self.retry_attempts):
            return False
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code is not None and status_code < 500:
            return False
        logger.warning("Embedding batch 调用失败，准备重试：attempt=%d error=%s", attempt, exc)
        return True


class LazyEmbeddingModel:
    """懒加载 Embedding 模型。

    用在长期记忆等只有在向量后端启用时才需要 embedding 的场景。
    底层模型可能是云端 OpenAI-compatible 接口，也可能是本地 HuggingFace 权重；
    延迟到第一次真正调用 embedding 方法时再构造，可以减少 FastAPI 启动阶段的外部依赖
    和本地模型加载成本。
    """

    def __init__(self, settings: object) -> None:
        self._settings = settings
        self._model = None

    def _get_model(self):
        """获取真实 embedding 模型，第一次调用时才加载。"""
        if self._model is None:
            self._model = create_embed_model(self._settings)
        return self._model

    def get_text_embedding(self, text: str):
        """LlamaIndex BaseEmbedding 风格接口。"""
        return self._get_model().get_text_embedding(text)

    def embed_query(self, text: str):
        """LangChain 风格接口。

        长期记忆 store 可能通过 embed_query 调用 embedding，
        所以这里做一层兼容。
        """
        model = self._get_model()
        if hasattr(model, "embed_query"):
            return model.embed_query(text)
        return model.get_text_embedding(text)

    def __call__(self, text: str):
        model = self._get_model()
        return model(text) if callable(model) else model.get_text_embedding(text)


def create_lazy_embed_model(settings: object) -> LazyEmbeddingModel:
    """创建懒加载 Embedding 代理。"""
    return LazyEmbeddingModel(settings)


def create_embed_model(settings: object):
    """根据 Settings 配置构造 LlamaIndex Embedding 实例。

    Args:
        settings: app.core.config.Settings 实例

    Returns:
        LlamaIndex BaseEmbedding 实例。创建失败时抛出 RuntimeError。
    """
    # 优先走企业级模型网关。这样 embedding 可以和 chat/reranker 一样由网关统一配置。
    gateway_embed = _create_gateway_embed(settings)
    if gateway_embed is not None:
        return gateway_embed

    provider = getattr(settings, "embed_provider", "local").strip().lower()
    if provider == "openai":
        return _create_openai_embed(settings)
    if provider == "mock":
        raise RuntimeError("embed_provider=mock 已禁用，请配置真实 embedding 模型。")
    # 走到这里说明模型网关没有成功创建 embedding，进入旧配置 fallback。
    # 旧配置默认 local（HuggingFace BGE），但当前项目主默认路径是模型网关里的阿里云 embedding。
    return _create_local_embed(settings)


def _create_gateway_embed(settings: object):
    """根据 ModelGateway 创建 embedding。

    支持两类常见企业配置：
      - 本地开源 embedding：HuggingFace / sentence-transformers。
      - 云端 OpenAI-compatible embedding：例如阿里云百炼、私有模型网关、
        兼容 OpenAI API 的服务。
    """
    try:
        from app.infrastructure.llm.model_gateway import ModelGateway

        gateway = ModelGateway(settings)
        profile = gateway.resolve_profile(use_case="embedding", model_type="embedding")
        if profile is None:
            return None
        provider = profile.provider.strip().lower()
        if provider in {"huggingface", "sentence_transformers", "local"}:
            return _create_local_embed(settings, model_name=profile.model)
        if provider == "openai":
            return _create_openai_embed(
                settings,
                model_name=profile.model,
                api_key=gateway.api_key_for(profile),
                api_base=profile.base_url or None,
                dimensions=profile.dimension,
            )
        if provider == "openai_compatible":
            return _create_openai_compatible_embed(
                model_name=profile.model,
                api_key=gateway.api_key_for(profile),
                api_base=profile.base_url,
                dimensions=profile.dimension,
            )
        if provider == "mock":
            raise RuntimeError("模型网关 embedding provider=mock 已禁用。")
        logger.warning("不支持的 Embedding provider=%s，回退旧配置。", provider)
        return None
    except Exception as exc:
        logger.warning("模型网关 Embedding 创建失败，回退旧配置：%s", exc)
        return None


def _create_local_embed(settings: object, model_name: str | None = None):
    """创建本地 HuggingFace embedding。

    当模型网关未给出 embedding profile，或显式选择本地 provider 时，
    默认模型是 BGE small 中文模型，适合中文供应链知识库，输出维度 512。
    当前项目主默认路径不是这里，而是模型网关中的阿里云 text-embedding-v4。
    如果本地依赖或权重不可用，会抛出 RuntimeError，避免系统静默使用伪向量。
    """
    model_name = model_name or getattr(settings, "embed_model_name", "BAAI/bge-small-zh-v1.5")
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed = HuggingFaceEmbedding(model_name=model_name)
        logger.info("Embedding 模型已加载：%s", model_name)
        return embed
    except ImportError:
        raise RuntimeError(
            "llama-index-embeddings-huggingface 未安装，无法创建本地 Embedding。"
            "请运行 `uv add llama-index-embeddings-huggingface` 安装，"
            "或在模型网关中配置可用的云端 embedding。"
        )
    except Exception as exc:
        raise RuntimeError(f"本地 Embedding 加载失败（{model_name}）：{exc}") from exc


def _create_openai_embed(
    settings: object,
    model_name: str = "text-embedding-3-small",
    api_key: str | None = None,
    api_base: str | None = None,
    dimensions: int | None = None,
):
    """创建 LlamaIndex OpenAIEmbedding。

    这条路径适合 LlamaIndex 官方支持的 OpenAI 模型名。
    api_base 允许接企业内部模型网关，而不一定是 OpenAI 官方地址。
    dimensions 用于 text-embedding-3-small 这类可配置维度的模型。
    对于阿里云 text-embedding-v4 这类 LlamaIndex 可能不认识的模型名，
    走 _create_openai_compatible_embed 的自定义适配器。
    """
    api_key = api_key if api_key is not None else getattr(settings, "llm_api_key", "")
    if not api_key:
        raise RuntimeError("Embedding 云模型缺少 API Key。")
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            dimensions=dimensions,
        )
    except ImportError:
        raise RuntimeError("llama-index-embeddings-openai 未安装，无法创建 OpenAIEmbedding。")


def _create_openai_compatible_embed(
    model_name: str,
    api_key: str,
    api_base: str,
    dimensions: int | None = None,
):
    """创建 OpenAI-compatible embedding，支持阿里云等非 OpenAI 官方模型名。"""
    if not api_key:
        raise RuntimeError("Embedding 云模型缺少 API Key。")
    if not api_base:
        raise RuntimeError("OpenAI-compatible Embedding 缺少 base_url。")
    logger.info("OpenAI-compatible Embedding 已配置：model=%s base_url=%s dim=%s", model_name, api_base, dimensions)
    return OpenAICompatibleEmbedding(
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        dimensions=dimensions,
    )
