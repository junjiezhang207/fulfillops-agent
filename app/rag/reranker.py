"""RAG 后处理组件（学习版注释）：Cross-Encoder 精排和上下文压缩。

在 RAG 里，召回阶段通常会拿到一批候选片段。
这些片段“可能相关”，但排序不一定足够准，也可能包含很多无关句子。

本文件做两件事：
1. CrossEncoderReranker：用更强但更慢的模型重新排序候选片段。
2. ContextualCompressor：从命中片段里抽取与问题直接相关的句子。

它们都是可选增强组件：加载失败或没配置模型时，主 RAG 链路仍能继续工作。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


@dataclass
class RankedPassage:
    """带分数的重排结果，主要用于调试和评测。"""

    # passage 原文。
    text: str
    # 原始召回分数，当前多数实现里没有传入，所以可能为 0。
    original_score: float
    # reranker 给出的相关性分数。
    rerank_score: float


class PassageReranker(Protocol):
    """Reranker 统一协议。

    KnowledgeRetrievalService 只依赖这个协议，不关心底层是本地 CrossEncoder
    还是云端 Jina API。这样更符合企业级“模型可替换”的设计。
    """

    def rerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        ...

    async def arerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        ...

    def rerank_with_scores(self, query: str, passages: list[str], top_k: int | None = None) -> list[RankedPassage]:
        ...


class CrossEncoderReranker:
    """使用 Cross-Encoder 对候选片段做精排。

    和普通 embedding 检索不同：
    - embedding 检索通常是先分别编码 query 和 passage，再算向量相似度。
    - Cross-Encoder 会把 query 和 passage 拼在一起输入模型，直接判断相关性。

    Cross-Encoder 更准，但更慢，所以一般只用于 top-N 候选的二次排序。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        top_k: int = 5,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._model = None

    def _load_model(self) -> bool:
        """懒加载模型。

        模型比较大，服务启动时不强制加载，可以减少启动压力。
        第一次真正 rerank 时再加载；加载失败则回退原始排序。
        """
        if self._model is not None:
            return True
        try:
            # sentence-transformers 的 CrossEncoder 会在本地加载模型权重。
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
            logger.info("CrossEncoder %s 加载成功", self._model_name)
            return True
        except ImportError:
            logger.warning("sentence-transformers 未安装，跳过 Cross-Encoder 重排。")
            return False
        except Exception as exc:
            logger.warning("CrossEncoder 加载失败（非致命）：%s", exc)
            return False

    def rerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        """同步重排，返回 top_k 最相关段落。"""
        # top_k 为空时使用构造函数默认值。
        k = top_k or self._top_k
        if not passages or not self._load_model():
            return passages[:k]
        try:
            # CrossEncoder 的输入是 (query, passage) pair。
            # 输出 score 越高，说明 passage 越适合作为这个 query 的上下文。
            pairs = [(query, p) for p in passages]
            scores = self._model.predict(pairs)
            ranked = sorted(zip(passages, scores), key=lambda x: float(x[1]), reverse=True)
            return [p for p, _ in ranked[:k]]
        except Exception as exc:
            logger.warning("Cross-Encoder 重排失败，使用原始顺序：%s", exc)
            return passages[:k]

    async def arerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        """异步重排：把 CPU/模型推理放到线程池，避免阻塞事件循环。"""
        # 本地 CrossEncoder 是同步计算，用 to_thread 包一层给 async 链路使用。
        return await asyncio.to_thread(self.rerank, query, passages, top_k)

    def rerank_with_scores(
        self, query: str, passages: list[str], top_k: int | None = None
    ) -> list[RankedPassage]:
        """返回带分数的重排结果。

        业务链路只需要排序后的文本；评测和调试时更关心分数。
        所以单独提供这个方法，不影响主流程。
        """
        k = top_k or self._top_k
        if not passages or not self._load_model():
            # 模型不可用时返回原文顺序，分数填 0，保证调用方不崩。
            return [RankedPassage(text=p, original_score=0.0, rerank_score=0.0) for p in passages[:k]]
        try:
            # 评分逻辑和 rerank 一样，只是这里保留分数。
            pairs = [(query, p) for p in passages]
            scores = list(self._model.predict(pairs))
            ranked = sorted(zip(passages, enumerate(scores)), key=lambda x: float(x[1][1]), reverse=True)
            return [
                RankedPassage(text=p, original_score=float(scores[orig_idx]), rerank_score=float(score))
                for p, (orig_idx, score) in ranked[:k]
            ]
        except Exception as exc:
            logger.warning("Cross-Encoder 评分失败：%s", exc)
            return [RankedPassage(text=p, original_score=0.0, rerank_score=0.0) for p in passages[:k]]


class JinaReranker:
    """Jina AI Rerank API 适配器。

    这个类和 CrossEncoderReranker 暴露同一组方法，便于在企业部署时
    在“本地开源 reranker”和“云端 reranker API”之间切换。
    """

    def __init__(
        self,
        model_name: str = "jina-reranker-v2-base-multilingual",
        api_key: str = "",
        base_url: str = "https://api.jina.ai/v1/rerank",
        top_k: int = 5,
    ) -> None:
        # 云端 reranker 模型名。
        self._model_name = model_name
        # API Key 只从后端配置读取，不应该给前端。
        self._api_key = api_key
        # 允许私有化部署/代理 Jina API。
        self._base_url = base_url or "https://api.jina.ai/v1/rerank"
        # 默认返回前 top_k 个。
        self._top_k = top_k

    def rerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        """返回重排后的文本列表，供主 RAG 链路直接使用。"""
        return [item.text for item in self.rerank_with_scores(query, passages, top_k)]

    async def arerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        """异步接口：Jina 调用目前通过线程池包同步请求，避免阻塞 async 链路。"""
        return await asyncio.to_thread(self.rerank, query, passages, top_k)

    def rerank_with_scores(
        self, query: str, passages: list[str], top_k: int | None = None
    ) -> list[RankedPassage]:
        """调用 Jina Rerank API，并保留 relevance_score。

        如果没有配置 API Key 或调用失败，返回原始顺序，主链路继续可用。
        """
        k = top_k or self._top_k
        if not passages or not self._api_key:
            # 没有 API Key 时降级为原始顺序。
            return [RankedPassage(text=p, original_score=0.0, rerank_score=0.0) for p in passages[:k]]
        try:
            import httpx

            # 调用 Jina rerank API。documents 是候选段落列表。
            response = httpx.post(
                self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model_name,
                    "query": query,
                    "documents": passages,
                    "top_n": k,
                },
                timeout=20.0,
            )
            response.raise_for_status()
            results = response.json().get("results") or []
            ranked: list[RankedPassage] = []
            for item in results:
                # Jina 返回 index，指向原 passages 中的下标。
                index = int(item.get("index", 0))
                score = float(item.get("relevance_score", 0.0))
                if 0 <= index < len(passages):
                    ranked.append(
                        RankedPassage(
                            text=passages[index],
                            original_score=0.0,
                            rerank_score=score,
                        )
                    )
            return ranked or [RankedPassage(text=p, original_score=0.0, rerank_score=0.0) for p in passages[:k]]
        except Exception as exc:
            logger.warning("Jina Reranker 调用失败，使用原始顺序：%s", exc)
            return [RankedPassage(text=p, original_score=0.0, rerank_score=0.0) for p in passages[:k]]


class DashScopeReranker:
    """阿里云百炼 DashScope Rerank API 适配器。

    qwen3-rerank 支持 OpenAI-compatible 的 ``/reranks`` 接口，也支持 DashScope
    文本重排序接口。两种接口的返回字段略有差异，所以解析时同时兼容
    ``results`` 和 ``output.results``。
    """

    def __init__(
        self,
        model_name: str = "qwen3-rerank",
        api_key: str = "",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
        top_k: int = 5,
    ) -> None:
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url or "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
        self._top_k = top_k

    def rerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        """返回重排后的文本列表，供主 RAG 链路直接使用。"""
        return [item.text for item in self.rerank_with_scores(query, passages, top_k)]

    async def arerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        """异步接口：通过线程池包同步 HTTP 请求。"""
        return await asyncio.to_thread(self.rerank, query, passages, top_k)

    def rerank_with_scores(
        self, query: str, passages: list[str], top_k: int | None = None
    ) -> list[RankedPassage]:
        """调用 DashScope Rerank API，并保留 relevance_score。"""
        k = top_k or self._top_k
        if not passages or not self._api_key:
            return [RankedPassage(text=p, original_score=0.0, rerank_score=0.0) for p in passages[:k]]
        try:
            import httpx

            response = httpx.post(
                self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model_name,
                    "query": query,
                    "documents": passages,
                    "top_n": k,
                },
                timeout=20.0,
            )
            response.raise_for_status()
            payload = response.json()
            results = (payload.get("output") or {}).get("results") or payload.get("results") or []
            ranked: list[RankedPassage] = []
            for item in results:
                index = int(item.get("index", 0))
                score = float(item.get("relevance_score", 0.0))
                if 0 <= index < len(passages):
                    ranked.append(
                        RankedPassage(
                            text=passages[index],
                            original_score=0.0,
                            rerank_score=score,
                        )
                    )
            return ranked or [RankedPassage(text=p, original_score=0.0, rerank_score=0.0) for p in passages[:k]]
        except Exception as exc:
            logger.warning("DashScope Reranker 调用失败，使用原始顺序：%s", exc)
            return [RankedPassage(text=p, original_score=0.0, rerank_score=0.0) for p in passages[:k]]


def create_reranker(settings: object) -> PassageReranker | None:
    """根据模型网关创建 reranker。

    默认使用本地 CrossEncoder；如果启用云端 reranker，则使用对应 API。
    创建失败时返回 None，RAG 主链路会保持原始排序。
    """
    try:
        from app.services.model_gateway import ModelGateway

        # 从模型网关找 use_case=reranker 的模型配置。
        gateway = ModelGateway(settings)
        profile = gateway.resolve_profile(use_case="reranker", model_type="reranker")
        if profile is None:
            return None
        provider = profile.provider.strip().lower()
        top_k = profile.top_k or 5
        if provider in {"sentence_transformers", "cross_encoder", "huggingface", "local"}:
            # 本地开源 reranker。
            return CrossEncoderReranker(model_name=profile.model, top_k=top_k)
        if provider == "jina":
            # 云端 Jina reranker。
            return JinaReranker(
                model_name=profile.model,
                api_key=gateway.api_key_for(profile),
                base_url=profile.base_url,
                top_k=top_k,
            )
        if provider in {"dashscope", "aliyun", "alibaba"}:
            # 阿里云百炼 DashScope reranker。
            return DashScopeReranker(
                model_name=profile.model,
                api_key=gateway.api_key_for(profile),
                base_url=profile.base_url,
                top_k=top_k,
            )
        logger.warning("不支持的 Reranker provider=%s，跳过 rerank。", provider)
        return None
    except Exception as exc:
        logger.warning("Reranker 创建失败，跳过 rerank：%s", exc)
        return None


# 上下文压缩 prompt：只抽取相关句子，不生成解释。
_COMPRESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个信息抽取助手。从给定段落中提取与问题直接相关的句子，"
        "删除无关内容。如果整段都不相关，返回空字符串。"
        "只返回提取的内容，不要添加解释或前缀。",
    ),
    ("human", "问题：{query}\n\n段落：{passage}"),
])


class ContextualCompressor:
    """抽取片段中与问题直接相关的句子，降低下游 prompt 噪声。

    召回出来的 chunk 往往比真正需要的证据更长。
    压缩器的作用是保留与 query 直接相关的句子，减少给 LLM 的无关上下文。
    """

    def __init__(self, chat_model: BaseChatModel | None = None) -> None:
        self._chat_model = chat_model

        # 没有传 chat_model 时，压缩器退化为原样返回 passages。
        # 这样 RAG 主链路不依赖压缩模型。
        self._chain = (
            _COMPRESS_PROMPT | chat_model | StrOutputParser()
            if chat_model is not None
            else None
        )

    def compress(self, query: str, passages: list[str]) -> list[str]:
        """同步压缩片段。"""
        if self._chain is None or not passages:
            return passages
        compressed = []
        for passage in passages:
            # 单段压缩失败时 _compress_one_sync 会返回原文。
            result = self._compress_one_sync(query, passage)
            if result is not None:
                compressed.append(result)
        # 如果全部被判定无关，保守返回原文，避免误删所有证据。
        return compressed or passages

    async def acompress(self, query: str, passages: list[str]) -> list[str]:
        """异步并行压缩片段。

        每个 passage 的压缩互不依赖，所以可以用 gather 同时发起多个 LLM 调用。
        """
        if self._chain is None or not passages:
            return passages

        results = await asyncio.gather(
            *[self._acompress_one(query, p) for p in passages],
            return_exceptions=True,
        )

        compressed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("异步压缩 chunk[%d] 失败，保留原文：%s", i, result)
                compressed.append(passages[i])
            elif result is not None:
                compressed.append(result)
            # result is None → 完全不相关，丢弃

        # 如果全部压缩为空，保守返回原文。
        return compressed or passages

    def _compress_one_sync(self, query: str, passage: str) -> str | None:
        """同步压缩单个 chunk，返回 None 表示完全不相关（丢弃）。"""
        try:
            # 让 LLM 从 passage 中抽取和 query 直接相关的句子。
            result = self._chain.invoke({"query": query, "passage": passage}).strip()
            if result and len(result) >= max(10, len(passage) * 0.1):
                return result
            if not result:
                return None

            # 如果模型只返回几个字，可能是压缩过度或回答异常。
            # 这种情况下保留原文比误删证据更安全。
            return passage  # 压缩结果太短，保留原文
        except Exception as exc:
            logger.warning("上下文压缩失败，保留原文：%s", exc)
            return passage

    async def _acompress_one(self, query: str, passage: str) -> str | None:
        """异步压缩单个 chunk，返回 None 表示完全不相关（丢弃）。"""
        # 异步版本逻辑和同步版本一致，只是调用 ainvoke。
        result = (await self._chain.ainvoke({"query": query, "passage": passage})).strip()
        if result and len(result) >= max(10, len(passage) * 0.1):
            return result
        if not result:
            return None
        return passage
