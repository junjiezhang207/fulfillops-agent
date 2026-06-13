"""RAG 后处理组件：候选片段精排与上下文压缩。

RAG 的第一阶段召回通常会拿到一批“可能相关”的 chunk。向量检索和 BM25
更擅长扩大召回覆盖，但它们不一定能把最适合回答用户问题的证据排在最前。
本文件提供两类可选增强能力：

1. Reranker：对召回后的候选片段做二次排序，把更相关的证据提前。
2. ContextualCompressor：从较长 chunk 中抽取和问题直接相关的句子，降低下游 prompt 噪声。

这两个能力都不是 RAG 主链路的硬依赖。模型不可用、API Key 缺失、调用失败时，
代码都会保守退回到原始候选顺序或原始文本，保证检索链路仍然可用。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.infrastructure.llm.model_gateway import get_model_gateway

logger = logging.getLogger(__name__)


@dataclass
class RankedPassage:
    """带分数的重排结果。

    KnowledgeRetrievalService 最终会把 rerank_score 写入 hit 的 score_detail，
    这样前端和 Trace Center 可以解释：某条证据是因为原始召回分高排在前面，
    还是经过 reranker 二次判断后被提前。
    """

    # 候选片段原文。
    text: str
    # 原始召回分。当前 CrossEncoder / DashScope 适配器没有外部传入原始分，
    # 因此这里通常填 0；保留字段是为了以后接入更完整的评分流水线。
    original_score: float
    # reranker 给出的相关性分数。分数越高，说明该 passage 越适合作为 query 的证据。
    rerank_score: float


class PassageReranker(Protocol):
    """Reranker 的统一协议。

    RAG 主服务只依赖这个协议，不关心底层具体是本地 CrossEncoder，
    还是云端 DashScope qwen3-rerank。这样模型来源可以通过模型网关配置切换，
    业务检索链路不需要跟着改。
    """

    def rerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        """同步重排，只返回文本列表。"""
        ...

    async def arerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        """异步重排，供异步 RAG 链路使用。"""
        ...

    def rerank_with_scores(self, query: str, passages: list[str], top_k: int | None = None) -> list[RankedPassage]:
        """同步重排并保留分数，供排序解释和测试使用。"""
        ...


class CrossEncoderReranker:
    """本地 CrossEncoder 精排器。

    CrossEncoder 和普通 embedding 检索的区别：
    - embedding 检索会分别编码 query 和 passage，再计算向量相似度，适合大规模召回。
    - CrossEncoder 会把 query 和 passage 作为一对输入模型，直接判断二者相关性，
      通常更准，但成本更高、速度更慢。

    因此它只适合处理召回后的前 N 个候选，不适合替代向量库做全量检索。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        top_k: int = 5,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        # 模型对象延迟加载。sentence-transformers 模型可能较大，不在服务启动时强制加载，
        # 避免为了一个可选增强能力拖慢后端启动。
        self._model = None

    def _load_model(self) -> bool:
        """懒加载本地 CrossEncoder 模型。

        返回 True 表示模型可用；返回 False 表示应该跳过 rerank。
        这里不向外抛异常，因为 reranker 是增强能力，不应该拖垮 RAG 主链路。
        """
        if self._model is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
            logger.info("CrossEncoder %s 加载成功", self._model_name)
            return True
        except ImportError:
            logger.warning("未安装 sentence-transformers，跳过 CrossEncoder 精排。")
            return False
        except Exception as exc:
            logger.warning("CrossEncoder 加载失败，跳过精排：%s", exc)
            return False

    def rerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        """同步精排候选片段，返回 top_k 个文本。

        如果模型不可用或推理失败，保守返回原始顺序的前 top_k 个片段。
        """
        k = top_k or self._top_k
        if not passages or not self._load_model():
            return passages[:k]
        try:
            # CrossEncoder 输入是 (query, passage) pair，模型会直接输出相关性分数。
            pairs = [(query, passage) for passage in passages]
            scores = self._model.predict(pairs)
            ranked = sorted(zip(passages, scores), key=lambda item: float(item[1]), reverse=True)
            return [passage for passage, _ in ranked[:k]]
        except Exception as exc:
            logger.warning("CrossEncoder 精排失败，使用原始顺序：%s", exc)
            return passages[:k]

    async def arerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        """异步精排。

        sentence-transformers 的 CrossEncoder 是同步推理，为了不阻塞 async RAG 链路，
        这里放到线程池中执行。
        """
        return await asyncio.to_thread(self.rerank, query, passages, top_k)

    def rerank_with_scores(
        self, query: str, passages: list[str], top_k: int | None = None
    ) -> list[RankedPassage]:
        """同步精排并返回分数明细。

        业务回答只需要排序后的文本，但测试、Trace 和前端 evidence 解释需要看到
        rerank_score，因此单独提供这个方法。
        """
        k = top_k or self._top_k
        if not passages or not self._load_model():
            return [RankedPassage(text=passage, original_score=0.0, rerank_score=0.0) for passage in passages[:k]]
        try:
            pairs = [(query, passage) for passage in passages]
            scores = list(self._model.predict(pairs))
            ranked = sorted(
                zip(passages, enumerate(scores)),
                key=lambda item: float(item[1][1]),
                reverse=True,
            )
            return [
                RankedPassage(text=passage, original_score=float(scores[original_index]), rerank_score=float(score))
                for passage, (original_index, score) in ranked[:k]
            ]
        except Exception as exc:
            logger.warning("CrossEncoder 评分失败，使用原始顺序：%s", exc)
            return [RankedPassage(text=passage, original_score=0.0, rerank_score=0.0) for passage in passages[:k]]


class DashScopeReranker:
    """阿里云百炼 DashScope qwen3-rerank 适配器。

    当前通过 OpenAI-compatible 的 `/reranks` 接口调用云端 reranker。
    它适合不想在本地加载 CrossEncoder 模型、或希望使用云端托管精排模型的场景。

    注意：没有 API Key 时不会报错中断，而是返回原始顺序；这是为了保证
    RAG 在开发环境和演示环境里不会因为精排能力缺失而整体不可用。
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
        """同步精排，只返回排序后的文本。"""
        return [item.text for item in self.rerank_with_scores(query, passages, top_k)]

    async def arerank(self, query: str, passages: list[str], top_k: int | None = None) -> list[str]:
        """异步精排。

        当前 httpx 调用是同步请求，放到线程池里执行，避免阻塞事件循环。
        """
        return await asyncio.to_thread(self.rerank, query, passages, top_k)

    def rerank_with_scores(
        self, query: str, passages: list[str], top_k: int | None = None
    ) -> list[RankedPassage]:
        """调用 DashScope rerank 接口，并把 relevance_score 转成 RankedPassage。"""
        k = top_k or self._top_k
        if not passages or not self._api_key:
            return [RankedPassage(text=passage, original_score=0.0, rerank_score=0.0) for passage in passages[:k]]
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

            # DashScope 兼容接口可能返回 {"results": [...]}，也可能返回
            # {"output": {"results": [...]}}。这里同时兼容两种格式。
            results = (payload.get("output") or {}).get("results") or payload.get("results") or []
            ranked: list[RankedPassage] = []
            for item in results:
                # API 返回的 index 指向原 passages 列表中的位置。
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

            # 如果接口返回为空，仍然回退原始顺序，避免 RAG 没有证据可用。
            return ranked or [
                RankedPassage(text=passage, original_score=0.0, rerank_score=0.0)
                for passage in passages[:k]
            ]
        except Exception as exc:
            logger.warning("DashScope 精排失败，使用原始顺序：%s", exc)
            return [RankedPassage(text=passage, original_score=0.0, rerank_score=0.0) for passage in passages[:k]]


def create_reranker(settings: object) -> PassageReranker | None:
    """根据模型网关配置创建 reranker。

    模型网关负责解析 `use_case=reranker` 的 profile，本函数只负责把 profile
    转成具体适配器：

    - sentence_transformers / cross_encoder / huggingface / local -> CrossEncoderReranker
    - dashscope / aliyun / alibaba -> DashScopeReranker

    如果没有配置、provider 不支持或创建失败，返回 None。调用方会跳过 rerank。
    """

    try:
        from app.infrastructure.llm.model_gateway import ModelGateway

        gateway = ModelGateway(settings)
        profile = gateway.resolve_profile(use_case="reranker", model_type="reranker")
        if profile is None:
            return None
        provider = profile.provider.strip().lower()
        top_k = profile.top_k or 5
        if provider in {"sentence_transformers", "cross_encoder", "huggingface", "local"}:
            return CrossEncoderReranker(model_name=profile.model, top_k=top_k)
        if provider in {"dashscope", "aliyun", "alibaba"}:
            return DashScopeReranker(
                model_name=profile.model,
                api_key=gateway.api_key_for(profile),
                base_url=profile.base_url,
                top_k=top_k,
            )
        logger.warning("不支持的 reranker provider=%s，跳过精排。", provider)
        return None
    except Exception as exc:
        logger.warning("Reranker 创建失败，跳过精排：%s", exc)
        return None


# 上下文压缩 prompt：只做中文证据抽取，不做总结、改写或补充解释。
# 目标是从较长的知识库片段里保留和用户问题直接相关的句子，减少下游 prompt 噪声。
_COMPRESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            get_model_gateway().prompt_system(use_case="rag_compress"),
        ),
        ("human", "用户问题：{query}\n\n候选段落：{passage}"),
    ]
)


class ContextualCompressor:
    """上下文压缩器：从召回片段中抽取和问题直接相关的句子。

    向量库召回出的 chunk 往往比最终回答需要的证据更长。压缩器的目标是：
    - 减少传给下游 LLM 的无关上下文。
    - 降低 prompt token 成本。
    - 保留最相关的证据句，避免答案被无关段落干扰。

    如果没有传入 chat_model，压缩器会退化为原样返回 passages。
    """

    def __init__(self, chat_model: BaseChatModel | None = None) -> None:
        self._chat_model = chat_model
        self._chain = (
            _COMPRESS_PROMPT | chat_model | StrOutputParser()
            if chat_model is not None
            else None
        )

    def compress(self, query: str, passages: list[str]) -> list[str]:
        """同步压缩多个 passage。

        单个 passage 被判定完全无关时返回 None，并从压缩结果里丢弃。
        如果全部 passage 都被丢弃，则保守返回原始 passages，避免误删所有证据。
        """
        if self._chain is None or not passages:
            return passages
        compressed = []
        for passage in passages:
            result = self._compress_one_sync(query, passage)
            if result is not None:
                compressed.append(result)
        return compressed or passages

    async def acompress(self, query: str, passages: list[str]) -> list[str]:
        """异步并行压缩多个 passage。

        每个 passage 的压缩互不依赖，可以用 asyncio.gather 并行调用模型。
        某个 passage 压缩失败时只保留该 passage 原文，不影响其它 passage。
        """
        if self._chain is None or not passages:
            return passages

        results = await asyncio.gather(
            *[self._acompress_one(query, passage) for passage in passages],
            return_exceptions=True,
        )

        compressed = []
        for index, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning("异步压缩 chunk[%d] 失败，保留原文：%s", index, result)
                compressed.append(passages[index])
            elif result is not None:
                compressed.append(result)
        return compressed or passages

    def _compress_one_sync(self, query: str, passage: str) -> str | None:
        """同步压缩单个 passage。

        返回值约定：
        - str：压缩后的证据文本，或在压缩结果过短时返回原文。
        - None：模型判断该 passage 和问题完全无关，可以丢弃。
        """
        try:
            result = self._chain.invoke({"query": query, "passage": passage}).strip()
            if result and len(result) >= max(10, len(passage) * 0.1):
                return result
            if not result:
                return None

            # 如果模型只返回非常短的文本，可能是过度压缩或输出异常。
            # 保留原文比误删关键证据更稳妥。
            return passage
        except Exception as exc:
            logger.warning("上下文压缩失败，保留原文：%s", exc)
            return passage

    async def _acompress_one(self, query: str, passage: str) -> str | None:
        """异步压缩单个 passage，逻辑和同步版本保持一致。"""
        result = (await self._chain.ainvoke({"query": query, "passage": passage})).strip()
        if result and len(result) >= max(10, len(passage) * 0.1):
            return result
        if not result:
            return None
        return passage
