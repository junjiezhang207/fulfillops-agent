"""LLM 查询改写器（学习版注释）。

Query Rewriting 是 RAG 里常见的召回增强手段。

举个例子：
用户问：“这个订单库存不够怎么办？”
模型可以改写成：
- “缺货订单处理规则”
- “库存不足跨仓调拨”
- “履约替代方案”

这些改写 query 会和原始问题一起检索，能提高召回率。
注意：本类只负责 LLM 改写；业务规则扩展在 KnowledgeRetrievalService 里完成。
"""

from __future__ import annotations

import re
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.infrastructure.llm.model_gateway import get_model_gateway

logger = logging.getLogger(__name__)

# Query Rewrite 的提示词。
# 它要求模型“一行一个 query”，是为了后面解析简单稳定。
_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        get_model_gateway().prompt_system(use_case="query_rewrite"),
    ),
    ("human", "请把原始问题改写成 {n} 个不同角度的检索查询。\n原始问题：{question}"),
])


class QueryRewriter:
    """把用户问题改写成几条更适合召回的短查询。

    这个类提供同步和异步两个接口：
    - rewrite(): 普通同步调用。
    - arewrite(): async 调用链中使用，避免阻塞事件循环。

    如果没有传入 chat_model，本类会自动退化为只返回原始问题。
    """

    def __init__(
        self,
        chat_model: BaseChatModel | None = None,
        max_variants: int = 3,
    ) -> None:
        # 可选 LLM。没传模型时，改写器自动变成 no-op。
        self._chat_model = chat_model
        # 默认最多生成几个改写 query。
        self._max_variants = max_variants

        # LCEL 链：Prompt -> LLM -> String Parser。
        # 有模型时才创建链；没有模型时整个 rewriter 就是一个安全的 no-op。
        self._chain = (
            _REWRITE_PROMPT | self._chat_model | StrOutputParser()
            if self._chat_model is not None
            else None
        )

    def rewrite(self, question: str, n: int | None = None) -> list[str]:
        """同步改写，返回原始问题 + N 条改写变体（去重后）。"""
        # 调用方没指定 n 时使用默认值。
        n = n or self._max_variants

        # 空问题或未配置 LLM 时不要报错，直接回退原始 query。
        # RAG 的召回链路应该尽量“可降级”，不能因为改写失败就整体不可用。
        if not question.strip() or self._chain is None:
            return [question]
        try:
            # LCEL 同步调用：Prompt -> LLM -> StrOutputParser。
            raw = self._chain.invoke({"question": question, "n": n})
            return self._build_result(question, raw, n)
        except Exception as exc:
            logger.warning("QueryRewriter 同步改写失败，回退到原始查询: %s", exc)
            return [question]

    async def arewrite(self, question: str, n: int | None = None) -> list[str]:
        """异步改写，供 async RAG 链路调用。"""
        n = n or self._max_variants
        if not question.strip() or self._chain is None:
            return [question]
        try:
            # async RAG 链路里用 ainvoke，避免阻塞事件循环。
            raw = await self._chain.ainvoke({"question": question, "n": n})
            return self._build_result(question, raw, n)
        except Exception as exc:
            logger.warning("QueryRewriter 异步改写失败，回退到原始查询: %s", exc)
            return [question]

    def _build_result(self, question: str, raw: str, n: int) -> list[str]:
        """把 LLM 原始输出整理成最终 query 列表。

        LLM 可能输出编号、项目符号或重复内容，所以这里统一解析和去重。
        原始问题永远放在第一位，防止改写方向跑偏。
        """
        # 先从 LLM 原始文本里解析出候选 query。
        variants = self._parse_variants(raw)
        # 原始问题永远放第一位，防止 LLM 改写偏题。
        all_queries = [question] + [v for v in variants if v != question]
        # dict.fromkeys 可以保持顺序去重；最后限制为 原问题 + n 条变体。
        return list(dict.fromkeys(all_queries))[:n + 1]

    @staticmethod
    def _parse_variants(raw: str) -> list[str]:
        """解析 LLM 输出的多行 query。"""
        # 去掉 “1.”、“-”、“*” 等常见列表前缀，只保留 query 文本。
        lines = [
            re.sub(r"^[\d\.\-\*．、]+\s*", "", line.strip()).strip()
            for line in raw.strip().splitlines()
        ]
        # 过滤过短行，避免模型输出空行或“好的”等无效内容。
        return [line for line in lines if len(line) >= 4]
