import pytest

from app.services.query_rewriter import QueryRewriter


class _SyncChain:
    def __init__(self, output: str | Exception) -> None:
        self.output = output
        self.inputs = None

    def invoke(self, inputs: dict):
        self.inputs = inputs
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class _AsyncChain:
    def __init__(self, output: str | Exception) -> None:
        self.output = output
        self.inputs = None

    async def ainvoke(self, inputs: dict):
        self.inputs = inputs
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def test_query_rewriter_without_model_returns_original_question():
    rewriter = QueryRewriter(chat_model=None)

    assert rewriter.rewrite("库存不足怎么办") == ["库存不足怎么办"]


def test_query_rewriter_parses_numbered_bullets_dedupes_and_limits():
    rewriter = QueryRewriter(chat_model=None)
    rewriter._chain = _SyncChain(
        """
        1. 缺货订单处理规则
        - 库存不足跨仓调拨
        * 缺货订单处理规则
        3、履约风险人工复核
        好
        """
    )

    result = rewriter.rewrite("库存不足怎么办", n=2)

    assert result == ["库存不足怎么办", "缺货订单处理规则", "库存不足跨仓调拨"]
    assert rewriter._chain.inputs == {"question": "库存不足怎么办", "n": 2}


def test_query_rewriter_falls_back_when_sync_chain_fails():
    rewriter = QueryRewriter(chat_model=None)
    rewriter._chain = _SyncChain(RuntimeError("model unavailable"))

    assert rewriter.rewrite("库存不足怎么办", n=3) == ["库存不足怎么办"]


@pytest.mark.asyncio
async def test_query_rewriter_async_parses_output():
    rewriter = QueryRewriter(chat_model=None)
    rewriter._chain = _AsyncChain("1. 缺货处理\n2. 仓库调拨\n3. 优先级规则")

    result = await rewriter.arewrite("订单缺货怎么办", n=2)

    assert result == ["订单缺货怎么办", "缺货处理", "仓库调拨"]
    assert rewriter._chain.inputs == {"question": "订单缺货怎么办", "n": 2}


@pytest.mark.asyncio
async def test_query_rewriter_async_falls_back_when_chain_fails():
    rewriter = QueryRewriter(chat_model=None)
    rewriter._chain = _AsyncChain(RuntimeError("timeout"))

    assert await rewriter.arewrite("订单缺货怎么办", n=2) == ["订单缺货怎么办"]
