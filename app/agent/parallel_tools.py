"""文件作用摘要：让多个互不依赖的 Agent 工具并发执行。

ReAct Agent 默认往往是串行调用工具：先查订单，再查库存，再查知识库。
但有些工具之间没有依赖关系，例如同时查库存和查规则，串行会浪费等待时间。
这个文件提供并发执行能力，目标是降低端到端延迟。

主要做的事：
1. ``ParallelToolRunner.run_parallel``：异步并发执行多个 LangChain 工具。
2. ``run_parallel_sync``：给同步场景提供一个薄封装，内部仍复用异步并发主流程。
3. ``make_parallel_query_tool``：创建一个名为 ``parallel_query`` 的元工具。
4. 对每个工具调用做独立异常捕获，避免一个工具失败影响全部结果。
5. 输出统一 JSON，方便 LLM 或前端理解每个工具的结果。

两种使用方式：
- 外部并行：业务代码提前知道要查哪些工具，直接用 ``ParallelToolRunner``。
- Agent 主动并行：把 ``parallel_query`` 注册给 Agent，让模型自己决定并行查什么。

学习时先看：
1. ``ParallelToolRunner.run_parallel``：asyncio 并发主流程。
2. ``run_parallel_sync``：同步入口如何复用 async-native 工具并发。
3. ``make_parallel_query_tool``：元工具的参数格式和返回格式。
"""

from __future__ import annotations

import asyncio
import json
import logging

from langchain_core.tools import BaseTool, StructuredTool, ToolException

logger = logging.getLogger(__name__)


# ── 模式 A：外部并行执行器 ────────────────────────────────────────────────────

# 面试官可能问：Agent 工具为什么需要并行执行？
# 回答：有些工具互不依赖，比如查库存和查规则可以同时做。并行执行能减少
# 总等待时间，尤其在工具访问外部系统时，比串行 ReAct 一步步查更快。
class ParallelToolRunner:
    """在 Agent 之外并行执行多个工具，结果注入会话上下文。

    使用方式：
        runner = ParallelToolRunner(max_workers=4)
        results = await runner.run_parallel([
            (check_inventory_tool, {"order_id": "SO123"}),
            (retrieve_knowledge_tool, {"order_id": "SO123", "question": "缺货处理"}),
        ])
        # results = [{"tool": "check_inventory", "result": "..."},...]
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._max_workers = max_workers

    async def run_parallel(
        self,
        tool_calls: list[tuple[BaseTool, dict]],
    ) -> list[dict]:
        """并发执行工具列表，返回按输入顺序排列的结果列表。

        Args:
            tool_calls: [(tool, args_dict), ...]

        Returns:
            [{"tool": tool_name, "result": output, "error": None | str}, ...]
        """
        semaphore = asyncio.Semaphore(self._max_workers)

        async def _invoke_one(tool: BaseTool, args: dict) -> dict:
            async with semaphore:
                try:
                    result = await tool.ainvoke(args)
                    return {"tool": tool.name, "result": str(result), "error": None}
                except Exception as exc:
                    logger.warning("并行工具 %s 执行失败：%s", tool.name, exc)
                    return {"tool": tool.name, "result": "", "error": str(exc)}

        tasks = [_invoke_one(tool, args) for tool, args in tool_calls]
        return await asyncio.gather(*tasks)

    def run_parallel_sync(
        self,
        tool_calls: list[tuple[BaseTool, dict]],
    ) -> list[dict]:
        """同步版本的并行执行（在无事件循环的上下文中复用异步主流程）。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_parallel(tool_calls))
        raise RuntimeError("当前已有事件循环，请使用 await run_parallel(...)")


# ── 模式 B：并行元工具（Agent 主动调用）──────────────────────────────────────

# 面试官可能问：parallel_query 元工具有什么风险？
# 回答：它让模型一次触发多个工具，提升效率，但也可能增加下游压力。
# 所以生产上要限制可并行工具白名单、最大并发数和参数校验，避免模型滥用。
def make_parallel_query_tool(tools_registry: dict[str, BaseTool]) -> StructuredTool:
    """创建 parallel_query 元工具，让 Agent 主动并发调用多个工具。

    Agent 调用示例：
        parallel_query(queries=[
            {"tool": "check_inventory", "args": {"order_id": "SO123"}},
            {"tool": "retrieve_knowledge", "args": {"order_id": "SO123",
              "question": "缺货规则", "categories": ""}},
        ])

    Args:
        tools_registry: {tool_name: BaseTool} 字典，Agent 可并行调用其中的工具

    Returns:
        StructuredTool 可直接注册到 Agent 的工具列表
    """
    runner = ParallelToolRunner()

    def _build_tool_calls(queries: list[dict]) -> list[tuple[BaseTool, dict]]:
        if not queries:
            raise ToolException("queries 不能为空")

        tool_calls: list[tuple[BaseTool, dict]] = []
        for q in queries:
            tool_name = q.get("tool", "")
            args = q.get("args", {})
            if tool_name not in tools_registry:
                raise ToolException(
                    f"工具 '{tool_name}' 不存在。"
                    f"可用工具：{list(tools_registry.keys())}"
                )
            tool_calls.append((tools_registry[tool_name], args))
        return tool_calls

    def _format_results(results: list[dict]) -> str:
        return json.dumps(
            {
                "parallel_results": results,
                "summary": f"并行执行了 {len(results)} 个工具",
            },
            ensure_ascii=False,
        )

    def _parallel_query(queries: list[dict]) -> str:
        """并行调用多个工具并返回合并结果。

        Args:
            queries: [{"tool": "工具名", "args": {参数字典}}, ...]
        """
        try:
            tool_calls = _build_tool_calls(queries)
        except ToolException as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

        results = runner.run_parallel_sync(tool_calls)
        return _format_results(results)

    async def _parallel_query_async(queries: list[dict]) -> str:
        try:
            tool_calls = _build_tool_calls(queries)
        except ToolException as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

        results = await runner.run_parallel(tool_calls)
        return _format_results(results)

    return StructuredTool.from_function(
        func=_parallel_query,
        coroutine=_parallel_query_async,
        name="parallel_query",
        description=(
            "并行调用多个独立工具，适合同时需要库存状态和检索规则等互相不依赖的查询。"
            "参数 queries 是列表，每项包含 tool（工具名）和 args（参数字典）。"
            f"可用工具：{list(tools_registry.keys())}"
        ),
        handle_tool_error=True,
        handle_validation_error=True,
    )
