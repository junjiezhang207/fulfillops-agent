"""MCP 客户端 — 将 MCP Server 的工具加载为 LangChain BaseTool。

面试亮点：
  "这个模块是 MCP 协议和 LangChain 框架之间的桥接层。
   核心思路是：
     1. MCP Client 连接到 MCP Server（subprocess + stdio）
     2. 从 Server 获取所有工具的 schema（名称/描述/参数）
     3. 用 langchain-mcp-adapters 将 MCP 工具转换为 LangChain BaseTool
     4. 在 MCP session 存活期间，把这些工具交给 Agent 或调试函数调用

   两种集成模式：
     Mode 1（MultiServer）：同时连接多个 MCP Server，适合生产环境
     Mode 2（单 Server）  ：只连接业务 Server，适合开发调试

   为什么用 async：
     MCP 协议基于 stdio 流，天然异步。
     MCP 工具绑定在 session 上，所以这里提供 run_with_*_tools()
     让调用逻辑在 session 生命周期内执行，避免拿到失效工具。"

使用方式：

  # 方式 1：异步调用业务 MCP 工具
  from app.mcp.client import MCPToolsClient

  async def work(tools):
      return await tools[0].ainvoke({"order_id": "SO202502140001"})

  result = await MCPToolsClient.run_with_business_tools(work)

  # 方式 2：同步脚本里调用
  result = MCPToolsClient.run_with_business_tools_sync(lambda tools: tools[0].ainvoke({"order_id": "SO202502140001"}))

  # 方式 3：同时使用业务 + 文件系统 MCP 工具
  result = await MCPToolsClient.run_with_all_tools(work)
"""

import asyncio
import inspect
import sys
from pathlib import Path
from typing import Optional

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

# 项目根目录
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_SERVER_SCRIPT = str(_PROJECT_ROOT / "app" / "mcp" / "server.py")
_FILESYSTEM_SERVER_SCRIPT = str(_PROJECT_ROOT / "app" / "mcp" / "filesystem_server.py")


# ============================================================================
# MCP Server 配置
# ============================================================================

def _get_business_server_config() -> dict:
    """业务 MCP Server 配置（以 subprocess stdio 方式连接）。"""
    return {
        "command": sys.executable,   # 使用当前 Python 解释器
        "args": [_SERVER_SCRIPT],
        "transport": "stdio",
        "env": None,
    }


def _get_filesystem_server_config() -> dict:
    """知识库文件系统 MCP Server 配置。"""
    return {
        "command": sys.executable,
        "args": [_FILESYSTEM_SERVER_SCRIPT],
        "transport": "stdio",
        "env": None,
    }


# ============================================================================
# MCP 工具加载
# ============================================================================

class MCPToolsClient:
    """MCP 工具客户端 — 连接 MCP Server 并加载工具为 LangChain BaseTool。

    设计思路：
      每次需要工具时建立新连接（无状态），适合服务启动时加载一次的场景。
      MCP 工具绑定在 session 上，不能简单返回后长期缓存；调用逻辑必须放在
      run_with_*_tools 的 factory 内执行。
    """

    @staticmethod
    async def run_with_business_tools(coro_factory):
        """在业务 MCP session 上下文内执行 coroutine。

        MCP 工具绑定在 session 上，session 关闭后工具不可用。
        此方法确保工具在 session 存活期间被使用。

        用法：
            async def my_work(tools):
                result = await tools[0].ainvoke({"order_id": "SO001"})
                return result

            result = await MCPToolsClient.run_with_business_tools(my_work)
        """
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[_SERVER_SCRIPT],
            env=None,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                return await coro_factory(tools)

    @staticmethod
    def run_with_business_tools_sync(func):
        """同步脚本入口：在业务 MCP session 内执行函数。

        langchain-mcp-adapters 生成的工具通常只支持 ``ainvoke``。
        因此 func 可以直接返回 coroutine，本方法会自动 await。
        """

        async def _runner(tools):
            result = func(tools)
            if inspect.isawaitable(result):
                return await result
            return result

        return asyncio.run(MCPToolsClient.run_with_business_tools(_runner))

    @staticmethod
    async def run_with_all_tools(coro_factory):
        """用 MultiServerMCPClient 同时加载两个 Server 的工具并执行。

        langchain-mcp-adapters 0.1.0 起，MultiServerMCPClient 不再是 context manager，
        直接创建实例并调用 get_tools()，内部自动管理连接生命周期。
        """
        configs = {
            "business": _get_business_server_config(),
            "filesystem": _get_filesystem_server_config(),
        }
        client = MultiServerMCPClient(configs)
        tools = await client.get_tools()
        return await coro_factory(tools)

    @staticmethod
    def run_with_all_tools_sync(func):
        """同步脚本入口：在业务 + 文件系统 MCP 工具上下文内执行函数。"""

        async def _runner(tools):
            result = func(tools)
            if inspect.isawaitable(result):
                return await result
            return result

        return asyncio.run(MCPToolsClient.run_with_all_tools(_runner))

    @staticmethod
    async def load_filesystem_tools_in_context(coro_factory):
        """在文件系统 MCP session 上下文内执行。"""
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[_FILESYSTEM_SERVER_SCRIPT],
            env=None,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                return await coro_factory(tools)

    @staticmethod
    def run_with_filesystem_tools_sync(func):
        """同步脚本入口：在文件系统 MCP session 内执行函数。"""

        async def _runner(tools):
            result = func(tools)
            if inspect.isawaitable(result):
                return await result
            return result

        return asyncio.run(MCPToolsClient.load_filesystem_tools_in_context(_runner))

    @staticmethod
    async def list_server_capabilities_async() -> dict:
        """列出业务 MCP Server 的所有能力（工具/资源/提示词）。

        用于服务发现和调试，返回 Server 完整的能力清单。
        """
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[_SERVER_SCRIPT],
            env=None,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 获取工具列表
                tools_response = await session.list_tools()
                tools = [
                    {"name": t.name, "description": t.description}
                    for t in tools_response.tools
                ]

                # 获取资源列表
                try:
                    resources_response = await session.list_resources()
                    resources = [
                        {"uri": r.uri, "name": r.name, "description": r.description}
                        for r in resources_response.resources
                    ]
                except Exception:
                    resources = []

                # 获取提示词列表
                try:
                    prompts_response = await session.list_prompts()
                    prompts = [
                        {"name": p.name, "description": p.description}
                        for p in prompts_response.prompts
                    ]
                except Exception:
                    prompts = []

                return {
                    "server": "multiship-fulfillment",
                    "tools": tools,
                    "resources": resources,
                    "prompts": prompts,
                    "total_tools": len(tools),
                    "total_resources": len(resources),
                    "total_prompts": len(prompts),
                }

    @staticmethod
    def list_server_capabilities_sync() -> dict:
        """同步获取 MCP Server 能力清单。"""
        try:
            return asyncio.run(MCPToolsClient.list_server_capabilities_async())
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# 资源和提示词访问
# ============================================================================

async def read_mcp_resource_async(uri: str) -> str:
    """从 MCP Server 读取指定 Resource。

    Args:
        uri: Resource URI，如 "inventory://warehouses" 或
             "knowledge://file/business_rules/customer_tiers.md"

    Returns:
        Resource 内容（字符串）
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[_SERVER_SCRIPT],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.read_resource(uri)
            contents = result.contents
            if contents:
                return str(contents[0].text) if hasattr(contents[0], "text") else str(contents[0])
            return ""


def read_mcp_resource(uri: str) -> str:
    """同步读取 MCP Resource。"""
    return asyncio.run(read_mcp_resource_async(uri))


async def get_mcp_prompt_async(name: str, arguments: Optional[dict] = None) -> str:
    """从 MCP Server 获取指定 Prompt 的内容。

    Args:
        name:      Prompt 名称，如 "fulfillment_analysis"
        arguments: Prompt 参数，如 {"order_id": "SO001"}
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[_SERVER_SCRIPT],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.get_prompt(name, arguments or {})
            messages = result.messages
            if messages:
                return str(messages[0].content.text) if hasattr(messages[0].content, "text") else str(messages[0])
            return ""


def get_mcp_prompt(name: str, arguments: Optional[dict] = None) -> str:
    """同步获取 MCP Prompt 内容。"""
    return asyncio.run(get_mcp_prompt_async(name, arguments))
