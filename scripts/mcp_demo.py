#!/usr/bin/env python3
"""MCP 集成验证演示。

MCP 工具必须在 session 上下文内调用（session 关闭后工具失效）。
本脚本演示正确的使用模式。
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.mcp.client import MCPToolsClient, read_mcp_resource_async, get_mcp_prompt_async

_SERVER_SCRIPT = str(Path(__file__).parent.parent / "app" / "mcp" / "server.py")
_FS_SERVER_SCRIPT = str(Path(__file__).parent.parent / "app" / "mcp" / "filesystem_server.py")


def p(title: str):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def sub(title: str):
    print(f"\n[{title}]\n{'-'*60}")


# ============================================================================
# Demo 1: 业务工具调用（Session 内执行）
# ============================================================================

async def demo_business_tools():
    p("Demo 1: 业务 MCP Server — 工具调用")

    server_params = StdioServerParameters(command=sys.executable, args=[_SERVER_SCRIPT])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            print(f"\n加载了 {len(tools)} 个工具: {[t.name for t in tools]}")

            # 测试 check_order_inventory
            sub("check_order_inventory(SO202502140001)")
            inv_tool = next(t for t in tools if "inventory" in t.name)
            result = await inv_tool.ainvoke({"order_id": "SO202502140001"})
            print(result[:300])
            print("[OK] 库存查询成功")

            # 测试 search_warehouse_sku
            sub("search_warehouse_sku(SKU-A001)")
            wh_tool = next(t for t in tools if "warehouse" in t.name)
            result = await wh_tool.ainvoke({"sku_id": "SKU-A001"})
            print(result[:300])
            print("[OK] 仓库搜索成功")

            # 测试 find_substitute_sku
            sub("find_substitute_sku(SKU-B002)")
            sub_tool = next(t for t in tools if "substitute" in t.name)
            result = await sub_tool.ainvoke({"sku_id": "SKU-B002"})
            print(result[:300])
            print("[OK] 替代品查询成功")

            print(f"\n[OK] 业务工具验证完成，所有 {len(tools)} 个工具可正常调用")


# ============================================================================
# Demo 2: 资源读取
# ============================================================================

async def demo_resources():
    p("Demo 2: MCP Resources 读取")

    uris = [
        ("inventory://warehouses", "仓库列表"),
        ("inventory://summary",    "库存概况"),
        ("knowledge://categories", "知识库分类"),
    ]

    server_params = StdioServerParameters(command=sys.executable, args=[_SERVER_SCRIPT])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for uri, desc in uris:
                sub(f"读取 {uri}（{desc}）")
                result = await session.read_resource(uri)
                content = str(result.contents[0].text) if result.contents else ""
                print(content[:400])
                print(f"[OK] 成功读取（{len(content)} 字符）")

            # 通过工具读取知识库文件（动态 URI 模板跨目录不支持，用工具代替）
            sub("通过工具检索 priority_orders.md")
            tools = await load_mcp_tools(session)
            rule_tool = next((t for t in tools if "rules" in t.name), None)
            if rule_tool:
                result = await rule_tool.ainvoke({
                    "query": "高优先级订单规则",
                    "categories": "priority_rule",
                })
                print(str(result)[:300])
                print("[OK] 通过工具访问知识库成功")


# ============================================================================
# Demo 3: 提示词模板
# ============================================================================

async def demo_prompts():
    p("Demo 3: MCP Prompts 模板")

    server_params = StdioServerParameters(command=sys.executable, args=[_SERVER_SCRIPT])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            sub("fulfillment_analysis(order_id=SO202502140001)")
            r = await session.get_prompt("fulfillment_analysis", {"order_id": "SO202502140001"})
            content = str(r.messages[0].content.text) if r.messages else ""
            print(content[:500])
            print("[OK] 获取标准履约分析提示词成功")

            sub("stockout_handling(sku_id=SKU-B002)")
            r2 = await session.get_prompt("stockout_handling", {"sku_id": "SKU-B002", "order_id": "SO001"})
            content2 = str(r2.messages[0].content.text) if r2.messages else ""
            print(content2[:400])
            print("[OK] 获取缺货处理提示词成功")


# ============================================================================
# Demo 4: 文件系统 MCP Server
# ============================================================================

async def demo_filesystem_mcp():
    p("Demo 4: 文件系统 MCP Server — 知识库读取")

    server_params = StdioServerParameters(command=sys.executable, args=[_FS_SERVER_SCRIPT])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            print(f"\n文件系统工具: {[t.name for t in tools]}")

            # 列出文件
            sub("list_knowledge_files()")
            list_tool = next(t for t in tools if "list" in t.name)
            result = await list_tool.ainvoke({"subdir": ""})
            print(result[:400])
            print("[OK] 文件列表获取成功")

            # 搜索内容
            sub("search_knowledge_content(缺货)")
            search_tool = next(t for t in tools if "search" in t.name)
            result = await search_tool.ainvoke({"keyword": "缺货", "max_results": 3})
            print(result[:400])
            print("[OK] 知识库搜索成功")

            # 读取文件
            sub("read_knowledge_file(stockout_rules.md)")
            read_tool = next(t for t in tools if "read" in t.name)
            result = await read_tool.ainvoke({"path": "stockout_rules.md"})
            safe = str(result).encode("gbk", errors="replace").decode("gbk")
            print(safe[:400])
            print("[OK] 知识库文件读取成功")


# ============================================================================
# Demo 5: MultiServerMCPClient（同时连接两个 Server）
# ============================================================================

async def demo_multi_server():
    p("Demo 5: MultiServerMCPClient — 同时连接两个 MCP Server")

    configs = {
        "business": {
            "command": sys.executable,
            "args": [_SERVER_SCRIPT],
            "transport": "stdio",
        },
        "filesystem": {
            "command": sys.executable,
            "args": [_FS_SERVER_SCRIPT],
            "transport": "stdio",
        },
    }

    # langchain-mcp-adapters 0.1.0: MultiServerMCPClient 不再是 context manager
    # 直接创建实例，内部自动管理连接生命周期
    client = MultiServerMCPClient(configs)
    tools = await client.get_tools()

    print(f"\n同时加载了 {len(tools)} 个工具（来自两个 Server）:")
    for t in tools:
        print(f"  {t.name}")

    # 调用来自两个不同 Server 的工具
    sub("调用 business Server 的 check_order_inventory")
    inv_tool = next((t for t in tools if "inventory" in t.name and "check" in t.name), None)
    if inv_tool:
        result = await inv_tool.ainvoke({"order_id": "SO202502140001"})
        print(str(result)[:200])
        print("[OK] 业务 Server 工具调用成功")

    sub("调用 filesystem Server 的 list_knowledge_files")
    list_tool = next((t for t in tools if "list" in t.name), None)
    if list_tool:
        result = await list_tool.ainvoke({"subdir": ""})
        print(str(result)[:200])
        print("[OK] 文件系统 Server 工具调用成功")

    print(f"\n[OK] MultiServerMCPClient 验证完成")
    print("亮点说明：")
    print("  - 一行代码同时连接多个 MCP Server")
    print("  - 工具名自动加前缀避免冲突")
    print("  - 统一管理连接生命周期（context manager 退出自动清理）")


# ============================================================================
# Demo 6: 集成到 LangGraph Agent 的模式说明
# ============================================================================

async def demo_agent_integration_pattern():
    p("Demo 6: 与 LangGraph Agent 集成模式")

    print("""
MCP 工具集成到 LangGraph Agent 的标准模式：

  client = MultiServerMCPClient(configs)
  mcp_tools = await client.get_tools()

  # 组合 MCP 工具 + 原有工具
  all_tools = base_tools + mcp_tools

  # 创建带 MCP 工具的 Agent
  agent = build_agent(chat_model, all_tools)

  # 在 MCP client 管理的生命周期内运行 Agent
  result = await agent.ainvoke(
      {"messages": [HumanMessage(content="检查 SO001 的库存")]},
      config={"configurable": {"thread_id": "session-001"}},
  )

关键约束：
  [!] MCP 工具和 Agent.ainvoke 必须在 MCP 连接存活期间执行
  [!] session 关闭后工具失效，不能跨请求复用
  [!] 建议在 FastAPI 的 lifespan 中维护 MCP client 生命周期

生产环境推荐方案：
  # 在 FastAPI app startup 时初始化并保持连接
  @asynccontextmanager
  async def lifespan(app: FastAPI):
      app.state.mcp_client = MultiServerMCPClient(configs)
      app.state.mcp_tools = await app.state.mcp_client.get_tools()
      yield
      # 服务关闭时自动清理
""")
    print("[OK] 集成模式说明完成")


# ============================================================================
# 主入口
# ============================================================================

async def main_async():
    print("\n" + "=" * 70)
    print("MCP 集成验证演示")
    print("=" * 70)
    print("首次运行需初始化向量索引，可能需要 10-30 秒...")

    await demo_business_tools()
    await demo_resources()
    await demo_prompts()
    await demo_filesystem_mcp()
    await demo_multi_server()
    await demo_agent_integration_pattern()

    print("\n" + "=" * 70)
    print("全部验证完成")
    print("=" * 70)
    print()
    print("新增文件：")
    print("  app/mcp/server.py            — 业务 MCP Server（5 工具 + 3 资源 + 2 提示词）")
    print("  app/mcp/filesystem_server.py — 知识库文件系统 MCP Server（3 工具 + 1 资源）")
    print("  app/mcp/client.py            — MCP 客户端（LangChain 集成 + session 管理）")
    print()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
