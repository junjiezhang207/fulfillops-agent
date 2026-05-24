"""MCP Server 启动参数。

本地 IDE / Claude Desktop 通常使用 stdio；企业内网共享 MCP 能力时，
更适合使用 streamable-http，并在网关层增加鉴权、限流和审计。
"""

from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP


def run_mcp_server(mcp: FastMCP, default_port: int) -> None:
    """按命令行参数启动 MCP Server。"""

    parser = argparse.ArgumentParser(description=f"Run MCP server: {mcp.name}")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="MCP 通信方式：stdio 适合本地客户端；streamable-http 适合远程/多客户端。",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP 模式监听地址。")
    parser.add_argument("--port", type=int, default=default_port, help="HTTP 模式监听端口。")
    parser.add_argument("--path", default="/mcp", help="Streamable HTTP endpoint path。")
    parser.add_argument(
        "--stateless-http",
        action="store_true",
        help="启用无状态 HTTP；适合放在负载均衡后，但会牺牲部分会话能力。",
    )
    parser.add_argument(
        "--json-response",
        action="store_true",
        help="HTTP 模式使用 JSON response，而不是默认 stream response。",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="MCP Server 日志级别。",
    )
    args = parser.parse_args()

    if args.transport == "streamable-http":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.settings.streamable_http_path = args.path
        mcp.settings.stateless_http = args.stateless_http
        mcp.settings.json_response = args.json_response
        mcp.settings.log_level = args.log_level

    mcp.run(transport=args.transport)
