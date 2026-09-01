"""知识库文件系统 MCP Server。

专门用于读取 Settings.knowledge_dir 配置的 markdown 文档。

面试亮点：
  "我们有两个 MCP Server：
     1. 业务 Server（server.py）：封装业务逻辑，返回结构化数据
     2. 文件系统 Server（这个）：直接暴露原始知识库文档

   分开的原因：
     - 职责分离：业务逻辑和文档访问是不同的关注点
     - 粒度控制：Agent 有时需要完整文档（文件系统），有时需要精准摘要（业务）
     - 安全隔离：文件系统 Server 只能访问配置的知识库目录，不能访问其他目录"

暴露的工具：
  read_knowledge_file(path)    — 读取指定文件内容
  list_knowledge_files(subdir) — 列出目录下的文件
  search_knowledge_content(keyword) — 在所有文件中搜索关键词
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from mcp.server.fastmcp import FastMCP
from app.core.config import get_settings
from app.mcp.runtime import run_mcp_server

mcp = FastMCP(
    "fulfillops-knowledge-filesystem",
    instructions="用于读取 fulfillops-agent 电商履约知识库的文件系统 MCP Server。",
)

_settings = get_settings()
_KNOWLEDGE_DIR = Path(_settings.knowledge_dir).resolve()
_MAX_FILE_CHARS = 10000
_ALLOWED_SUFFIXES = {".md"}


def _safe_path(relative_path: str) -> Path | None:
    """将相对路径解析为安全的绝对路径（防止路径穿越攻击）。"""
    target = (_KNOWLEDGE_DIR / (relative_path or ".")).resolve()
    try:
        target.relative_to(_KNOWLEDGE_DIR)
    except ValueError:
        return None
    return target


@mcp.tool()
def read_knowledge_file(path: str) -> str:
    """读取知识库中的指定文件。

    Args:
        path: 相对于 Settings.knowledge_dir 的文件路径，
              如 "stockout_rules.md" 或 "after_sales_rules.md"
    """
    target = _safe_path(path)
    if target is None:
        return "[ERROR] 不允许访问知识库目录之外的文件"
    if not target.exists():
        return f"[NOT FOUND] 文件不存在: {path}\n可用文件请调用 list_knowledge_files()"
    if not target.is_file():
        return f"[ERROR] {path} 不是文件"
    if target.suffix.lower() not in _ALLOWED_SUFFIXES:
        return "[ERROR] 只允许读取 Markdown 知识库文件"

    content = target.read_text(encoding="utf-8")
    if len(content) > _MAX_FILE_CHARS:
        return content[:_MAX_FILE_CHARS] + f"\n\n...[文件过长，已截断，总长 {len(content)} 字符]"
    return content


@mcp.tool()
def list_knowledge_files(subdir: str = "") -> str:
    """列出知识库目录下的所有文件。

    Args:
        subdir: 子目录名（可选），如 "business_rules"。
                留空则列出所有文件。
    """
    base = _KNOWLEDGE_DIR
    if subdir:
        target = _safe_path(subdir)
        if target is None:
            return "[ERROR] 路径不合法"
        base = target

    if not base.exists():
        return f"[NOT FOUND] 目录不存在: {subdir}"

    lines = [f"【知识库文件列表】 {'/' + subdir if subdir else '（全部）'}\n"]
    for p in sorted(base.rglob("*.md")):
        rel = p.relative_to(_KNOWLEDGE_DIR)
        size_kb = p.stat().st_size / 1024
        lines.append(f"  {rel}  ({size_kb:.1f} KB)")

    return "\n".join(lines) if len(lines) > 1 else "未找到任何 .md 文件"


@mcp.tool()
def search_knowledge_content(keyword: str, max_results: int = 5) -> str:
    """在所有知识库文件中搜索包含关键词的段落。

    Args:
        keyword:     搜索关键词，如 "VIP 客户" 或 "缺货处理"
        max_results: 最多返回结果数（默认 5）
    """
    results = []

    for md_file in sorted(_KNOWLEDGE_DIR.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
        except Exception:
            continue

        lines = content.split("\n")
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                # 提取上下文（前后各 1 行）
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                context = "\n".join(lines[start:end]).strip()

                rel_path = md_file.relative_to(_KNOWLEDGE_DIR)
                results.append(f"[{rel_path}] 第 {i+1} 行:\n  {context}")

                if len(results) >= max_results:
                    break

        if len(results) >= max_results:
            break

    if not results:
        return f"未找到包含 '{keyword}' 的内容"

    return f"找到 {len(results)} 处包含 '{keyword}'：\n\n" + "\n\n".join(results)


@mcp.resource("knowledge://index")
def get_knowledge_index() -> str:
    """知识库完整索引（所有文件的标题和摘要）。"""
    lines = ["【知识库索引】\n"]

    for md_file in sorted(_KNOWLEDGE_DIR.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
            first_lines = "\n".join(content.split("\n")[:3]).strip()
            rel = md_file.relative_to(_KNOWLEDGE_DIR)
            lines.append(f"📄 {rel}\n   {first_lines[:100]}\n")
        except Exception:
            continue

    return "\n".join(lines)


if __name__ == "__main__":
    run_mcp_server(mcp, default_port=9001)
