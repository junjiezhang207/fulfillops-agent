"""MultiShip 供应链履约 MCP Server。

这是项目的核心 MCP Server，以标准 MCP 协议暴露所有业务能力。

面试亮点：
  "MCP（Model Context Protocol）是 Anthropic 提出的 AI 工具标准协议。
   它解决了'每个项目都要从头实现工具适配'的重复工作问题——
   只要实现一次 MCP Server，任何支持 MCP 的客户端（Claude Desktop、
   Cursor、其他 AI Agent）都能直接使用，无需任何额外适配代码。

   与直接用 LangChain @tool 的区别：
     @tool：只能在 LangChain Agent 里用，跨系统不可复用
     MCP：协议标准化，跨语言/跨框架/跨客户端复用"

MCP Server 三种能力类型：
  Tool     — LLM 可以调用的函数（有副作用，如查询数据库）
  Resource — 可读的数据端点（无副作用，如读取配置/文件）
  Prompt   — 可复用的提示词模板（标准化 LLM 输入格式）

本 Server 暴露的能力：
  Tools：
    check_order_inventory    — 检查订单库存状态
    search_warehouse_sku     — 跨仓库搜索 SKU 可用量
    query_fulfillment_rules  — 检索履约规则知识库
    find_substitute_sku      — 查找替代 SKU
    generate_fulfillment_plan — 生成完整履约方案

  Resources：
    inventory://warehouses        — 所有仓库列表
    inventory://summary           — 整体库存概况
    knowledge://categories        — 知识库分类
    knowledge://file/{filename}   — 读取具体知识库文件

  Prompts：
    fulfillment_analysis   — 标准履约分析提示词
    stockout_handling      — 缺货处理决策提示词

启动方式：
  python app/mcp/server.py
  python app/mcp/server.py --transport streamable-http --host 0.0.0.0 --port 9000
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 路径（以 subprocess 方式运行时需要）
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from mcp.server.fastmcp import FastMCP

from app.core.config import get_settings
from app.core.service_registry import (
    get_inventory_analysis_service,
    get_knowledge_retrieval_service,
    get_order_analysis_service,
)
from app.services.warehouse_service import WarehouseService
from app.services.substitute_sku_service import SubstituteSkuService
from app.services.fulfillment_plan_service import FulfillmentPlanService
from app.mcp.runtime import run_mcp_server

# ============================================================================
# 初始化 FastMCP Server
# ============================================================================

mcp = FastMCP(
    "multiship-fulfillment",
    instructions="""
你正在使用 MultiShip 供应链履约 MCP Server。

该 Server 提供以下能力：
  1. 订单和库存查询（check_order_inventory, search_warehouse_sku）
  2. 知识库检索（query_fulfillment_rules）
  3. 替代品和履约方案（find_substitute_sku, generate_fulfillment_plan）
  4. 仓库数据读取（inventory://warehouses, inventory://summary）
  5. 知识库文件访问（knowledge://categories, knowledge://file/{filename}）

典型工作流：
  1. check_order_inventory → 了解库存状态
  2. query_fulfillment_rules → 查找相关规则
  3. generate_fulfillment_plan → 生成最优方案
""",
)

# ============================================================================
# 初始化业务服务（复用项目统一 service registry）
# ============================================================================

_settings = get_settings()
_order_svc = get_order_analysis_service()
_inv_svc = get_inventory_analysis_service()
_warehouse_svc = WarehouseService()
_substitute_svc = SubstituteSkuService()
_fulfillment_svc = FulfillmentPlanService(
    inventory_service=_inv_svc,
    warehouse_service=_warehouse_svc,
    substitute_service=_substitute_svc,
)


def _get_knowledge_svc():
    """通过统一注册表获取知识检索服务。

    这样 MCP、FastAPI、Agent、Workflow 会共享同一套企业数据源和 RAG 配置。
    """
    return get_knowledge_retrieval_service()


# ============================================================================
# Tools（LLM 可调用的函数）
# ============================================================================

@mcp.tool()
def check_order_inventory(order_id: str) -> str:
    """检查订单的库存状态，判断能否全量履约。

    返回信息包含：
    - 每个 SKU 的可用库存量 vs 订单需求量
    - 是否可以完整履约（fulfillment_ready）
    - 缺货 SKU 列表（insufficient_skus）

    Args:
        order_id: 订单 ID，格式如 SO202502140001
    """
    try:
        result = _inv_svc.analyze_inventory(order_id)
        return result.summary
    except Exception as e:
        return f"库存查询失败: {e}"


@mcp.tool()
def search_warehouse_sku(sku_id: str) -> str:
    """在全国所有仓库中搜索某个 SKU 的可用库存分布。

    返回信息包含：
    - 每个仓库的可用数量
    - 地理覆盖范围
    - 总可用量

    Args:
        sku_id: 商品 SKU 编码，格式如 SKU-IPHONE-CASE-001
    """
    try:
        result = _warehouse_svc.search_sku_inventory(sku_id)
        lines = [result.summary]
        for wh in result.warehouse_list:
            lines.append(
                f"  {wh.warehouse_name}: {wh.available_quantity} 件可用"
                f"（预留 {wh.reserved_quantity} 件）"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"仓库搜索失败: {e}"


@mcp.tool()
def query_fulfillment_rules(query: str, categories: str = "") -> str:
    """检索履约规则知识库，获取与问题相关的业务规则和处理建议。

    知识库包含：缺货处理、优先级规则、区域调度、售后处理、拆合单规则。

    Args:
        query:      检索问题，如"VIP 客户缺货时应如何处理？"
        categories: 逗号分隔的类别过滤（可选），如 "stockout,priority"
    """
    try:
        svc = _get_knowledge_svc()
        filter_cats = [c.strip() for c in categories.split(",") if c.strip()]
        result = svc.retrieve(
            order_id="",
            question=query,
            filter_categories=filter_cats,
        )
        if result.answer_summary:
            rules = "\n".join(f"  - {r}" for r in result.answer_summary.key_rules)
            actions = "\n".join(f"  - {a}" for a in result.answer_summary.suggested_actions)
            return f"命中规则：\n{rules}\n\n建议动作：\n{actions}"
        return f"找到 {len(result.hits)} 条相关记录，但未生成摘要。"
    except Exception as e:
        return f"知识检索失败: {e}"


@mcp.tool()
def find_substitute_sku(sku_id: str) -> str:
    """查询 SKU 缺货时的替代方案。

    返回信息包含：
    - 替代品 SKU 列表
    - 每个替代品的兼容程度
    - 价格差异说明

    Args:
        sku_id: 原始 SKU 编码，如 SKU-IPHONE-CASE-001
    """
    try:
        result = _substitute_svc.search_substitutes(sku_id)
        return result.summary
    except Exception as e:
        return f"替代方案查询失败: {e}"


@mcp.tool()
def generate_fulfillment_plan(order_id: str) -> str:
    """为订单生成完整的履约方案，包括库存配置、替代方案和发货策略。

    返回信息包含：
    - 每个 SKU 的发货仓库
    - 需要替代的 SKU 及其替代品
    - 预计发货时间
    - 分批发货安排（如有）

    Args:
        order_id: 订单 ID，格式如 SO202502140001
    """
    try:
        plan = _fulfillment_svc.generate_plan(order_id)
        lines = [plan.summary, "\n详细履约动作："]
        for action in plan.actions:
            lines.append(f"  - {action.product_name}（{action.quantity}件）→ {action.action_type}")
            if action.warehouse_name:
                lines.append(f"    从 {action.warehouse_name} 发货")
            if action.substitute_product:
                lines.append(f"    用 {action.substitute_product} 替代")
            if action.estimated_days:
                lines.append(f"    预计 {action.estimated_days} 天")
        return "\n".join(lines)
    except Exception as e:
        return f"履约方案生成失败: {e}"


# ============================================================================
# Resources（可读数据端点，无副作用）
# ============================================================================

@mcp.resource("inventory://warehouses")
def get_warehouses() -> str:
    """所有仓库的基本信息列表。

    MCP Resource：适合 AI 在开始分析前"先了解有哪些仓库"。
    与 Tool 的区别：Resource 是只读数据，不触发业务操作。
    """
    try:
        warehouses = _warehouse_svc.list_warehouses()
        if not warehouses:
            return "暂无仓库数据（使用内存模拟数据）"
        lines = ["【仓库列表】"]
        for wh in warehouses:
            lines.append(f"  {wh.warehouse_id}: {wh.warehouse_name} ({wh.region})")
        return "\n".join(lines)
    except Exception:
        return """【仓库列表（示例）】
  WH-SH: 上海浦东仓库（华东区）
  WH-GZ: 广州番禺仓库（华南区）
  WH-BJ: 北京顺义仓库（华北区）
  WH-CD: 成都高新仓库（西南区）"""


@mcp.resource("inventory://summary")
def get_inventory_summary() -> str:
    """全仓库库存整体概况（库存健康度、总库存量、预警 SKU 等）。

    适合 AI 在分析订单前先了解整体库存状态。
    """
    return """【库存整体概况】

仓库使用率：
  WH-SH（上海）：60%（正常）
  WH-GZ（广州）：78%（偏紧）
  WH-BJ（北京）：45%（充裕）
  WH-CD（成都）：55%（正常）

重点关注：
  - SKU-B002：全国缺货，建议找替代品
  - SKU-C003：广州仓积压（在库 90 天），建议优先发货
  - SKU-A001：上海仓库存充裕，可支持大单

更新时间：实时（内存数据）"""


@mcp.resource("knowledge://categories")
def get_knowledge_categories() -> str:
    """知识库的分类目录，帮助了解有哪些业务规则可以查询。"""
    try:
        svc = _get_knowledge_svc()
        paths = svc.knowledge_repository.list_knowledge_paths()
        categories = set()
        for p in paths:
            parts = Path(p).parts
            if len(parts) >= 2:
                categories.add(parts[-2])
        if categories:
            return "【知识库分类】\n" + "\n".join(f"  - {c}" for c in sorted(categories))
    except Exception:
        pass
    return """【知识库分类】
  - stockout_rules       缺货处理规则
  - priority_orders      优先级订单处理
  - regional_fulfillment 区域履约策略
  - after_sales_rules    售后处理规则
  - split_merge_orders   拆合单规则
  - business_rules       客户分级与风控
  - market_context       市场状况与定价"""


@mcp.resource("knowledge://file/{filename}")
def get_knowledge_file(filename: str) -> str:
    """读取指定的知识库文件内容。

    Args:
        filename: 文件名（含子目录），如 business_rules/customer_tiers.md
    """
    try:
        knowledge_dir = Path(_settings.knowledge_dir).resolve()
        file_path = knowledge_dir / filename
        resolved = file_path.resolve()
        try:
            resolved.relative_to(knowledge_dir)
        except ValueError:
            return "[ERROR] 不允许访问知识库目录之外的文件"
        if resolved.suffix.lower() != ".md":
            return "[ERROR] 只允许读取 Markdown 知识库文件"
        if not resolved.exists():
            return f"[NOT FOUND] 文件不存在: {filename}"
        content = resolved.read_text(encoding="utf-8")
        # 截断过长内容
        if len(content) > 8000:
            content = content[:8000] + "\n\n...[内容过长，已截断]..."
        return content
    except Exception as e:
        return f"[ERROR] 读取文件失败: {e}"


# ============================================================================
# Prompts（可复用的提示词模板）
# ============================================================================

@mcp.prompt()
def fulfillment_analysis(order_id: str) -> str:
    """标准履约分析提示词模板。

    MCP Prompt 的作用：
      将常用的分析步骤标准化为可复用模板，
      确保每次分析都遵循相同的最佳实践流程。
    """
    return f"""请对订单 {order_id} 进行完整的履约分析，按以下步骤进行：

**第 1 步：库存核查**
调用 check_order_inventory(order_id="{order_id}") 了解当前库存状态。

**第 2 步：规则检索**
如果有缺货，调用 query_fulfillment_rules(query="缺货处理规则") 获取处理规范。

**第 3 步：替代方案**
对每个缺货 SKU，调用 find_substitute_sku(sku_id="...") 查找替代品。

**第 4 步：方案生成**
调用 generate_fulfillment_plan(order_id="{order_id}") 生成完整履约方案。

**第 5 步：输出结论**
综合以上信息，给出：
  - 能否按时完整履约（是/否）
  - 如有缺货，推荐的处理方案
  - 预计最终交货时间
  - 需要客户确认的事项（如有）
"""


@mcp.prompt()
def stockout_handling(sku_id: str, order_id: str = "") -> str:
    """缺货处理决策提示词模板。"""
    order_context = f"，订单 {order_id}" if order_id else ""
    return f"""SKU {sku_id}{order_context} 出现缺货，请按以下流程处理：

**1. 查找替代品**
调用 find_substitute_sku(sku_id="{sku_id}")

**2. 搜索其他仓库**
调用 search_warehouse_sku(sku_id="{sku_id}") 查看是否有其他仓库有货

**3. 检索处理规则**
调用 query_fulfillment_rules(query="缺货时客户优先级处理规则", categories="stockout,priority")

**4. 给出决策**
基于以上信息，建议选择以下处理方式之一：
  A. 替换为替代品（需客户确认）
  B. 从其他仓库调货（关注运费和时效）
  C. 部分发货 + 余量补发（需告知客户延误）
  D. 取消缺货部分 + 退款（最后选项）
"""


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    run_mcp_server(mcp, default_port=9000)
