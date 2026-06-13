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
    analyze_order             — 查询订单结构化详情
    check_inventory           — 检查订单库存状态
    search_warehouse_inventory — 跨仓库搜索 SKU 可用量
    retrieve_knowledge        — 检索履约规则知识库
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

import os
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径（以 subprocess 方式运行时需要）
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from mcp.server.fastmcp import FastMCP

from app.core.config import get_settings
from app.core.service_registry import (
    get_inventory_analysis_service,
    get_order_analysis_service,
    get_knowledge_retrieval_service,
)
from app.agents.tools.contracts import (
    DEFAULT_AGENT_TOOL_PERMISSIONS,
    ToolRuntimeContext,
    error_envelope,
    tool_runtime_context,
)
from app.agents.tools.registry import ToolServiceBundle, get_tool_registry
from app.agents.tools.wrapper import wrap_tool_with_resilience
from app.domain.inventory.warehouse_service import WarehouseService
from app.domain.fulfillment.substitute_sku import SubstituteSkuService
from app.domain.fulfillment.plan_service import FulfillmentPlanService
from app.mcp.runtime import run_mcp_server

# ============================================================================
# 初始化 FastMCP Server
# ============================================================================

mcp = FastMCP(
    "multiship-fulfillment",
    instructions="""
你正在使用 MultiShip 供应链履约 MCP Server。

该 Server 提供以下能力：
  1. 订单和库存查询（analyze_order, check_inventory, search_warehouse_inventory）
  2. 知识库检索（retrieve_knowledge）
  3. 替代品和履约方案（find_substitute_sku, generate_fulfillment_plan）
  4. 仓库数据读取（inventory://warehouses, inventory://summary）
  5. 知识库文件访问（knowledge://categories, knowledge://file/{filename}）

典型工作流：
  1. check_inventory → 了解库存状态
  2. retrieve_knowledge → 查找相关规则
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


def _csv_env(name: str, default: list[str]) -> list[str]:
    """读取逗号分隔的 MCP 运行时配置。

    MCP Server 经常以 subprocess 或 HTTP 服务方式启动，不一定能像内部 Agent
    一样拿到完整的会话对象。这里用环境变量提供一个轻量入口，让部署方可以
    为外部 MCP 客户端设置租户、用户和权限；没有配置时使用项目默认的只读/规划权限。
    """
    raw = os.getenv(name, "")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(default)


def _mcp_runtime_context() -> ToolRuntimeContext:
    """构造外部 MCP 调用进入内部工具层时使用的权限上下文。

    关键点是：MCP tool 本身只是协议适配器，真正的权限判断仍然交给
    ``authorize_tool_call``。这样内部 Agent 和外部 MCP 调用看到的是同一套
    ``ToolManifest.required_permissions``，不会出现两套安全规则各管各的情况。
    """
    return ToolRuntimeContext(
        tenant_id=os.getenv("MCP_TENANT_ID", "default"),
        user_id=os.getenv("MCP_USER_ID", "mcp-client"),
        roles=_csv_env("MCP_TOOL_ROLES", ["mcp_client"]),
        permissions=_csv_env("MCP_TOOL_PERMISSIONS", DEFAULT_AGENT_TOOL_PERMISSIONS),
        request_id=os.getenv("MCP_REQUEST_ID", ""),
    )


def _build_mcp_tool_map():
    """从 ToolRegistry 构建 MCP Server 要暴露的业务工具。

    以前 MCP Server 在本文件里手写业务函数，因此它和内部 Agent 工具有两套
    名称、错误格式和可观测性。现在这里直接复用 registry 的定义：

    1. ``ToolRegistry`` 决定有哪些工具、manifest 是什么、需要哪些 service。
    2. ``factory.py`` 负责把 service 包装成 LangChain ``StructuredTool``。
    3. ``wrap_tool_with_resilience`` 统一加权限、错误 envelope、缓存、熔断和 telemetry。
    4. MCP 函数只把协议参数转成 dict，再调用这些包装后的工具。

    这层拆分的好处是：以后新增内部 Agent 工具时，只要在 registry 里标记
    ``use_case="mcp"``，MCP 侧就能复用同一套治理能力。
    """
    services = ToolServiceBundle(
        order_service=_order_svc,
        inventory_service=_inv_svc,
        knowledge_service=_get_knowledge_svc(),
        warehouse_service=_warehouse_svc,
        substitute_service=_substitute_svc,
        fulfillment_service=_fulfillment_svc,
    )
    registry = get_tool_registry()
    tools = {}
    for definition in registry.definitions(use_case="mcp"):
        base_tool = definition.builder(services)
        tools[definition.name] = wrap_tool_with_resilience(
            base_tool,
            enable_cache=definition.cache_enabled,
            enable_circuit_breaker=True,
        )
    return tools


_TOOL_REGISTRY = get_tool_registry()
_MCP_TOOL_MAP = _build_mcp_tool_map()


def _tool_description(tool_name: str) -> str:
    """把 ToolManifest 展开成 MCP 客户端可读的工具说明。

    MCP 客户端主要通过工具名、描述和参数 schema 让模型决定何时调用工具。
    描述文本直接来自 registry manifest，可以避免 MCP 和 Agent 两边文案漂移。
    """
    definition = _TOOL_REGISTRY.get(tool_name)
    if definition is None:
        return ""
    manifest = definition.manifest
    permissions = ", ".join(manifest.required_permissions) or "none"
    return (
        f"{manifest.description}\n"
        f"Owner: {manifest.owner}; risk: {manifest.risk_level.value}; "
        f"permissions: {permissions}; freshness: {manifest.data_freshness}."
    )


def _tool_meta(tool_name: str) -> dict:
    """给 MCP tool 附带机器可读治理元数据。

    不是所有 MCP 客户端都会展示 ``meta``，但把 manifest 放进去有两个价值：
    调试工具发现结果时能看到权限/风险/缓存策略；未来如果接入网关或审计系统，
    也可以直接读取这些字段，而不用再反查 Python 代码。
    """
    definition = _TOOL_REGISTRY.get(tool_name)
    if definition is None:
        return {}
    return {
        "tool_registry": definition.metadata(),
        "exposed_via": "mcp",
    }


def _invoke_registry_tool(tool_name: str, arguments: dict) -> str:
    """执行 registry 工具并保持统一 envelope。

    这里故意不捕获并改写正常的工具返回值，因为 ``factory.py`` 和
    ``wrapper.py`` 已经保证结果是 ``ToolEnvelope`` JSON 字符串。MCP 只需要
    在找不到工具这种适配层错误时返回同样的 error envelope。
    """
    tool = _MCP_TOOL_MAP.get(tool_name)
    if tool is None:
        return error_envelope(
            "mcp_tool_not_registered",
            f"MCP tool {tool_name} is not registered in ToolRegistry.",
            retryable=False,
            details={"tool_name": tool_name},
        )
    with tool_runtime_context(_mcp_runtime_context()):
        return str(tool.invoke(arguments))


# ============================================================================
# Tools（LLM 可调用的函数）
# ============================================================================

# 这一组 MCP tools 的函数体都很薄：它们只负责把 MCP 协议传进来的参数整理成
# dict，然后交给 ``_invoke_registry_tool``。真正的业务逻辑、权限检查、错误
# envelope、缓存、熔断和 telemetry 都在 ToolRegistry / wrapper 那条链路里。


@mcp.tool(
    name="analyze_order",
    description=_tool_description("analyze_order"),
    meta=_tool_meta("analyze_order"),
)
def analyze_order(order_id: str) -> str:
    """查询订单结构化详情。

    Args:
        order_id: 订单 ID，格式如 SO202502140001。
    """
    return _invoke_registry_tool("analyze_order", {"order_id": order_id})


@mcp.tool(
    name="check_inventory",
    description=_tool_description("check_inventory"),
    meta=_tool_meta("check_inventory"),
)
def check_inventory(order_id: str) -> str:
    """检查订单的库存状态，判断能否全量履约。

    返回信息包含：
    - 每个 SKU 的可用库存量 vs 订单需求量
    - 是否可以完整履约（fulfillment_ready）
    - 缺货 SKU 列表（insufficient_skus）

    Args:
        order_id: 订单 ID，格式如 SO202502140001
    """
    return _invoke_registry_tool("check_inventory", {"order_id": order_id})


@mcp.tool(
    name="search_warehouse_inventory",
    description=_tool_description("search_warehouse_inventory"),
    meta=_tool_meta("search_warehouse_inventory"),
)
def search_warehouse_inventory(sku_id: str) -> str:
    """在全国所有仓库中搜索某个 SKU 的可用库存分布。

    返回信息包含：
    - 每个仓库的可用数量
    - 地理覆盖范围
    - 总可用量

    Args:
        sku_id: 商品 SKU 编码，格式如 SKU-IPHONE-CASE-001
    """
    return _invoke_registry_tool("search_warehouse_inventory", {"sku_id": sku_id})


@mcp.tool(
    name="retrieve_knowledge",
    description=_tool_description("retrieve_knowledge"),
    meta=_tool_meta("retrieve_knowledge"),
)
def retrieve_knowledge(question: str, order_id: str = "", categories: str = "") -> str:
    """检索履约规则知识库，获取与问题相关的业务规则和处理建议。

    知识库包含：缺货处理、优先级规则、区域调度、售后处理、拆合单规则。

    Args:
        order_id:   订单 ID，可选；不传时只检索知识库文档。
        question:   检索问题，如"VIP 客户缺货时应如何处理？"
        categories: 逗号分隔的类别过滤（可选），如 "stockout,priority"
    """
    return _invoke_registry_tool(
        "retrieve_knowledge",
        {"order_id": order_id, "question": question, "categories": categories},
    )


@mcp.tool(
    name="find_substitute_sku",
    description=_tool_description("find_substitute_sku"),
    meta=_tool_meta("find_substitute_sku"),
)
def find_substitute_sku(sku_id: str) -> str:
    """查询 SKU 缺货时的替代方案。

    返回信息包含：
    - 替代品 SKU 列表
    - 每个替代品的兼容程度
    - 价格差异说明

    Args:
        sku_id: 原始 SKU 编码，如 SKU-IPHONE-CASE-001
    """
    return _invoke_registry_tool("find_substitute_sku", {"sku_id": sku_id})


@mcp.tool(
    name="generate_fulfillment_plan",
    description=_tool_description("generate_fulfillment_plan"),
    meta=_tool_meta("generate_fulfillment_plan"),
)
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
    return _invoke_registry_tool("generate_fulfillment_plan", {"order_id": order_id})


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
调用 check_inventory(order_id="{order_id}") 了解当前库存状态。

**第 2 步：规则检索**
如果有缺货，调用 retrieve_knowledge(order_id="{order_id}", question="缺货处理规则") 获取处理规范。

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
调用 search_warehouse_inventory(sku_id="{sku_id}") 查看是否有其他仓库有货

**3. 检索处理规则**
调用 retrieve_knowledge(order_id="{order_id or 'MCP-ADHOC'}", question="缺货时客户优先级处理规则", categories="stockout,priority")

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
