"""文件作用摘要：把业务能力封装成 Agent 可调用的 LangChain 工具。

这个文件是 Agent 和业务服务之间的接口层。LLM 不直接访问数据库、
repository 或 service，而是看到一组带名称、参数 schema 和 description 的工具。
当模型判断“需要查订单 / 查库存 / 查规则”时，LangChain 会调用这里创建的
``StructuredTool``，再把工具结果作为 observation 返回给模型。

主要做的事：
1. ``make_order_tool``：把订单分析服务封装成 ``analyze_order`` 工具。
2. ``make_inventory_tool``：把库存分析服务封装成 ``check_inventory`` 工具。
3. ``make_knowledge_tool``：把 RAG 知识检索封装成 ``retrieve_knowledge`` 工具。
4. ``make_warehouse_tool``：查询 SKU 在各仓库的库存分布。
5. ``make_substitute_tool``：查询缺货 SKU 的替代品。
6. ``make_fulfillment_plan_tool``：生成履约方案建议。

工具统一返回 JSON 字符串：
- ``status``：程序可判断成功或失败。
- ``data``：保留结构化业务字段，方便测试和后续处理。
- ``summary``：给 LLM 快速阅读，减少它理解复杂 JSON 的负担。

学习时先看：
1. ``_ok``：理解工具返回格式。
2. ``_as_structured_tool``：理解普通函数如何变成 LangChain Tool。
3. 按顺序看各个 ``make_*_tool``，它们就是 Agent 的业务能力清单。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable

from langchain_core.tools import StructuredTool, ToolException

from app.domain.inventory.analysis import InventoryAnalysisService
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
from app.domain.orders.analysis import OrderAnalysisService

if TYPE_CHECKING:
    from app.domain.fulfillment.plan_service import FulfillmentPlanService
    from app.domain.fulfillment.substitute_sku import SubstituteSkuService
    from app.domain.inventory.warehouse_service import WarehouseService


# 面试官可能问：为什么工具返回 JSON 字符串，而不是直接返回 dict？
# 回答：LangChain 工具结果最终会进入 LLM 上下文，字符串最通用；但字符串内部
# 用 JSON 组织，程序、测试、前端和后续结构化抽取都能稳定解析。
def _ok(data: dict, summary: str) -> str:
    """统一成功输出格式。

    为什么工具返回字符串而不是 dict？
    LangChain Tool 的输出最终会进入 LLM 上下文，字符串最通用；
    但字符串内容用 JSON 组织，程序和测试又能稳定解析。
    """
    return json.dumps({"status": "ok", "data": data, "summary": summary}, ensure_ascii=False)


# 面试官可能问：StructuredTool 的价值是什么？
# 回答：它会根据 Python 函数签名生成参数 schema，LLM 能知道工具需要哪些参数。
# 这样工具不是“随便拼字符串调用”，而是有名称、描述和参数约束的可控接口。
def _as_structured_tool(func: Callable[..., str], name: str, description: str) -> StructuredTool:
    """把普通 Python 函数注册成 LangChain StructuredTool。

    ``StructuredTool`` 会根据函数签名推断参数 schema。例如：
    ``def _analyze_order(order_id: str)`` 会变成一个需要 ``order_id`` 的工具。
    LLM 看到 description 后，会自己决定何时调用以及传什么参数。
    """
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
        handle_tool_error=True,
        handle_validation_error=True,
    )


# 面试官可能问：为什么订单、库存、知识库都做成工具，而不是让 Agent 直接访问 service？
# 回答：工具层是 Agent 和业务系统的边界。模型只能看到工具描述和参数 schema，
# 不能随意访问内部对象；同时工具输出可测试、可审计、可加缓存/超时/安全净化。
def make_order_tool(service: OrderAnalysisService) -> StructuredTool:
    """创建订单分析工具。

    作用：把“订单分析 service”包装成 Agent 能调用的 ``analyze_order`` 工具。
    典型问题：用户问“订单 SOxxx 是什么情况？”时，Agent 应该先调用它拿订单事实。
    """

    def _analyze_order(order_id: str) -> str:
        try:
            result = service.analyze_order(order_id)
            return _ok(
                data={
                    "order_id": order_id,
                    "item_count": result.item_count,
                    "total_quantity": result.total_quantity,
                    "platform": getattr(result, "platform", ""),
                    "region": getattr(result, "region", ""),
                    "customer_level": getattr(result, "customer_level", ""),
                    "priority": getattr(result, "priority", ""),
                },
                summary=result.summary,
            )
        except Exception as exc:
            # 抛 ToolException 而不是普通异常，是为了让 LangChain 的
            # handle_tool_error=True 接住错误并返回给模型，Agent 还有机会继续回答。
            raise ToolException(f"订单查询失败：{exc}") from exc

    return _as_structured_tool(
        _analyze_order,
        "analyze_order",
        "根据订单 ID 查询订单详情，返回商品数量、总件数、平台、区域等结构化信息。",
    )


# 面试官可能问：库存工具为什么按 order_id 查询，而不是让模型自己传 SKU 列表？
# 回答：履约判断要结合订单中的多个 SKU、数量、仓库库存和锁定库存。让 service
# 根据 order_id 做完整分析，比让模型自己拆 SKU 更稳定，也减少参数遗漏。
def make_inventory_tool(service: InventoryAnalysisService) -> StructuredTool:
    """创建库存检查工具。

    作用：回答“能不能发货、缺哪些 SKU”这类强依赖实时库存的问题。
    这个工具通常不应该缓存，因为库存变化很快。
    """

    def _check_inventory(order_id: str) -> str:
        try:
            result = service.analyze_inventory(order_id)
            return _ok(
                data={
                    "order_id": order_id,
                    "fulfillment_ready": result.fulfillment_ready,
                    "insufficient_skus": list(result.insufficient_skus),
                    "item_count": getattr(result, "item_count", 0),
                },
                summary=result.summary,
            )
        except Exception as exc:
            raise ToolException(f"库存查询失败：{exc}") from exc

    return _as_structured_tool(
        _check_inventory,
        "check_inventory",
        "检查指定订单的库存状态，返回是否可全量履约、缺货 SKU 列表等结构化信息。",
    )


# 面试官可能问：RAG 为什么也是一个工具？
# 回答：Agent 需要在开放式追问中按需检索企业规则。把 RAG 封成工具后，
# Agent 可以先查订单/库存，再根据问题选择是否查规则，实现结构化数据和知识库结合。
def make_knowledge_tool(service: KnowledgeRetrievalService) -> StructuredTool:
    """创建知识库检索工具。

    作用：让 Agent 在回答规则/政策/处理建议时有依据，而不是凭模型记忆编。
    这里连接的是项目里的 RAG 服务。
    """

    def _retrieve_knowledge(order_id: str, question: str, categories: str = "") -> str:
        """检索履约知识库。

        Args:
            order_id:   订单 ID。
            question:   检索问题，如"缺货时应如何处理？"。
            categories: 逗号分隔的规则类别过滤（可选）。
        """
        try:
            # categories 设计成字符串，是为了让 LLM 更容易传参；
            # 进入 service 前再转成 list，保持业务层接口干净。
            filter_categories = [c.strip() for c in categories.split(",") if c.strip()]
            result = service.retrieve(
                order_id=order_id,
                question=question,
                filter_categories=filter_categories,
            )
            rules = []
            actions = []
            if result.answer_summary:
                rules = list(result.answer_summary.key_rules)
                actions = list(result.answer_summary.suggested_actions)

            return _ok(
                data={
                    "hit_count": len(result.hits),
                    "matched_categories": list(result.matched_categories),
                    "key_rules": rules,
                    "suggested_actions": actions,
                },
                summary=(
                    f"命中 {len(result.hits)} 条规则：" + "；".join(rules[:3])
                    if rules
                    else f"命中 {len(result.hits)} 条记录，未生成摘要。"
                ),
            )
        except Exception as exc:
            raise ToolException(f"知识检索失败：{exc}") from exc

    return _as_structured_tool(
        _retrieve_knowledge,
        "retrieve_knowledge",
        "检索履约知识库，获取缺货处理规则和建议动作，返回结构化规则列表。",
    )


# 面试官可能问：为什么要单独做仓库库存工具，已有 check_inventory 不够吗？
# 回答：check_inventory 面向订单整体是否可履约；仓库工具面向 SKU 在各仓的分布。
# 当用户追问“哪个仓有货”“能不能跨仓调拨”时，SKU 维度工具更直接。
def make_warehouse_tool(service: "WarehouseService") -> StructuredTool:
    """创建仓库库存搜索工具。

    作用：按 SKU 查全国各仓可用量，适合处理“哪个仓还能发”的问题。
    它和 ``check_inventory`` 的粒度不同：前者看订单整体，后者看单个 SKU 的仓库分布。
    """

    def _search_warehouse_inventory(sku_id: str) -> str:
        """查询某个 SKU 在全国仓库的库存分布。

        Args:
            sku_id: 商品 SKU 编码，如 "SKU-IPHONE-CASE-001"。
        """
        try:
            result = service.search_sku_inventory(sku_id)
            warehouses = [
                {
                    "warehouse": wh.warehouse_name,
                    "available": wh.available_quantity,
                    "reserved": wh.reserved_quantity,
                }
                for wh in result.warehouse_list
            ]
            return _ok(
                data={
                    "sku_id": sku_id,
                    "total_available": sum(w["available"] for w in warehouses),
                    "warehouses": warehouses,
                },
                summary=result.summary,
            )
        except Exception as exc:
            raise ToolException(f"仓库查询失败：{exc}") from exc

    return _as_structured_tool(
        _search_warehouse_inventory,
        "search_warehouse_inventory",
        "查询某个 SKU 在全国仓库的库存分布和地域覆盖，返回各仓可用量结构化数据。",
    )


# 面试官可能问：替代 SKU 工具在业务上解决什么问题？
# 回答：当主 SKU 缺货时，运营不只需要知道“不能发”，还需要候选替代品。
# 这个工具把商品目录里的替代关系暴露给 Agent，便于生成可执行的补救方案。
def make_substitute_tool(service: "SubstituteSkuService") -> StructuredTool:
    """创建替代 SKU 查询工具。

    作用：缺货时查可替代商品。它属于目录类数据，通常可以短时间缓存。
    """

    def _find_substitute_sku(sku_id: str) -> str:
        """查询某个 SKU 缺货时的替代方案。

        Args:
            sku_id: 原始 SKU 编码，如 "SKU-IPHONE-CASE-001"。
        """
        try:
            result = service.search_substitutes(sku_id)
            return _ok(
                data={
                    "original_sku": sku_id,
                    "has_substitutes": bool(getattr(result, "substitutes", [])),
                    "substitutes": [
                        {
                            "sku_id": s.sku_id if hasattr(s, "sku_id") else str(s),
                            "compatibility": getattr(s, "compatibility", ""),
                        }
                        for s in (getattr(result, "substitutes", []) or [])
                    ],
                },
                summary=result.summary,
            )
        except Exception as exc:
            raise ToolException(f"替代方案查询失败：{exc}") from exc

    return _as_structured_tool(
        _find_substitute_sku,
        "find_substitute_sku",
        "查询某个 SKU 缺货时的替代方案，返回可用替代品及兼容说明的结构化列表。",
    )


# 面试官可能问：为什么履约方案生成也是工具，而不是直接让 LLM 编方案？
# 回答：履约方案要结合库存、仓库、替代品和业务规则。封成工具后，规则和计算
# 留在后端服务里，LLM 负责解释和组织语言，避免凭空生成不可执行方案。
def make_fulfillment_plan_tool(service: "FulfillmentPlanService") -> StructuredTool:
    """创建履约方案生成工具。

    作用：当订单、库存、替代品等事实都比较清楚后，生成具体执行方案。
    这个工具的输出通常是 Agent 最终建议的重要依据。
    """

    def _generate_fulfillment_plan(order_id: str) -> str:
        """为订单生成完整履约方案。

        Args:
            order_id: 订单 ID，如 "SO202502140001"。
        """
        try:
            plan = service.generate_plan(order_id)
            actions = [
                {
                    "product": action.product_name,
                    "quantity": action.quantity,
                    "action_type": action.action_type,
                    "warehouse": action.warehouse_name or "",
                    "substitute": action.substitute_product or "",
                    "estimated_days": action.estimated_days or 0,
                    "note": action.note or "",
                }
                for action in plan.actions
            ]
            return _ok(
                data={
                    "order_id": order_id,
                    "action_count": len(actions),
                    "actions": actions,
                },
                summary=plan.summary,
            )
        except Exception as exc:
            raise ToolException(f"履约方案生成失败：{exc}") from exc

    return _as_structured_tool(
        _generate_fulfillment_plan,
        "generate_fulfillment_plan",
        "为订单生成完整履约方案，返回包含仓库分配、替代品、时间线的结构化动作列表。",
    )
