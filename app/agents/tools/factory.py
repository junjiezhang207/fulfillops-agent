"""Agent 业务工具工厂。

本模块是 Agent 和业务服务之间的接口层。LLM 不直接访问数据库、
repository 或 service，而是看到一组带名称、参数 schema 和 description 的工具。
当模型判断“需要查订单 / 查库存 / 查规则”时，LangChain 会调用这里创建的
``StructuredTool``，再把工具结果作为 observation 返回给模型。

主要工具：
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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from langchain_core.tools import StructuredTool, ToolException

from app.agents.tools.contracts import (
    AnalyzeOrderArgs,
    CheckInventoryArgs,
    FindSubstituteSkuArgs,
    GenerateFulfillmentPlanArgs,
    GetContextDetailArgs,
    RetrieveKnowledgeArgs,
    SearchWarehouseInventoryArgs,
    SourceContextDetailArgs,
    error_envelope,
    ok_envelope,
)
from app.domain.inventory.analysis import InventoryAnalysisService
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
from app.domain.orders.analysis import OrderAnalysisService

if TYPE_CHECKING:
    from app.domain.fulfillment.plan_service import FulfillmentPlanService
    from app.domain.fulfillment.substitute_sku import SubstituteSkuService
    from app.domain.inventory.warehouse_service import WarehouseService
    from app.application.routing.order_context_service import OrderContextService


# 工具返回 JSON 字符串，兼容 LLM 上下文，同时保留可解析的结构化字段。
def _ok(data: dict, summary: str) -> str:
    """统一成功输出格式。

    LangChain Tool 的输出最终会进入 LLM 上下文，字符串最通用；
    但字符串内容用 JSON 组织，程序和测试又能稳定解析。
    """
    return ok_envelope(data=data, summary=summary)


def _error(
    code: str,
    message: str,
    *,
    retryable: bool = False,
    details: dict | None = None,
) -> str:
    """统一工具失败输出，避免 LangChain 把异常转成不可解析的自由文本。"""
    return error_envelope(code=code, message=message, retryable=retryable, details=details)


def _handle_tool_error(exc: ToolException) -> str:
    return _error("tool_exception", str(exc), retryable=True)


def _handle_validation_error(exc) -> str:
    return _error("validation_error", f"工具参数校验失败：{exc}", retryable=False)


# StructuredTool 将函数封装成有名称、描述和参数约束的可控接口。
def _as_structured_tool(
    func: Callable[..., str],
    name: str,
    description: str,
    args_schema: type | None = None,
) -> StructuredTool:
    """把普通 Python 函数注册成 LangChain StructuredTool。

    ``StructuredTool`` 会根据函数签名推断参数 schema。例如：
    ``def _analyze_order(order_id: str)`` 会变成一个需要 ``order_id`` 的工具。
    LLM 看到 description 后，会自己决定何时调用以及传什么参数。
    """
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
        args_schema=args_schema,
        handle_tool_error=_handle_tool_error,
        handle_validation_error=_handle_validation_error,
        infer_schema=args_schema is None,
    )


# 工具层是 Agent 和业务系统的边界，便于测试、审计和统一加弹性包装。
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
            return _error("order_query_failed", f"订单查询失败：{exc}", retryable=True)

    return _as_structured_tool(
        _analyze_order,
        "analyze_order",
        "根据订单 ID 查询订单详情，返回商品数量、总件数、平台、区域等结构化信息。",
        AnalyzeOrderArgs,
    )


# 库存分析按订单执行，由 service 统一处理多 SKU、数量、仓库和锁定库存。
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
            return _error("inventory_query_failed", f"库存查询失败：{exc}", retryable=True)

    return _as_structured_tool(
        _check_inventory,
        "check_inventory",
        "检查指定订单的库存状态，返回是否可全量履约、缺货 SKU 列表等结构化信息。",
        CheckInventoryArgs,
    )


# RAG 作为工具接入，便于 Agent 在开放式追问中按需检索企业规则。
def make_knowledge_tool(service: KnowledgeRetrievalService) -> StructuredTool:
    """创建知识库检索工具。

    作用：让 Agent 在回答规则/政策/处理建议时有依据，而不是凭模型记忆编。
    这里连接的是项目里的 RAG 服务。
    """

    def _retrieve_knowledge(question: str, order_id: str = "", categories: str = "") -> str:
        """检索履约知识库。

        参数：
            order_id:   订单 ID，可选；不传时只检索知识库文档。
            question:   检索问题，如"缺货时应如何处理？"。
            categories: 逗号分隔的规则类别过滤（可选）。
        """
        def _safe_float(value) -> float:
            try:
                return float(value or 0.0)
            except (TypeError, ValueError):
                return 0.0

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
            evidence = []
            for hit in list(result.hits)[:3]:
                metadata = getattr(hit, "metadata", None)
                score_detail = getattr(hit, "score_detail", None)
                evidence.append({
                    "source_file": str(getattr(hit, "source_file", "")),
                    "category": str(getattr(hit, "category", "")),
                    "chunk_id": str(getattr(metadata, "chunk_id", "")) if metadata else "",
                    "title": str(getattr(metadata, "title", "")) if metadata else "",
                    "score": _safe_float(getattr(hit, "score", 0.0)),
                    "score_detail": {
                        "semantic": _safe_float(getattr(score_detail, "semantic_score", 0.0)) if score_detail else 0.0,
                        "keyword": _safe_float(getattr(score_detail, "keyword_score", 0.0)) if score_detail else 0.0,
                        "business_rule": _safe_float(getattr(score_detail, "business_rule_score", 0.0)) if score_detail else 0.0,
                        "rerank": _safe_float(getattr(score_detail, "rerank_score", 0.0)) if score_detail else 0.0,
                    },
                    "matched_terms": list(getattr(hit, "matched_terms", []) or []),
                    "excerpt": str(getattr(hit, "text", ""))[:180],
                })

            return _ok(
                data={
                    "hit_count": len(result.hits),
                    "matched_categories": list(result.matched_categories),
                    "key_rules": rules,
                    "suggested_actions": actions,
                    "coverage_note": str(getattr(result.answer_summary, "coverage_note", "")) if result.answer_summary else "",
                    "expanded_queries": list(getattr(result, "expanded_queries", []) or [])[:6],
                    "evidence": evidence,
                },
                summary=(
                    f"命中 {len(result.hits)} 条规则：" + "；".join(rules[:3])
                    if rules
                    else f"命中 {len(result.hits)} 条记录，未生成摘要。"
                ),
            )
        except Exception as exc:
            return _error("knowledge_retrieval_failed", f"知识检索失败：{exc}", retryable=True)

    return _as_structured_tool(
        _retrieve_knowledge,
        "retrieve_knowledge",
        "检索履约知识库，获取缺货处理规则和建议动作，返回结构化规则列表。",
        RetrieveKnowledgeArgs,
    )


# 仓库库存工具面向 SKU 维度，补充订单整体库存检查无法覆盖的仓库分布问题。
def make_warehouse_tool(service: "WarehouseService") -> StructuredTool:
    """创建仓库库存搜索工具。

    作用：按 SKU 查全国各仓可用量，适合处理“哪个仓还能发”的问题。
    它和 ``check_inventory`` 的粒度不同：前者看订单整体，后者看单个 SKU 的仓库分布。
    """

    def _search_warehouse_inventory(sku_id: str) -> str:
        """查询某个 SKU 在全国仓库的库存分布。

        参数：
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
            return _error("warehouse_query_failed", f"仓库查询失败：{exc}", retryable=True)

    return _as_structured_tool(
        _search_warehouse_inventory,
        "search_warehouse_inventory",
        "查询某个 SKU 在全国仓库的库存分布和地域覆盖，返回各仓可用量结构化数据。",
        SearchWarehouseInventoryArgs,
    )


# 替代 SKU 工具暴露商品目录中的替代关系，用于缺货补救方案。
def make_substitute_tool(service: "SubstituteSkuService") -> StructuredTool:
    """创建替代 SKU 查询工具。

    作用：缺货时查可替代商品。它属于目录类数据，通常可以短时间缓存。
    """

    def _find_substitute_sku(sku_id: str) -> str:
        """查询某个 SKU 缺货时的替代方案。

        参数：
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
            return _error("substitute_query_failed", f"替代方案查询失败：{exc}", retryable=True)

    return _as_structured_tool(
        _find_substitute_sku,
        "find_substitute_sku",
        "查询某个 SKU 缺货时的替代方案，返回可用替代品及兼容说明的结构化列表。",
        FindSubstituteSkuArgs,
    )


# 履约方案由后端服务生成，LLM 只负责解释和组织语言。
def make_fulfillment_plan_tool(service: "FulfillmentPlanService") -> StructuredTool:
    """创建履约方案生成工具。

    作用：当订单、库存、替代品等事实都比较清楚后，生成具体执行方案。
    这个工具的输出通常是 Agent 最终建议的重要依据。
    """

    def _generate_fulfillment_plan(order_id: str) -> str:
        """为订单生成完整履约方案。

        参数：
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
            return _error("fulfillment_plan_failed", f"履约方案生成失败：{exc}", retryable=True)

    return _as_structured_tool(
        _generate_fulfillment_plan,
        "generate_fulfillment_plan",
        "为订单生成完整履约方案，返回包含仓库分配、替代品、时间线的结构化动作列表。",
        GenerateFulfillmentPlanArgs,
    )


def make_context_detail_tool(service: "OrderContextService") -> StructuredTool:
    """创建 OrderContext 明细展开工具。

    作用：Planner 已经拿到核心字段和意图字段组后，如果还缺大体量明细，
    只能按路径展开，不把完整 OrderContext 一次性塞回模型。
    """

    def _get_context_detail(order_id: str, detail_path: str, intent: str = "fulfillment_action", question: str = "") -> str:
        try:
            result = service.get_context_detail(
                order_id=order_id,
                detail_path=detail_path,
                intent=intent,
                question=question or None,
            )
            detail = result["detail"]
            if not detail["found"]:
                return _error(
                    "context_detail_not_found",
                    f"OrderContext 路径不存在：{detail_path}",
                    retryable=False,
                    details={"available_paths": result["context_completeness"].get("large_details_expandable_by_path", [])},
                )
            return _ok(
                data={
                    "order_id": result["order_id"],
                    "intent": result["intent"],
                    "field_groups_loaded": result["field_groups_loaded"],
                    "context_completeness": result["context_completeness"],
                    "path": detail["path"],
                    "value": detail["value"],
                },
                summary=f"已展开 OrderContext 路径 {detail_path}，字段组：{', '.join(result['field_groups_loaded'])}",
            )
        except Exception as exc:
            return _error("context_detail_failed", f"上下文明细读取失败：{exc}", retryable=True)

    return _as_structured_tool(
        _get_context_detail,
        "get_context_detail",
        "按路径展开订单履约上下文大体量明细，例如 inventory/logistics/product_restrictions/customer_risk。",
        GetContextDetailArgs,
    )


def make_source_context_tool(
    service: "OrderContextService",
    *,
    name: str,
    source_system: str,
    description: str,
) -> StructuredTool:
    """创建按源系统拆分的渐进式只读上下文工具。"""

    def _get_source_context(order_id: str, intent: str = "fulfillment_action", question: str = "") -> str:
        try:
            result = service.get_source_detail(
                order_id=order_id,
                source_system=source_system,
                intent=intent,
                question=question or None,
            )
            return _ok(
                data=result,
                summary=(
                    f"已从 {source_system} 只读加载 {order_id} 的 "
                    f"{', '.join(result['detail_paths'])}；不包含短期记忆中的旧业务快照。"
                ),
            )
        except Exception as exc:
            return _error(f"{source_system.lower()}_context_failed", f"{source_system} 上下文读取失败：{exc}", retryable=True)

    return _as_structured_tool(
        _get_source_context,
        name,
        description,
        SourceContextDetailArgs,
    )
