"""Planner RAG Context：SOP 与优秀案例分开召回。"""

from __future__ import annotations

import re
from typing import Any

from app.core.config import get_settings
from app.rag.knowledge_retrieval_service import KnowledgeRetrievalService
from app.schemas.rag_context import PlannerRAGContext, RAGEvidenceItem, RAGSourceSet


SOP_FILTERS = [
    "stockout_rule",
    "priority_rule",
    "regional_strategy",
    "split_merge_rule",
    "after_sales_rule",
    "general",
]
CASE_FILTERS = ["excellent_case"]
DIRECT_MUTATION_PATTERNS = (
    "直接修改订单",
    "直接改订单",
    "直接扣减库存",
    "直接修改库存",
    "直接生成运单",
    "直接调拨库存",
)
SOP_BYPASS_PATTERNS = ("无需审批", "无需人工", "不用人工", "直接执行", "不用确认")
HITL_SOP_PATTERNS = ("审批", "人工", "确认", "HITL")
SPLIT_PATTERNS = ("拆单", "分批", "多包裹", "split")


class PlannerRAGContextService:
    """为 Planner 生成可引用的双知识源上下文。

    当前项目物理向量后端使用 PGVector；这里做的是逻辑源隔离，避免 SOP 和
    历史案例进入同一个候选池混合排名。后续要拆成两个 PGVector table 时，
    只需要替换底层 retrieval 实现。
    """

    def __init__(self, knowledge_service: KnowledgeRetrievalService) -> None:
        self.knowledge_service = knowledge_service

    def build(
        self,
        *,
        order_id: str,
        question: str,
        session_memory: dict[str, Any] | None = None,
        current_business_context: dict[str, Any] | None = None,
    ) -> PlannerRAGContext:
        query = self._query_with_memory(question, session_memory)
        settings = get_settings()
        source_collections = {
            "sop": getattr(settings, "sop_collection", "sop_collection") or "sop_collection",
            "excellent_case": getattr(settings, "case_collection", "case_collection") or "case_collection",
        }
        context = PlannerRAGContext(
            order_id=order_id,
            question=question,
            vector_backend=getattr(settings, "vector_store_type", "pgvector") or "pgvector",
            source_collections=source_collections,
        )
        try:
            sop_result = self.knowledge_service.retrieve(
                order_id=order_id,
                question=question,
                filter_categories=SOP_FILTERS,
                session_memory=session_memory,
            )
            context.sop_evidence = self._source_set(
                result=sop_result,
                source_type="sop",
                collection_name=source_collections["sop"],
                fallback_query=query,
            )
        except Exception as exc:
            context.warnings.append(f"SOP evidence unavailable: {exc}")

        try:
            case_result = self.knowledge_service.retrieve(
                order_id=order_id,
                question=question,
                filter_categories=CASE_FILTERS,
                session_memory=session_memory,
            )
            context.similar_cases = self._source_set(
                result=case_result,
                source_type="excellent_case",
                collection_name=source_collections["excellent_case"],
                fallback_query=query,
                current_business_context=current_business_context,
                sop_key_points=context.sop_evidence.key_points,
            )
            if context.similar_cases.ignored_evidence:
                context.warnings.append(
                    f"已忽略 {len(context.similar_cases.ignored_evidence)} 条与当前业务数据或 SOP 冲突的历史案例。"
                )
        except Exception as exc:
            context.warnings.append(f"Similar case evidence unavailable: {exc}")

        return context

    @staticmethod
    def _query_with_memory(question: str, session_memory: dict[str, Any] | None) -> str:
        structured = (session_memory or {}).get("structured") or {}
        hints: list[str] = []
        topic = structured.get("current_topic")
        if topic:
            hints.append(f"当前讨论主题：{topic}")
        preferences = structured.get("user_preferences") or {}
        if preferences:
            hints.append(f"运营偏好：{preferences}")
        constraints = structured.get("confirmed_constraints") or {}
        if constraints:
            hints.append(f"已确认约束：{constraints}")
        feedback = structured.get("plan_feedback") or {}
        if feedback:
            hints.append(f"方案反馈：{feedback}")
        if not hints:
            return question or "请检索当前订单相关履约规则和优秀案例。"
        return "\n".join([question or "请检索当前订单相关履约规则和优秀案例。", *hints])

    @staticmethod
    def _source_set(
        result: Any,
        source_type: str,
        fallback_query: str,
        collection_name: str,
        current_business_context: dict[str, Any] | None = None,
        sop_key_points: list[str] | None = None,
    ) -> RAGSourceSet:
        summary = getattr(result, "answer_summary", None)
        key_points = list(getattr(summary, "key_rules", []) or [])
        if not key_points and getattr(summary, "conclusion", ""):
            key_points = [summary.conclusion]
        hits: list[RAGEvidenceItem] = []
        ignored_hits: list[RAGEvidenceItem] = []
        for hit in list(getattr(result, "hits", []) or [])[:3]:
            item = PlannerRAGContextService._evidence_item(hit, source_type)
            if source_type == "excellent_case":
                reasons = PlannerRAGContextService._case_conflict_reasons(
                    item=item,
                    current_business_context=current_business_context,
                    sop_key_points=sop_key_points or [],
                )
                if reasons:
                    item = item.model_copy(
                        update={
                            "conflict_status": "ignored_conflict",
                            "conflict_reasons": reasons,
                        }
                    )
                    ignored_hits.append(item)
                    continue
            hits.append(item)
        safe_key_points = key_points[:5]
        if source_type == "excellent_case":
            safe_key_points = [
                item for item in key_points[:5]
                if not PlannerRAGContextService._case_text_conflict_reasons(
                    item,
                    current_business_context=current_business_context,
                    sop_key_points=sop_key_points or [],
                )
            ]
        return RAGSourceSet(
            query=fallback_query,
            collection_name=collection_name,
            applied_filters=list(getattr(result, "applied_filters", []) or []),
            key_points=safe_key_points,
            coverage_note=getattr(summary, "coverage_note", "") if summary else "",
            evidence=hits,
            ignored_evidence=ignored_hits,
        )

    @staticmethod
    def _evidence_item(hit: Any, source_type: str) -> RAGEvidenceItem:
        metadata = getattr(hit, "metadata", None)
        text = str(getattr(hit, "text", "") or "")
        return RAGEvidenceItem(
            source_type=source_type,
            source_file=str(getattr(hit, "source_file", "") or getattr(metadata, "source_file", "") or ""),
            category=str(getattr(hit, "category", "") or ""),
            title=str(getattr(metadata, "title", "") or ""),
            chunk_id=str(getattr(metadata, "chunk_id", "") or ""),
            source_case_id=str(getattr(metadata, "source_case_id", "") or ""),
            score=float(getattr(hit, "score", 0.0) or 0.0),
            text_excerpt=text[:180],
        )

    @staticmethod
    def _case_conflict_reasons(
        *,
        item: RAGEvidenceItem,
        current_business_context: dict[str, Any] | None,
        sop_key_points: list[str],
    ) -> list[str]:
        text = "\n".join([item.title, item.text_excerpt])
        return PlannerRAGContextService._case_text_conflict_reasons(
            text,
            current_business_context=current_business_context,
            sop_key_points=sop_key_points,
        )

    @staticmethod
    def _case_text_conflict_reasons(
        text: str,
        *,
        current_business_context: dict[str, Any] | None,
        sop_key_points: list[str],
    ) -> list[str]:
        reasons: list[str] = []
        if any(pattern in text for pattern in DIRECT_MUTATION_PATTERNS):
            reasons.append("case_suggests_direct_business_mutation")
        if any(pattern in text for pattern in SOP_BYPASS_PATTERNS) and any(
            pattern in "\n".join(sop_key_points) for pattern in HITL_SOP_PATTERNS
        ):
            reasons.append("case_conflicts_with_active_sop")
        disallowed_skus = PlannerRAGContextService._split_disallowed_skus(current_business_context or {})
        if disallowed_skus and any(pattern in text for pattern in SPLIT_PATTERNS):
            matched_skus = [sku for sku in disallowed_skus if re.search(re.escape(sku), text, re.IGNORECASE)]
            if matched_skus:
                reasons.append("case_conflicts_with_current_product_restriction")
        return list(dict.fromkeys(reasons))

    @staticmethod
    def _split_disallowed_skus(current_business_context: dict[str, Any]) -> list[str]:
        rows = list(current_business_context.get("product_restrictions") or [])
        if not rows:
            rows = list(current_business_context.get("order_items") or [])
        skus: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("split_allowed") is False and row.get("sku_id"):
                skus.append(str(row["sku_id"]))
        return skus
