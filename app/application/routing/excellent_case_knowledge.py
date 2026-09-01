"""Write successful fulfillment cases into the RAG knowledge source."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.schemas.fulfillment_case import ExcellentCaseRecord


class ExcellentCaseKnowledgeWriter:
    """Persist excellent cases as Markdown documents for PGVector indexing."""

    def __init__(self, knowledge_dir: str, *, collection_name: str = "case_collection") -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.collection_name = collection_name

    def write(self, record: ExcellentCaseRecord) -> str:
        directory = self.knowledge_dir / "excellent_cases"
        directory.mkdir(parents=True, exist_ok=True)
        doc_id = self._safe_doc_id(record.excellent_case_id)
        path = directory / f"{doc_id}.md"
        path.write_text(self._markdown(record, doc_id, self.collection_name), encoding="utf-8")
        return str(path)

    @staticmethod
    def _safe_doc_id(value: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_\-]+", "-", value.strip()).strip("-").lower()
        return safe or "excellent-case"

    @staticmethod
    def _markdown(record: ExcellentCaseRecord, doc_id: str, collection_name: str) -> str:
        title = ExcellentCaseKnowledgeWriter._front_matter_scalar(record.scenario)
        prompt_version = ExcellentCaseKnowledgeWriter._front_matter_scalar(record.prompt_version or "")
        extraction_warnings = ExcellentCaseKnowledgeWriter._front_matter_list(record.extraction_warnings)
        heading = ExcellentCaseKnowledgeWriter._single_line(record.scenario) or "优秀案例"
        conditions = "\n".join(f"- {ExcellentCaseKnowledgeWriter._single_line(item)}" for item in record.key_conditions) or "- 无"
        plan = "\n".join(
            "- {action_type} / SKU={sku_id} / qty={quantity} / from={from_warehouse} / to={to_warehouse} / carrier={carrier} / reason={reason}".format(
                action_type=ExcellentCaseKnowledgeWriter._single_line(item.get("action_type", "")),
                sku_id=ExcellentCaseKnowledgeWriter._single_line(item.get("sku_id", "")),
                quantity=ExcellentCaseKnowledgeWriter._single_line(item.get("quantity", "")),
                from_warehouse=ExcellentCaseKnowledgeWriter._single_line(item.get("from_warehouse", "")),
                to_warehouse=ExcellentCaseKnowledgeWriter._single_line(item.get("to_warehouse", "")),
                carrier=ExcellentCaseKnowledgeWriter._single_line(item.get("carrier", "")),
                reason=ExcellentCaseKnowledgeWriter._single_line(item.get("reason", "")),
            )
            for item in record.final_plan
        ) or "- 无"
        evidence = "\n".join(f"- {ExcellentCaseKnowledgeWriter._single_line(item)}" for item in record.sop_evidence) or "- 无"
        human_changes = "\n".join(f"- {ExcellentCaseKnowledgeWriter._single_line(item)}" for item in record.human_changes) or "- 无"
        final_result = "\n".join(
            f"- {ExcellentCaseKnowledgeWriter._single_line(key)}: {ExcellentCaseKnowledgeWriter._single_line(value)}"
            for key, value in record.final_result.items()
        ) or "- 无"
        result_items = "\n".join(
            f"- {ExcellentCaseKnowledgeWriter._single_line(key)}: {ExcellentCaseKnowledgeWriter._single_line(value)}"
            for key, value in record.execution_result.items()
        ) or "- 无"
        return f"""---
document_id: {doc_id}
category: excellent_case
knowledge_source: case
collection_name: {collection_name}
version_status: active
is_active: true
title: {title}
business_scope: [fulfillment, abnormal_order]
vector_backend: {record.vector_backend}
source_case_id: {record.case_id}
extraction_use_case: {record.extraction_use_case}
prompt_version: {prompt_version}
extraction_mode: {record.extraction_mode}
extraction_warnings: {extraction_warnings}
---

# {heading}

## 异常场景

订单号已脱敏：{record.order_id_masked}

## 关键条件

{conditions}

## 最终方案

{plan}

## SOP 依据

{evidence}

## 人工调整

{human_changes}

## 最终结果

{final_result}

## 执行结果

{result_items}
"""

    @staticmethod
    def _single_line(value: object) -> str:
        return re.sub(r"\s+", " ", str(value)).replace("---", "- - -").strip()

    @classmethod
    def _front_matter_scalar(cls, value: object) -> str:
        return json.dumps(cls._single_line(value), ensure_ascii=False)

    @classmethod
    def _front_matter_list(cls, values: list[str]) -> str:
        return json.dumps([cls._single_line(item) for item in values], ensure_ascii=False)
