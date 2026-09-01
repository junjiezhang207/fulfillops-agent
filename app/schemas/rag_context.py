"""Planner 使用的双知识源 RAG Context schema。"""

from pydantic import BaseModel, Field


class RAGEvidenceItem(BaseModel):
    source_type: str = Field(..., description="sop / excellent_case")
    source_file: str = ""
    category: str = ""
    title: str = ""
    chunk_id: str = ""
    source_case_id: str = ""
    score: float = 0.0
    text_excerpt: str = ""
    conflict_status: str = Field(default="usable", description="usable / ignored_conflict")
    conflict_reasons: list[str] = Field(default_factory=list)


class RAGSourceSet(BaseModel):
    query: str = ""
    collection_name: str = ""
    applied_filters: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    coverage_note: str = ""
    evidence: list[RAGEvidenceItem] = Field(default_factory=list)
    ignored_evidence: list[RAGEvidenceItem] = Field(default_factory=list)


class PlannerRAGContext(BaseModel):
    order_id: str
    question: str
    vector_backend: str = "pgvector"
    retrieval_strategy: str = "sop_and_case_separate"
    priority_rule: str = "current_business_state > current_sop > historical_case"
    source_collections: dict[str, str] = Field(
        default_factory=lambda: {
            "sop": "sop_collection",
            "excellent_case": "case_collection",
        }
    )
    sop_evidence: RAGSourceSet = Field(default_factory=RAGSourceSet)
    similar_cases: RAGSourceSet = Field(default_factory=RAGSourceSet)
    warnings: list[str] = Field(default_factory=list)
