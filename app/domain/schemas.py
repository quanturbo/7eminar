from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Confidence = Literal["high", "medium", "low", "fallback"]
FallbackReason = Literal[
    "no_relevant_context",
    "missing_exact_date",
    "missing_calculation_inputs",
    "context_says_not_enough_data",
    "llm_schema_error",
    "unsupported_answer_detected",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AskRequest(StrictModel):
    question: str = Field(min_length=3, max_length=2000)


class Source(StrictModel):
    section: str
    chunk: str
    score: float = Field(ge=0.0, le=1.0)


class AskResponse(StrictModel):
    answer: str
    sources: list[Source]
    confidence: Confidence
    fallback_reason: FallbackReason | None
    trace_id: str
    latency_ms: int = Field(ge=0)


class Chunk(StrictModel):
    chunk_id: str
    section: str
    text: str
    language_hint: Literal["uk", "en", "mixed"]
    source_path: str


class RetrievalHit(StrictModel):
    chunk: Chunk
    bm25_score: float = 0.0
    vector_score: float = 0.0
    fusion_score: float = 0.0
    score: float = Field(ge=0.0, le=1.0)


class NormalizedQuery(StrictModel):
    original: str
    expanded_query: str
    additions: list[str]


class ContextDecision(StrictModel):
    answerable: bool
    fallback_reason: FallbackReason | None = None
    confidence: Confidence = "low"
    note: str


class LLMStructuredAnswer(StrictModel):
    answerable: bool
    answer: str
    used_source_ids: list[str]
    missing_information: list[str] = Field(default_factory=list)
    grounding_note: str = ""


class TraceRecord(StrictModel):
    trace_id: str
    question: str
    stages: list[dict[str, object]]
    retrieved: list[dict[str, object]]
    final_response: dict[str, object]
    confidence: Confidence
    fallback_reason: FallbackReason | None
    latency_ms: int
    errors: list[str]