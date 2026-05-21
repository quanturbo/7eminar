from __future__ import annotations

import time
import uuid
from typing import Any, TypedDict, cast

from langgraph.graph import END, StateGraph

from app.core.guardrails import Guardrails
from app.domain.schemas import (
    AskResponse,
    ContextDecision,
    LLMStructuredAnswer,
    NormalizedQuery,
    RetrievalHit,
    Source,
    TraceRecord,
)
from app.generation.base import LLMClient
from app.observability.trace_logger import TraceLogger
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query import QueryNormalizer


class PipelineState(TypedDict, total=False):
    trace_id: str
    question: str
    started_at: float
    normalized: NormalizedQuery
    hits: list[RetrievalHit]
    context_decision: ContextDecision
    llm_answer: LLMStructuredAnswer
    validation: ContextDecision
    response: AskResponse
    stages: list[dict[str, Any]]
    errors: list[str]


class RagPipeline:
    def __init__(
        self,
        normalizer: QueryNormalizer,
        retriever: HybridRetriever,
        guardrails: Guardrails,
        llm_client: LLMClient,
        trace_logger: TraceLogger,
        top_k: int = 4,
    ) -> None:
        self.normalizer = normalizer
        self.retriever = retriever
        self.guardrails = guardrails
        self.llm_client = llm_client
        self.trace_logger = trace_logger
        self.top_k = top_k
        self.graph: Any = self._build_graph()

    def ask(self, question: str) -> AskResponse:
        initial: PipelineState = {
            "trace_id": str(uuid.uuid4()),
            "question": question,
            "started_at": time.monotonic(),
            "stages": [{"name": "load_context", "status": "ready"}],
            "errors": [],
        }
        final_state = cast(PipelineState, self.graph.invoke(initial))
        return final_state["response"]

    def _build_graph(self) -> Any:
        graph = StateGraph(PipelineState)
        graph.add_node("normalize_query", self._normalize_query)
        graph.add_node("retrieve_context", self._retrieve_context)
        graph.add_node("grade_context", self._grade_context)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("validate_answer", self._validate_answer)
        graph.add_node("build_response", self._build_response)
        graph.add_node("write_trace", self._write_trace)

        graph.set_entry_point("normalize_query")
        graph.add_edge("normalize_query", "retrieve_context")
        graph.add_edge("retrieve_context", "grade_context")
        graph.add_edge("grade_context", "generate_answer")
        graph.add_edge("generate_answer", "validate_answer")
        graph.add_conditional_edges(
            "validate_answer",
            lambda state: "ok" if state["validation"].answerable else "fallback",
            {"ok": "build_response", "fallback": "build_response"},
        )
        graph.add_edge("build_response", "write_trace")
        graph.add_edge("write_trace", END)
        return graph.compile()

    def _normalize_query(self, state: PipelineState) -> PipelineState:
        normalized = self.normalizer.normalize(state["question"])
        return self._merge(
            state,
            normalized=normalized,
            stage={"name": "normalize_query", "additions": normalized.additions},
        )

    def _retrieve_context(self, state: PipelineState) -> PipelineState:
        hits = self.retriever.retrieve(state["normalized"].expanded_query, top_k=self.top_k)
        return self._merge(
            state,
            hits=hits,
            stage={"name": "retrieve_context", "hit_count": len(hits)},
        )

    def _grade_context(self, state: PipelineState) -> PipelineState:
        decision = self.guardrails.evaluate_context(state["question"], state["hits"])
        return self._merge(
            state,
            context_decision=decision,
            stage={
                "name": "grade_context",
                "answerable": decision.answerable,
                "reason": decision.fallback_reason,
            },
        )

    def _generate_answer(self, state: PipelineState) -> PipelineState:
        try:
            context_decision = state["context_decision"]
            generation_hits = (
                [] if context_decision.fallback_reason == "no_relevant_context" else state["hits"]
            )
            answer = self.llm_client.generate(
                state["question"],
                generation_hits,
                force_answerable_retry=context_decision.answerable,
            )
            return self._merge(
                state,
                llm_answer=answer,
                stage={
                    "name": "generate_answer",
                    "answerable": answer.answerable,
                    "input_hit_count": len(generation_hits),
                },
            )
        except Exception as exc:  # pragma: no cover - live provider safety path
            errors = [*state.get("errors", []), f"llm_error: {exc.__class__.__name__}"]
            decision = ContextDecision(
                answerable=False,
                fallback_reason="llm_schema_error",
                confidence="fallback",
                note="LLM generation failed or returned invalid schema.",
            )
            return self._merge(
                state,
                validation=decision,
                errors=errors,
                stage={"name": "generate_answer", "error": str(exc)},
            )

    def _validate_answer(self, state: PipelineState) -> PipelineState:
        if "llm_answer" not in state:
            return state
        context_decision = state["context_decision"]
        if not context_decision.answerable:
            return self._merge(
                state,
                validation=context_decision,
                stage={
                    "name": "validate_answer",
                    "answerable": False,
                    "reason": context_decision.fallback_reason,
                },
            )
        validation = self.guardrails.validate_answer(state["llm_answer"], state["hits"])
        return self._merge(
            state,
            validation=validation,
            stage={
                "name": "validate_answer",
                "answerable": validation.answerable,
                "reason": validation.fallback_reason,
            },
        )

    def _build_response(self, state: PipelineState) -> PipelineState:
        decision = state.get("validation") or state["context_decision"]
        llm_answer = state.get("llm_answer")
        latency_ms = int((time.monotonic() - state["started_at"]) * 1000)
        response_hits = state.get("hits", [])
        if decision.fallback_reason == "no_relevant_context":
            response_hits = []
        sources = [
            Source(section=hit.chunk.section, chunk=hit.chunk.text, score=round(hit.score, 4))
            for hit in response_hits
        ]

        if decision.answerable and llm_answer is not None:
            answer = llm_answer.answer
            confidence = state["context_decision"].confidence
            fallback_reason = None
        else:
            reason = decision.fallback_reason or "no_relevant_context"
            answer = _fallback_answer(reason)
            confidence = "fallback"
            fallback_reason = reason

        response = AskResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            fallback_reason=fallback_reason,
            trace_id=state["trace_id"],
            latency_ms=latency_ms,
        )
        return self._merge(
            state,
            response=response,
            stage={"name": "build_response", "confidence": confidence},
        )

    def _write_trace(self, state: PipelineState) -> PipelineState:
        response = state["response"]
        record = TraceRecord(
            trace_id=state["trace_id"],
            question=state["question"],
            stages=state.get("stages", []),
            retrieved=[
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "section": hit.chunk.section,
                    "score": hit.score,
                    "bm25_score": hit.bm25_score,
                    "vector_score": hit.vector_score,
                }
                for hit in state.get("hits", [])
            ],
            final_response=response.model_dump(),
            confidence=response.confidence,
            fallback_reason=response.fallback_reason,
            latency_ms=response.latency_ms,
            errors=state.get("errors", []),
        )
        self.trace_logger.write(record)
        return self._merge(
            state,
            stage={"name": "write_trace", "path": str(self.trace_logger.path)},
        )

    @staticmethod
    def _merge(
        state: PipelineState,
        stage: dict[str, Any] | None = None,
        **updates: Any,
    ) -> PipelineState:
        stages = [*state.get("stages", [])]
        if stage is not None:
            stages.append(stage)
        return cast(PipelineState, {**dict(state), **updates, "stages": stages})


def _fallback_answer(reason: str) -> str:
    base = "У наданій базі знань недостатньо інформації для точної відповіді."
    details = {
        "missing_exact_date": " Точна дата відсутня у джерелі, тому я не можу її вигадувати.",
        "missing_calculation_inputs": (
            " Для розрахунку бракує необхідних вхідних даних або числових правил."
        ),
        "no_relevant_context": " Релевантний контекст не знайдено.",
        "context_says_not_enough_data": " Знайдений контекст прямо вказує на обмеження даних.",
        "llm_schema_error": " Модель не повернула коректну структуровану відповідь.",
        "unsupported_answer_detected": (
            " Згенерована відповідь не була достатньо підтверджена джерелами."
        ),
    }
    return base + details.get(reason, "")