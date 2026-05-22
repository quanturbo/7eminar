from __future__ import annotations

import re

from app.domain.schemas import ContextDecision, FallbackReason, LLMStructuredAnswer, RetrievalHit

EXACT_DATE_RE = re.compile(r"точн\w*\s+дат|exact\s+date", re.I)
TAX_RE = re.compile(r"єсв|подат|tax", re.I)
CALC_RE = re.compile(r"порах|розрах|скільки|calculate|calculation", re.I)
INDEXATION_RE = re.compile(r"індексац|indexation", re.I)
EXCEPTION_RE = re.compile(r"винят|exception", re.I)
MAX_DURATION_RE = re.compile(r"максимальн\w*\s+тривал|скільки\s+.*трива", re.I)
NOTICE_PERIOD_RE = re.compile(r"строк\w*\s+поперед|поперед\w+\s+роботодав", re.I)
COMPENSATION_RE = re.compile(r"компенсац|компенсу|amount|сума|скільки", re.I)


class Guardrails:
    def __init__(self, min_score: float = 0.08, min_vector_only_score: float = 0.35) -> None:
        self.min_score = min_score
        self.min_vector_only_score = min_vector_only_score

    def evaluate_context(self, question: str, hits: list[RetrievalHit]) -> ContextDecision:
        if not hits or hits[0].score < self.min_score:
            return ContextDecision(
                answerable=False,
                fallback_reason="no_relevant_context",
                confidence="fallback",
                note="No retrieved chunk exceeded the relevance threshold.",
            )

        if (
            all(hit.bm25_score <= 0 for hit in hits)
            and hits[0].vector_score < self.min_vector_only_score
        ):
            return ContextDecision(
                answerable=False,
                fallback_reason="no_relevant_context",
                confidence="fallback",
                note=(
                    "No retrieved chunk had lexical evidence or a strong multilingual "
                    "vector match."
                ),
            )

        combined_context = "\n".join(hit.chunk.text.lower() for hit in hits)
        lowered_question = question.lower()

        if EXACT_DATE_RE.search(lowered_question) and TAX_RE.search(lowered_question):
            if "точні календарні дати" in combined_context or "точну дату" in combined_context:
                return self._fallback(
                    "missing_exact_date",
                    "Exact ESВ dates are absent from the KB.",
                )

        if CALC_RE.search(lowered_question) and INDEXATION_RE.search(lowered_question):
            missing_inputs = ["базовий місяць", "додаткові дані", "немає достатніх числових"]
            if (
                any(marker in combined_context for marker in missing_inputs)
                or "невідом" in lowered_question
            ):
                return self._fallback(
                    "missing_calculation_inputs",
                    "The KB lacks numeric inputs required for indexation calculation.",
                )

        if EXCEPTION_RE.search(lowered_question) and (
            "не описані винятки" in combined_context or "does not contain" in combined_context
        ):
            return self._fallback(
                "context_says_not_enough_data",
                "The KB explicitly does not describe requested exceptions.",
            )

        if (
            MAX_DURATION_RE.search(lowered_question)
            and "максимальної тривалості" in combined_context
        ):
            return self._fallback(
                "context_says_not_enough_data",
                "The KB does not provide the requested maximum duration.",
            )

        if (
            NOTICE_PERIOD_RE.search(lowered_question)
            and "точний строк попередження" in combined_context
        ):
            return self._fallback(
                "context_says_not_enough_data",
                "The KB does not provide the requested notice period.",
            )

        if COMPENSATION_RE.search(lowered_question) and any(
            marker in combined_context
            for marker in [
                "does not contain information about the exact amount",
                "does not contain rules about compensation",
                "does not contain rules for calculating",
            ]
        ):
            return self._fallback(
                "context_says_not_enough_data",
                "The KB does not provide the requested compensation amount or rule.",
            )

        if hits[0].score >= 0.72:
            confidence = "high"
        elif hits[0].score >= 0.38:
            confidence = "medium"
        else:
            confidence = "low"
        return ContextDecision(
            answerable=True,
            fallback_reason=None,
            confidence=confidence,
            note="Retrieved context is sufficient for a grounded answer.",
        )

    def validate_answer(
        self,
        llm_answer: LLMStructuredAnswer,
        hits: list[RetrievalHit],
    ) -> ContextDecision:
        retrieved_ids = {hit.chunk.chunk_id for hit in hits}
        used_ids = set(llm_answer.used_source_ids)
        if not llm_answer.answerable:
            return self._fallback(
                "context_says_not_enough_data",
                "LLM marked answer as not answerable.",
            )
        if not used_ids or not used_ids.issubset(retrieved_ids):
            return self._fallback(
                "unsupported_answer_detected",
                "Answer cited unsupported sources.",
            )
        return ContextDecision(
            answerable=True,
            fallback_reason=None,
            confidence="high",
            note=llm_answer.grounding_note or "Answer cites retrieved chunks.",
        )

    @staticmethod
    def _fallback(reason: FallbackReason, note: str) -> ContextDecision:
        return ContextDecision(
            answerable=False,
            fallback_reason=reason,
            confidence="fallback",
            note=note,
        )