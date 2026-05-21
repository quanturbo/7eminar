from __future__ import annotations

from typing import Protocol

from app.domain.schemas import LLMStructuredAnswer, RetrievalHit


class LLMClient(Protocol):
    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        force_answerable_retry: bool = True,
    ) -> LLMStructuredAnswer: ...