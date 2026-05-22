from __future__ import annotations

from app.domain.schemas import NormalizedQuery


class QueryNormalizer:
    def normalize(self, question: str) -> NormalizedQuery:
        compact = " ".join(question.strip().split())
        return NormalizedQuery(original=compact, expanded_query=compact, additions=[])