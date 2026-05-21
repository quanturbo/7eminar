from __future__ import annotations

import re

from app.domain.schemas import NormalizedQuery


class QueryNormalizer:
    def __init__(self) -> None:
        self._synonyms: list[tuple[re.Pattern[str], list[str]]] = [
            (
                re.compile(r"відпуст|vacation|leave", re.I),
                ["щорічна оплачувана відпустка", "annual paid leave", "6 місяців"],
            ),
            (
                re.compile(r"лікарн|захвор|медич|sick", re.I),
                ["sick leave", "medical certificate", "paid sick leave"],
            ),
            (
                re.compile(r"індексац|indexation|payroll", re.I),
                ["salary indexation", "base month", "індекс споживчих цін"],
            ),
            (
                re.compile(r"єсв|подат|tax", re.I),
                ["ЄСВ", "податкові строки", "точні календарні дати"],
            ),
            (
                re.compile(r"дистанц|remote", re.I),
                ["remote work", "communication channels", "internal company policies"],
            ),
            (
                re.compile(r"звільнен|попереджен", re.I),
                ["звільнення за власним бажанням", "строк попередження"],
            ),
        ]

    def normalize(self, question: str) -> NormalizedQuery:
        compact = " ".join(question.strip().split())
        additions: list[str] = []
        for pattern, terms in self._synonyms:
            if pattern.search(compact):
                additions.extend(terms)

        deduped = list(dict.fromkeys(additions))
        expanded = " ".join([compact, *deduped])
        return NormalizedQuery(original=compact, expanded_query=expanded, additions=deduped)