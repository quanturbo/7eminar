from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

from rank_bm25 import BM25Plus

from app.domain.schemas import Chunk, RetrievalHit

TOKEN_RE = re.compile(r"[\wА-Яа-яІіЇїЄєҐґ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class HybridRetriever:
    chunks: list[Chunk]
    bm25: BM25Plus
    tokenized_corpus: list[list[str]]
    vectors: list[list[float]]

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> HybridRetriever:
        if not chunks:
            raise ValueError("HybridRetriever requires at least one chunk")
        tokenized = [tokenize(f"{chunk.section} {chunk.text}") for chunk in chunks]
        return cls(
            chunks=chunks,
            bm25=BM25Plus(tokenized),
            tokenized_corpus=tokenized,
            vectors=[_hashed_vector(tokens) for tokens in tokenized],
        )

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievalHit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        bm25_scores = list(self.bm25.get_scores(query_tokens))
        query_vector = _hashed_vector(query_tokens)
        vector_scores = [_cosine(query_vector, vector) for vector in self.vectors]

        bm25_ranks = _ranks_desc(bm25_scores)
        vector_ranks = _ranks_desc(vector_scores)
        raw_fusion = []
        for index in range(len(self.chunks)):
            if bm25_scores[index] <= 0 and vector_scores[index] <= 0:
                fusion = 0.0
            else:
                fusion = 0.62 / (60 + bm25_ranks[index]) + 0.38 / (60 + vector_ranks[index])
            raw_fusion.append(fusion)

        max_fusion = max(raw_fusion) or 1.0
        hits = [
            RetrievalHit(
                chunk=chunk,
                bm25_score=float(bm25_scores[index]),
                vector_score=float(vector_scores[index]),
                fusion_score=float(raw_fusion[index]),
                score=float(raw_fusion[index] / max_fusion),
            )
            for index, chunk in enumerate(self.chunks)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]


def _ranks_desc(scores: list[float]) -> list[int]:
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    ranks = [0] * len(scores)
    for rank, index in enumerate(ranked, start=1):
        ranks[index] = rank
    return ranks


def _hashed_vector(tokens: list[str], dimensions: int = 192) -> list[float]:
    counts = Counter(tokens)
    vector = [0.0] * dimensions
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    return max(0.0, sum(a * b for a, b in zip(left, right, strict=True)))