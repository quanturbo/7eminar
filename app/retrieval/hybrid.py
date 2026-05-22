from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol, cast

import faiss
import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray
from rank_bm25 import BM25Okapi

from app.domain.schemas import Chunk, RetrievalHit
from app.retrieval.cache import RetrievalIndexCache
from app.retrieval.config import DEFAULT_EMBEDDING_MODEL_NAME, RetrievalConfig

TOKEN_RE = re.compile(r"[\wА-Яа-яІіЇїЄєҐґ]+", re.UNICODE)
STOPWORDS = frozenset(
    {
        "а",
        "але",
        "або",
        "без",
        "в",
        "від",
        "він",
        "вона",
        "вони",
        "для",
        "до",
        "за",
        "із",
        "і",
        "й",
        "на",
        "не",
        "но",
        "при",
        "про",
        "та",
        "у",
        "чи",
        "як",
        "якщо",
        "база",
        "базі",
        "знань",
        "компанія",
        "компанії",
        "може",
        "можемо",
        "можна",
        "наданій",
        "працівник",
        "працівника",
        "працівники",
        "роботодавець",
        "роботодавцю",
        "роботодавця",
        "роботі",
        "роботу",
        "роботи",
        "треба",
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "by",
        "for",
        "from",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "company",
        "employee",
        "employer",
        "knowledge",
        "base",
    }
)


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in STOPWORDS
    ]


class EmbeddingModel(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    def embed_query(self, text: str) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class FastEmbedEmbeddingModel:
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    batch_size: int = 256

    @property
    def dimensions(self) -> int:
        return int(TextEmbedding.get_embedding_size(self.model_name))

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        model = _fastembed_model(self.model_name)
        embeddings = list(model.passage_embed(list(texts), batch_size=self.batch_size))
        return _normalized_matrix(embeddings)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        model = _fastembed_model(self.model_name)
        embeddings = list(model.query_embed(text))
        if not embeddings:
            return cast(NDArray[np.float32], np.zeros(self.dimensions, dtype=np.float32))
        return cast(NDArray[np.float32], _normalized_matrix(embeddings)[0])


@dataclass(frozen=True)
class HybridRetriever:
    chunks: list[Chunk]
    bm25: BM25Okapi
    tokenized_corpus: list[list[str]]
    embedding_model: EmbeddingModel
    vector_index: Any
    embedding_dimensions: int
    candidate_pool_size: int = 64
    bm25_weight: float = 0.35
    vector_weight: float = 0.65
    cache_key: str | None = None
    cache_status: str = "disabled"

    @classmethod
    def from_chunks(
        cls,
        chunks: list[Chunk],
        config: RetrievalConfig | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> HybridRetriever:
        if not chunks:
            raise ValueError("HybridRetriever requires at least one chunk")
        active_config = config or RetrievalConfig()
        tokenized = [tokenize(f"{chunk.section} {chunk.text}") for chunk in chunks]
        active_embedding_model = embedding_model or FastEmbedEmbeddingModel(
            model_name=active_config.embedding_model_name,
            batch_size=active_config.embedding_batch_size,
        )
        embedding_dimensions = active_embedding_model.dimensions
        cache = (
            RetrievalIndexCache(active_config.cache_dir)
            if active_config.cache_enabled
            else None
        )
        cache_model_name = _cache_embedding_model_name(active_config, active_embedding_model)
        cached = (
            cache.load(chunks, cache_model_name, embedding_dimensions)
            if cache is not None
            else None
        )
        if cached is not None:
            vector_index = cached.index
            cache_key = cached.metadata.cache_key
            cache_status = cached.status
        else:
            embeddings = active_embedding_model.embed_documents(
                [_embedding_text(chunk) for chunk in chunks]
            )
            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            if embeddings.shape != (len(chunks), embedding_dimensions):
                raise ValueError(
                    "Embedding model returned matrix with shape "
                    f"{embeddings.shape}; expected {(len(chunks), embedding_dimensions)}."
                )
            vector_index = _build_faiss_index(embeddings, embedding_dimensions)
            if cache is not None:
                metadata = cache.save(
                    chunks,
                    cache_model_name,
                    embedding_dimensions,
                    embeddings,
                    vector_index,
                )
                cache_key = metadata.cache_key
                cache_status = "miss"
            else:
                cache_key = None
                cache_status = "disabled"
        return cls(
            chunks=chunks,
            bm25=BM25Okapi(tokenized),
            tokenized_corpus=tokenized,
            embedding_model=active_embedding_model,
            vector_index=vector_index,
            embedding_dimensions=embedding_dimensions,
            candidate_pool_size=active_config.candidate_pool_size,
            bm25_weight=active_config.bm25_weight,
            vector_weight=active_config.vector_weight,
            cache_key=cache_key,
            cache_status=cache_status,
        )

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievalHit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        bm25_scores = list(self.bm25.get_scores(query_tokens))
        candidate_count = min(max(top_k * 8, self.candidate_pool_size), len(self.chunks))
        bm25_candidates = _top_indexes(bm25_scores, candidate_count)
        vector_scores, vector_candidates = self._search_vectors(query, candidate_count)

        candidates = sorted(set(bm25_candidates) | set(vector_candidates))
        if not candidates:
            return []

        max_bm25_score = max((score for score in bm25_scores if score > 0.0), default=0.0)
        max_vector_score = max(vector_scores, default=0.0)
        raw_fusion: dict[int, float] = {}
        for index in candidates:
            bm25_component = bm25_scores[index] / max_bm25_score if max_bm25_score else 0.0
            vector_component = vector_scores[index] / max_vector_score if max_vector_score else 0.0
            fusion = self.bm25_weight * bm25_component + self.vector_weight * vector_component
            raw_fusion[index] = fusion

        max_fusion = max(raw_fusion.values()) or 1.0
        hits = [
            RetrievalHit(
                chunk=self.chunks[index],
                bm25_score=float(bm25_scores[index]),
                vector_score=float(vector_scores[index]),
                fusion_score=float(raw_fusion[index]),
                score=float(raw_fusion[index] / max_fusion),
            )
            for index in candidates
            if raw_fusion[index] > 0.0
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _search_vectors(self, query: str, candidate_count: int) -> tuple[list[float], list[int]]:
        vector_scores = [0.0] * len(self.chunks)
        query_vector = self.embedding_model.embed_query(query).reshape(1, -1).astype(np.float32)
        query_vector = _normalized_matrix(query_vector)
        distances, indexes = self.vector_index.search(query_vector, candidate_count)
        vector_candidates: list[int] = []
        for raw_score, raw_index in zip(distances[0], indexes[0], strict=True):
            index = int(raw_index)
            if index < 0:
                continue
            vector_scores[index] = max(0.0, float(raw_score))
            vector_candidates.append(index)
        return vector_scores, vector_candidates


def _top_indexes(scores: Sequence[float], limit: int) -> list[int]:
    if limit <= 0:
        return []
    score_array = np.asarray(scores, dtype=np.float32)
    positive_indexes = np.flatnonzero(score_array > 0.0)
    if positive_indexes.size == 0:
        return []
    if positive_indexes.size > limit:
        selected_positions = np.argpartition(
            -score_array[positive_indexes], kth=limit - 1
        )[:limit]
        selected_indexes = positive_indexes[selected_positions]
    else:
        selected_indexes = positive_indexes
    return [
        int(index)
        for index in sorted(
            selected_indexes,
            key=lambda index: float(score_array[index]),
            reverse=True,
        )
    ]


def _embedding_text(chunk: Chunk) -> str:
    return f"{chunk.section}\n{chunk.text}"


def _build_faiss_index(embeddings: NDArray[np.float32], embedding_dimensions: int) -> Any:
    vector_index = cast(Any, faiss.IndexFlatIP(embedding_dimensions))
    vector_index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    return vector_index


def _cache_embedding_model_name(
    config: RetrievalConfig,
    embedding_model: EmbeddingModel,
) -> str:
    model_name = getattr(embedding_model, "model_name", None)
    return str(model_name) if model_name else config.embedding_model_name


@lru_cache(maxsize=2)
def _fastembed_model(model_name: str) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*now uses mean pooling instead of CLS embedding.*",
            category=UserWarning,
        )
        return TextEmbedding(model_name=model_name)


def _normalized_matrix(
    vectors: Sequence[NDArray[np.float32]] | NDArray[np.float32],
) -> NDArray[np.float32]:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = matrix / safe_norms
    return cast(NDArray[np.float32], normalized.astype(np.float32))
