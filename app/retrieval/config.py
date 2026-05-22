from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_BATCH_SIZE = 256
DEFAULT_CANDIDATE_POOL_SIZE = 64
DEFAULT_BM25_WEIGHT = 0.35
DEFAULT_VECTOR_WEIGHT = 0.65
DEFAULT_CACHE_DIR = Path(".cache/retrieval")


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE
    bm25_weight: float = DEFAULT_BM25_WEIGHT
    vector_weight: float = DEFAULT_VECTOR_WEIGHT
    cache_enabled: bool = True
    cache_dir: Path = DEFAULT_CACHE_DIR

    def __post_init__(self) -> None:
        if not self.embedding_model_name.strip():
            raise ValueError("embedding_model_name must not be empty")
        if self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be at least 1")
        if self.candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be at least 1")
        if self.bm25_weight < 0.0 or self.vector_weight < 0.0:
            raise ValueError("retrieval weights must be non-negative")
        if self.bm25_weight + self.vector_weight <= 0.0:
            raise ValueError("at least one retrieval weight must be positive")
        if not str(self.cache_dir).strip():
            raise ValueError("cache_dir must not be empty")