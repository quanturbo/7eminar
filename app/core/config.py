from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.retrieval.config import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_CACHE_DIR,
    DEFAULT_CANDIDATE_POOL_SIZE,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_VECTOR_WEIGHT,
    RetrievalConfig,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="OPENAI_BASE_URL",
    )
    openai_model: str = Field(default="openai/gpt-4o-mini", validation_alias="OPENAI_MODEL")
    knowledge_base_path: Path = Field(
        default=Path("knowledge_base.md"),
        validation_alias="KNOWLEDGE_BASE_PATH",
    )
    trace_path: Path = Field(default=Path("traces.jsonl"), validation_alias="TRACE_PATH")
    retrieval_top_k: int = Field(default=4, ge=1, le=10, validation_alias="RETRIEVAL_TOP_K")
    min_retrieval_score: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        validation_alias="MIN_RETRIEVAL_SCORE",
    )
    min_vector_only_score: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        validation_alias="MIN_VECTOR_ONLY_SCORE",
    )
    embedding_model_name: str = Field(
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        min_length=1,
        validation_alias="EMBEDDING_MODEL_NAME",
    )
    embedding_batch_size: int = Field(
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        ge=1,
        validation_alias="EMBEDDING_BATCH_SIZE",
    )
    retrieval_candidate_pool_size: int = Field(
        default=DEFAULT_CANDIDATE_POOL_SIZE,
        ge=1,
        validation_alias="RETRIEVAL_CANDIDATE_POOL_SIZE",
    )
    retrieval_bm25_weight: float = Field(
        default=DEFAULT_BM25_WEIGHT,
        ge=0.0,
        validation_alias="RETRIEVAL_BM25_WEIGHT",
    )
    retrieval_vector_weight: float = Field(
        default=DEFAULT_VECTOR_WEIGHT,
        ge=0.0,
        validation_alias="RETRIEVAL_VECTOR_WEIGHT",
    )
    retrieval_cache_enabled: bool = Field(
        default=True,
        validation_alias="RETRIEVAL_CACHE_ENABLED",
    )
    retrieval_cache_dir: Path = Field(
        default=DEFAULT_CACHE_DIR,
        validation_alias="RETRIEVAL_CACHE_DIR",
    )

    def retrieval_config(self) -> RetrievalConfig:
        return RetrievalConfig(
            embedding_model_name=self.embedding_model_name,
            embedding_batch_size=self.embedding_batch_size,
            candidate_pool_size=self.retrieval_candidate_pool_size,
            bm25_weight=self.retrieval_bm25_weight,
            vector_weight=self.retrieval_vector_weight,
            cache_enabled=self.retrieval_cache_enabled,
            cache_dir=self.retrieval_cache_dir,
        )


def get_settings() -> Settings:
    return Settings()