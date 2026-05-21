from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


def get_settings() -> Settings:
    return Settings()