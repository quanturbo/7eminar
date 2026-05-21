from __future__ import annotations

import logging
from typing import cast

from fastapi import FastAPI

from app.core.config import Settings, get_settings
from app.core.guardrails import Guardrails
from app.core.pipeline import RagPipeline
from app.domain.schemas import AskRequest, AskResponse
from app.generation.openai_client import OpenAICompatibleLLMClient
from app.knowledge.loader import load_knowledge_base
from app.observability.trace_logger import TraceLogger
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query import QueryNormalizer
from app.testing.fakes import FakeLLMClient

LOGGER = logging.getLogger("uvicorn.error")


def build_pipeline(settings: Settings | None = None) -> RagPipeline:
    settings = settings or get_settings()
    chunks = load_knowledge_base(settings.knowledge_base_path)
    llm_client = (
        OpenAICompatibleLLMClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
        )
        if settings.openai_api_key
        else FakeLLMClient()
    )
    LOGGER.info(
        "AI provider=%s model=%s base_url=%s",
        llm_client.__class__.__name__,
        _llm_model_id(llm_client),
        settings.openai_base_url if settings.openai_api_key else "offline",
    )
    return RagPipeline(
        normalizer=QueryNormalizer(),
        retriever=HybridRetriever.from_chunks(chunks),
        guardrails=Guardrails(min_score=settings.min_retrieval_score),
        llm_client=llm_client,
        trace_logger=TraceLogger(settings.trace_path),
        top_k=settings.retrieval_top_k,
    )


def create_app(pipeline: RagPipeline | None = None) -> FastAPI:
    app = FastAPI(title="Controlled AI Consultant", version="0.1.0")
    app.state.pipeline = pipeline or build_pipeline()

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        active_pipeline = cast(RagPipeline, app.state.pipeline)
        return active_pipeline.ask(request.question)

    @app.get("/health")
    def health() -> dict[str, object]:
        active_pipeline: RagPipeline = app.state.pipeline
        return {
            "status": "ok",
            "chunks": len(active_pipeline.retriever.chunks),
            "retrieval_top_k": active_pipeline.top_k,
            "llm_provider": active_pipeline.llm_client.__class__.__name__,
            "llm_model": _llm_model_id(active_pipeline.llm_client),
        }

    return app


def _llm_model_id(llm_client: object) -> str:
    model = getattr(llm_client, "model", None)
    return str(model) if model else "offline-fake"


app = create_app()