import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.core.guardrails import Guardrails
from app.core.pipeline import RagPipeline
from app.domain.schemas import LLMStructuredAnswer, RetrievalHit
from app.generation.openai_client import _normalize_payload, _parse_json_payload
from app.knowledge.loader import load_knowledge_base
from app.observability.trace_logger import TraceLogger
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query import QueryNormalizer
from app.testing.fakes import FakeLLMClient

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "knowledge_base.md"


class SpyLLMClient:
    def __init__(self) -> None:
        self.received_hits: list[RetrievalHit] | None = None
        self._fake = FakeLLMClient()

    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        force_answerable_retry: bool = True,
    ) -> LLMStructuredAnswer:
        self.received_hits = hits
        return self._fake.generate(question, hits, force_answerable_retry)


def build_pipeline(tmp_path: Path) -> RagPipeline:
    chunks = load_knowledge_base(KB_PATH)
    retriever = HybridRetriever.from_chunks(chunks)
    return RagPipeline(
        normalizer=QueryNormalizer(),
        retriever=retriever,
        guardrails=Guardrails(),
        llm_client=FakeLLMClient(),
        trace_logger=TraceLogger(tmp_path / "traces.jsonl"),
    )


def test_kb_loader_discovers_sections_and_chunks() -> None:
    chunks = load_knowledge_base(KB_PATH)

    sections = {chunk.section for chunk in chunks}

    assert len(sections) == 10
    assert "1. Щорічна відпустка" in sections
    assert "2. Sick leave policy" in sections
    assert all(chunk.chunk_id for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)


def test_hybrid_retriever_finds_expected_sections_for_company_questions() -> None:
    chunks = load_knowledge_base(KB_PATH)
    retriever = HybridRetriever.from_chunks(chunks)
    normalizer = QueryNormalizer()

    expectations = [
        (
            "Я працюю в компанії 3 місяці. Чи можу вже піти у щорічну оплачувану "
            "відпустку?",
            "1. Щорічна відпустка",
        ),
        (
            "Працівник захворів, але ще не надав медичний документ. Чи можна одразу "
            "оплатити лікарняний?",
            "2. Sick leave policy",
        ),
        (
            "У нас є payroll-система. Чи можемо ми автоматично індексувати зарплату "
            "всім працівникам однаково?",
            "3. Індексація зарплати / Salary indexation",
        ),
        ("Яка точна дата сплати ЄСВ у цьому місяці?", "4. Податкові строки"),
        (
            "Порахуй індексацію для працівника із зарплатою 25000 грн, якщо базовий "
            "місяць невідомий.",
            "3. Індексація зарплати / Salary indexation",
        ),
    ]

    for question, expected_section in expectations:
        expanded = normalizer.normalize(question).expanded_query
        hits = retriever.retrieve(expanded, top_k=3)
        assert hits, question
        assert any(hit.chunk.section == expected_section for hit in hits), question


def test_guardrails_detect_missing_exact_dates_and_calculation_inputs() -> None:
    guardrails = Guardrails()
    chunks = load_knowledge_base(KB_PATH)
    retriever = HybridRetriever.from_chunks(chunks)
    normalizer = QueryNormalizer()

    date_question = "Яка точна дата сплати ЄСВ у цьому місяці?"
    date_hits = retriever.retrieve(normalizer.normalize(date_question).expanded_query)
    date_result = guardrails.evaluate_context(date_question, date_hits)
    assert not date_result.answerable
    assert date_result.fallback_reason == "missing_exact_date"

    calc_question = (
        "Порахуй індексацію для працівника із зарплатою 25000 грн, "
        "якщо базовий місяць невідомий."
    )
    calc_hits = retriever.retrieve(normalizer.normalize(calc_question).expanded_query)
    calc_result = guardrails.evaluate_context(calc_question, calc_hits)
    assert not calc_result.answerable
    assert calc_result.fallback_reason == "missing_calculation_inputs"


def test_pipeline_fallbacks_for_out_of_scope_question(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)

    response = pipeline.ask("Як скачати ігру онлайн?")

    assert response.confidence == "fallback"
    assert response.fallback_reason == "no_relevant_context"
    assert "релевантний контекст" in response.answer.lower()
    assert response.sources == []


def test_pipeline_fallbacks_for_missing_facts_exceptions_and_amounts(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)

    cases = [
        "Які винятки дозволяють піти у щорічну відпустку до 6 місяців?",
        "Яка максимальна тривалість випробувального строку?",
        "Скільки днів треба попередити роботодавця про звільнення за власним бажанням?",
        "Скільки компанія компенсує за інтернет при дистанційній роботі?",
    ]

    for question in cases:
        response = pipeline.ask(question)

        assert response.confidence == "fallback", question
        assert response.fallback_reason == "context_says_not_enough_data", question
        assert "недостатньо інформації" in response.answer.lower(), question
        assert response.sources, question


def test_fallback_trace_records_context_limitation(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)

    response = pipeline.ask("Яка максимальна тривалість випробувального строку?")

    trace = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert trace["trace_id"] == response.trace_id
    assert trace["fallback_reason"] == "context_says_not_enough_data"
    assert trace["retrieved"]
    assert any(
        stage["name"] == "grade_context" and not stage["answerable"]
        for stage in trace["stages"]
    )
    assert trace["final_response"]["fallback_reason"] == "context_says_not_enough_data"


def test_live_adapter_normalizes_common_model_schema_variance() -> None:
    chunks = load_knowledge_base(KB_PATH)
    hit = HybridRetriever.from_chunks(chunks).retrieve("лікарняний медичний документ", top_k=1)[0]
    payload = {
        "answerable": "true",
        "answer": "Ні, лікарняний не можна оплатити без медичного документа.",
        "used_source_ids": [],
        "missing_information": "",
        "grounding_note": "Based on retrieved context.",
    }

    normalized = _normalize_payload(payload, [hit])

    assert normalized["answerable"] is True
    assert normalized["used_source_ids"] == [hit.chunk.chunk_id]
    assert normalized["missing_information"] == []


def test_live_adapter_extracts_json_from_markdown_response() -> None:
    payload = _parse_json_payload(
        "```json\n"
        '{"answerable": true, "answer": "Так", "used_source_ids": ["s01-c01"]}'
        "\n```"
    )

    assert payload["answerable"] is True
    assert payload["used_source_ids"] == ["s01-c01"]


def test_pipeline_returns_grounded_answer_and_trace(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)

    response = pipeline.ask(
        "Я працюю в компанії 3 місяці. Чи можу вже піти у щорічну оплачувану "
        "відпустку?"
    )

    assert response.confidence == "high"
    assert response.fallback_reason is None
    assert "6 місяців" in response.answer
    assert "3 міся" in response.answer
    assert any(source.section == "1. Щорічна відпустка" for source in response.sources)

    trace_text = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip()
    trace = json.loads(trace_text)
    assert trace["trace_id"] == response.trace_id
    assert [stage["name"] for stage in trace["stages"]] == [
        "load_context",
        "normalize_query",
        "retrieve_context",
        "grade_context",
        "generate_answer",
        "validate_answer",
        "build_response",
    ]
    assert trace["retrieved"]


def test_pipeline_fallback_for_unsupported_exact_date(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)

    response = pipeline.ask("Яка точна дата сплати ЄСВ у цьому місяці?")

    assert response.confidence == "fallback"
    assert response.fallback_reason == "missing_exact_date"
    assert "недостатньо" in response.answer.lower()
    assert "точної" in response.answer.lower()


def test_pipeline_sends_only_retrieved_chunks_to_llm(tmp_path: Path) -> None:
    chunks = load_knowledge_base(KB_PATH)
    spy_llm = SpyLLMClient()
    pipeline = RagPipeline(
        normalizer=QueryNormalizer(),
        retriever=HybridRetriever.from_chunks(chunks),
        guardrails=Guardrails(),
        llm_client=spy_llm,
        trace_logger=TraceLogger(tmp_path / "traces.jsonl"),
        top_k=2,
    )

    response = pipeline.ask(
        "Я працюю в компанії 3 місяці. Чи можу вже піти у щорічну оплачувану "
        "відпустку?"
    )

    assert response.fallback_reason is None
    assert spy_llm.received_hits is not None
    assert len(spy_llm.received_hits) == 2
    assert len(spy_llm.received_hits) < len(chunks)
    trace = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8"))
    assert {hit.chunk.chunk_id for hit in spy_llm.received_hits} == {
        retrieved["chunk_id"] for retrieved in trace["retrieved"]
    }


def test_api_endpoint_returns_structured_json(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)
    app = create_app(pipeline=pipeline)
    client = TestClient(app)

    response = client.post(
        "/ask",
        json={
            "question": "Працівник захворів, але ще не надав медичний документ. "
            "Чи можна одразу оплатити лікарняний?"
        },
    )

    assert response.status_code == 200
    data: dict[str, Any] = response.json()
    assert set(data) == {
        "answer",
        "sources",
        "confidence",
        "fallback_reason",
        "trace_id",
        "latency_ms",
    }
    assert data["confidence"] in {"high", "medium", "low", "fallback"}
    assert data["fallback_reason"] is None
    assert data["sources"]
    assert data["trace_id"]


def test_health_reports_provider_and_model(tmp_path: Path) -> None:
    pipeline = build_pipeline(tmp_path)
    app = create_app(pipeline=pipeline)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_provider"] == "FakeLLMClient"
    assert data["llm_model"] == "offline-fake"