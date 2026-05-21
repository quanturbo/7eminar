from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.guardrails import Guardrails  # noqa: E402
from app.kb_loader import load_knowledge_base  # noqa: E402
from app.llm_client import FakeLLMClient  # noqa: E402
from app.pipeline import RagPipeline  # noqa: E402
from app.query import QueryNormalizer  # noqa: E402
from app.retriever import HybridRetriever  # noqa: E402
from app.schemas import AskResponse  # noqa: E402
from app.trace_logger import TraceLogger  # noqa: E402

DEFAULT_QUESTIONS = ROOT / "test_questions.json"
DEFAULT_KB = ROOT / "knowledge_base.md"
DEFAULT_REPORT = ROOT / "evaluation_report.md"
REQUIRED_RESPONSE_KEYS = {
    "answer",
    "sources",
    "confidence",
    "fallback_reason",
    "trace_id",
    "latency_ms",
}
LIVE_PROBE = {
    "question": "Чи може працівник взяти щорічну відпустку після 3 місяців роботи?",
    "section": "1. Щорічна відпустка",
    "must_contain": ["6 місяців"],
}

EXPECTED: dict[str, dict[str, Any]] = {
    "q001": {
        "section": "1. Щорічна відпустка",
        "fallback": None,
        "must_contain": ["6 місяців"],
    },
    "q002": {"section": "2. Sick leave policy", "fallback": None, "must_contain": ["медич"]},
    "q003": {
        "section": "3. Індексація зарплати / Salary indexation",
        "fallback": None,
        "must_contain": ["базов"],
    },
    "q004": {
        "section": "4. Податкові строки",
        "fallback": "missing_exact_date",
        "must_contain": ["недостатньо"],
    },
    "q005": {
        "section": "3. Індексація зарплати / Salary indexation",
        "fallback": "missing_calculation_inputs",
        "must_contain": ["розрах"],
    },
    "q006": {
        "section": None,
        "fallback": "no_relevant_context",
        "must_contain": ["релевантний контекст"],
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run evaluation for the controlled RAG service."
    )
    parser.add_argument(
        "--base-url",
        help="Optional running API base URL, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Fail unless /health reports a live provider and the model answers an in-scope probe.",
    )
    parser.add_argument(
        "--expected-model",
        help="Optional exact llm_model expected from /health, e.g. openai/gpt-4o-mini.",
    )
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    pipeline = None if args.base_url else _build_direct_pipeline()
    health: dict[str, Any] | None = None
    base_url = args.base_url.rstrip("/") if args.base_url else None

    if base_url:
        health = _get_health(base_url, args.timeout)
        provider = str(health.get("llm_provider", "unknown"))
        model = str(health.get("llm_model", "unknown"))
        print(
            f"health: status={health.get('status')} "
            f"chunks={health.get('chunks')} provider={provider} model={model}"
        )
        if args.require_live and provider != "OpenAICompatibleLLMClient":
            print("FAIL: expected live provider OpenAICompatibleLLMClient")
            return 1
        if args.expected_model and model != args.expected_model:
            print(f"FAIL: expected model {args.expected_model!r}, got {model!r}")
            return 1
        local_expected_model = _local_expected_model(base_url) if args.require_live else None
        if local_expected_model and model != local_expected_model:
            print(
                f"FAIL: running server model {model!r} does not match current local "
                f"settings model {local_expected_model!r}. Restart Uvicorn after editing .env."
            )
            return 1
        if args.require_live:
            probe_errors = _validate_live_probe(base_url, args.timeout)
            if probe_errors:
                print("FAIL: live LLM probe did not produce a grounded in-scope answer")
                for error in probe_errors:
                    print(f"  - {error}")
                return 1

    rows = []

    for item in questions:
        started = time.monotonic()
        raw_response: dict[str, Any] | AskResponse
        if base_url:
            raw_response = _ask_api(base_url, item["question"], args.timeout)
        else:
            if pipeline is None:
                raise RuntimeError("Direct evaluation pipeline was not initialized.")
            raw_response = pipeline.ask(item["question"])
        latency_ms = int((time.monotonic() - started) * 1000)
        payload = raw_response if isinstance(raw_response, dict) else raw_response.model_dump()
        rows.append(_evaluate_row(item["id"], item["question"], payload, latency_ms))

    args.report.write_text(_render_report(rows, bool(base_url), health), encoding="utf-8")
    failures = [row for row in rows if not row["passed"]]
    print(
        f"evaluated={len(rows)} passed={len(rows) - len(failures)} "
        f"failed={len(failures)} report={args.report}"
    )
    return 1 if failures else 0


def _build_direct_pipeline() -> RagPipeline:
    chunks = load_knowledge_base(DEFAULT_KB)
    return RagPipeline(
        normalizer=QueryNormalizer(),
        retriever=HybridRetriever.from_chunks(chunks),
        guardrails=Guardrails(),
        llm_client=FakeLLMClient(),
        trace_logger=TraceLogger(ROOT / "traces.jsonl"),
    )


def _get_health(base_url: str, timeout: float) -> dict[str, Any]:
    with httpx.Client(timeout=timeout) as client:
        response = client.get(f"{base_url}/health")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("Health response must be a JSON object.")
        return payload


def _ask_api(base_url: str, question: str, timeout: float) -> dict[str, Any]:
    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{base_url}/ask", json={"question": question})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("API response must be a JSON object.")
        return payload


def _validate_live_probe(base_url: str, timeout: float) -> list[str]:
    payload = _ask_api(base_url, str(LIVE_PROBE["question"]), timeout)
    errors: list[str] = []
    if set(payload) != REQUIRED_RESPONSE_KEYS:
        return [f"unexpected probe response keys: {sorted(payload)}"]
    if payload["fallback_reason"] is not None:
        errors.append(f"probe fallback_reason expected None, got {payload['fallback_reason']!r}")
    sections = _source_sections(payload)
    if LIVE_PROBE["section"] not in sections:
        errors.append(f"probe expected source section {LIVE_PROBE['section']!r}, got {sections!r}")
    answer_lower = str(payload["answer"]).lower()
    missing_terms = [
        term for term in LIVE_PROBE["must_contain"] if str(term).lower() not in answer_lower
    ]
    if missing_terms:
        errors.append(f"probe answer missing expected terms: {missing_terms}")
    return errors


def _local_expected_model(base_url: str) -> str | None:
    hostname = urlparse(base_url).hostname
    if hostname not in {"127.0.0.1", "localhost"}:
        return None
    return Settings().openai_model


def _evaluate_row(
    question_id: str,
    question: str,
    payload: dict[str, Any],
    latency_ms: int,
) -> dict[str, Any]:
    expected = EXPECTED.get(question_id)
    answer = str(payload.get("answer", ""))
    answer_lower = answer.lower()
    source_sections = _source_sections(payload)
    if expected is None:
        checks = {"expected_case_configured": False}
    else:
        expected_section = expected["section"]
        section_ok = (
            not source_sections if expected_section is None else expected_section in source_sections
        )
        checks = {
            "response_schema_correct": set(payload) == REQUIRED_RESPONSE_KEYS,
            "expected_section_found": section_ok,
            "fallback_correct": payload.get("fallback_reason") == expected["fallback"],
            "answer_contains_expected_trait": all(
                term.lower() in answer_lower for term in expected["must_contain"]
            ),
            "no_empty_trace_id": bool(payload.get("trace_id")),
        }
    return {
        "id": question_id,
        "question": question,
        "answer": answer,
        "sources": source_sections,
        "confidence": payload.get("confidence", "<missing>"),
        "fallback_reason": payload.get("fallback_reason"),
        "latency_ms": payload.get("latency_ms", latency_ms),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _source_sections(payload: dict[str, Any]) -> list[str]:
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        return []
    sections: list[str] = []
    for source in sources:
        if isinstance(source, dict) and isinstance(source.get("section"), str):
            sections.append(source["section"])
    return sections


def _render_report(
    rows: list[dict[str, Any]],
    api_mode: bool,
    health: dict[str, Any] | None,
) -> str:
    passed = sum(1 for row in rows if row["passed"])
    avg_latency = statistics.mean(row["latency_ms"] for row in rows) if rows else 0
    grounded_ids = [row["id"] for row in rows if row["fallback_reason"] is None]
    fallback_ids = [row["id"] for row in rows if row["fallback_reason"] is not None]
    grounded_summary = ", ".join(grounded_ids) if grounded_ids else "none"
    fallback_summary = ", ".join(fallback_ids) if fallback_ids else "none"
    lines = [
        "# Evaluation Report",
        "",
        f"Mode: {'running HTTP API' if api_mode else 'direct pipeline with FakeLLMClient'}.",
        f"Health provider: {health.get('llm_provider') if health else 'direct FakeLLMClient'}.",
        f"Health model: {health.get('llm_model') if health else 'offline-fake'}.",
        f"Questions evaluated: {len(rows)}.",
        f"Passed checks: {passed}/{len(rows)}.",
        f"Average latency: {avg_latency:.1f} ms.",
        "",
        "## Per-question results",
        "",
    ]
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        lines.extend(
            [
                f"### {row['id']} - {status}",
                "",
                f"Question: {row['question']}",
                f"Confidence: {row['confidence']}",
                f"Fallback reason: {row['fallback_reason']}",
                f"Retrieved sections: {', '.join(row['sources'])}",
                f"Answer: {row['answer']}",
                "Checks:",
            ]
        )
        for name, value in row["checks"].items():
            lines.append(f"- {name}: {'yes' if value else 'no'}")
        lines.append("")

    lines.extend(
        [
            "## Hallucination and fallback review",
            "",
            f"Grounded-answer cases: {grounded_summary}. Fallback cases: {fallback_summary}. "
            "The fallback checks verify that the service does not invent exact dates, "
            "numeric calculations, or unsupported facts when the knowledge base does not "
            "contain enough information. The deterministic offline adapter lets these "
            "checks run without network access or API keys when direct mode is used.",
            "",
            "## Remaining weak spots",
            "",
            "- The bundled fake LLM is deterministic for testability; live model quality "
            "still depends on the configured OpenAI-compatible model.",
            "- The local vector scorer is deterministic and lightweight. Installing the "
            "optional `vector` extra enables future FAISS-backed persistence for larger corpora.",
            f"- The eval set has only {len(rows)} questions; production should add adversarial, "
            "paraphrase, and prompt-injection cases.",
            "",
            "## Production improvements",
            "",
            "Add Docker/CI, trace rotation, API authentication, a larger labeled eval "
            "set, optional LangSmith tracing, persisted FAISS indexes, a reranker, and "
            "monitoring for fallback rate and latency.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())