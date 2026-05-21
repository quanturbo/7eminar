from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402

DEFAULT_QUESTIONS = ROOT / "test_questions.json"

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
    "answer_terms": ["6 місяців"],
}

EXPECTED: dict[str, dict[str, Any]] = {
    "q001": {
        "section": "1. Щорічна відпустка",
        "fallback_reason": None,
        "answer_terms": ["6 місяців"],
    },
    "q002": {
        "section": "2. Sick leave policy",
        "fallback_reason": None,
        "answer_terms": ["медич"],
    },
    "q003": {
        "section": "3. Індексація зарплати / Salary indexation",
        "fallback_reason": None,
        "answer_terms": ["базов"],
    },
    "q004": {
        "section": "4. Податкові строки",
        "fallback_reason": "missing_exact_date",
        "answer_terms": ["недостатньо"],
    },
    "q005": {
        "section": "3. Індексація зарплати / Salary indexation",
        "fallback_reason": "missing_calculation_inputs",
        "answer_terms": ["розрах"],
    },
    "q006": {
        "section": None,
        "fallback_reason": "no_relevant_context",
        "answer_terms": ["релевантний контекст"],
    },
}


def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="Check a running /ask endpoint with requests and test_questions.json."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Fail unless /health reports OpenAICompatibleLLMClient.",
    )
    parser.add_argument(
        "--expected-model",
        help="Optional exact llm_model expected from /health, e.g. openai/gpt-4o-mini.",
    )
    args = parser.parse_args()

    questions = _load_questions(args.questions)
    base_url = args.base_url.rstrip("/")

    with requests.Session() as session:
        health = _get_json(session, "GET", f"{base_url}/health", timeout=args.timeout)
        provider = str(health.get("llm_provider", "unknown"))
        model = str(health.get("llm_model", "unknown"))
        print(
            f"health: status={health.get('status')} "
            f"chunks={health.get('chunks')} provider={provider} model={model}"
        )

        if args.require_live and provider != "OpenAICompatibleLLMClient":
            print(
                "FAIL: server is not using OpenAICompatibleLLMClient. "
                "Restart it with OPENAI_API_KEY set in the server process."
            )
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
            probe_errors = _validate_live_probe(session, base_url, args.timeout)
            if probe_errors:
                print("FAIL: live LLM probe did not produce a grounded in-scope answer")
                for error in probe_errors:
                    print(f"  - {error}")
                return 1

        failures = 0
        for item in questions:
            question_id = str(item["id"])
            payload = _get_json(
                session,
                "POST",
                f"{base_url}/ask",
                timeout=args.timeout,
                json_body={"question": str(item["question"])},
            )
            errors = _validate_payload(question_id, payload)
            if errors:
                failures += 1
                print(f"{question_id}: FAIL")
                for error in errors:
                    print(f"  - {error}")
                continue

            sections = _source_sections(payload)
            answer = str(payload["answer"]).replace("\n", " ")
            preview = answer[:160] + ("..." if len(answer) > 160 else "")
            print(
                f"{question_id}: PASS {_outcome_label(payload)} confidence={payload['confidence']} "
                f"fallback={payload['fallback_reason']} sections={sections}"
            )
            print(f"  answer: {preview}")

    passed = len(questions) - failures
    print(f"summary: evaluated={len(questions)} passed={passed} failed={failures}")
    return 1 if failures else 0


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")


def _load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("questions file must contain a JSON array")
    questions: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise TypeError("each question item must be a JSON object")
        if "id" not in item or "question" not in item:
            raise KeyError("each question item must contain id and question")
        questions.append(cast(dict[str, Any], item))
    return questions


def _get_json(
    session: requests.Session,
    method: str,
    url: str,
    timeout: float,
    json_body: dict[str, str] | None = None,
) -> dict[str, Any]:
    response = session.request(method, url, json=json_body, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError(f"{url} did not return a JSON object")
    return cast(dict[str, Any], payload)


def _validate_payload(question_id: str, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(payload) != REQUIRED_RESPONSE_KEYS:
        errors.append(f"unexpected response keys: {sorted(payload)}")
        return errors

    if not isinstance(payload["answer"], str) or not payload["answer"].strip():
        errors.append("answer must be a non-empty string")
    if payload["confidence"] not in {"high", "medium", "low", "fallback"}:
        errors.append(f"unexpected confidence: {payload['confidence']}")
    if not isinstance(payload["trace_id"], str) or not payload["trace_id"]:
        errors.append("trace_id must be non-empty")
    if not isinstance(payload["latency_ms"], int) or payload["latency_ms"] < 0:
        errors.append("latency_ms must be a non-negative integer")
    if not isinstance(payload["sources"], list):
        errors.append("sources must be a list")
        return errors

    expected = EXPECTED.get(question_id)
    if expected is None:
        errors.append(
            f"no expected checks configured for {question_id!r}; "
            "add this case to EXPECTED or remove it from the questions file"
        )
        return errors

    if payload["fallback_reason"] != expected["fallback_reason"]:
        errors.append(
            f"fallback_reason expected {expected['fallback_reason']!r}, "
            f"got {payload['fallback_reason']!r}"
        )
    sections = _source_sections(payload)
    if expected["section"] is None:
        if sections:
            errors.append(f"expected no response sources, got {sections!r}")
    elif expected["section"] not in sections:
        errors.append(f"expected source section {expected['section']!r}, got {sections!r}")
    answer_lower = str(payload["answer"]).lower()
    missing_terms = [term for term in expected["answer_terms"] if term.lower() not in answer_lower]
    if missing_terms:
        errors.append(f"answer missing expected terms: {missing_terms}")
    return errors


def _validate_live_probe(
    session: requests.Session,
    base_url: str,
    timeout: float,
) -> list[str]:
    payload = _get_json(
        session,
        "POST",
        f"{base_url}/ask",
        timeout=timeout,
        json_body={"question": str(LIVE_PROBE["question"])},
    )
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
        term for term in LIVE_PROBE["answer_terms"] if str(term).lower() not in answer_lower
    ]
    if missing_terms:
        errors.append(f"probe answer missing expected terms: {missing_terms}")
    return errors


def _local_expected_model(base_url: str) -> str | None:
    hostname = urlparse(base_url).hostname
    if hostname not in {"127.0.0.1", "localhost"}:
        return None
    return Settings().openai_model


def _source_sections(payload: dict[str, Any]) -> list[str]:
    sections: list[str] = []
    for source in payload["sources"]:
        if isinstance(source, dict) and isinstance(source.get("section"), str):
            sections.append(source["section"])
    return sections


def _outcome_label(payload: dict[str, Any]) -> str:
    return "expected_fallback" if payload["fallback_reason"] else "grounded_answer"


if __name__ == "__main__":
    raise SystemExit(main())