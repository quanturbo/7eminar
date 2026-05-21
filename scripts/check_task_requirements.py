from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402

DEFAULT_TRACE_FILE = ROOT / "traces.jsonl"

REQUIRED_RESPONSE_KEYS = {
    "answer",
    "sources",
    "confidence",
    "fallback_reason",
    "trace_id",
    "latency_ms",
}
REQUIRED_TRACE_KEYS = {
    "trace_id",
    "question",
    "stages",
    "retrieved",
    "final_response",
    "confidence",
    "fallback_reason",
    "latency_ms",
    "errors",
}
LIVE_PROBE = {
    "question": "Чи може працівник взяти щорічну відпустку після 3 місяців роботи?",
    "section": "1. Щорічна відпустка",
    "answer_terms": ["6 місяців"],
}

CASES: list[dict[str, Any]] = [
    {
        "id": "core_leave_after_3_months",
        "question": (
            "Я працюю в компанії 3 місяці. Чи можу вже піти у щорічну "
            "оплачувану відпустку?"
        ),
        "section": "1. Щорічна відпустка",
        "fallback_reason": None,
        "answer_terms": ["6 місяців"],
    },
    {
        "id": "core_sick_leave_certificate",
        "question": (
            "Працівник захворів, але ще не надав медичний документ. "
            "Чи можна одразу оплатити лікарняний?"
        ),
        "section": "2. Sick leave policy",
        "fallback_reason": None,
        "answer_terms": ["медич"],
    },
    {
        "id": "core_salary_indexation_base_month",
        "question": (
            "У нас є payroll-система. Чи можемо ми автоматично індексувати "
            "зарплату всім працівникам однаково?"
        ),
        "section": "3. Індексація зарплати / Salary indexation",
        "fallback_reason": None,
        "answer_terms": ["базов"],
    },
    {
        "id": "no_exact_esv_date",
        "question": "Яка точна дата сплати ЄСВ у цьому місяці?",
        "section": "4. Податкові строки",
        "fallback_reason": "missing_exact_date",
        "answer_terms": ["недостатньо"],
        "forbidden_patterns": [r"\b20\d{2}\b", r"\b\d{1,2}\.\d{1,2}"],
    },
    {
        "id": "no_indexation_calculation_without_inputs",
        "question": (
            "Порахуй індексацію для працівника із зарплатою 25000 грн, "
            "якщо базовий місяць невідомий."
        ),
        "section": "3. Індексація зарплати / Salary indexation",
        "fallback_reason": "missing_calculation_inputs",
        "answer_terms": ["розрах"],
    },
    {
        "id": "no_early_leave_exceptions",
        "question": "Які винятки дозволяють піти у щорічну відпустку до 6 місяців?",
        "section": "1. Щорічна відпустка",
        "fallback_reason": "context_says_not_enough_data",
        "answer_terms": ["недостатньо"],
    },
    {
        "id": "no_probation_max_duration",
        "question": "Яка максимальна тривалість випробувального строку?",
        "section": "6. Випробувальний строк",
        "fallback_reason": "context_says_not_enough_data",
        "answer_terms": ["недостатньо"],
        "forbidden_patterns": [r"\b\d+\s*(дн|місяц|тижн)"],
    },
    {
        "id": "no_resignation_notice_period",
        "question": "Скільки днів треба попередити роботодавця про звільнення за власним бажанням?",
        "section": "8. Звільнення за власним бажанням",
        "fallback_reason": "context_says_not_enough_data",
        "answer_terms": ["недостатньо"],
        "forbidden_patterns": [r"\b\d+\s*(дн|тижн|місяц)"],
    },
    {
        "id": "no_remote_work_compensation_amount",
        "question": "Скільки компанія компенсує за інтернет при дистанційній роботі?",
        "section": "7. Remote work policy",
        "fallback_reason": "context_says_not_enough_data",
        "answer_terms": ["недостатньо"],
    },
    {
        "id": "english_context_answers_in_ukrainian",
        "question": (
            "Чи скасовує дистанційна робота обов'язок виконувати внутрішні "
            "політики компанії?"
        ),
        "section": "7. Remote work policy",
        "fallback_reason": None,
        "answer_terms": ["не", "політик"],
    },
    {
        "id": "out_of_scope_question_fallback",
        "question": "Як скачати ігру онлайн?",
        "section": None,
        "fallback_reason": "no_relevant_context",
        "answer_terms": ["релевантний контекст"],
    },
]

CYRILLIC_RE = re.compile(r"[А-Яа-яІіЇїЄєҐґ]")


def main() -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="Run task-requirement acceptance checks against a local /ask endpoint."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--trace-file", type=Path, default=DEFAULT_TRACE_FILE)
    parser.add_argument("--timeout", type=float, default=25.0)
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

    base_url = args.base_url.rstrip("/")
    failures = 0

    with requests.Session() as session:
        health = _get_json(session, "GET", f"{base_url}/health", timeout=args.timeout)
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
            probe_errors = _validate_live_probe(session, base_url, args.timeout)
            if probe_errors:
                print("FAIL: live LLM probe did not produce a grounded in-scope answer")
                for error in probe_errors:
                    print(f"  - {error}")
                return 1

        for case in CASES:
            payload = _get_json(
                session,
                "POST",
                f"{base_url}/ask",
                timeout=args.timeout,
                json_body={"question": str(case["question"])},
            )
            trace = _find_trace(args.trace_file, str(payload.get("trace_id")))
            errors = _validate_case(case, payload, trace)
            if errors:
                failures += 1
                print(f"{case['id']}: FAIL")
                for error in errors:
                    print(f"  - {error}")
                continue
            print(
                f"{case['id']}: PASS {_outcome_label(payload)} "
                f"confidence={payload['confidence']} "
                f"fallback={payload['fallback_reason']}"
            )

    passed = len(CASES) - failures
    print(f"summary: evaluated={len(CASES)} passed={passed} failed={failures}")
    return 1 if failures else 0


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")


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


def _validate_case(
    case: dict[str, Any],
    payload: dict[str, Any],
    trace: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    if set(payload) != REQUIRED_RESPONSE_KEYS:
        errors.append(f"unexpected response keys: {sorted(payload)}")
        return errors

    answer = str(payload["answer"])
    answer_lower = answer.lower()
    sections = _source_sections(payload)

    if not answer.strip():
        errors.append("answer is empty")
    if not CYRILLIC_RE.search(answer):
        errors.append("answer is not Ukrainian/Cyrillic text")
    if payload["confidence"] not in {"high", "medium", "low", "fallback"}:
        errors.append(f"unexpected confidence: {payload['confidence']}")
    if not isinstance(payload["latency_ms"], int) or payload["latency_ms"] < 0:
        errors.append("latency_ms must be a non-negative integer")
    if not isinstance(payload["trace_id"], str) or not payload["trace_id"]:
        errors.append("trace_id must be non-empty")
    if payload["fallback_reason"] != case["fallback_reason"]:
        errors.append(
            f"fallback_reason expected {case['fallback_reason']!r}, "
            f"got {payload['fallback_reason']!r}"
        )
    if case["section"] is None:
        if sections:
            errors.append(f"expected no response sources, got {sections!r}")
    elif case["section"] not in sections:
        errors.append(f"expected source section {case['section']!r}, got {sections!r}")
    missing_terms = [
        term for term in case.get("answer_terms", []) if str(term).lower() not in answer_lower
    ]
    if missing_terms:
        errors.append(f"answer missing expected terms: {missing_terms}")
    for pattern in case.get("forbidden_patterns", []):
        if re.search(str(pattern), answer_lower):
            errors.append(f"answer matched forbidden hallucination pattern: {pattern}")

    if trace is None:
        errors.append("trace_id was not found in trace file")
    else:
        errors.extend(_validate_trace(case, payload, trace))

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


def _validate_trace(
    case: dict[str, Any],
    payload: dict[str, Any],
    trace: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not REQUIRED_TRACE_KEYS.issubset(trace):
        errors.append(f"trace missing keys: {sorted(REQUIRED_TRACE_KEYS - set(trace))}")
        return errors
    if trace["trace_id"] != payload["trace_id"]:
        errors.append("trace_id mismatch between response and trace")
    if trace["question"] != case["question"]:
        errors.append("trace question mismatch")
    if trace["final_response"].get("trace_id") != payload["trace_id"]:
        errors.append("trace final_response does not contain response trace_id")
    if trace["fallback_reason"] != payload["fallback_reason"]:
        errors.append("trace fallback_reason mismatch")
    stage_names = [stage.get("name") for stage in trace["stages"] if isinstance(stage, dict)]
    required_stages = [
        "load_context",
        "normalize_query",
        "retrieve_context",
        "grade_context",
        "build_response",
    ]
    for required_stage in required_stages:
        if required_stage not in stage_names:
            errors.append(f"trace missing stage {required_stage}")
    if payload["fallback_reason"] is None and "generate_answer" not in stage_names:
        errors.append("answerable response trace missing generate_answer stage")
    if not isinstance(trace["retrieved"], list):
        errors.append("trace retrieved must be a list")
    return errors


def _source_sections(payload: dict[str, Any]) -> list[str]:
    sections: list[str] = []
    for source in payload["sources"]:
        if isinstance(source, dict) and isinstance(source.get("section"), str):
            sections.append(source["section"])
    return sections


def _outcome_label(payload: dict[str, Any]) -> str:
    return "expected_fallback" if payload["fallback_reason"] else "grounded_answer"


def _find_trace(path: Path, trace_id: str) -> dict[str, Any] | None:
    if not path.exists() or not trace_id:
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if trace_id not in line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and payload.get("trace_id") == trace_id:
            return cast(dict[str, Any], payload)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
