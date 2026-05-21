from __future__ import annotations

import json
from typing import Any, cast

from openai import BadRequestError, OpenAI

from app.domain.schemas import LLMStructuredAnswer, RetrievalHit

SYSTEM_PROMPT = (
    "Ти контрольований AI-консультант для RAG-системи. Відповідай лише українською "
    "мовою і тільки на основі переданого контексту. Якщо контекст містить правило, "
    "яке дозволяє відповісти 'так' або 'ні', це answerable=true навіть коли фінальна "
    "відповідь негативна. answerable=false використовуй тільки коли потрібного факту "
    "немає в контексті. Не вигадуй точні дати, суми, формули або винятки. "
    "Поверни тільки JSON без Markdown."
)


class OpenAICompatibleLLMClient:
    def __init__(self, api_key: str, base_url: str | None, model: str) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        force_answerable_retry: bool = True,
    ) -> LLMStructuredAnswer:
        answer = self._generate_once(question, hits, force_answerable=False)
        if force_answerable_retry and hits and not answer.answerable:
            return self._generate_once(question, hits, force_answerable=True)
        return answer

    def _generate_once(
        self,
        question: str,
        hits: list[RetrievalHit],
        force_answerable: bool,
    ) -> LLMStructuredAnswer:
        context = "\n\n".join(
            f"[{hit.chunk.chunk_id}] {hit.chunk.section}: {hit.chunk.text}" for hit in hits
        )
        allowed_ids = ", ".join(hit.chunk.chunk_id for hit in hits)
        force_instruction = ""
        if force_answerable:
            force_instruction = (
                "\nВажливо: попередня відповідь помилково позначила достатній контекст "
                "як недостатній. Знайди правило у контексті і дай обґрунтовану "
                "відповідь з answerable=true. Негативна відповідь 'ні' не є fallback."
            )
        prompt = (
            "Поверни JSON об'єкт строго з ключами:\n"
            "- answerable: boolean\n"
            "- answer: string\n"
            "- used_source_ids: array of strings, тільки chunk id з дозволеного списку\n"
            "- missing_information: array of strings, порожній список якщо answerable=true\n"
            "- grounding_note: string\n\n"
            f"Дозволені chunk id: {allowed_ids}\n"
            f"{force_instruction}\n\n"
            f"Питання: {question}\n\nКонтекст:\n{context}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = self._create_completion(messages)
        content = response.choices[0].message.content or "{}"
        payload = _parse_json_payload(content)
        return LLMStructuredAnswer.model_validate(_normalize_payload(payload, hits))

    def _create_completion(self, messages: list[dict[str, str]]) -> Any:
        completions = cast(Any, self.client.chat.completions)
        try:
            return completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0,
            )
        except BadRequestError:
            return completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )


def _parse_json_payload(content: str) -> dict[str, Any]:
    stripped = content.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise TypeError("LLM response must be a JSON object.")
    return cast(dict[str, Any], payload)


def _normalize_payload(payload: dict[str, Any], hits: list[RetrievalHit]) -> dict[str, Any]:
    allowed_ids = {hit.chunk.chunk_id for hit in hits}
    used_ids = _as_string_list(payload.get("used_source_ids"))
    used_ids = [chunk_id for chunk_id in used_ids if chunk_id in allowed_ids]

    answerable = _as_bool(payload.get("answerable"))
    if answerable and not used_ids and hits:
        used_ids = [hits[0].chunk.chunk_id]

    return {
        "answerable": answerable,
        "answer": str(payload.get("answer", "")).strip(),
        "used_source_ids": used_ids,
        "missing_information": _as_string_list(payload.get("missing_information")),
        "grounding_note": str(payload.get("grounding_note", "")).strip(),
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "так", "yes", "1"}
    return False


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []