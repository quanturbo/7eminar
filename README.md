# Controlled AI Consultant Pipeline

This repository implements the test task as a controlled hybrid RAG API service. It is not a plain prompt wrapper: the service parses the knowledge base into chunks, retrieves only relevant context, routes the request through a LangGraph pipeline, validates whether the context is sufficient, returns structured JSON, and writes a trace for every request.

## Run Locally

Prerequisite: Python 3.11-3.13.

From a fresh Windows clone, create the local virtual environment and install the project:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

If the Windows `py` launcher is unavailable, or if you need to choose a specific installed version, use any Python 3.11-3.13 executable for the first command:

```powershell
python -m venv .venv
```

macOS/Linux equivalent:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Provider Modes

The service supports two provider modes. Use `/health` or the endpoint checker to verify which one is active.

### Fake Provider

Use fake mode when you want deterministic local tests without spending tokens or requiring a secret. Leave `OPENAI_API_KEY` empty or unset, then start the server.

Expected health provider:

```text
llm_provider: FakeLLMClient
```

Fake mode is useful for unit tests, offline evaluation, and CI. It does not call OpenRouter, so no activity appears in the OpenRouter dashboard.

### Real OpenRouter Provider

Use real mode for final end-to-end validation. Copy the template and put the real key only in `.env`:

```powershell
Copy-Item .env.example .env
notepad .env
```

Required `.env` values:

```dotenv
OPENAI_API_KEY=
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-4o-mini
```

Then restart Uvicorn. The key and model are read at server startup, so changing `.env` after startup is not enough. If you use a custom OpenRouter model, verify that the model id is valid in OpenRouter before final testing.

Expected health provider:

```text
llm_provider: OpenAICompatibleLLMClient
llm_model: openai/gpt-4o-mini
```

Never commit `.env` or paste a real key into `.env.example`. If `/health` still shows an old model after editing `.env`, stop the running Uvicorn process and start it again; settings are loaded into memory at process startup.

## Endpoint

`POST http://127.0.0.1:8000/ask`

Request:

```json
{
  "question": "Чи може працівник взяти щорічну відпустку після 3 місяців роботи?"
}
```

Response:

```json
{
  "answer": "За наданою базою знань, щорічну оплачувану відпустку можна використати після 6 місяців безперервної роботи у компанії...",
  "sources": [
    {
      "section": "1. Щорічна відпустка",
      "chunk": "Працівник може використати щорічну оплачувану відпустку після 6 місяців безперервної роботи у компанії.",
      "score": 1.0
    }
  ],
  "confidence": "high",
  "fallback_reason": null,
  "trace_id": "generated-uuid",
  "latency_ms": 20
}
```

## Examples for `test_questions.json`

The repository includes five bundled test questions. `trace_id`, `latency_ms`, and exact retrieval scores are generated at runtime, so the values below are representative examples.

### q001

Request:

```json
{"question": "Я працюю в компанії 3 місяці. Чи можу вже піти у щорічну оплачувану відпустку?"}
```

Response:

```json
{
  "answer": "За наданою базою знань, щорічну оплачувану відпустку можна використати після 6 місяців безперервної роботи у компанії. Тому після 3 місяців таке право в наданому контексті не підтверджене.",
  "sources": [
    {
      "section": "1. Щорічна відпустка",
      "chunk": "Працівник може використати щорічну оплачувану відпустку після 6 місяців безперервної роботи у компанії.",
      "score": 1.0
    }
  ],
  "confidence": "high",
  "fallback_reason": null,
  "trace_id": "generated-uuid",
  "latency_ms": 20
}
```

### q002

Request:

```json
{"question": "Працівник захворів, але ще не надав медичний документ. Чи можна одразу оплатити лікарняний?"}
```

Response:

```json
{
  "answer": "Ні. Лікарняний має бути підтверджений офіційним медичним документом, а за його відсутності відсутність не може автоматично вважатися оплачуваним лікарняним.",
  "sources": [
    {
      "section": "2. Sick leave policy",
      "chunk": "Sick leave must be confirmed by an official medical certificate.",
      "score": 1.0
    }
  ],
  "confidence": "high",
  "fallback_reason": null,
  "trace_id": "generated-uuid",
  "latency_ms": 20
}
```

### q003

Request:

```json
{"question": "У нас є payroll-система. Чи можемо ми автоматично індексувати зарплату всім працівникам однаково?"}
```

Response:

```json
{
  "answer": "Ні. Індексацію зарплати не слід застосовувати автоматично однаково для всіх: потрібно перевіряти базовий місяць індексації, який має зберігатися окремо для кожного працівника.",
  "sources": [
    {
      "section": "3. Індексація зарплати / Salary indexation",
      "chunk": "For payroll systems, the base month must be stored separately for each employee.",
      "score": 1.0
    }
  ],
  "confidence": "high",
  "fallback_reason": null,
  "trace_id": "generated-uuid",
  "latency_ms": 20
}
```

### q004

Request:

```json
{"question": "Яка точна дата сплати ЄСВ у цьому місяці?"}
```

Response:

```json
{
  "answer": "У наданій базі знань недостатньо інформації для точної відповіді. Точна дата відсутня у джерелі, тому я не можу її вигадувати.",
  "sources": [
    {
      "section": "4. Податкові строки",
      "chunk": "У цій базі знань точні календарні дати сплати ЄСВ не наведені.",
      "score": 1.0
    }
  ],
  "confidence": "fallback",
  "fallback_reason": "missing_exact_date",
  "trace_id": "generated-uuid",
  "latency_ms": 20
}
```

### q005

Request:

```json
{"question": "Порахуй індексацію для працівника із зарплатою 25000 грн, якщо базовий місяць невідомий."}
```

Response:

```json
{
  "answer": "У наданій базі знань недостатньо інформації для точної відповіді. Для розрахунку бракує необхідних вхідних даних або числових правил.",
  "sources": [
    {
      "section": "3. Індексація зарплати / Salary indexation",
      "chunk": "Для розрахунку індексації потрібні додаткові дані: базовий місяць працівника, актуальний індекс споживчих цін та сума доходу, яка підлягає індексації.",
      "score": 1.0
    }
  ],
  "confidence": "fallback",
  "fallback_reason": "missing_calculation_inputs",
  "trace_id": "generated-uuid",
  "latency_ms": 20
}
```

## Retrieval Approach

`knowledge_base.md` is the only knowledge source. `app/knowledge/loader.py` splits it by Markdown sections and paragraph chunks, preserving section names and source paths. `app/retrieval/query.py` expands Ukrainian/English HR terms. `app/retrieval/hybrid.py` combines BM25Plus lexical search with a deterministic local hashed vector scorer, then merges rankings with weighted Reciprocal Rank Fusion. Every request goes through the generation step. Only the top retrieved chunks are passed to the model; for `no_relevant_context` fallback, no irrelevant chunks are passed. The whole KB is never sent to the model.

## Project Structure

Production ownership is separated by layer:

- `app/api/` contains FastAPI app construction and routes.
- `app/core/` contains settings, guardrails, and the LangGraph RAG pipeline.
- `app/domain/` contains the Pydantic request, response, retrieval, and trace contracts.
- `app/knowledge/` parses the Markdown knowledge base into logical chunks.
- `app/retrieval/` owns query normalization and hybrid retrieval.
- `app/generation/` owns the live OpenAI-compatible generation adapter.
- `app/observability/` owns JSONL trace writing.
- `app/testing/` contains deterministic test doubles used only when no API key is configured.

The legacy top-level modules such as `app/retriever.py` and `app/pipeline.py` are thin compatibility exports so older imports keep working while the real implementation lives in the layered packages.

The optional `vector` dependency group is reserved for FAISS-backed persistence when the corpus grows:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,vector]"
```

## Confidence

Confidence is service-owned, not model-owned. The context gate maps top retrieval score to `high`, `medium`, or `low`; any unsupported or insufficient request becomes `fallback`. Post-generation validation also requires cited source ids to be among retrieved chunks.

## Fallback Rules

Fallback returns HTTP 200 with the same JSON schema. It triggers when no relevant context is found, the KB explicitly lacks exact dates, the user asks for a calculation without required numeric inputs, the LLM returns invalid structure, or the answer cites unsupported sources.

Examples required by the task:

- q004 must not invent the exact ESВ payment date.
- q005 must not calculate salary indexation when the base month is unknown.
- Additional audit cases verify no-context fallback, missing exceptions, missing durations, missing compensation amounts, and Ukrainian answers for English source context.

## LLM API

The live client uses the OpenAI Python SDK against an OpenAI-compatible endpoint. `.env.example` defaults `OPENAI_BASE_URL` to the OpenRouter-compatible format and `OPENAI_MODEL` to an OpenAI-compatible model name. Secrets are read only from environment variables and are not logged.

When `--require-live` is used, the check scripts verify both `/health` and a real in-scope `/ask` probe. This catches invalid model IDs that would otherwise only fail when the service reaches LLM generation.

## Trace Logging

Every request appends one JSON object to `traces.jsonl` with `trace_id`, question, pipeline stages, retrieved chunks and scores, final response, confidence, fallback reason, latency, and errors. API keys and environment variables are never written to traces.

## Evaluation Report

The current run over all questions from `test_questions.json` is recorded in `evaluation_report.md`.

## Laravel + Python AI Layer Integration

Laravel should own authentication, authorization, product API validation, user/session persistence, and billing/audit concerns. Laravel calls this Python AI service over HTTP or through a queue worker, stores the returned `trace_id`, answer, sources, confidence, and fallback reason, and exposes the result to the product UI. The Python layer owns LangGraph orchestration, knowledge-base indexing, retrieval, LLM calls, guardrails, trace logging, and evaluation.

## 1-2 Week Roadmap

- Add Docker Compose and CI for `pytest`, `ruff`, and `mypy`.
- Persist a FAISS index and embedding cache keyed by KB content hash.
- Add reranking and a larger labeled eval dataset with paraphrases and prompt-injection tests.
- Add trace rotation, request auth, rate limiting, and production metrics.
- Add optional LangSmith tracing and dashboard alerts for fallback rate and latency.