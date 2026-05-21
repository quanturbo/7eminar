# Controlled AI Consultant Pipeline

This repository implements the test task as a controlled hybrid RAG API service. It is not a plain prompt wrapper: the service parses the knowledge base into chunks, retrieves only relevant context, routes the request through a LangGraph pipeline, validates whether the context is sufficient, returns structured JSON, and writes a trace for every request.

## Run Locally

Install the project into the local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
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

## Evaluation

Run deterministic unit/integration tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run the provided `test_questions.json` evaluation directly:

```powershell
.\.venv\Scripts\python.exe scripts\run_eval.py
```

Or against a running API:

```powershell
.\.venv\Scripts\python.exe scripts\run_eval.py --base-url http://127.0.0.1:8000
```

The script writes `evaluation_report.md`.

For a simple `requests`-based endpoint smoke test that reads `test_questions.json`:

```powershell
.\.venv\Scripts\python.exe scripts\check_endpoint_requests.py --base-url http://127.0.0.1:8000
```

To prove the running server is using the live OpenAI-compatible adapter rather than the deterministic offline adapter, and that the configured model can generate a grounded in-scope answer:

```powershell
.\.venv\Scripts\python.exe scripts\check_endpoint_requests.py --base-url http://127.0.0.1:8000 --require-live --expected-model openai/gpt-4o-mini
```

Run the broader task audit that checks additional missing-fact, missing-exception, missing-duration, missing-compensation, English-context, response-schema, and trace requirements:

```powershell
.\.venv\Scripts\python.exe scripts\check_task_requirements.py --base-url http://127.0.0.1:8000 --require-live --expected-model openai/gpt-4o-mini
```

If `--require-live` fails with `provider=FakeLLMClient`, stop the server, set `OPENAI_API_KEY` in the same terminal process, and start Uvicorn again. `/health` should then report `OpenAICompatibleLLMClient`.

For final real-provider validation, the expected result is:

```text
health: status=ok chunks=46 provider=OpenAICompatibleLLMClient
summary: evaluated=5 passed=5 failed=0
```

The broader task audit should report:

```text
health: status=ok chunks=46 provider=OpenAICompatibleLLMClient
summary: evaluated=11 passed=11 failed=0
```

## Laravel + Python AI Layer Integration

Laravel should own authentication, authorization, product API validation, user/session persistence, and billing/audit concerns. Laravel calls this Python AI service over HTTP or through a queue worker, stores the returned `trace_id`, answer, sources, confidence, and fallback reason, and exposes the result to the product UI. The Python layer owns LangGraph orchestration, knowledge-base indexing, retrieval, LLM calls, guardrails, trace logging, and evaluation.

## 1-2 Week Roadmap

- Add Docker Compose and CI for `pytest`, `ruff`, and `mypy`.
- Persist a FAISS index and embedding cache keyed by KB content hash.
- Add reranking and a larger labeled eval dataset with paraphrases and prompt-injection tests.
- Add trace rotation, request auth, rate limiting, and production metrics.
- Add optional LangSmith tracing and dashboard alerts for fallback rate and latency.