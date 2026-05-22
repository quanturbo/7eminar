# Evaluation Report

Mode: running HTTP API.
Health provider: OpenAICompatibleLLMClient.
Health model: openai/gpt-5.4-mini.
Retrieval cache: hit.
Retrieval cache key: aaf16050a2544310538034f3eeea46a83dd25c1b5a553edaf1c67edb42577f88.
Questions evaluated: 5.
Passed checks: 5/5.
Average latency: 1608.6 ms.

## Per-question results

### q001 - PASS

Question: Я працюю в компанії 3 місяці. Чи можу вже піти у щорічну оплачувану відпустку?
Confidence: high
Fallback reason: None
Retrieved sections: 1. Щорічна відпустка, 1. Щорічна відпустка, 1. Щорічна відпустка, 3. Індексація зарплати / Salary indexation
Answer: Ні, ще не можете: щорічну оплачувану відпустку можна використати після 6 місяців безперервної роботи в компанії.
Checks:
- response_schema_correct: yes
- expected_section_found: yes
- fallback_correct: yes
- answer_contains_expected_trait: yes
- no_empty_trace_id: yes

### q002 - PASS

Question: Працівник захворів, але ще не надав медичний документ. Чи можна одразу оплатити лікарняний?
Confidence: high
Fallback reason: None
Retrieved sections: 2. Sick leave policy, 2. Sick leave policy, 5. Документи при прийнятті на роботу, 2. Sick leave policy
Answer: Ні. Лікарняний має бути підтверджений офіційним медичним документом, а без нього відсутність не може автоматично вважатися оплачуваним лікарняним. Також перед оплатою потрібно зареєструвати запис про лікарняний.
Checks:
- response_schema_correct: yes
- expected_section_found: yes
- fallback_correct: yes
- answer_contains_expected_trait: yes
- no_empty_trace_id: yes

### q003 - PASS

Question: У нас є payroll-система. Чи можемо ми автоматично індексувати зарплату всім працівникам однаково?
Confidence: high
Fallback reason: None
Retrieved sections: 3. Індексація зарплати / Salary indexation, 3. Індексація зарплати / Salary indexation, 4. Податкові строки, 3. Індексація зарплати / Salary indexation
Answer: Ні, не слід автоматично індексувати зарплату всім працівникам однаково, оскільки для індексації потрібно окремо враховувати базовий місяць кожного працівника та перевіряти актуальну базу індексації.
Checks:
- response_schema_correct: yes
- expected_section_found: yes
- fallback_correct: yes
- answer_contains_expected_trait: yes
- no_empty_trace_id: yes

### q004 - PASS

Question: Яка точна дата сплати ЄСВ у цьому місяці?
Confidence: fallback
Fallback reason: missing_exact_date
Retrieved sections: 4. Податкові строки, 4. Податкові строки, 4. Податкові строки, 4. Податкові строки
Answer: У наданій базі знань недостатньо інформації для точної відповіді. Точна дата відсутня у джерелі, тому я не можу її вигадувати.
Checks:
- response_schema_correct: yes
- expected_section_found: yes
- fallback_correct: yes
- answer_contains_expected_trait: yes
- no_empty_trace_id: yes

### q005 - PASS

Question: Порахуй індексацію для працівника із зарплатою 25000 грн, якщо базовий місяць невідомий.
Confidence: fallback
Fallback reason: missing_calculation_inputs
Retrieved sections: 3. Індексація зарплати / Salary indexation, 3. Індексація зарплати / Salary indexation, 4. Податкові строки, 3. Індексація зарплати / Salary indexation
Answer: У наданій базі знань недостатньо інформації для точної відповіді. Для розрахунку бракує необхідних вхідних даних або числових правил.
Checks:
- response_schema_correct: yes
- expected_section_found: yes
- fallback_correct: yes
- answer_contains_expected_trait: yes
- no_empty_trace_id: yes

## Hallucination and fallback review

Grounded-answer cases: q001, q002, q003. Fallback cases: q004, q005. The fallback checks verify that the service does not invent exact dates, numeric calculations, or unsupported facts when the knowledge base does not contain enough information. The deterministic offline adapter lets these checks run without network access or API keys when direct mode is used.

## Executable before/after retrieval audit

Before is a BM25-only baseline run inside this script with the same question text and no synonym dictionary. After is the current FAISS+BM25 hybrid pipeline output from this eval run.

| Case | Expected | BM25-only before | Before ok | Hybrid after | After ok |
| --- | --- | --- | --- | --- | --- |
| q001 | 1. Щорічна відпустка | 1. Щорічна відпустка, 3. Індексація зарплати / Salary indexation | yes | 1. Щорічна відпустка, 3. Індексація зарплати / Salary indexation | yes |
| q002 | 2. Sick leave policy | 5. Документи при прийнятті на роботу | no | 2. Sick leave policy, 5. Документи при прийнятті на роботу | yes |
| q003 | 3. Індексація зарплати / Salary indexation | 3. Індексація зарплати / Salary indexation | yes | 3. Індексація зарплати / Salary indexation, 4. Податкові строки | yes |
| q004 | 4. Податкові строки | 4. Податкові строки, 3. Індексація зарплати / Salary indexation | yes | 4. Податкові строки | yes |
| q005 | 3. Індексація зарплати / Salary indexation | 3. Індексація зарплати / Salary indexation | yes | 3. Індексація зарплати / Salary indexation, 4. Податкові строки | yes |

## Current implementation audit

- Query normalization is whitespace-only; tests assert `additions` remains empty.
- Vector search uses a FAISS `IndexFlatIP` built from FastEmbed multilingual vectors.
- The legacy hashed character-ngram fallback and transliteration table are absent.
- BM25 remains as the lexical side of hybrid retrieval with configurable fusion weights.
- FAISS index and normalized embeddings are cached on disk by KB content hash and embedding configuration.

## Remaining weak spots

- The bundled fake LLM is deterministic for testability; live model quality still depends on the configured OpenAI-compatible model.
- The retrieval cache is local filesystem storage; production deployments should mount `RETRIEVAL_CACHE_DIR` on persistent storage and prewarm it before switching traffic.
- The first run may download the configured FastEmbed model into the local cache; production deployments should prewarm that artifact.
- The eval set has only 5 questions; production should add adversarial, paraphrase, and prompt-injection cases.

## Production improvements

Add Docker/CI, trace rotation, API authentication, a larger labeled eval set, optional LangSmith tracing, a reranker, and monitoring for fallback rate and latency.
