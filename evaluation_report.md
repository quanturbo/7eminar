# Evaluation Report

Mode: direct pipeline with FakeLLMClient.
Health provider: direct FakeLLMClient.
Health model: offline-fake.
Questions evaluated: 5.
Passed checks: 5/5.
Average latency: 7.8 ms.

## Per-question results

### q001 - PASS

Question: Я працюю в компанії 3 місяці. Чи можу вже піти у щорічну оплачувану відпустку?
Confidence: high
Fallback reason: None
Retrieved sections: 1. Щорічна відпустка, 1. Щорічна відпустка, 1. Щорічна відпустка, 1. Щорічна відпустка
Answer: За наданою базою знань, щорічну оплачувану відпустку можна використати після 6 місяців безперервної роботи у компанії. Тому після 3 місяців таке право в наданому контексті не підтверджене. Джерело: секція 1.
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
Retrieved sections: 2. Sick leave policy, 2. Sick leave policy, 2. Sick leave policy, 2. Sick leave policy
Answer: Ні. Лікарняний має бути підтверджений офіційним медичним документом, а за його відсутності відсутність не може автоматично вважатися оплачуваним лікарняним. Джерело: секція 2.
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
Retrieved sections: 3. Індексація зарплати / Salary indexation, 3. Індексація зарплати / Salary indexation, 3. Індексація зарплати / Salary indexation, 3. Індексація зарплати / Salary indexation
Answer: Ні. Індексацію зарплати не слід застосовувати автоматично однаково для всіх: потрібно перевіряти базовий місяць індексації, який має зберігатися окремо для кожного працівника. Джерело: секція 3.
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
Retrieved sections: 4. Податкові строки, 4. Податкові строки, 4. Податкові строки, 10. Правила відповіді AI-консультанта
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
Retrieved sections: 3. Індексація зарплати / Salary indexation, 3. Індексація зарплати / Salary indexation, 3. Індексація зарплати / Salary indexation, 3. Індексація зарплати / Salary indexation
Answer: У наданій базі знань недостатньо інформації для точної відповіді. Для розрахунку бракує необхідних вхідних даних або числових правил.
Checks:
- response_schema_correct: yes
- expected_section_found: yes
- fallback_correct: yes
- answer_contains_expected_trait: yes
- no_empty_trace_id: yes

## Hallucination and fallback review

Grounded-answer cases: q001, q002, q003. Fallback cases: q004, q005. The fallback checks verify that the service does not invent exact dates, numeric calculations, or unsupported facts when the knowledge base does not contain enough information. The deterministic offline adapter lets these checks run without network access or API keys when direct mode is used.

## Remaining weak spots

- The bundled fake LLM is deterministic for testability; live model quality still depends on the configured OpenAI-compatible model.
- The local vector scorer is deterministic and lightweight. Installing the optional `vector` extra enables future FAISS-backed persistence for larger corpora.
- The eval set has only 5 questions; production should add adversarial, paraphrase, and prompt-injection cases.

## Production improvements

Add Docker/CI, trace rotation, API authentication, a larger labeled eval set, optional LangSmith tracing, persisted FAISS indexes, a reranker, and monitoring for fallback rate and latency.
