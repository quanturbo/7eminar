from __future__ import annotations

from app.domain.schemas import LLMStructuredAnswer, RetrievalHit


class FakeLLMClient:
    def generate(
        self,
        question: str,
        hits: list[RetrievalHit],
        force_answerable_retry: bool = True,
    ) -> LLMStructuredAnswer:
        if not hits:
            return LLMStructuredAnswer(
                answerable=False,
                answer="У наданій базі знань недостатньо інформації для точної відповіді.",
                used_source_ids=[],
                missing_information=["relevant context"],
                grounding_note="No hits.",
            )

        top_section = hits[0].chunk.section
        used_ids = [hits[0].chunk.chunk_id]
        lower_question = question.lower()

        if top_section.startswith("1."):
            answer = (
                "За наданою базою знань, щорічну оплачувану відпустку можна використати "
                "після 6 місяців безперервної роботи у компанії. Тому після 3 місяців "
                "таке право в наданому контексті не підтверджене. Джерело: секція 1."
            )
        elif top_section.startswith("2."):
            answer = (
                "Ні. Лікарняний має бути підтверджений офіційним медичним документом, "
                "а за його відсутності відсутність не може автоматично вважатися оплачуваним "
                "лікарняним. Джерело: секція 2."
            )
        elif top_section.startswith("3.") and "автомат" in lower_question:
            answer = (
                "Ні. Індексацію зарплати не слід застосовувати автоматично однаково для всіх: "
                "потрібно перевіряти базовий місяць індексації, який має зберігатися окремо "
                "для кожного працівника. Джерело: секція 3."
            )
        elif top_section.startswith("7."):
            answer = (
                "Дистанційна робота можлива за згодою працівника і роботодавця, але вона не "
                "скасовує обов'язок дотримуватися внутрішніх політик компанії. Джерело: секція 7."
            )
        else:
            answer = (
                "За наданою базою знань, відповідь має спиратися на знайдений контекст. "
                f"Найрелевантніше джерело: секція {top_section}."
            )

        return LLMStructuredAnswer(
            answerable=True,
            answer=answer,
            used_source_ids=used_ids,
            grounding_note=f"Grounded in {top_section}.",
        )