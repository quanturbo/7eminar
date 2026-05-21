from __future__ import annotations

import re
from pathlib import Path

from app.domain.schemas import Chunk

SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")
CYRILLIC_RE = re.compile(r"[А-Яа-яІіЇїЄєҐґ]")
LATIN_RE = re.compile(r"[A-Za-z]")


def load_knowledge_base(path: str | Path) -> list[Chunk]:
    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8")
    sections = _split_sections(text)
    chunks: list[Chunk] = []

    for section_index, (section, body) in enumerate(sections, start=1):
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
        for paragraph_index, paragraph in enumerate(paragraphs, start=1):
            normalized = " ".join(paragraph.split())
            chunks.append(
                Chunk(
                    chunk_id=f"s{section_index:02d}-c{paragraph_index:02d}",
                    section=section,
                    text=normalized,
                    language_hint=_language_hint(normalized),
                    source_path=str(source_path),
                )
            )

    if not chunks:
        raise ValueError(f"No knowledge-base chunks found in {source_path}")
    return chunks


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        match = SECTION_RE.match(line)
        if match:
            if current_title is not None:
                sections.append((current_title, current_lines))
            current_title = match.group(1).strip()
            current_lines = []
            continue
        if current_title is not None:
            current_lines.append(line)

    if current_title is not None:
        sections.append((current_title, current_lines))

    return [(title, "\n".join(lines).strip()) for title, lines in sections if title.strip()]


def _language_hint(text: str) -> str:
    has_cyrillic = bool(CYRILLIC_RE.search(text))
    has_latin = bool(LATIN_RE.search(text))
    if has_cyrillic and has_latin:
        return "mixed"
    if has_cyrillic:
        return "uk"
    return "en"