from __future__ import annotations

import json
from pathlib import Path

from app.domain.schemas import TraceRecord


class TraceLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: TraceRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
