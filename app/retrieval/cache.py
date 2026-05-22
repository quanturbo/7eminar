from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np
from numpy.typing import NDArray

from app.domain.schemas import Chunk

CACHE_VERSION = 1
INDEX_TYPE = "IndexFlatIP"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalCacheMetadata:
    cache_version: int
    cache_key: str
    content_hash: str
    embedding_model_name: str
    embedding_dimensions: int
    chunk_count: int
    index_type: str = INDEX_TYPE

    def to_dict(self) -> dict[str, str | int]:
        return {
            "cache_version": self.cache_version,
            "cache_key": self.cache_key,
            "content_hash": self.content_hash,
            "embedding_model_name": self.embedding_model_name,
            "embedding_dimensions": self.embedding_dimensions,
            "chunk_count": self.chunk_count,
            "index_type": self.index_type,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RetrievalCacheMetadata:
        return cls(
            cache_version=int(payload["cache_version"]),
            cache_key=str(payload["cache_key"]),
            content_hash=str(payload["content_hash"]),
            embedding_model_name=str(payload["embedding_model_name"]),
            embedding_dimensions=int(payload["embedding_dimensions"]),
            chunk_count=int(payload["chunk_count"]),
            index_type=str(payload["index_type"]),
        )


@dataclass(frozen=True)
class RetrievalCacheArtifacts:
    index: Any
    embeddings: NDArray[np.float32]
    metadata: RetrievalCacheMetadata
    status: str


class RetrievalIndexCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def expected_metadata(
        self,
        chunks: list[Chunk],
        embedding_model_name: str,
        embedding_dimensions: int,
    ) -> RetrievalCacheMetadata:
        content_hash = chunk_content_hash(chunks)
        cache_key = _cache_key(
            content_hash=content_hash,
            embedding_model_name=embedding_model_name,
            embedding_dimensions=embedding_dimensions,
        )
        return RetrievalCacheMetadata(
            cache_version=CACHE_VERSION,
            cache_key=cache_key,
            content_hash=content_hash,
            embedding_model_name=embedding_model_name,
            embedding_dimensions=embedding_dimensions,
            chunk_count=len(chunks),
        )

    def load(
        self,
        chunks: list[Chunk],
        embedding_model_name: str,
        embedding_dimensions: int,
    ) -> RetrievalCacheArtifacts | None:
        expected = self.expected_metadata(chunks, embedding_model_name, embedding_dimensions)
        metadata_path, embeddings_path, index_path = self._paths(expected.cache_key)
        if not metadata_path.exists() or not embeddings_path.exists():
            return None

        try:
            metadata = RetrievalCacheMetadata.from_dict(
                json.loads(metadata_path.read_text(encoding="utf-8"))
            )
            if metadata != expected:
                return None
            embeddings = _load_embeddings(embeddings_path, metadata)
            if index_path.exists():
                index = cast(Any, faiss.read_index(str(index_path)))
                if _index_matches(index, metadata):
                    return RetrievalCacheArtifacts(index, embeddings, metadata, "hit")
            index = _build_index(embeddings, metadata.embedding_dimensions)
            self._write_index(index_path, index)
            return RetrievalCacheArtifacts(index, embeddings, metadata, "embeddings_hit")
        except Exception as exc:
            LOGGER.warning("Ignoring invalid retrieval cache %s: %s", expected.cache_key, exc)
            return None

    def save(
        self,
        chunks: list[Chunk],
        embedding_model_name: str,
        embedding_dimensions: int,
        embeddings: NDArray[np.float32],
        index: Any,
    ) -> RetrievalCacheMetadata:
        metadata = self.expected_metadata(chunks, embedding_model_name, embedding_dimensions)
        metadata_path, embeddings_path, index_path = self._paths(metadata.cache_key)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._write_embeddings(embeddings_path, embeddings)
        self._write_index(index_path, index)
        self._write_metadata(metadata_path, metadata)
        return metadata

    def _paths(self, cache_key: str) -> tuple[Path, Path, Path]:
        return (
            self.cache_dir / f"{cache_key}.json",
            self.cache_dir / f"{cache_key}.embeddings.npy",
            self.cache_dir / f"{cache_key}.faiss",
        )

    @staticmethod
    def _write_embeddings(path: Path, embeddings: NDArray[np.float32]) -> None:
        tmp_path = path.with_name(f"{path.name}.tmp")
        with tmp_path.open("wb") as file:
            np.save(file, np.ascontiguousarray(embeddings, dtype=np.float32), allow_pickle=False)
        tmp_path.replace(path)

    @staticmethod
    def _write_index(path: Path, index: Any) -> None:
        tmp_path = path.with_name(f"{path.name}.tmp")
        faiss.write_index(index, str(tmp_path))
        tmp_path.replace(path)

    @staticmethod
    def _write_metadata(path: Path, metadata: RetrievalCacheMetadata) -> None:
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(
            json.dumps(metadata.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)


def chunk_content_hash(chunks: list[Chunk]) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"chunk-count:{len(chunks)}\n".encode())
    for chunk in chunks:
        hasher.update(
            _canonical_json(
                {
                    "chunk_id": chunk.chunk_id,
                    "language_hint": chunk.language_hint,
                    "section": chunk.section,
                    "source_path": chunk.source_path,
                    "text": chunk.text,
                }
            ).encode("utf-8")
        )
        hasher.update(b"\n")
    return hasher.hexdigest()


def _cache_key(
    content_hash: str,
    embedding_model_name: str,
    embedding_dimensions: int,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "cache_version": CACHE_VERSION,
                "content_hash": content_hash,
                "embedding_dimensions": embedding_dimensions,
                "embedding_model_name": embedding_model_name,
                "index_type": INDEX_TYPE,
            }
        ).encode("utf-8")
    ).hexdigest()


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_embeddings(
    path: Path,
    metadata: RetrievalCacheMetadata,
) -> NDArray[np.float32]:
    embeddings = np.load(path, allow_pickle=False)
    matrix: NDArray[np.float32] = np.ascontiguousarray(embeddings, dtype=np.float32)
    expected_shape = (metadata.chunk_count, metadata.embedding_dimensions)
    if matrix.shape != expected_shape:
        raise ValueError(f"cached embeddings shape {matrix.shape} != {expected_shape}")
    return matrix


def _build_index(embeddings: NDArray[np.float32], embedding_dimensions: int) -> Any:
    index = cast(Any, faiss.IndexFlatIP(embedding_dimensions))
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    return index


def _index_matches(index: Any, metadata: RetrievalCacheMetadata) -> bool:
    return (
        int(index.ntotal) == metadata.chunk_count
        and int(index.d) == metadata.embedding_dimensions
    )