from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


DEFAULT_STORAGE_DIR = Path("storage")


@dataclass
class SearchResult:
    text: str
    score: float
    source: str


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def save_index(
    embeddings: list[list[float]],
    chunks: list[str],
    source_name: str,
    storage_dir: Path = DEFAULT_STORAGE_DIR,
) -> None:
    storage_dir.mkdir(parents=True, exist_ok=True)

    vectors = np.array(embeddings, dtype="float32")
    vectors = _normalize(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(storage_dir / "documents.faiss"))
    with (storage_dir / "chunks.pkl").open("wb") as f:
        pickle.dump(
            [{"text": chunk, "source": source_name} for chunk in chunks],
            f,
        )


def load_index(storage_dir: Path = DEFAULT_STORAGE_DIR):
    index_path = storage_dir / "documents.faiss"
    chunks_path = storage_dir / "chunks.pkl"

    if not index_path.exists() or not chunks_path.exists():
        raise RuntimeError("No index found. Run ingestion first.")

    index = faiss.read_index(str(index_path))
    with chunks_path.open("rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def search(
    query_embedding: list[float],
    top_k: int = 4,
    storage_dir: Path = DEFAULT_STORAGE_DIR,
) -> list[SearchResult]:
    index, chunks = load_index(storage_dir)

    query = np.array([query_embedding], dtype="float32")
    query = _normalize(query)

    scores, indices = index.search(query, top_k)
    results: list[SearchResult] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        results.append(
            SearchResult(
                text=chunk["text"],
                score=float(score),
                source=chunk["source"],
            )
        )

    return results

