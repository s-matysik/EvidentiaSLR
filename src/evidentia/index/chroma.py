"""Chroma backend.

Included for coverage of what practitioners actually reach for. Chroma
exposes fewer index knobs than Qdrant or FAISS, which is itself a finding
worth recording: a user cannot tune, or even inspect, the parameters that
determine whether their evidence set is reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..determinism import DEFAULT_SCORE_PRECISION
from ..exceptions import PipelineStateError
from .base import BaseIndex, require_module


class ChromaIndex(BaseIndex):
    exact = False

    def __init__(
        self,
        collection: str = "evidentia",
        persist_directory: str | None = None,
        score_precision: int = DEFAULT_SCORE_PRECISION,
        seed: int = 42,
    ):
        require_module("chromadb", "chroma")
        super().__init__(seed=seed)
        self.collection = collection
        self.persist_directory = persist_directory
        self.score_precision = score_precision
        self._collection = None

    @property
    def spec(self) -> str:
        return f"chroma/v1(hnsw-default,seed={self.seed})"

    def build(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        import chromadb

        matrix = self._register(vectors, ids)
        client = (
            chromadb.PersistentClient(path=self.persist_directory)
            if self.persist_directory
            else chromadb.EphemeralClient()
        )
        try:
            client.delete_collection(self.collection)
        except Exception:
            pass
        self._collection = client.create_collection(
            name=self.collection, metadata={"hnsw:space": "cosine"}
        )
        self._collection.add(
            ids=list(self.ids),
            embeddings=[matrix[i].tolist() for i in range(len(self.ids))],
        )

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._collection is None:
            raise PipelineStateError("index not built")
        result = self._collection.query(
            query_embeddings=[np.asarray(query, dtype=np.float64).reshape(-1).tolist()],
            n_results=k,
        )
        pairs = [
            (chunk_id, round(1.0 - float(distance), self.score_precision))
            for chunk_id, distance in zip(result["ids"][0], result["distances"][0], strict=True)
        ]
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        return pairs
