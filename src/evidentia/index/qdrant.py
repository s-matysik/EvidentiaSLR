"""Qdrant backend.

HNSW under the hood, so approximate and rebuild-dependent in the same way as
the in-process FAISS backend. `m` and `ef_construct` are exposed rather than
left at library defaults, because those are exactly the variables the
stability experiments manipulate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..determinism import DEFAULT_SCORE_PRECISION
from ..exceptions import PipelineStateError
from .base import BaseIndex, require_module


class QdrantIndex(BaseIndex):
    exact = False

    def __init__(
        self,
        collection: str = "evidentia",
        url: str = "http://localhost:6333",
        m: int = 16,
        ef_construct: int = 100,
        hnsw_ef: int = 64,
        score_precision: int = DEFAULT_SCORE_PRECISION,
        seed: int = 42,
    ):
        require_module("qdrant_client", "qdrant")
        super().__init__(seed=seed)
        self.collection = collection
        self.url = url
        self.m = m
        self.ef_construct = ef_construct
        self.hnsw_ef = hnsw_ef
        self.score_precision = score_precision
        self._client = None

    @property
    def spec(self) -> str:
        return (
            f"qdrant/v1(M={self.m},efC={self.ef_construct},"
            f"hnsw_ef={self.hnsw_ef},seed={self.seed})"
        )

    def build(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams

        matrix = self._register(vectors, ids)
        self._client = QdrantClient(url=self.url)
        self._client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=self.m, ef_construct=self.ef_construct),
        )
        points = [
            PointStruct(id=position, vector=matrix[position].tolist(), payload={"chunk_id": chunk_id})
            for position, chunk_id in enumerate(self.ids)
        ]
        self._client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        from qdrant_client.models import SearchParams

        if self._client is None:
            raise PipelineStateError("index not built")
        hits = self._client.search(
            collection_name=self.collection,
            query_vector=np.asarray(query, dtype=np.float64).reshape(-1).tolist(),
            limit=k,
            search_params=SearchParams(hnsw_ef=self.hnsw_ef),
        )
        pairs = [
            (hit.payload["chunk_id"], round(float(hit.score), self.score_precision))
            for hit in hits
        ]
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        return pairs
