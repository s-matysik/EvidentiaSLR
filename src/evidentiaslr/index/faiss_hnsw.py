"""FAISS HNSW backend.

HNSW assigns each node a level from a random draw, and a multi-threaded build
visits nodes in a non-fixed order, so two builds of the same data produce
different graphs. Pinning `threads=1` removes the ordering component; the
level draw remains seed-dependent, which is why `seed` is a first-class
argument here rather than buried in global state.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..determinism import DEFAULT_SCORE_PRECISION
from ..exceptions import BackendUnavailableError, PipelineStateError
from .base import BaseIndex, require_module


def _require_faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover
        raise BackendUnavailableError(
            "faiss is required for this backend; install faiss-cpu"
        ) from exc
    return faiss


class FaissHNSW(BaseIndex):
    exact = False

    def __init__(
        self,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        threads: int = 1,
        score_precision: int = DEFAULT_SCORE_PRECISION,
        seed: int = 42,
    ):
        require_module("faiss", "faiss")
        super().__init__(seed=seed)
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.threads = threads
        self.score_precision = score_precision
        self._index = None

    @property
    def spec(self) -> str:
        return (
            f"faiss-hnsw/v1(M={self.m},efC={self.ef_construction},"
            f"efS={self.ef_search},threads={self.threads},seed={self.seed})"
        )

    def build(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        faiss = _require_faiss()
        matrix = self._register(vectors, ids)
        faiss.omp_set_num_threads(self.threads)

        index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        # Level assignment is drawn from FAISS's RNG; expose it to the caller.
        try:
            index.hnsw.rng.seed = self.seed
        except AttributeError:  # pragma: no cover - version dependent
            pass

        normalised = matrix.astype(np.float32)
        faiss.normalize_L2(normalised)
        index.add(normalised)
        self._index = index

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        faiss = _require_faiss()
        if self._index is None:
            raise PipelineStateError("index not built")
        vector = np.asarray(query, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vector)
        scores, positions = self._index.search(vector, k)

        pairs = [
            (self.ids[int(position)], round(float(score), self.score_precision))
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0
        ]
        pairs.sort(key=lambda pair: (-pair[1], pair[0]))
        return pairs
