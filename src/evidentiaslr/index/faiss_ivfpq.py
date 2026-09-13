"""FAISS IVF-PQ backend.

Doubly non-deterministic: coarse centroids and product-quantiser codebooks
are both trained by k-means from a random initialisation, and the
quantisation is lossy on top of that. `nprobe` then trades recall for latency
at search time on the already-approximate structure.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..determinism import DEFAULT_SCORE_PRECISION
from ..exceptions import BackendUnavailableError, ConfigurationError, PipelineStateError
from .base import BaseIndex, require_module


def _require_faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover
        raise BackendUnavailableError(
            "faiss is required for this backend; install faiss-cpu"
        ) from exc
    return faiss


class FaissIVFPQ(BaseIndex):
    exact = False

    def __init__(
        self,
        nlist: int = 100,
        m: int = 8,
        nbits: int = 8,
        nprobe: int = 8,
        threads: int = 1,
        score_precision: int = DEFAULT_SCORE_PRECISION,
        seed: int = 42,
    ):
        require_module("faiss", "faiss")
        super().__init__(seed=seed)
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self.threads = threads
        self.score_precision = score_precision
        self._index = None

    @property
    def spec(self) -> str:
        return (
            f"faiss-ivfpq/v1(nlist={self.nlist},m={self.m},nbits={self.nbits},"
            f"nprobe={self.nprobe},threads={self.threads},seed={self.seed})"
        )

    def build(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        faiss = _require_faiss()
        matrix = self._register(vectors, ids)
        faiss.omp_set_num_threads(self.threads)

        if self.dimension % self.m != 0:
            raise ConfigurationError(f"dimension {self.dimension} not divisible by m={self.m}")

        quantiser = faiss.IndexFlatIP(self.dimension)
        index = faiss.IndexIVFPQ(
            quantiser, self.dimension, self.nlist, self.m, self.nbits, faiss.METRIC_INNER_PRODUCT
        )
        index.cp.seed = self.seed  # k-means initialisation
        index.nprobe = self.nprobe

        normalised = matrix.astype(np.float32)
        faiss.normalize_L2(normalised)
        index.train(normalised)
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
