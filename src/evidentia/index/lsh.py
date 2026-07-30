"""Random-projection LSH index.

Pure numpy, no external service, and — critically — seed-dependent. It exists
so that the fidelity and stability experiments can be executed in any
environment, including CI, without FAISS or a running database. It shares the
defining property of every production ANN index: the recall it achieves and
the particular documents it drops both depend on a random draw made at build
time.

Use it to establish the *shape* of the effect. Use the FAISS and service
backends to establish the magnitude on the systems people actually deploy.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from ..determinism import DEFAULT_SCORE_PRECISION, rng
from ..exceptions import PipelineStateError
from .base import BaseIndex


class RandomProjectionLSH(BaseIndex):
    exact = False

    def __init__(
        self,
        n_bits: int = 16,
        n_tables: int = 4,
        probe_radius: int = 1,
        score_precision: int = DEFAULT_SCORE_PRECISION,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.n_bits = n_bits
        self.n_tables = n_tables
        self.probe_radius = probe_radius
        self.score_precision = score_precision
        self._planes: list[np.ndarray] = []
        self._tables: list[dict[int, list[int]]] = []
        self._matrix: np.ndarray | None = None

    @property
    def spec(self) -> str:
        return (
            f"lsh/v1(bits={self.n_bits},tables={self.n_tables},"
            f"probe={self.probe_radius},seed={self.seed})"
        )

    def _signature(self, vectors: np.ndarray, planes: np.ndarray) -> np.ndarray:
        bits = (vectors @ planes.T) > 0
        weights = (1 << np.arange(self.n_bits)).astype(np.int64)
        return bits.astype(np.int64) @ weights

    def build(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        matrix = self._register(vectors, ids)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self._matrix = matrix / norms

        self._planes = []
        self._tables = []
        for table in range(self.n_tables):
            generator = rng(self.seed, "lsh", table)
            planes = generator.normal(size=(self.n_bits, self.dimension))
            buckets: dict[int, list[int]] = defaultdict(list)
            for position, code in enumerate(self._signature(self._matrix, planes)):
                buckets[int(code)].append(position)
            self._planes.append(planes)
            self._tables.append(dict(buckets))

    def _candidates(self, vector: np.ndarray) -> set[int]:
        found: set[int] = set()
        for planes, buckets in zip(self._planes, self._tables, strict=True):
            code = int(self._signature(vector.reshape(1, -1), planes)[0])
            found.update(buckets.get(code, ()))
            for bit in range(self.n_bits * min(self.probe_radius, 1)):
                found.update(buckets.get(code ^ (1 << bit), ()))
        return found

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._matrix is None:
            raise PipelineStateError("index not built")
        vector = np.asarray(query, dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(vector)
        vector = vector / (norm if norm else 1.0)

        candidates = self._candidates(vector)
        if not candidates:
            return []

        positions = sorted(candidates)
        scores = np.round(self._matrix[positions] @ vector, self.score_precision)
        order = sorted(
            range(len(positions)),
            key=lambda i: (-float(scores[i]), self.ids[positions[i]]),
        )
        return [(self.ids[positions[i]], float(scores[i])) for i in order[:k]]
