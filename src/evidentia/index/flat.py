"""Exact brute-force index.

This is the reference implementation against which every approximate backend
is measured, and — on the corpus sizes typical of a systematic review — it is
also the recommendation. A 10^6-chunk corpus at 384 dimensions is a 3 GB
float64 matrix; at float32 and with the quantisation grid applied it fits in
memory on a laptop and answers a query in well under a second.

Ties are broken by chunk id, not by insertion order. Two chunks with cosine
similarity equal to the ninth decimal are ordered by a value that does not
depend on how the corpus was loaded, which removes the last input-order
dependency from the retrieval step.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..determinism import DEFAULT_SCORE_PRECISION
from ..exceptions import ConfigurationError, PipelineStateError
from .base import BaseIndex


class FlatIndex(BaseIndex):
    exact = True

    def __init__(self, metric: str = "cosine", score_precision: int = DEFAULT_SCORE_PRECISION, seed: int = 42):
        super().__init__(seed=seed)
        if metric not in {"cosine", "ip", "l2"}:
            raise ConfigurationError(f"unsupported metric: {metric}")
        self.metric = metric
        self.score_precision = score_precision
        self._matrix: np.ndarray | None = None

    @property
    def spec(self) -> str:
        return f"flat/v1(metric={self.metric},prec={self.score_precision},exact=true)"

    def build(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        matrix = self._register(vectors, ids)
        if self.metric == "cosine":
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms
        self._matrix = matrix

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._matrix is None:
            raise PipelineStateError("index not built")
        vector = np.asarray(query, dtype=np.float64).reshape(-1)

        if self.metric == "l2":
            scores = -np.linalg.norm(self._matrix - vector, axis=1)
        else:
            if self.metric == "cosine":
                norm = np.linalg.norm(vector)
                vector = vector / (norm if norm else 1.0)
            scores = self._matrix @ vector

        scores = np.round(scores, self.score_precision)
        return self._top_k(scores, k)

    def _top_k(self, scores: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Select and order the top k without sorting the whole corpus.

        A full Python sort is O(n log n) with a comparison callback and, at a
        million chunks, dominates the search by orders of magnitude — the
        matrix product itself is fast. `argpartition` finds the k-th largest
        score in linear time; everything at or above it is then collected and
        only that small set is sorted with the deterministic
        `(-score, chunk_id)` key.

        Collecting on `>= threshold` rather than taking argpartition's output
        directly is what keeps ties correct: when several chunks share the
        k-th score, argpartition picks among them arbitrarily, which would
        reintroduce exactly the input-order dependence this index exists to
        remove.
        """
        n = len(self.ids)
        if k >= n:
            candidates = range(n)
        else:
            partition = np.argpartition(-scores, k - 1)[:k]
            threshold = scores[partition].min()
            candidates = np.flatnonzero(scores >= threshold)

        ordered = sorted(candidates, key=lambda i: (-float(scores[i]), self.ids[i]))
        return [(self.ids[i], float(scores[i])) for i in ordered[:k]]
