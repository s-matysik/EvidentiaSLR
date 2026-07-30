"""Deterministic BM25.

Implemented in-package rather than pulled from a library so that the tokeniser,
the IDF variant and the tie-breaking rule are all pinned and recorded in the
certificate. A hybrid run is only reproducible if both of its halves are.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from .determinism import DEFAULT_SCORE_PRECISION
from .exceptions import ConfigurationError

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        score_precision: int = DEFAULT_SCORE_PRECISION,
    ):
        self.k1 = k1
        self.b = b
        self.score_precision = score_precision
        self.ids: tuple[str, ...] = ()
        self._documents: list[Counter] = []
        self._lengths: list[int] = []
        self._avg_length: float = 0.0
        self._document_frequency: Counter = Counter()

    @property
    def spec(self) -> str:
        return f"bm25/v1(k1={self.k1},b={self.b},tok=alnum-lower)"

    def build(self, texts: Sequence[str], ids: Sequence[str]) -> None:
        if len(texts) != len(ids):
            raise ConfigurationError("texts and ids length mismatch")
        self.ids = tuple(ids)
        self._documents = [Counter(tokenise(text)) for text in texts]
        self._lengths = [sum(document.values()) for document in self._documents]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._document_frequency = Counter()
        for document in self._documents:
            self._document_frequency.update(document.keys())

    def _idf(self, term: str) -> float:
        n = len(self._documents)
        df = self._document_frequency.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = tokenise(query)
        if not terms or not self._documents:
            return []

        scores = [0.0] * len(self._documents)
        for term in terms:
            if term not in self._document_frequency:
                continue
            idf = self._idf(term)
            for position, document in enumerate(self._documents):
                frequency = document.get(term, 0)
                if not frequency:
                    continue
                length_ratio = (self._lengths[position] / self._avg_length) if self._avg_length else 0.0
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[position] += idf * (frequency * (self.k1 + 1)) / denominator

        rounded = [round(score, self.score_precision) for score in scores]
        order = sorted(
            range(len(self.ids)),
            key=lambda i: (-rounded[i], self.ids[i]),
        )
        return [(self.ids[i], rounded[i]) for i in order[:k] if rounded[i] > 0.0]
