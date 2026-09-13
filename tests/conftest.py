"""Shared fixtures.

`synthetic_corpus` is built from a fixed topic rotation so that every test
sees the same records with the same ids, and so that a query is guaranteed to
have both strong and weak matches — which is what makes tie-breaking and
approximation effects observable.
"""

from __future__ import annotations

import pytest

from evidentiaslr import Corpus, HashEmbedder, Record

from .helpers import TEI_SAMPLE

TOPICS = (
    "convolutional neural networks for eeg emotion recognition",
    "transformer models applied to clinical text classification",
    "systematic review methodology and screening automation",
    "approximate nearest neighbour search in vector databases",
)

def build_corpus(n: int = 40) -> Corpus:
    records = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        records.append(
            Record.build(
                title=f"Study {i:03d} on {topic}",
                abstract=(
                    f"This paper investigates {topic}. Sample size {100 + i}. "
                    f"We report accuracy and discuss limitations of {topic}."
                ),
                doi=f"10.1234/study.{i:03d}",
                year=2015 + (i % 10),
                authors=[f"Author {i}", f"Coauthor {i}"],
            )
        )
    return Corpus(records)


@pytest.fixture
def tei() -> str:
    return TEI_SAMPLE


@pytest.fixture
def corpus_factory():
    return build_corpus


@pytest.fixture
def synthetic_corpus() -> Corpus:
    return build_corpus(40)


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dimension=128)
