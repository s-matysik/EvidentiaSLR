"""BM25."""

import pytest

from evidentiaslr.exceptions import ConfigurationError
from evidentiaslr.lexical import BM25, tokenise


def build() -> BM25:
    bm25 = BM25()
    bm25.build(
        ["vector databases and ann search", "eeg emotion recognition", "prisma screening"],
        ["a", "b", "c"],
    )
    return bm25


def test_tokeniser_is_lowercase_alphanumeric():
    assert tokenise("EEG-based, 32 channels!") == ["eeg", "based", "32", "channels"]


def test_ranks_the_matching_document_first():
    assert build().search("vector databases", k=1)[0][0] == "a"


def test_unmatched_query_returns_nothing():
    assert build().search("astrophysics", k=5) == []


def test_empty_query_returns_nothing():
    assert build().search("", k=5) == []


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ConfigurationError):
        BM25().build(["one", "two"], ["only-one"])


def test_spec_records_parameters():
    assert "k1=1.2" in BM25().spec
