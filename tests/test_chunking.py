"""Chunker determinism and IMRaD awareness."""

import pytest

from evidentiaslr import Corpus, RecordChunker, TEISectionChunker
from evidentiaslr.chunking import FixedWindowChunker, SentenceChunker, canonical_section
from evidentiaslr.exceptions import ConfigurationError
from evidentiaslr.tei import parse_tei


def test_chunk_ids_are_stable_across_runs(corpus_factory):
    corpus = corpus_factory(6)
    a = SentenceChunker().chunk_corpus(corpus)
    b = SentenceChunker().chunk_corpus(corpus)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_different_chunkers_yield_disjoint_chunk_ids(corpus_factory):
    corpus = corpus_factory(4)
    fixed = {c.chunk_id for c in FixedWindowChunker(size=20, overlap=5).chunk_corpus(corpus)}
    sentence = {c.chunk_id for c in SentenceChunker().chunk_corpus(corpus)}
    assert not fixed & sentence


def test_chunk_order_is_stable_regardless_of_corpus_iteration(corpus_factory):
    corpus = corpus_factory(8)
    chunks = SentenceChunker().chunk_corpus(corpus)
    assert chunks == sorted(chunks, key=lambda c: (c.record_id, c.ordinal))


def test_fixed_window_rejects_overlap_larger_than_size():
    with pytest.raises(ConfigurationError):
        FixedWindowChunker(size=10, overlap=10)


def test_record_chunker_produces_one_chunk_per_record(corpus_factory):
    assert len(RecordChunker().chunk_corpus(corpus_factory(7))) == 7


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Materials and Methods", "methods"),
        ("3. Results", "results"),
        ("Related Work", "background"),
        ("Acknowledgements", "other"),
    ],
)
def test_canonical_section_maps_variants(raw, expected):
    assert canonical_section(raw) == expected


def test_tei_section_chunker_filters_to_methods(tei):
    chunks = TEISectionChunker(keep_sections=("methods",)).chunk_corpus(Corpus([parse_tei(tei)]))
    sections = {c.section for c in chunks}
    assert "methods" in sections
    assert "results" not in sections


def test_tei_section_chunker_falls_back_without_sections(corpus_factory):
    chunks = TEISectionChunker().chunk_corpus(corpus_factory(3))
    assert chunks
    assert {c.section for c in chunks} == {"unknown"}


def test_spec_is_reflected_in_chunk_ids(corpus_factory):
    corpus = corpus_factory(3)
    a = {c.chunk_id for c in SentenceChunker(max_words=50).chunk_corpus(corpus)}
    b = {c.chunk_id for c in SentenceChunker(max_words=180).chunk_corpus(corpus)}
    assert not a & b
