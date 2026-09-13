"""Retrieval pipeline: modes, fusion, section filtering."""

import pytest

from evidentiaslr import FlatIndex, HashEmbedder, Retriever, TEISectionChunker
from evidentiaslr.corpus import Corpus
from evidentiaslr.exceptions import ConfigurationError, PipelineStateError
from evidentiaslr.retrieve import reciprocal_rank_fusion
from evidentiaslr.tei import parse_tei


def test_retriever_is_bit_identical_across_repeated_runs(corpus_factory):
    corpus = corpus_factory(30)
    query = "systematic review screening automation"
    runs = [
        Retriever(corpus, HashEmbedder(dimension=128), index=FlatIndex())
        .prepare()
        .retrieve(query, k=10)
        .chunk_ids
        for _ in range(5)
    ]
    assert all(run == runs[0] for run in runs)


def test_hybrid_mode_requires_lexical_preparation(corpus_factory, embedder):
    retriever = Retriever(corpus_factory(10), embedder).prepare(with_lexical=False)
    with pytest.raises(PipelineStateError):
        retriever.retrieve("anything", mode="hybrid")


def test_unknown_mode_is_rejected(corpus_factory, embedder):
    retriever = Retriever(corpus_factory(10), embedder).prepare()
    with pytest.raises(ConfigurationError):
        retriever.retrieve("anything", mode="telepathy")


@pytest.mark.parametrize("mode", ["dense", "lexical", "hybrid"])
def test_every_mode_returns_k_results(corpus_factory, embedder, mode):
    retriever = Retriever(corpus_factory(30), embedder).prepare(with_lexical=True)
    evidence = retriever.retrieve("eeg emotion recognition", k=8, mode=mode)
    assert len(evidence.chunks) == 8
    assert evidence.mode == mode


def test_ranks_are_contiguous_and_ordered(corpus_factory, embedder):
    evidence = Retriever(corpus_factory(20), embedder).prepare().retrieve("eeg", k=5)
    assert [c.rank for c in evidence.chunks] == [1, 2, 3, 4, 5]
    scores = [c.score for c in evidence.chunks]
    assert scores == sorted(scores, reverse=True)


def test_record_ids_deduplicate_while_preserving_order(corpus_factory, embedder):
    evidence = Retriever(corpus_factory(20), embedder).prepare().retrieve("eeg", k=5)
    assert len(evidence.record_ids) == len(set(evidence.record_ids))


def test_rrf_is_deterministic_and_order_independent():
    a = [("x", 0.9), ("y", 0.8)]
    b = [("y", 0.7), ("x", 0.6)]
    assert reciprocal_rank_fusion([a, b]) == reciprocal_rank_fusion([b, a])


def test_rrf_rewards_agreement_between_rankings():
    dense = [("shared", 0.9), ("dense_only", 0.8)]
    lexical = [("shared", 5.0), ("lexical_only", 4.0)]
    fused = reciprocal_rank_fusion([dense, lexical])
    assert fused[0][0] == "shared"


def test_section_filter_restricts_evidence(tei):
    corpus = Corpus([parse_tei(tei)])
    retriever = Retriever(
        corpus, HashEmbedder(dimension=64), chunker=TEISectionChunker()
    ).prepare()
    evidence = retriever.retrieve("convolutional network training", k=5, section_filter=("methods",))
    assert evidence.chunks
    assert {c.section for c in evidence.chunks} == {"methods"}
    assert evidence.section_filter == ("methods",)


def test_empty_corpus_is_rejected(embedder):
    with pytest.raises(ConfigurationError):
        Retriever(Corpus([]), embedder).prepare()


def test_evidence_set_serialises(corpus_factory, embedder):
    evidence = Retriever(corpus_factory(15), embedder).prepare().retrieve("eeg", k=3)
    payload = evidence.as_dict()
    assert payload["k"] == 3
    assert len(payload["chunks"]) == 3
    assert payload["corpus_hash"]
