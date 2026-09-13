"""Certificate issuance and verification."""

import json

from evidentiaslr import FlatIndex, HashEmbedder, Retriever, issue, verify
from evidentiaslr.certificate import evidence_digest


def evidence_for(corpus, dimension=128, query="transformer models clinical text", k=10):
    return (
        Retriever(corpus, HashEmbedder(dimension=dimension), index=FlatIndex())
        .prepare()
        .retrieve(query, k=k)
    )


def test_digest_is_order_sensitive():
    assert evidence_digest(["a", "b"]) != evidence_digest(["b", "a"])


def test_certificate_verifies_against_an_identical_rerun(tmp_path, corpus_factory):
    corpus = corpus_factory(25)
    path = issue(evidence_for(corpus), corpus_size=len(corpus)).save(tmp_path / "cert.json")

    result = verify(json.loads(path.read_text()), evidence_for(corpus), corpus_size=len(corpus))
    assert result.ok
    assert result.mismatches == []


def test_certificate_records_the_pipeline(corpus_factory):
    corpus = corpus_factory(20)
    payload = issue(evidence_for(corpus), corpus_size=len(corpus)).as_dict()
    assert payload["pipeline"].keys() == {"chunker", "embedder", "index"}
    assert payload["corpus"]["hash"] == corpus.corpus_hash
    assert payload["environment"]["versions"]["python"]


def test_certificate_detects_a_changed_embedder(corpus_factory):
    corpus = corpus_factory(25)
    certificate = issue(evidence_for(corpus, 128), corpus_size=len(corpus)).as_dict()

    result = verify(certificate, evidence_for(corpus, 64), corpus_size=len(corpus))
    assert not result.ok
    assert "pipeline.embedder" in result.mismatches
    assert result.detail["pipeline.embedder"]["expected"] != result.detail["pipeline.embedder"]["actual"]


def test_certificate_detects_a_changed_corpus(corpus_factory):
    small, large = corpus_factory(20), corpus_factory(21)
    certificate = issue(evidence_for(small, k=5), corpus_size=len(small)).as_dict()

    result = verify(certificate, evidence_for(large, k=5), corpus_size=len(large))
    assert not result.ok
    assert "corpus.hash" in result.mismatches


def test_certificate_flags_reordering_separately_from_membership(corpus_factory):
    corpus = corpus_factory(20)
    evidence = evidence_for(corpus, k=5)
    certificate = issue(evidence, corpus_size=len(corpus)).as_dict()

    evidence.chunks = list(reversed(evidence.chunks))
    result = verify(certificate, evidence, corpus_size=len(corpus))
    assert not result.ok
    assert result.detail["evidence.digest"]["reordered_only"] is True
    assert result.detail["evidence.digest"]["missing"] == []
