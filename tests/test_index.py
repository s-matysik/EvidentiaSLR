"""Index backends: exactness, tie-breaking, seed sensitivity."""

import numpy as np
import pytest

from evidentiaslr import FlatIndex, RandomProjectionLSH, Retriever, build_index
from evidentiaslr.exceptions import (
    BackendUnavailableError,
    ConfigurationError,
    PipelineStateError,
    UnknownBackendError,
)
from evidentiaslr.index import EXACT_BACKENDS, INDEX_REGISTRY


def test_flat_index_is_exact_and_ties_break_on_id():
    index = FlatIndex()
    index.build(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), ["zzz", "aaa", "mmm"])
    result = index.search(np.array([1.0, 0.0]), k=3)
    assert [chunk_id for chunk_id, _ in result][:2] == ["aaa", "zzz"]


def test_exactness_flags_match_the_registry():
    assert FlatIndex().exact is True
    assert RandomProjectionLSH().exact is False
    assert "flat" in EXACT_BACKENDS


def test_registry_lists_every_backend():
    assert set(INDEX_REGISTRY) == {
        "flat", "lsh", "faiss-hnsw", "faiss-ivfpq", "qdrant", "pgvector", "chroma",
    }


def test_build_index_rejects_unknown_backend():
    with pytest.raises(UnknownBackendError):
        build_index("does-not-exist")


def test_optional_backend_reports_missing_dependency():
    try:
        import faiss  # noqa: F401
    except ImportError:
        with pytest.raises(BackendUnavailableError):
            build_index("faiss-hnsw").build(np.eye(4), list("abcd"))
    else:  # pragma: no cover - only when faiss is installed
        pytest.skip("faiss installed")


def test_searching_before_building_is_an_error():
    with pytest.raises(PipelineStateError):
        FlatIndex().search(np.array([1.0, 0.0]), k=1)


def test_flat_index_rejects_unsupported_metric():
    with pytest.raises(ConfigurationError):
        FlatIndex(metric="hamming")


def test_mismatched_ids_and_vectors_are_rejected():
    with pytest.raises(ConfigurationError):
        FlatIndex().build(np.eye(3), ["a", "b"])


def test_lsh_recall_is_seed_dependent(corpus_factory, embedder):
    """The core empirical premise: same data, same parameters, different build
    seed, different evidence set."""
    corpus = corpus_factory(60)
    query = "vector databases and approximate nearest neighbour search"

    reference = Retriever(corpus, embedder, index=FlatIndex()).prepare().retrieve(query, k=10)

    runs = [
        Retriever(corpus, embedder, index=RandomProjectionLSH(n_bits=10, n_tables=1, seed=s))
        .prepare()
        .retrieve(query, k=10)
        .chunk_ids
        for s in range(6)
    ]

    assert any(run != reference.chunk_ids for run in runs)
    assert len({tuple(run) for run in runs}) > 1


def test_lsh_is_reproducible_for_a_fixed_seed(corpus_factory, embedder):
    corpus = corpus_factory(40)
    query = "screening automation"
    runs = [
        Retriever(corpus, embedder, index=RandomProjectionLSH(n_bits=10, seed=3))
        .prepare()
        .retrieve(query, k=10)
        .chunk_ids
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


def test_index_spec_records_the_seed():
    assert "seed=7" in RandomProjectionLSH(seed=7).spec
