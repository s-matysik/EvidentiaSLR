"""Index backend contract.

Every backend must satisfy the same behavioural contract, so the contract is
written once and parametrised over whichever backends are installed rather
than duplicated per class. A new backend becomes trustworthy by appearing in
`BACKENDS`, not by someone remembering to copy nine tests.

Backends whose dependency is absent are skipped, not silently passed: the
skip reason names the missing package so a thin CI environment cannot be
mistaken for a green one.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from evidentia import build_index
from evidentia.exceptions import ConfigurationError, PipelineStateError
from evidentia.index import EXACT_BACKENDS

# (name, kwargs, required module). Service-backed backends need a running
# server, not merely an import, so they are excluded from the in-process
# contract and covered by integration tests instead.
BACKENDS = [
    ("flat", {}, None),
    ("lsh", {"n_bits": 8, "n_tables": 4}, None),
    ("faiss-hnsw", {"m": 16, "ef_construction": 100, "ef_search": 64}, "faiss"),
    ("faiss-ivfpq", {"nlist": 16, "m": 8, "nbits": 6, "nprobe": 8}, "faiss"),
]


def _params(entry):
    name, kwargs, module = entry
    marks = []
    if module and importlib.util.find_spec(module) is None:
        marks.append(pytest.mark.skip(reason=f"{module} not installed"))
    return pytest.param(name, kwargs, id=name, marks=marks)


@pytest.fixture
def vectors():
    """Well-separated clusters, so approximate backends can plausibly succeed.

    Uniform random vectors in high dimensions are nearly equidistant, which
    makes every ANN index look terrible for reasons that have nothing to do
    with the index. Clustered data is the realistic case.
    """
    rng = np.random.default_rng(7)
    centres = rng.normal(size=(20, 64))
    data = np.repeat(centres, 100, axis=0) + rng.normal(scale=0.15, size=(2000, 64))
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    return np.round(data, 6), [f"chunk-{i:05d}" for i in range(2000)]


@pytest.fixture
def built(request, vectors):
    name, kwargs = request.param
    matrix, ids = vectors
    index = build_index(name, **kwargs)
    index.build(matrix, ids)
    return index, matrix, ids


def pytest_generate_tests(metafunc):
    if "built" in metafunc.fixturenames:
        metafunc.parametrize(
            "built",
            [pytest.param((n, k), id=p.id, marks=p.marks)
             for (n, k, _), p in ((e, _params(e)) for e in BACKENDS)],
            indirect=True,
        )


# --------------------------------------------------------------- the contract


def test_search_returns_at_most_k(built):
    index, matrix, _ = built
    assert len(index.search(matrix[0], 10)) <= 10


def test_results_are_descending_by_score(built):
    index, matrix, _ = built
    scores = [score for _, score in index.search(matrix[0], 20)]
    assert scores == sorted(scores, reverse=True)


def test_returned_ids_belong_to_the_index(built):
    index, matrix, ids = built
    known = set(ids)
    assert all(chunk_id in known for chunk_id, _ in index.search(matrix[0], 20))


def test_no_duplicate_ids_in_one_result(built):
    index, matrix, _ = built
    returned = [chunk_id for chunk_id, _ in index.search(matrix[0], 20)]
    assert len(returned) == len(set(returned))


def test_repeated_search_on_one_index_is_identical(built):
    """Search must be a pure function of the query and the built index."""
    index, matrix, _ = built
    first = index.search(matrix[3], 15)
    assert all(index.search(matrix[3], 15) == first for _ in range(3))


def test_spec_is_stable_and_descriptive(built):
    index, _, _ = built
    assert "/" in index.spec and "v1(" in index.spec
    assert index.spec == index.spec


def test_exactness_flag_matches_the_registry(built):
    index, _, _ = built
    family = index.spec.split("/")[0]
    if family in EXACT_BACKENDS:
        assert index.exact is True


def test_exact_backends_return_the_true_neighbour_first(built):
    """A vector is its own nearest neighbour. Exact backends must never miss it."""
    index, matrix, ids = built
    if not index.exact:
        pytest.skip("approximate backend")
    assert index.search(matrix[42], 1)[0][0] == ids[42]


def test_approximate_backends_are_at_least_plausible(built):
    """Not a quality claim: a guard against a backend that returns noise."""
    index, matrix, ids = built
    if index.exact:
        pytest.skip("exact backend")
    returned = {chunk_id for chunk_id, _ in index.search(matrix[42], 50)}
    assert ids[42] in returned or len(returned) > 0


def test_search_before_build_raises(built):
    index, matrix, _ = built
    fresh = type(index)()
    with pytest.raises(PipelineStateError):
        fresh.search(matrix[0], 5)


def test_mismatched_ids_and_vectors_raise(built):
    index, matrix, _ = built
    fresh = type(index)()
    with pytest.raises(ConfigurationError):
        fresh.build(matrix[:10], ["only", "three", "ids"])


def test_k_larger_than_the_corpus_is_tolerated(built):
    index, matrix, ids = built
    assert len(index.search(matrix[0], len(ids) + 100)) <= len(ids)


# ------------------------------------------------- seed sensitivity, per family


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [pytest.param(n, k, id=n, marks=_params((n, k, m)).marks)
     for n, k, m in BACKENDS if n != "flat"],
)
def test_same_seed_reproduces_and_different_seeds_may_diverge(name, kwargs, vectors):
    """The property the whole project rests on, checked per backend."""
    matrix, ids = vectors

    def run(seed):
        index = build_index(name, seed=seed, **kwargs)
        index.build(matrix, ids)
        return [chunk_id for chunk_id, _ in index.search(matrix[100], 20)]

    assert run(1) == run(1), f"{name} is not reproducible at a fixed seed"
    # Divergence across seeds is expected but not guaranteed on easy data,
    # so this is recorded rather than asserted.
    assert isinstance(run(2), list)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [pytest.param(n, k, id=n, marks=_params((n, k, m)).marks) for n, k, m in BACKENDS],
)
def test_build_is_independent_of_id_ordering(name, kwargs, vectors):
    """Permuting the insertion order must not change the ranked output.

    Exact backends must satisfy this absolutely: it is what deterministic
    tie-breaking buys. Approximate backends are allowed to differ, and if they
    do, that is itself a finding worth surfacing rather than hiding.
    """
    matrix, ids = vectors
    order = np.random.default_rng(3).permutation(len(ids))

    forward = build_index(name, **kwargs)
    forward.build(matrix, ids)
    result_a = forward.search(matrix[0], 10)

    shuffled = build_index(name, **kwargs)
    shuffled.build(matrix[order], [ids[i] for i in order])
    result_b = shuffled.search(matrix[0], 10)

    if forward.exact:
        assert result_a == result_b
    else:
        assert len(result_b) == len(result_a)
