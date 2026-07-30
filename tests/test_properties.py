"""Property-based tests.

Example-based tests check the cases someone thought of. For a library whose
entire claim is "the same inputs always produce the same output", that is the
wrong shape of evidence: the interesting failures are the inputs nobody
imagined — a title that is only punctuation, an abstract with a lone surrogate,
two records whose scores collide at the ninth decimal.

Hypothesis generates those. Each test below states an invariant that must hold
for *every* input, which is exactly the form the reproducibility claim takes.
"""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis", reason="pip install hypothesis")

from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from evidentia import Corpus, FlatIndex, HashEmbedder, Record  # noqa: E402
from evidentia.chunking import FixedWindowChunker, RecordChunker, SentenceChunker  # noqa: E402
from evidentia.determinism import l2_normalise, quantise, stable_hash  # noqa: E402
from evidentia.lexical import BM25  # noqa: E402
from evidentia.metrics import (  # noqa: E402
    evidence_set_fidelity,
    evidence_set_stability,
    jaccard,
    kendall_tau_on_common,
    rank_biased_overlap,
)
from evidentia.retrieve import reciprocal_rank_fusion  # noqa: E402

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

text = st.text(min_size=1, max_size=120).filter(lambda s: s.strip())
identifier = st.text(alphabet="abcdefghijklmnop0123456789", min_size=4, max_size=8)


@st.composite
def records(draw):
    return Record.build(
        title=draw(text),
        abstract=draw(st.text(max_size=200)),
        doi=draw(st.one_of(st.just(""), st.text(alphabet="0123456789./abcdef", min_size=5, max_size=20))),
        year=draw(st.one_of(st.none(), st.integers(1900, 2030))),
        authors=draw(st.lists(text, max_size=4)),
    )


@st.composite
def corpora(draw, min_size=1, max_size=12):
    return Corpus(draw(st.lists(records(), min_size=min_size, max_size=max_size)))


# --------------------------------------------------------------- canonical id


@SETTINGS
@given(records())
def test_record_id_is_a_function_of_content_only(record):
    assert record.record_id == Record.build(**record.identity()).record_id


@SETTINGS
@given(st.lists(records(), min_size=1, max_size=10), st.randoms())
def test_corpus_hash_is_permutation_invariant(items, random):
    shuffled = list(items)
    random.shuffle(shuffled)
    assert Corpus(items).corpus_hash == Corpus(shuffled).corpus_hash


@SETTINGS
@given(st.lists(records(), min_size=1, max_size=10))
def test_corpus_hash_changes_when_a_record_is_added(items):
    base = Corpus(items)
    extra = Record.build(title="a title that will not collide", abstract="unique body text")
    assume(extra.dedup_key not in {r.dedup_key for r in base})
    assert Corpus([*items, extra]).corpus_hash != base.corpus_hash


@SETTINGS
@given(corpora())
def test_corpus_is_free_of_duplicate_keys(corpus):
    keys = [record.dedup_key for record in corpus]
    assert len(keys) == len(set(keys))


@SETTINGS
@given(st.lists(st.text(max_size=30), min_size=1, max_size=6))
def test_stable_hash_is_deterministic_and_order_sensitive(parts):
    assert stable_hash(*parts) == stable_hash(*parts)
    # A palindromic argument list reverses to itself, so order sensitivity
    # is only meaningful when reversing actually changes the sequence.
    if parts != list(reversed(parts)):
        assert stable_hash(*parts) != stable_hash(*reversed(parts))


# ----------------------------------------------------------------- chunking


@SETTINGS
@given(corpora(), st.sampled_from([RecordChunker, SentenceChunker, FixedWindowChunker]))
def test_chunking_is_deterministic(corpus, chunker_cls):
    a = chunker_cls().chunk_corpus(corpus)
    b = chunker_cls().chunk_corpus(corpus)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


@SETTINGS
@given(corpora(), st.sampled_from([RecordChunker, SentenceChunker, FixedWindowChunker]))
def test_chunk_ids_are_unique_within_a_corpus(corpus, chunker_cls):
    ids = [c.chunk_id for c in chunker_cls().chunk_corpus(corpus)]
    assert len(ids) == len(set(ids))


@SETTINGS
@given(corpora())
def test_chunks_are_globally_ordered(corpus):
    chunks = SentenceChunker().chunk_corpus(corpus)
    assert chunks == sorted(chunks, key=lambda c: (c.record_id, c.ordinal))


# --------------------------------------------------------------- determinism


@SETTINGS
@given(st.lists(st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=32))
def test_quantise_is_idempotent(values):
    once = quantise(np.array(values), 6)
    assert np.array_equal(once, quantise(once, 6))


@SETTINGS
@given(st.lists(st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=32))
def test_quantise_bounds_the_effect_of_a_sub_grid_perturbation(values):
    """Quantisation reduces sub-grid noise; it cannot abolish it.

    A value sitting within the perturbation of a grid boundary still flips to
    the neighbouring grid point. What quantisation guarantees is that the flip
    is at most one grid step, so the disturbance is bounded by the grid size
    rather than propagating at full float precision. Claiming exact absorption
    would be false, and Hypothesis finds the counterexample immediately.
    """
    array = np.array(values)
    perturbed = array + np.full_like(array, 1e-9)
    difference = np.abs(quantise(array, 6) - quantise(perturbed, 6))
    assert np.all(difference <= 1e-6 + 1e-12)


@SETTINGS
@given(st.lists(st.floats(-100, 100, allow_nan=False, allow_infinity=False),
                min_size=2, max_size=16))
def test_l2_normalise_produces_unit_rows(values):
    assume(any(abs(v) > 1e-6 for v in values))
    normalised = l2_normalise(np.array([values]))
    assert abs(float(np.linalg.norm(normalised[0])) - 1.0) < 1e-9


@SETTINGS
@given(st.lists(text, min_size=1, max_size=8))
def test_embedding_is_a_pure_function_of_the_text(texts):
    assert np.array_equal(
        HashEmbedder(dimension=32).encode(texts),
        HashEmbedder(dimension=32).encode(texts),
    )


@SETTINGS
@given(st.lists(text, min_size=1, max_size=8))
def test_embedding_rows_are_independent_of_batch_order(texts):
    embedder = HashEmbedder(dimension=32)
    together = embedder.encode(texts)
    apart = np.vstack([embedder.encode([t]) for t in texts])
    assert np.array_equal(together, apart)


# ---------------------------------------------------------------- retrieval


@SETTINGS
@given(st.lists(identifier, min_size=2, max_size=25, unique=True), st.randoms())
def test_exact_search_is_independent_of_insertion_order(ids, random):
    """The property deterministic tie-breaking exists to guarantee.

    Every vector is identical, so every score ties — the hardest case, and the
    one where insertion order would decide the ranking in a naive index.
    """
    vectors = np.ones((len(ids), 4))

    forward = FlatIndex()
    forward.build(vectors, ids)
    result_a = forward.search(np.ones(4), min(5, len(ids)))

    shuffled_ids = list(ids)
    random.shuffle(shuffled_ids)
    backward = FlatIndex()
    backward.build(vectors, shuffled_ids)
    result_b = backward.search(np.ones(4), min(5, len(ids)))

    assert result_a == result_b
    assert [i for i, _ in result_a] == sorted(ids)[: len(result_a)]


@SETTINGS
@given(st.lists(identifier, min_size=1, max_size=20, unique=True),
       st.integers(min_value=1, max_value=30))
def test_search_never_returns_more_than_k_or_more_than_the_corpus(ids, k):
    rng = np.random.default_rng(0)
    index = FlatIndex()
    index.build(quantise(rng.normal(size=(len(ids), 8))), ids)
    assert len(index.search(rng.normal(size=8), k)) <= min(k, len(ids))


@SETTINGS
@given(st.lists(st.tuples(identifier, st.floats(0, 1, allow_nan=False)),
                min_size=1, max_size=10, unique_by=lambda p: p[0]),
       st.lists(st.tuples(identifier, st.floats(0, 1, allow_nan=False)),
                min_size=1, max_size=10, unique_by=lambda p: p[0]))
def test_rrf_is_commutative(a, b):
    assert reciprocal_rank_fusion([a, b]) == reciprocal_rank_fusion([b, a])


@SETTINGS
@given(st.lists(st.tuples(identifier, st.floats(0, 1, allow_nan=False)),
                min_size=1, max_size=10, unique_by=lambda p: p[0]))
def test_rrf_output_is_strictly_ordered(ranking):
    fused = reciprocal_rank_fusion([ranking])
    assert fused == sorted(fused, key=lambda pair: (-pair[1], pair[0]))


@SETTINGS
@given(st.lists(text, min_size=1, max_size=10), st.text(min_size=1, max_size=30))
def test_bm25_never_returns_zero_scored_documents(texts, query):
    bm25 = BM25()
    bm25.build(texts, [f"d{i}" for i in range(len(texts))])
    assert all(score > 0 for _, score in bm25.search(query, k=len(texts)))


# ------------------------------------------------------------------ metrics


@SETTINGS
@given(st.lists(identifier, min_size=1, max_size=15, unique=True),
       st.lists(identifier, min_size=1, max_size=15, unique=True))
def test_agreement_measures_stay_in_range(a, b):
    assert 0.0 <= jaccard(a, b) <= 1.0
    assert 0.0 <= rank_biased_overlap(a, b) <= 1.0 + 1e-9
    assert -1.0 <= kendall_tau_on_common(a, b) <= 1.0


@SETTINGS
@given(st.lists(identifier, min_size=1, max_size=15, unique=True))
def test_agreement_measures_are_one_for_identical_lists(ids):
    assert jaccard(ids, ids) == 1.0
    assert rank_biased_overlap(ids, ids) == pytest.approx(1.0)
    assert kendall_tau_on_common(ids, ids) == 1.0


@SETTINGS
@given(st.lists(identifier, min_size=1, max_size=15, unique=True),
       st.lists(identifier, min_size=1, max_size=15, unique=True))
def test_jaccard_is_symmetric(a, b):
    assert jaccard(a, b) == jaccard(b, a)


@SETTINGS
@given(st.lists(identifier, min_size=1, max_size=12, unique=True),
       st.lists(identifier, min_size=1, max_size=12, unique=True))
def test_fidelity_partitions_the_reference_set(exact, approx):
    result = evidence_set_fidelity(exact, approx)
    assert 0.0 <= result.esf <= 1.0
    assert set(result.missing) | (set(exact) & set(approx)) == set(exact)
    assert set(result.missing).isdisjoint(result.spurious)


@SETTINGS
@given(st.lists(st.lists(identifier, min_size=1, max_size=10, unique=True),
                min_size=2, max_size=6))
def test_stability_core_and_volatile_partition_the_union(runs):
    result = evidence_set_stability(runs)
    union = set().union(*(set(run) for run in runs))
    assert set(result.core) | set(result.volatile) == union
    assert set(result.core).isdisjoint(result.volatile)
    assert 0.0 <= result.ess <= 1.0


@SETTINGS
@given(st.lists(identifier, min_size=1, max_size=10, unique=True),
       st.integers(min_value=2, max_value=5))
def test_stability_is_perfect_when_every_run_agrees(ids, repeats):
    result = evidence_set_stability([ids] * repeats)
    assert result.ess == 1.0
    assert result.identical_fraction == 1.0
    assert result.volatile == []
