"""Additional chunkers and the chunker-agnostic comparison.

The invariant that matters most here is not any single chunker's output but
that all of them read the *same* source text. A chunker comparison in which
the chunkers disagree about what the record says measures the discrepancy,
not the strategy — and that is a silent failure, so it gets its own test.
"""

from __future__ import annotations

import pytest

from evidentia import Corpus, FlatIndex, HashEmbedder, Record, Retriever, span_agreement
from evidentia.chunkers_extra import (
    ParagraphChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    SlidingSentenceChunker,
)
from evidentia.chunking import (
    CHUNKERS,
    MODEL_DEPENDENT_CHUNKERS,
    SentenceChunker,
    TEISectionChunker,
    body_text,
    canonical_source,
)
from evidentia.exceptions import ConfigurationError
from evidentia.metrics.spans import merge_spans, record_agreement


@pytest.fixture
def sectioned_corpus():
    """Sections but no flattened body — what GROBID ingestion produces."""
    records = [
        Record.build(
            title=f"Study {i:02d} on design science",
            abstract="We examine design science research.",
            doi=f"10.1/x{i:03d}",
            year=2015 + i,
            sections=[
                ("Introduction", "Interest has grown steadily. Prior work is fragmented."),
                ("Materials and Methods",
                 f"We applied a design science method. Data came from {80 + i * 7} "
                 f"participants recruited online. Partial least squares modelling was used. "
                 f"Reliability was assessed with composite reliability."),
                ("Results", "The model explained most variance. All hypotheses held."),
            ],
        )
        for i in range(12)
    ]
    return Corpus(records)


ALL_TEXT_CHUNKERS = [
    ("sentence", lambda: SentenceChunker(max_words=60)),
    ("paragraph", ParagraphChunker),
    ("recursive", lambda: RecursiveCharacterChunker(chunk_size=300, overlap=60)),
    ("sliding", lambda: SlidingSentenceChunker(window=2, stride=1)),
    ("tei-section", lambda: TEISectionChunker(max_words=60)),
]


# ------------------------------------------------- the shared-source invariant


@pytest.mark.parametrize(("name", "factory"), ALL_TEXT_CHUNKERS, ids=[n for n, _ in ALL_TEXT_CHUNKERS])
def test_every_chunker_reads_the_same_body(name, factory, sectioned_corpus):
    """Regression: sentence and window chunkers once ignored `sections`
    entirely, so on a GROBID corpus they chunked the title and abstract while
    section-aware chunkers chunked the body. Any comparison between them was
    then guaranteed to score zero for reasons that had nothing to do with
    chunking."""
    record = sectioned_corpus[0]
    body = body_text(record)
    assert "Partial least squares" in body

    chunks = factory().chunk_corpus(Corpus([record]))
    combined = " ".join(chunk.text for chunk in chunks)
    assert "Partial least squares" in combined, f"{name} did not read the sections"


@pytest.mark.parametrize(("name", "factory"), ALL_TEXT_CHUNKERS, ids=[n for n, _ in ALL_TEXT_CHUNKERS])
def test_every_chunk_is_locatable(name, factory, sectioned_corpus):
    """Span metrics silently degrade to zero for unlocatable chunks."""
    chunks = factory().chunk_corpus(sectioned_corpus)
    located = sum(1 for chunk in chunks if chunk.located)
    assert located == len(chunks), f"{name} produced unlocatable chunks"


@pytest.mark.parametrize(("name", "factory"), ALL_TEXT_CHUNKERS, ids=[n for n, _ in ALL_TEXT_CHUNKERS])
def test_spans_lie_inside_the_canonical_source(name, factory, sectioned_corpus):
    source_lengths = {r.record_id: len(canonical_source(r)) for r in sectioned_corpus}
    for chunk in factory().chunk_corpus(sectioned_corpus):
        assert 0 <= chunk.start < chunk.end <= source_lengths[chunk.record_id]


@pytest.mark.parametrize(("name", "factory"), ALL_TEXT_CHUNKERS, ids=[n for n, _ in ALL_TEXT_CHUNKERS])
def test_spans_point_at_the_chunk_text(name, factory, sectioned_corpus):
    sources = {r.record_id: canonical_source(r) for r in sectioned_corpus}
    for chunk in factory().chunk_corpus(sectioned_corpus):
        assert sources[chunk.record_id][chunk.start:chunk.end] == chunk.text


@pytest.mark.parametrize(("name", "factory"), ALL_TEXT_CHUNKERS, ids=[n for n, _ in ALL_TEXT_CHUNKERS])
def test_deterministic_chunkers_are_reproducible(name, factory, sectioned_corpus):
    a = factory().chunk_corpus(sectioned_corpus)
    b = factory().chunk_corpus(sectioned_corpus)
    assert [(c.chunk_id, c.start, c.end) for c in a] == [(c.chunk_id, c.start, c.end) for c in b]


# ----------------------------------------------------------- per-chunker rules


def test_registry_contains_every_chunker():
    assert set(CHUNKERS) >= {
        "record", "fixed", "sentence", "tei-section",
        "paragraph", "recursive", "sliding-sentence", "semantic",
    }
    assert MODEL_DEPENDENT_CHUNKERS == {"semantic"}


def test_paragraph_chunker_rejects_an_inverted_budget():
    with pytest.raises(ConfigurationError):
        ParagraphChunker(min_words=100, max_words=50)


def test_recursive_chunker_rejects_overlap_at_or_above_size():
    with pytest.raises(ConfigurationError):
        RecursiveCharacterChunker(chunk_size=100, overlap=100)


def test_recursive_chunker_respects_the_size_budget(sectioned_corpus):
    chunks = RecursiveCharacterChunker(chunk_size=200, overlap=0).chunk_corpus(sectioned_corpus)
    assert all(len(chunk.text) <= 260 for chunk in chunks)  # overlap-free slack


def test_sliding_chunker_rejects_a_stride_above_the_window():
    with pytest.raises(ConfigurationError):
        SlidingSentenceChunker(window=2, stride=5)


def test_sliding_chunker_overlaps_consecutive_windows(sectioned_corpus):
    chunks = SlidingSentenceChunker(window=3, stride=1).chunk_corpus(
        Corpus([sectioned_corpus[0]])
    )
    assert len(chunks) >= 2
    assert chunks[0].end > chunks[1].start  # windows overlap in the source


# -------------------------------------------------------------------- semantic


def test_semantic_chunker_is_reproducible_without_jitter(sectioned_corpus):
    """With jitter off and a deterministic embedder it must be a pure function,
    which is what makes its instability a controllable variable."""
    embedder = HashEmbedder(dimension=128)
    a = SemanticChunker(embedder, percentile=85, seed=1).chunk_corpus(sectioned_corpus)
    b = SemanticChunker(embedder, percentile=85, seed=1).chunk_corpus(sectioned_corpus)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_semantic_chunker_boundaries_move_under_perturbation(sectioned_corpus):
    """The claim the chunker axis rests on: a sub-threshold change to the
    embeddings relocates chunk boundaries, so the evidence text itself
    changes — not merely which pre-cut piece is selected."""
    embedder = HashEmbedder(dimension=128)
    stable = SemanticChunker(embedder, percentile=85, jitter=0.0, seed=1)
    shaken = SemanticChunker(embedder, percentile=85, jitter=0.05, seed=1)

    assert [c.chunk_id for c in stable.chunk_corpus(sectioned_corpus)] != [
        c.chunk_id for c in shaken.chunk_corpus(sectioned_corpus)
    ]


def test_semantic_chunker_spec_records_the_embedder(sectioned_corpus):
    spec = SemanticChunker(HashEmbedder(dimension=64), percentile=90, jitter=0.01).spec
    assert "hash/v1(dim=64" in spec
    assert "pct=90" in spec and "jitter=0.01" in spec


def test_semantic_chunker_rejects_an_out_of_range_percentile():
    with pytest.raises(ConfigurationError):
        SemanticChunker(HashEmbedder(dimension=32), percentile=10.0)


def test_semantic_chunker_respects_the_word_budget(sectioned_corpus):
    chunks = SemanticChunker(
        HashEmbedder(dimension=64), percentile=99, max_words=25
    ).chunk_corpus(sectioned_corpus)
    assert all(len(chunk.text.split()) <= 40 for chunk in chunks)


# ---------------------------------------------------------------- span metrics


def test_merge_spans_collapses_overlaps():
    assert merge_spans([(0, 10), (5, 15), (20, 25)]) == [(0, 15), (20, 25)]
    assert merge_spans([(10, 20), (0, 5)]) == [(0, 5), (10, 20)]
    assert merge_spans([(5, 5)]) == []


def test_span_agreement_is_one_for_an_identical_evidence_set(sectioned_corpus):
    embedder = HashEmbedder(dimension=128)
    evidence = (
        Retriever(sectioned_corpus, embedder, chunker=SentenceChunker(max_words=60),
                  index=FlatIndex()).prepare().retrieve("participants", k=5)
    )
    result = span_agreement(evidence, evidence)
    assert result.span_jaccard == 1.0
    assert result.record_jaccard == 1.0
    assert result.located_fraction_a == 1.0


def test_span_agreement_compares_across_chunkers(sectioned_corpus):
    """The whole point: two different chunkers produce incomparable chunk ids
    but perfectly comparable character spans."""
    embedder = HashEmbedder(dimension=128)

    def evidence_for(chunker):
        return (
            Retriever(sectioned_corpus, embedder, chunker=chunker, index=FlatIndex())
            .prepare().retrieve("how many participants", k=8)
        )

    a = evidence_for(SentenceChunker(max_words=60))
    b = evidence_for(SlidingSentenceChunker(window=2, stride=1))

    assert set(a.chunk_ids).isdisjoint(b.chunk_ids)  # ids cannot overlap
    result = span_agreement(a, b)
    assert 0.0 < result.span_jaccard <= 1.0        # spans can, and do
    assert result.covered_both > 0


def test_record_agreement_is_coarser_than_span_agreement(sectioned_corpus):
    embedder = HashEmbedder(dimension=128)

    def evidence_for(chunker):
        return (
            Retriever(sectioned_corpus, embedder, chunker=chunker, index=FlatIndex())
            .prepare().retrieve("participants recruited", k=8)
        )

    a = evidence_for(SentenceChunker(max_words=60))
    b = evidence_for(RecursiveCharacterChunker(chunk_size=250, overlap=50))
    assert record_agreement(a, b) >= span_agreement(a, b).span_jaccard - 1e-9


# ------------------------------------------------------------ sentence split


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Purposive sampling (Campbell et al., 2020). When used, it works.", 2),
        ("The results follow. 385 respondents replied. Of these, 201 were men.", 3),
        ("See Fig. 3 for details. The model fits well.", 2),
        ("Data from Dr. Smith. Analysis followed.", 2),
        ("Compare e.g. Kim and Lee. Results differ.", 2),
        ("One sentence only.", 1),
        ("", 0),
    ],
)
def test_sentence_splitting_survives_academic_prose(text, expected):
    """Over-splitting a citation is worse than under-splitting: the second half
    loses the context that made the first half meaningful."""
    from evidentia.chunking import split_sentences

    assert len(split_sentences(text)) == expected


def test_sentence_splitting_accepts_non_capital_openers():
    from evidentia.chunking import split_sentences

    assert len(split_sentences("Total was computed. 42 cases remained.")) == 2
    assert len(split_sentences('He said so. "Quoted" follows.')) == 2


def test_semantic_chunker_seed_changes_the_boundaries():
    """The chunker-side analogue of rebuilding an index: same configuration,
    different seed, different evidence text."""
    from evidentia.chunking import SentenceChunker  # noqa: F401

    sentences = " ".join(
        f"Sentence {i} discusses topic {i % 5} in considerable detail." for i in range(40)
    )
    corpus = Corpus([Record.build(title="T", abstract="A.", full_text=sentences)])
    embedder = HashEmbedder(dimension=256)

    runs = [
        [c.start for c in SemanticChunker(embedder, percentile=90, jitter=0.05, seed=s)
         .chunk_corpus(corpus)]
        for s in (1, 2, 3)
    ]
    assert len({tuple(run) for run in runs}) > 1


# ------------------------------------------------------------ build_chunker


def test_build_chunker_injects_the_pipeline_embedder():
    """Regression: the CLI and the HTTP service each constructed chunkers on
    their own and forgot that a model-dependent one needs the embedder — the
    CLI died with a bare TypeError at the prompt, the service with an
    unhandled 500. One factory, one rule."""
    from evidentia.chunking import build_chunker

    embedder = HashEmbedder(dimension=64)
    chunker = build_chunker("semantic", embedder=embedder, percentile=90)
    assert embedder.fingerprint in chunker.spec


def test_build_chunker_without_an_embedder_fails_with_guidance():
    from evidentia.chunking import build_chunker
    from evidentia.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="embedding model"):
        build_chunker("semantic")


def test_build_chunker_rejects_unknown_names():
    from evidentia.chunking import build_chunker
    from evidentia.exceptions import UnknownBackendError

    with pytest.raises(UnknownBackendError, match="telepathy"):
        build_chunker("telepathy")


def test_build_chunker_leaves_pure_chunkers_alone():
    from evidentia.chunking import build_chunker

    assert build_chunker("sentence", max_words=50).spec.startswith("sentence/")
