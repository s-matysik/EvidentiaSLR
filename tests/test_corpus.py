"""Canonicalisation, identity and hashing."""

from evidentia import Corpus, Record
from evidentia.corpus import normalise_doi, normalise_text


def test_normalisation_collapses_cosmetic_differences():
    assert normalise_text("  Deep   learning\n\nfor  EEG ") == "Deep learning for EEG"
    assert normalise_doi("https://doi.org/10.1000/ABC.123") == "10.1000/abc.123"
    assert normalise_doi("doi:10.1000/abc.123") == "10.1000/abc.123"


def test_record_id_is_invariant_to_author_order_and_whitespace():
    a = Record.build(title="A Title", abstract="Body", authors=["Zoe X", "Adam Y"])
    b = Record.build(title="A  Title ", abstract=" Body", authors=["Adam Y", "Zoe X"])
    assert a.record_id == b.record_id


def test_corpus_hash_is_independent_of_input_order(corpus_factory):
    records = list(corpus_factory(12))
    assert Corpus(records).corpus_hash == Corpus(list(reversed(records))).corpus_hash


def test_corpus_deduplicates_on_doi():
    corpus = Corpus([
        Record.build(title="One", doi="10.1/x", abstract="a"),
        Record.build(title="One but reformatted", doi="https://doi.org/10.1/X", abstract="b"),
    ])
    assert len(corpus) == 1
    assert len(corpus.duplicates) == 1


def test_corpus_deduplicates_on_title_and_year_without_doi():
    corpus = Corpus([
        Record.build(title="Same Study", year=2020, abstract="a"),
        Record.build(title="same  study!", year=2020, abstract="b"),
    ])
    assert len(corpus) == 1


def test_records_differing_in_year_are_kept_apart():
    corpus = Corpus([
        Record.build(title="Same Study", year=2020),
        Record.build(title="Same Study", year=2021),
    ])
    assert len(corpus) == 2


def test_corpus_roundtrips_through_jsonl(tmp_path, synthetic_corpus):
    path = tmp_path / "corpus.jsonl"
    synthetic_corpus.to_jsonl(path)
    assert Corpus.from_jsonl(path).corpus_hash == synthetic_corpus.corpus_hash


def test_by_id_finds_a_record(synthetic_corpus):
    first = synthetic_corpus[0]
    assert synthetic_corpus.by_id(first.record_id) is first
    assert synthetic_corpus.by_id("nope") is None


def test_screening_text_joins_title_and_abstract():
    record = Record.build(title="Title", abstract="Abstract body")
    assert record.screening_text() == "Title. Abstract body"


def test_duplicate_resolution_is_independent_of_input_order():
    """Two exports of one work can disagree; the survivor must not depend on
    which file was read first, or the corpus hash stops being a function of
    content."""
    sparse = Record.build(title="Same Study", year=2020)
    rich = Record.build(title="Same Study", year=2020, abstract="A full abstract.")

    forward = Corpus([sparse, rich])
    backward = Corpus([rich, sparse])

    assert forward.corpus_hash == backward.corpus_hash
    assert len(forward) == 1
    assert forward[0].abstract == "A full abstract."  # the richer record wins


def test_duplicate_resolution_ties_break_on_record_id():
    a = Record.build(title="Tie", year=2020, abstract="aaaa")
    b = Record.build(title="Tie", year=2020, abstract="bbbb")
    expected = min([a, b], key=lambda r: r.record_id).record_id
    assert Corpus([a, b])[0].record_id == expected
    assert Corpus([b, a])[0].record_id == expected


def test_all_discarded_duplicates_are_reported():
    records = [
        Record.build(title="Same", year=2020, abstract="x" * i) for i in range(1, 5)
    ]
    corpus = Corpus(records)
    assert len(corpus) == 1
    assert len(corpus.duplicates) == 3
