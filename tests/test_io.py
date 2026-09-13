"""Bibliographic import: Scopus/WoS CSV, RIS, BibTeX."""

import pytest

from evidentiaslr.exceptions import CorpusError, UnknownBackendError
from evidentiaslr.io import load_corpus, load_scopus_csv, parse_bibtex, parse_ris

SCOPUS_CSV = (
    "Authors,Title,Year,Source title,DOI,Abstract,Author Keywords,Document Type,Cited by,EID\n"
    '"Kowalska A., Nowak B.",EEG emotion recognition,2021,J Neural Eng,10.1000/ABC.123,'
    '"We study convolutional networks.","eeg; deep learning",Article,12,2-s2.0-1\n'
    '"Smith J.",Book chapter on screening,2019,Handbook,10.1000/def,'
    '"[No abstract available]",,Book chapter,0,2-s2.0-2\n'
    '"Lee K.",Automated screening review,2022,Syst Rev,,'
    '"A review of screening automation.","prisma",Review,5,2-s2.0-3\n'
)

RIS = """TY  - JOUR
TI  - EEG emotion recognition
AU  - Kowalska, Anna
AU  - Nowak, Bartosz
PY  - 2021
DO  - 10.1000/abc.123
JO  - Journal of Neural Engineering
AB  - We study convolutional networks
      on EEG signals.
KW  - eeg
KW  - deep learning
ER  -

TY  - CONF
TI  - Screening automation at scale
AU  - Lee, Kim
PY  - 2022
AB  - A conference paper about screening.
ER  -
"""

BIBTEX = """
@article{kowalska2021,
  title = {{EEG} emotion recognition},
  author = {Kowalska, Anna and Nowak, Bartosz},
  year = {2021},
  journal = {Journal of Neural Engineering},
  doi = {10.1000/abc.123},
  abstract = {We study convolutional networks on EEG signals.},
  keywords = {eeg, deep learning}
}

@inproceedings{lee2022,
  title = "Screening automation at scale",
  author = "Lee, Kim",
  year = "2022",
  booktitle = "Proceedings of Something",
  abstract = "A conference paper about screening."
}
"""


@pytest.fixture
def scopus_file(tmp_path):
    path = tmp_path / "scopus.csv"
    path.write_text(SCOPUS_CSV, encoding="utf-8")
    return path


# ------------------------------------------------------------------ scopus


def test_scopus_loads_all_rows(scopus_file):
    corpus, report = load_scopus_csv(scopus_file)
    assert len(corpus) == 3
    assert report.rows_read == 3
    assert report.unmapped_fields == []


def test_scopus_placeholder_abstract_becomes_empty(scopus_file):
    corpus, report = load_scopus_csv(scopus_file)
    chapter = next(r for r in corpus if "Book chapter" in r.title)
    assert chapter.abstract == ""
    assert report.missing_abstract == 1


def test_scopus_counts_missing_doi(scopus_file):
    _, report = load_scopus_csv(scopus_file)
    assert report.missing_doi == 1


def test_scopus_normalises_doi_case(scopus_file):
    corpus, _ = load_scopus_csv(scopus_file)
    assert any(r.doi == "10.1000/abc.123" for r in corpus)


def test_scopus_document_type_filter(scopus_file):
    corpus, _ = load_scopus_csv(scopus_file, keep_types=("Article", "Review"))
    assert len(corpus) == 2


def test_scopus_require_abstract_drops_placeholder_rows(scopus_file):
    corpus, _ = load_scopus_csv(scopus_file, require_abstract=True)
    assert len(corpus) == 2


def test_scopus_reports_document_type_counts(scopus_file):
    _, report = load_scopus_csv(scopus_file)
    assert report.document_types == {"Article": 1, "Book chapter": 1, "Review": 1}


def test_scopus_authors_are_split_and_sorted(scopus_file):
    corpus, _ = load_scopus_csv(scopus_file)
    record = next(r for r in corpus if r.doi == "10.1000/abc.123")
    assert record.authors == ("Kowalska A.", "Nowak B.")


def test_unmapped_columns_produce_an_actionable_error(tmp_path):
    """A renamed column must fail loudly and name the columns it looked for."""
    path = tmp_path / "weird.csv"
    path.write_text("Headline,Summary\nA title,A summary\n", encoding="utf-8")
    with pytest.raises(CorpusError) as exc:
        load_scopus_csv(path)
    message = str(exc.value)
    assert "title" in message and "abstract" in message
    assert "Headline" in message  # the actual header is echoed back


def test_optional_columns_are_not_reported_as_unmapped(scopus_file):
    """Index Keywords is routinely absent; that is not a mapping failure."""
    _, report = load_scopus_csv(scopus_file)
    assert "index_keywords" not in report.unmapped_fields


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(CorpusError):
        load_scopus_csv(path)


def test_scopus_bom_is_stripped(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_text("\ufeff" + SCOPUS_CSV, encoding="utf-8")
    corpus, report = load_scopus_csv(path)
    assert report.unmapped_fields == []
    assert len(corpus) == 3


# --------------------------------------------------------------------- ris


def test_ris_parses_entries_and_continuations():
    records, report = parse_ris(RIS)
    assert report.rows_read == 2
    article = records[0]
    assert article.title == "EEG emotion recognition"
    assert "on EEG signals" in article.abstract
    assert article.authors == ("Kowalska, Anna", "Nowak, Bartosz")
    assert article.year == 2021


def test_ris_tolerates_a_missing_closing_tag():
    records, _ = parse_ris("TY  - JOUR\nTI  - First\nAB  - One.\nTY  - JOUR\nTI  - Second\nAB  - Two.\n")
    assert len(records) == 2


def test_ris_require_abstract():
    records, _ = parse_ris("TY  - JOUR\nTI  - No abstract here\nER  -\n", require_abstract=True)
    assert records == []


# ------------------------------------------------------------------ bibtex


def test_bibtex_parses_both_quoting_styles():
    records, report = parse_bibtex(BIBTEX)
    assert report.rows_read == 2
    assert {r.title for r in records} == {
        "EEG emotion recognition",
        "Screening automation at scale",
    }


def test_bibtex_strips_protective_braces():
    records, _ = parse_bibtex(BIBTEX)
    assert records[0].title == "EEG emotion recognition"


def test_bibtex_splits_authors_on_and():
    records, _ = parse_bibtex(BIBTEX)
    assert records[0].authors == ("Kowalska, Anna", "Nowak, Bartosz")


# ---------------------------------------------------------------- dispatch


def test_load_corpus_infers_format(tmp_path, scopus_file):
    corpus, _ = load_corpus(scopus_file)
    assert len(corpus) == 3

    ris_path = tmp_path / "refs.ris"
    ris_path.write_text(RIS, encoding="utf-8")
    assert len(load_corpus(ris_path)[0]) == 2

    bib_path = tmp_path / "refs.bib"
    bib_path.write_text(BIBTEX, encoding="utf-8")
    assert len(load_corpus(bib_path)[0]) == 2


def test_load_corpus_rejects_unknown_extension(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("nothing", encoding="utf-8")
    with pytest.raises(UnknownBackendError):
        load_corpus(path)


def test_imported_corpus_hash_is_stable(scopus_file):
    a, _ = load_scopus_csv(scopus_file)
    b, _ = load_scopus_csv(scopus_file)
    assert a.corpus_hash == b.corpus_hash


def test_report_serialises(scopus_file):
    _, report = load_scopus_csv(scopus_file)
    payload = report.as_dict()
    assert payload["records_built"] == 3
    assert "document_types" in payload
