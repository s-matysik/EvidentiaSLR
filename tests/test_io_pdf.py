"""PDF ingestion: GROBID client, local fallback, metadata merge."""

import shutil

import pytest

from evidentia import Corpus, Record
from evidentia.exceptions import ConfigurationError, CorpusError
from evidentia.io import corpus_from_pdf_dir, merge_pdf_corpus
from evidentia.io.pdf import GrobidClient, LocalPdfExtractor, PdfIngestReport, file_digest

from .helpers import TEI_SAMPLE

# Smallest structurally valid PDF with an extractable text layer.
MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
    b"/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n"
    b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"5 0 obj<</Length 90>>stream\n"
    b"BT /F1 12 Tf 72 720 Td (Deep learning for EEG emotion recognition) Tj ET\n"
    b"endstream endobj\n"
    b"trailer<</Root 1 0 R>>\n%%EOF\n"
)


class FakeGrobid:
    """Stands in for a running GROBID service."""

    engine = "grobid"

    def __init__(self, tei: str = TEI_SAMPLE, fail_on: set[str] | None = None):
        self.tei = tei
        self.fail_on = fail_on or set()
        self.calls: list[str] = []

    def alive(self) -> bool:
        return True

    def process(self, path):
        self.calls.append(path.name)
        if path.name in self.fail_on:
            raise RuntimeError("GROBID returned 500")
        return self.tei, False


@pytest.fixture
def pdf_dir(tmp_path):
    directory = tmp_path / "pdf"
    directory.mkdir()
    for i in range(3):
        (directory / f"paper_{i:02d}.pdf").write_bytes(MINIMAL_PDF)
    return directory


# ------------------------------------------------------------------ grobid


def test_ingests_every_pdf_in_a_directory(pdf_dir):
    corpus, report = corpus_from_pdf_dir(pdf_dir, client=FakeGrobid())
    assert report.files_found == 3
    assert report.processed == 3
    assert report.engine == "grobid"
    # All three files carry identical TEI, so canonicalisation collapses them.
    assert len(corpus) == 1


def test_sections_are_recovered_from_tei(pdf_dir):
    corpus, report = corpus_from_pdf_dir(pdf_dir, client=FakeGrobid())
    assert report.with_sections == 3
    assert report.without_sections == 0
    assert corpus[0].sections


def test_files_are_processed_in_sorted_order(pdf_dir):
    client = FakeGrobid()
    corpus_from_pdf_dir(pdf_dir, client=client)
    assert client.calls == sorted(client.calls)


def test_limit_yields_a_stable_prefix(pdf_dir):
    client_a, client_b = FakeGrobid(), FakeGrobid()
    corpus_from_pdf_dir(pdf_dir, client=client_a, limit=2)
    corpus_from_pdf_dir(pdf_dir, client=client_b, limit=2)
    assert client_a.calls == client_b.calls == ["paper_00.pdf", "paper_01.pdf"]


def test_a_failing_file_does_not_abort_the_batch(pdf_dir):
    client = FakeGrobid(fail_on={"paper_01.pdf"})
    _, report = corpus_from_pdf_dir(pdf_dir, client=client)
    assert report.processed == 2
    assert report.failed == 1
    assert report.errors[0]["file"] == "paper_01.pdf"


def test_malformed_tei_fails_loudly_with_a_count(pdf_dir):
    """Per-file TEI errors are collected; a batch that yields nothing raises."""
    with pytest.raises(CorpusError, match="3 failed"):
        corpus_from_pdf_dir(pdf_dir, client=FakeGrobid(tei="<not-xml"))


def test_partially_malformed_tei_still_yields_a_corpus(pdf_dir):
    class Flaky(FakeGrobid):
        def process(self, path):
            self.calls.append(path.name)
            return ("<not-xml" if path.name == "paper_01.pdf" else self.tei), False

    corpus, report = corpus_from_pdf_dir(pdf_dir, client=Flaky())
    assert len(corpus) == 1
    assert report.processed == 2
    assert report.failed == 1


def test_require_sections_drops_unsegmented_records(pdf_dir):
    bare_tei = (
        '<?xml version="1.0"?><TEI xmlns="http://www.tei-c.org/ns/1.0">'
        "<teiHeader><fileDesc><titleStmt><title>Bare</title></titleStmt>"
        "<sourceDesc/></fileDesc>"
        "<profileDesc><abstract><p>Only an abstract here.</p></abstract></profileDesc>"
        "</teiHeader><text><body/></text></TEI>"
    )
    with pytest.raises(CorpusError):
        corpus_from_pdf_dir(pdf_dir, client=FakeGrobid(tei=bare_tei), require_sections=True)


def test_empty_directory_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CorpusError):
        corpus_from_pdf_dir(empty, client=FakeGrobid())


def test_missing_directory_raises(tmp_path):
    with pytest.raises(ConfigurationError):
        corpus_from_pdf_dir(tmp_path / "nope", client=FakeGrobid())


# ------------------------------------------------------------------- cache


def test_tei_is_cached_by_content_hash(pdf_dir, tmp_path):
    cache = tmp_path / "tei"
    client = GrobidClient(tei_cache=cache)
    digest = file_digest(pdf_dir / "paper_00.pdf")
    (cache / f"{digest}.tei.xml").write_text(TEI_SAMPLE, encoding="utf-8")

    tei, cached = client.process(pdf_dir / "paper_00.pdf")
    assert cached is True
    assert "EEG emotion recognition" in tei


def test_identical_files_share_a_cache_entry(pdf_dir):
    a = file_digest(pdf_dir / "paper_00.pdf")
    b = file_digest(pdf_dir / "paper_01.pdf")
    assert a == b  # same bytes, different names


# ------------------------------------------------------------- local engine


@pytest.mark.skipif(
    not shutil.which("pdftotext"), reason="poppler not installed"
)
def test_local_engine_extracts_text_but_no_sections(pdf_dir):
    corpus, report = corpus_from_pdf_dir(pdf_dir, client=LocalPdfExtractor())
    assert report.engine == "pdftotext"
    assert report.with_sections == 0
    assert report.without_sections == 3
    assert "no section structure" in report.summary()
    assert corpus[0].full_text


@pytest.mark.skipif(not shutil.which("pdftotext"), reason="poppler not installed")
def test_local_engine_recovers_a_title_from_the_text(pdf_dir):
    corpus, _ = corpus_from_pdf_dir(pdf_dir, client=LocalPdfExtractor())
    assert "EEG emotion recognition" in corpus[0].title


# ------------------------------------------------------------------- merge


def test_merge_prefers_metadata_and_takes_text_from_pdfs():
    metadata = Corpus([
        Record.build(title="EEG study", doi="10.1/a", year=2021, venue="J Neural Eng")
    ])
    pdfs = Corpus([
        Record.build(
            title="EEG study", doi="10.1/a", full_text="Full body text.",
            sections=[("Methods", "We recorded 32 channels.")], source="grobid",
        )
    ])
    merged, stats = merge_pdf_corpus(metadata, pdfs)

    assert stats["matched"] == 1
    assert stats["match_rate"] == 1.0
    record = merged[0]
    assert record.year == 2021           # from metadata
    assert record.venue == "J Neural Eng"  # from metadata
    assert record.sections                # from the PDF


def test_merge_falls_back_to_title_matching():
    metadata = Corpus([Record.build(title="A Study Of Things", year=2020)])
    pdfs = Corpus([Record.build(title="a study of things!", full_text="Body.")])
    _, stats = merge_pdf_corpus(metadata, pdfs)
    assert stats["matched"] == 1


def test_unmatched_metadata_records_are_kept():
    metadata = Corpus([
        Record.build(title="Has a PDF", doi="10.1/a"),
        Record.build(title="Has no PDF", doi="10.1/b"),
    ])
    pdfs = Corpus([Record.build(title="Has a PDF", doi="10.1/a", full_text="Body.")])
    merged, stats = merge_pdf_corpus(metadata, pdfs)

    assert len(merged) == 2
    assert stats["matched"] == 1
    assert stats["match_rate"] == 0.5


# ------------------------------------------------------------------- stats


def test_corpus_stats_recommends_a_chunker():
    abstracts = Corpus([Record.build(title="T", abstract="An abstract.")])
    assert abstracts.stats()["recommended_chunker"] == "record"

    full_text = Corpus([Record.build(title="T", full_text="A long body.")])
    assert full_text.stats()["recommended_chunker"] == "sentence"

    structured = Corpus([
        Record.build(title="T", full_text="Body.", sections=[("Methods", "We did things.")])
    ])
    assert structured.stats()["recommended_chunker"] == "tei-section"


def test_report_serialises_and_reports_coverage():
    report = PdfIngestReport(files_found=10, processed=7, engine="grobid")
    assert report.coverage == 0.7
    assert report.as_dict()["engine"] == "grobid"


# ------------------------------------------------------------ TEI recovery


def test_corpus_rebuilds_from_cached_tei(tmp_path):
    """The TEI cache is the durable artefact: a corpus must be reconstructible
    from it alone, with neither the source PDFs nor a running GROBID."""
    from evidentia.io import corpus_from_tei_dir

    cache = tmp_path / "tei"
    cache.mkdir()
    for i in range(3):
        tei = TEI_SAMPLE.replace(
            "Deep learning for EEG emotion recognition", f"Study {i:02d} on screening"
        ).replace("10.1000/ABC.123", f"10.1000/abc.{i:03d}")
        (cache / f"digest{i}.tei.xml").write_text(tei, encoding="utf-8")

    corpus, report = corpus_from_tei_dir(cache)

    assert len(corpus) == 3
    assert report.engine == "tei-cache"
    assert report.from_cache == 3
    assert report.with_sections == 3
    assert "TEI files" in report.summary()
    assert all(record.sections for record in corpus)


def test_tei_recovery_matches_the_original_ingestion(tmp_path, pdf_dir):
    """Recovering from cache must give the same corpus hash as ingesting the
    PDFs did, or the cache is not a faithful record of the run."""
    from evidentia.io import corpus_from_tei_dir

    cache = tmp_path / "tei"
    client = GrobidClient(tei_cache=cache)
    original, _ = corpus_from_pdf_dir(pdf_dir, client=FakeGrobid())

    # FakeGrobid does not write the cache, so populate it as the real client would.
    for path in sorted(pdf_dir.glob("*.pdf")):
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{file_digest(path)}.tei.xml").write_text(TEI_SAMPLE, encoding="utf-8")

    recovered, _ = corpus_from_tei_dir(cache)
    assert recovered.corpus_hash == original.corpus_hash
    assert client.tei_cache == cache


def test_tei_recovery_reports_malformed_files(tmp_path):
    from evidentia.io import corpus_from_tei_dir

    cache = tmp_path / "tei"
    cache.mkdir()
    (cache / "good.tei.xml").write_text(TEI_SAMPLE, encoding="utf-8")
    (cache / "broken.tei.xml").write_text("<not-xml", encoding="utf-8")

    corpus, report = corpus_from_tei_dir(cache)
    assert len(corpus) == 1
    assert report.failed == 1
    assert report.errors[0]["file"] == "broken.tei.xml"


def test_tei_recovery_rejects_an_empty_directory(tmp_path):
    from evidentia.io import corpus_from_tei_dir

    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(CorpusError):
        corpus_from_tei_dir(empty)
