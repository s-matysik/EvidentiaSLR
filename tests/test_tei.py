"""GROBID TEI parsing."""

from evidentiaslr.chunking import canonical_section
from evidentiaslr.tei import parse_tei, record_from_extracted


def test_parse_tei_recovers_metadata_and_sections(tei):
    record = parse_tei(tei)
    assert record.title == "Deep learning for EEG emotion recognition"
    assert record.doi == "10.1000/abc.123"
    assert record.year == 2021
    assert record.authors == ("Anna Kowalska",)
    assert "convolutional networks" in record.abstract
    assert any(canonical_section(name) == "methods" for name, _ in record.sections)


def test_record_from_extracted_prefers_tei(tei):
    record = record_from_extracted("flattened text", {"tei": tei})
    assert record.title == "Deep learning for EEG emotion recognition"
    assert record.sections


def test_record_from_extracted_falls_back_on_broken_tei():
    record = record_from_extracted("flattened text", {"tei": "<not-xml"})
    assert record.full_text == "flattened text"
    assert record.source == "litrev-text"


def test_record_from_extracted_handles_missing_metadata():
    record = record_from_extracted("body only", None)
    assert record.full_text == "body only"
