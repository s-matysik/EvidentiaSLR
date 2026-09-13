"""LitRev ingestion and protocol handling."""

from evidentiaslr.litrev import LitRevReview, corpus_from_source_files, partition_source_files


def test_corpus_from_source_files_uses_tei_and_skips_failures(tei):
    rows = [
        {"extraction_status": "extracted", "extracted_text": "flat", "extracted_metadata": {"tei": tei}},
        {"extraction_status": "failed", "extracted_text": "", "extracted_metadata": {}},
        {"extraction_status": "pending", "extracted_text": "x", "extracted_metadata": {}},
    ]
    corpus = corpus_from_source_files(rows)
    assert len(corpus) == 1
    assert corpus[0].title == "Deep learning for EEG emotion recognition"


def test_rows_without_usable_content_are_dropped():
    rows = [{"extraction_status": "extracted", "extracted_text": "", "extracted_metadata": {}}]
    assert len(corpus_from_source_files(rows)) == 0


def test_partition_reports_coverage():
    rows = [
        {"extraction_status": "extracted"},
        {"extraction_status": "extracted"},
        {"extraction_status": "failed"},
        {},
    ]
    buckets = partition_source_files(rows)
    assert len(buckets["extracted"]) == 2
    assert len(buckets["failed"]) == 1
    assert len(buckets["unknown"]) == 1


def test_review_builds_a_pico_query():
    review = LitRevReview.from_payload({
        "id": 1,
        "title": "Review",
        "research_question": "Does AI help screening?",
        "population": "researchers",
        "intervention": "embedding based screening",
        "outcome": "recall",
        "extraction_fields": ["sample size", "model"],
    })
    assert "embedding based screening" in review.pico_query()
    assert len(review.extraction_queries()) == 2


def test_review_tolerates_a_sparse_protocol():
    review = LitRevReview.from_payload({"id": 2, "title": "Empty"})
    assert review.pico_query() == ""
    assert review.extraction_queries() == []
