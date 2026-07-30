"""IMRaD classification of real-world headings.

Measured on a corpus of published information-systems papers, the original
exact-match table left 77% of headings unclassified, so `--section methods`
discarded most of the method text while appearing to work. These tests pin the
vocabulary and the two inheritance rules that fixed it.
"""

from __future__ import annotations

import pytest

from evidentia import Corpus, Record
from evidentia.chunking import TEISectionChunker, canonical_section
from evidentia.sections import (
    SECTION_LABELS,
    classify_record_sections,
    classify_section,
    classify_with_reason,
    section_report,
)

# Headings taken from the structure of real IS and management papers.
REAL_HEADINGS = [
    "1. Introduction",
    "2. Theoretical background",
    "2.1 Non-fungible tokens",
    "2.2 Hypothesis development",
    "3. Research methodology",
    "3.1 Sample and data collection",
    "3.2 Measurement",
    "3.3 Common method bias",
    "4. Results",
    "4.1 Measurement model",
    "4.2 Structural model",
    "5. Discussion",
    "5.1 Theoretical implications",
    "5.2 Managerial implications",
    "5.3 Limitations and future research",
    "6. Conclusion",
    "Acknowledgements",
    "Declaration of competing interest",
    "Appendix A. Survey items",
]


@pytest.fixture
def realistic_corpus():
    records = [
        Record.build(
            title=f"Paper {i} on token affordances",
            abstract="We examine platform affordances and loyalty.",
            doi=f"10.1/p{i:03d}",
            sections=[
                (heading, f"Body text for {heading}. It runs to several sentences of content.")
                for heading in REAL_HEADINGS
            ],
        )
        for i in range(6)
    ]
    return Corpus(records)


# ------------------------------------------------------------------ numbering


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("3.1 Sample and data collection", "methods"),
        ("3. Research methodology", "methods"),
        ("IV. Experimental setup", "methods"),
        ("(2) Data analysis", "methods"),
        ("2.1.3 Measurement model", "methods"),
        ("A. Questionnaire design", "methods"),
    ],
)
def test_numbering_is_stripped_before_matching(heading, expected):
    assert classify_section(heading) == expected


def test_trailing_punctuation_is_ignored():
    assert classify_section("Results:") == "results"
    assert classify_section("Conclusion —") == "conclusion"


# ----------------------------------------------------------------- vocabulary


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        # methods vocabulary that a naive table misses entirely
        ("Measurement model", "methods"),
        ("Common method bias", "methods"),
        ("Partial least squares analysis", "methods"),
        ("Participants and procedure", "methods"),
        ("Data collection", "methods"),
        ("Data analysis", "methods"),
        ("Instrument development", "methods"),
        ("Pilot study", "methods"),
        ("Reliability and validity", "methods"),
        ("Sampling strategy", "methods"),
        # background
        ("Theoretical background", "background"),
        ("Literature review", "background"),
        ("Hypothesis development", "background"),
        ("Research model", "background"),
        # results
        ("Structural model", "results"),
        ("Hypothesis testing", "results"),
        ("Descriptive statistics", "results"),
        ("Empirical results", "results"),
        # discussion
        ("Managerial implications", "discussion"),
        ("Limitations and future research", "discussion"),
        # front and back matter
        ("Acknowledgements", "other"),
        ("Declaration of competing interest", "other"),
        ("References", "references"),
        ("Appendix A", "appendix"),
    ],
)
def test_real_world_vocabulary(heading, expected):
    assert classify_section(heading) == expected


def test_specific_patterns_beat_general_ones():
    """Order is the design: "data analysis" is a method, "analysis of
    results" is a result, and a naive substring table gets both wrong."""
    assert classify_section("Data analysis") == "methods"
    assert classify_section("Analysis of results") == "results"
    assert classify_section("Results and discussion") == "results"


def test_every_label_is_declared():
    for heading in REAL_HEADINGS:
        assert classify_section(heading) in SECTION_LABELS


def test_classification_reports_its_rule():
    assert classify_with_reason("Materials and Methods").rule == "exact"
    assert classify_with_reason("3.1 Sample and data").rule.startswith("contains:")
    assert classify_with_reason("Zzzz qqqq").rule == "unmatched"


def test_empty_heading_is_other():
    assert classify_section("") == "other"
    assert classify_section("   ") == "other"


# ---------------------------------------------------------------- inheritance


def test_numbered_subsection_inherits_its_parent():
    """"2.1 Non-fungible tokens" is unclassifiable alone and obviously
    background under "2. Theoretical background"."""
    matches = classify_record_sections([
        ("2. Theoretical background", "x"),
        ("2.1 Non-fungible tokens", "y"),
    ])
    assert matches[1].label == "background"
    assert matches[1].rule == "inherited:2"


def test_unnumbered_subsection_continues_the_previous_section():
    matches = classify_record_sections([
        ("3. Research methodology", "x"),
        ("Credamo panel recruitment", "y"),
    ])
    assert matches[1].label == "methods"
    assert matches[1].rule == "inherited:previous"


def test_inheritance_never_overrides_an_explicit_match():
    """"4.1 Measurement model" stays methods under a "4. Results" parent: the
    heading states what it is more reliably than its position does."""
    matches = classify_record_sections([
        ("4. Results", "x"),
        ("4.1 Measurement model", "y"),
    ])
    assert matches[0].label == "results"
    assert matches[1].label == "methods"
    assert matches[1].rule.startswith("contains:")


def test_inheritance_skips_back_matter():
    matches = classify_record_sections([
        ("6. Conclusion", "x"),
        ("Appendix A", "y"),
        ("Zzzz qqqq", "z"),
    ])
    assert matches[2].label == "conclusion"  # not appendix


def test_first_heading_cannot_inherit():
    matches = classify_record_sections([("Zzzz qqqq", "x")])
    assert matches[0].label == "other"
    assert matches[0].rule == "unmatched"


# --------------------------------------------------------------------- report


def test_report_on_a_realistic_corpus_leaves_little_unmatched(realistic_corpus):
    """Regression against the 77% failure observed on real GROBID output."""
    report = section_report(realistic_corpus)
    assert report["total_headings"] == len(REAL_HEADINGS) * len(realistic_corpus)
    assert report["unmatched_fraction"] < 0.20
    assert report["unmatched_headings"] == {}
    assert report["labels"]["methods"] > 0


def test_report_lists_headings_no_rule_matched():
    corpus = Corpus([
        Record.build(title="P", abstract="a", sections=[("Qqqq zzzz", "body text here")])
    ])
    report = section_report(corpus)
    assert "qqqq zzzz" in report["unmatched_headings"]


# ------------------------------------------------------- chunker integration


def test_section_filter_captures_subsections(realistic_corpus):
    """The point of the whole exercise: a methods filter must pick up
    "3.1 Sample and data collection", not only "3. Research methodology"."""
    chunks = TEISectionChunker(keep_sections=("methods",)).chunk_corpus(realistic_corpus)
    per_record = len(chunks) / len(realistic_corpus)
    assert per_record >= 3  # methodology + sample + measurement + bias
    assert {chunk.section for chunk in chunks} == {"methods"}


def test_canonical_section_delegates_to_the_classifier():
    assert canonical_section("3.1 Data collection") == classify_section("3.1 Data collection")
