"""Section classification.

Mapping a paper's headings onto IMRaD is the step that decides whether
section-filtered retrieval works. A narrow alias table looks fine on synthetic
records and collapses on real ones: measured against a corpus of published
information-systems papers, an exact-match table left 77% of headings
unclassified, so a filter on `methods` discarded most of the method text.

Real headings are numbered ("3.1 Sample and data collection"), compound
("Results and discussion"), and domain-specific ("Measurement model",
"Common method bias", "Partial least squares analysis"). The classifier below
handles all three, and `classify_with_reason` reports *why* a heading was
mapped, so a corpus can be audited rather than trusted.

Ordering is significant. Patterns are checked most-specific first, because
"data analysis" is a method and "analysis of results" is a result, and a
naive substring table gets both wrong.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .corpus import normalise_text

__all__ = [
    "SECTION_LABELS",
    "SectionMatch",
    "classify_section",
    "classify_record_sections",
    "classify_with_reason",
    "section_report",
]

#: The canonical labels. `other` is the explicit failure case, not a category.
SECTION_LABELS = (
    "abstract",
    "introduction",
    "background",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "references",
    "appendix",
    "other",
)

# Leading numbering: "3", "3.1", "3.1.2", "IV.", "A.", "(2)"
_NUMBERING = re.compile(r"^\s*[(\[]?\s*(?:\d+(?:\.\d+)*|[ivxlcdm]+|[a-z])\s*[.)\]:]?\s+", re.I)
_TRAILING_PUNCT = re.compile(r"[\s.:;,\-–—]+$")


def _normalise_heading(heading: str) -> str:
    text = normalise_text(heading).lower()
    # Strip numbering repeatedly: "3.1. 2 Data collection" happens.
    for _ in range(3):
        stripped = _NUMBERING.sub("", text)
        if stripped == text:
            break
        text = stripped
    return _TRAILING_PUNCT.sub("", text).strip()


# Exact matches take absolute priority.
_EXACT: dict[str, str] = {
    "abstract": "abstract",
    "summary": "abstract",
    "introduction": "introduction",
    "background": "background",
    "related work": "background",
    "literature review": "background",
    "theoretical background": "background",
    "theory": "background",
    "methods": "methods",
    "method": "methods",
    "methodology": "methods",
    "materials and methods": "methods",
    "research methodology": "methods",
    "research method": "methods",
    "research design": "methods",
    "results": "results",
    "findings": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "references": "references",
    "bibliography": "references",
    "appendix": "appendix",
}

# Ordered, most specific first. A heading is assigned the first pattern it
# contains. Order is the whole design: "data analysis" must be tested before
# "analysis", and "analysis of results" before both.
_PATTERNS: tuple[tuple[str, str], ...] = (
    # results, when the wording would otherwise read as method
    ("analysis of results", "results"),
    ("results and discussion", "results"),
    ("empirical results", "results"),
    ("descriptive statistics", "results"),
    ("hypothesis testing", "results"),
    ("hypotheses testing", "results"),
    ("structural model", "results"),
    ("model fit", "results"),
    ("path coefficient", "results"),
    # methods — sampling and participants
    ("data collection", "methods"),
    ("data gathering", "methods"),
    ("data analysis", "methods"),
    ("analytical approach", "methods"),
    ("analysis strategy", "methods"),
    ("sample and", "methods"),
    ("sampling", "methods"),
    ("sample", "methods"),
    ("participant", "methods"),
    ("respondent", "methods"),
    ("procedure", "methods"),
    ("protocol", "methods"),
    ("experimental setup", "methods"),
    ("experimental design", "methods"),
    ("study design", "methods"),
    ("research design", "methods"),
    ("research context", "methods"),
    ("research approach", "methods"),
    ("research setting", "methods"),
    # methods — instruments and psychometrics
    ("measurement model", "methods"),
    ("measurement", "methods"),
    ("instrument", "methods"),
    ("questionnaire", "methods"),
    ("survey design", "methods"),
    ("scale development", "methods"),
    ("construct", "methods"),
    ("operationalis", "methods"),
    ("operationaliz", "methods"),
    ("reliability", "methods"),
    ("validity", "methods"),
    ("common method bias", "methods"),
    ("method bias", "methods"),
    ("manipulation check", "methods"),
    ("pilot study", "methods"),
    ("pre-test", "methods"),
    ("pretest", "methods"),
    # methods — analytical techniques
    ("partial least squares", "methods"),
    ("structural equation", "methods"),
    ("pls-sem", "methods"),
    ("regression", "methods"),
    ("factor analysis", "methods"),
    ("statistical", "methods"),
    ("estimation", "methods"),
    ("simulation setup", "methods"),
    ("implementation", "methods"),
    ("algorithm", "methods"),
    ("dataset", "methods"),
    ("corpus", "methods"),
    ("evaluation metric", "methods"),
    ("evaluation protocol", "methods"),
    ("experiment", "methods"),
    ("materials", "methods"),
    ("method", "methods"),
    # background and theory
    ("literature", "background"),
    ("related work", "background"),
    ("prior research", "background"),
    ("theoretical", "background"),
    ("theory", "background"),
    ("conceptual framework", "background"),
    ("research model", "background"),
    ("hypothesis development", "background"),
    ("hypotheses development", "background"),
    ("hypothes", "background"),
    ("propositions", "background"),
    ("research question", "background"),
    # results
    ("result", "results"),
    ("finding", "results"),
    ("evaluation", "results"),
    ("performance", "results"),
    # discussion and closing
    ("theoretical implication", "discussion"),
    ("practical implication", "discussion"),
    ("managerial implication", "discussion"),
    ("implication", "discussion"),
    ("limitation", "discussion"),
    ("future research", "discussion"),
    ("future work", "discussion"),
    ("discussion", "discussion"),
    ("concluding", "conclusion"),
    ("conclusion", "conclusion"),
    # front and back matter
    ("abstract", "abstract"),
    ("introduction", "introduction"),
    ("motivation", "introduction"),
    ("overview", "introduction"),
    ("reference", "references"),
    ("bibliograph", "references"),
    ("appendix", "appendix"),
    ("acknowledg", "other"),
    ("funding", "other"),
    ("declaration", "other"),
    ("conflict of interest", "other"),
    ("author contribution", "other"),
)


@dataclass(frozen=True)
class SectionMatch:
    heading: str
    normalised: str
    label: str
    rule: str

    def as_dict(self) -> dict:
        return {
            "heading": self.heading,
            "normalised": self.normalised,
            "label": self.label,
            "rule": self.rule,
        }


def classify_with_reason(heading: str) -> SectionMatch:
    """Classify a heading and report which rule fired.

    The rule matters when auditing a corpus: a heading landing in `methods`
    because it contains the word "sample" is a weaker assignment than one that
    matched "materials and methods" exactly, and a reviewer is entitled to see
    the difference.
    """
    normalised = _normalise_heading(heading)

    if not normalised:
        return SectionMatch(heading, normalised, "other", "empty")

    if normalised in _EXACT:
        return SectionMatch(heading, normalised, _EXACT[normalised], "exact")

    for pattern, label in _PATTERNS:
        if pattern in normalised:
            return SectionMatch(heading, normalised, label, f"contains:{pattern}")

    return SectionMatch(heading, normalised, "other", "unmatched")


def classify_section(heading: str) -> str:
    """Canonical IMRaD label for a heading, or `other`."""
    return classify_with_reason(heading).label


_NUMBER_PREFIX = re.compile(r"^\s*[(\[]?\s*(\d+(?:\.\d+)*)")


def _numbering(heading: str) -> tuple[int, ...]:
    """Section number as a tuple, e.g. "3.1.2 Data" -> (3, 1, 2)."""
    match = _NUMBER_PREFIX.match(normalise_text(heading))
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split(".") if part.isdigit())


def classify_record_sections(
    sections: Sequence[tuple[str, str]],
) -> list[SectionMatch]:
    """Classify a record's headings using their order and numbering.

    A heading in isolation is often unclassifiable: "Non-fungible tokens"
    could be anything. In context it is subsection 2.1 under "2. Theoretical
    background", and therefore background. Two inheritance rules recover that:

    1. A numbered heading that matched no rule inherits from the nearest
       preceding heading whose number is its prefix — 2.1 inherits from 2.
    2. An unnumbered heading that matched no rule continues the previous
       classified heading, because an unlabelled subsection belongs to the
       section it follows.

    Inheritance never overrides a rule that fired: an explicit "4.1 Measurement
    model" stays `methods` even under a "4. Results" parent, since the heading
    says what it is more reliably than its position does.
    """
    matches = [classify_with_reason(heading) for heading, _ in sections]
    numbers = [_numbering(heading) for heading, _ in sections]

    resolved: list[SectionMatch] = []
    for position, match in enumerate(matches):
        if match.rule != "unmatched":
            resolved.append(match)
            continue

        inherited: SectionMatch | None = None
        number = numbers[position]

        if len(number) > 1:
            parent = number[:-1]
            for earlier in range(position - 1, -1, -1):
                if numbers[earlier] == parent and resolved[earlier].label != "other":
                    inherited = SectionMatch(
                        match.heading, match.normalised, resolved[earlier].label,
                        f"inherited:{'.'.join(map(str, parent))}",
                    )
                    break

        if inherited is None:
            for earlier in range(position - 1, -1, -1):
                if resolved[earlier].label not in {"other", "references", "appendix"}:
                    inherited = SectionMatch(
                        match.heading, match.normalised, resolved[earlier].label,
                        "inherited:previous",
                    )
                    break

        resolved.append(inherited or match)

    return resolved


def section_report(corpus) -> dict:
    """Audit how a corpus's headings were classified.

    Returns the label distribution plus the headings that fell through, which
    is the actionable part: an unmatched heading is method text that a
    `--section methods` filter will silently discard.
    """
    counts: dict[str, int] = {}
    unmatched: dict[str, int] = {}
    rules: dict[str, int] = {}

    for record in corpus:
        for match in classify_record_sections(record.sections):
            counts[match.label] = counts.get(match.label, 0) + 1
            rules[match.rule.split(":")[0]] = rules.get(match.rule.split(":")[0], 0) + 1
            if match.rule == "unmatched":
                unmatched[match.normalised] = unmatched.get(match.normalised, 0) + 1

    total = sum(counts.values())
    return {
        "total_headings": total,
        "labels": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "rules": dict(sorted(rules.items(), key=lambda kv: -kv[1])),
        "unmatched_fraction": (counts.get("other", 0) / total) if total else 0.0,
        "unmatched_headings": dict(
            sorted(unmatched.items(), key=lambda kv: -kv[1])[:50]
        ),
    }
