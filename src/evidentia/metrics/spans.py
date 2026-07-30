"""Chunker-agnostic agreement.

Comparing two evidence sets by chunk identifier only works when both came from
the same chunker: the identifier hashes the chunker spec, so a sentence chunk
and a section chunk covering the very same words are, by construction,
different objects. Asking "did these two configurations retrieve the same
evidence?" across chunkers therefore needs a different unit.

Two are provided, at different resolutions:

`record_agreement`
    Which *papers* did each configuration surface? Coarse, but it is the
    question a reviewer actually asks, and it is defined even for chunks that
    could not be located in the source text.

`span_agreement`
    Which *characters* of the source did each configuration put in front of
    the reader? Jaccard over covered character positions, so a 200-word window
    and the three sentences inside it score as near-identical rather than as
    complete disagreement.

Span agreement is the one that makes a chunker comparison meaningful. Without
it, changing the chunker guarantees zero overlap and the experiment measures
nothing but its own construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .agreement import jaccard

if TYPE_CHECKING:  # pragma: no cover
    from ..retrieve import EvidenceSet

__all__ = [
    "SpanAgreementResult",
    "covered_spans",
    "merge_spans",
    "record_agreement",
    "span_agreement",
    "total_covered",
]

Span = tuple[int, int]


def merge_spans(spans: Iterable[Span]) -> list[Span]:
    """Collapse overlapping intervals so covered length is not double counted."""
    ordered = sorted(span for span in spans if span[1] > span[0])
    if not ordered:
        return []

    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def covered_spans(evidence: EvidenceSet) -> dict[str, list[Span]]:
    """Merged character intervals covered per record.

    Chunks that could not be located are skipped; use `located_fraction` to
    judge how much of the comparison rests on exact positions.
    """
    by_record: dict[str, list[Span]] = {}
    for chunk in evidence.chunks:
        if chunk.start < 0 or chunk.end <= chunk.start:
            continue
        by_record.setdefault(chunk.record_id, []).append((chunk.start, chunk.end))
    return {record: merge_spans(spans) for record, spans in by_record.items()}


def total_covered(spans_by_record: dict[str, list[Span]]) -> int:
    return sum(end - start for spans in spans_by_record.values() for start, end in spans)


def _intersect(a: Sequence[Span], b: Sequence[Span]) -> int:
    """Length of the intersection of two merged, sorted interval lists."""
    total = 0
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


@dataclass
class SpanAgreementResult:
    span_jaccard: float
    record_jaccard: float
    covered_a: int
    covered_b: int
    covered_both: int
    located_fraction_a: float
    located_fraction_b: float

    def as_dict(self) -> dict:
        return {
            "span_jaccard": self.span_jaccard,
            "record_jaccard": self.record_jaccard,
            "covered_a": self.covered_a,
            "covered_b": self.covered_b,
            "covered_both": self.covered_both,
            "located_fraction_a": self.located_fraction_a,
            "located_fraction_b": self.located_fraction_b,
        }


def _located_fraction(evidence: EvidenceSet) -> float:
    if not evidence.chunks:
        return 1.0
    located = sum(1 for chunk in evidence.chunks if chunk.start >= 0 and chunk.end > chunk.start)
    return located / len(evidence.chunks)


def record_agreement(a: EvidenceSet, b: EvidenceSet) -> float:
    """Jaccard over the records each configuration surfaced."""
    return jaccard(a.record_ids, b.record_ids)


def span_agreement(a: EvidenceSet, b: EvidenceSet) -> SpanAgreementResult:
    """Compare two evidence sets by the source text they actually cover.

    Defined across chunkers, indexes, embedders and k, because character
    positions in the record are the one thing all of those share.
    """
    spans_a = covered_spans(a)
    spans_b = covered_spans(b)

    covered_a = total_covered(spans_a)
    covered_b = total_covered(spans_b)

    both = sum(
        _intersect(spans_a[record], spans_b[record])
        for record in set(spans_a) & set(spans_b)
    )
    union = covered_a + covered_b - both

    return SpanAgreementResult(
        span_jaccard=both / union if union else 1.0,
        record_jaccard=record_agreement(a, b),
        covered_a=covered_a,
        covered_b=covered_b,
        covered_both=both,
        located_fraction_a=_located_fraction(a),
        located_fraction_b=_located_fraction(b),
    )
