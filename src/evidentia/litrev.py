"""LitRev integration.

Evidentia plugs into LitRev at the point where its pipeline currently stops.
LitRev already ingests PDFs into `folder_items`, extracts them through GROBID
in the `extractor` service and stores TEI in
`review_source_files.extracted_metadata->tei`. Everything after that — the
data extraction and synthesis stages — is unimplemented.

This module reads that existing state and turns it into a canonical corpus,
so no second extraction pass and no schema change are required to start.

Two entry points:

`LitRevClient`
    Talks to the existing REST API with a Sanctum token.

`corpus_from_source_files`
    Pure function over already-fetched rows, so the ingestion path is
    testable without a running LitRev instance.

The recommended deployment mirrors `extractor`: a sibling container exposing
`evidentia.service:app`, with a Laravel job posting to it exactly as
`ExtractReviewSourceFileJob` posts to the extractor today.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .corpus import Corpus, Record
from .tei import record_from_extracted


@dataclass
class LitRevReview:
    """The protocol fields LitRev stores on a review."""

    id: int
    title: str
    research_question: str = ""
    population: str = ""
    intervention: str = ""
    comparison: str = ""
    outcome: str = ""
    research_questions: tuple[str, ...] = ()
    inclusion_criteria: tuple[str, ...] = ()
    exclusion_criteria: tuple[str, ...] = ()
    extraction_fields: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: dict) -> LitRevReview:
        def tup(key: str) -> tuple[str, ...]:
            value = payload.get(key) or []
            return tuple(str(v) for v in value)

        return cls(
            id=int(payload["id"]),
            title=payload.get("title") or "",
            research_question=payload.get("research_question") or "",
            population=payload.get("population") or "",
            intervention=payload.get("intervention") or "",
            comparison=payload.get("comparison") or "",
            outcome=payload.get("outcome") or "",
            research_questions=tup("research_questions"),
            inclusion_criteria=tup("inclusion_criteria"),
            exclusion_criteria=tup("exclusion_criteria"),
            extraction_fields=tup("extraction_fields"),
            keywords=tup("keywords"),
        )

    def pico_query(self) -> str:
        """Build a retrieval query from the PICO fields the protocol declares.

        Using the stored protocol rather than a free-text prompt keeps the
        query auditable: it is derived from a versioned record that LitRev's
        audit log already tracks.
        """
        parts = [
            self.research_question,
            self.population,
            self.intervention,
            self.comparison,
            self.outcome,
        ]
        parts.extend(self.research_questions)
        return " ".join(part for part in parts if part).strip()

    def extraction_queries(self) -> list[tuple[str, str]]:
        """One query per declared extraction field."""
        base = self.pico_query()
        return [(field, f"{field}. {base}".strip()) for field in self.extraction_fields]


def corpus_from_source_files(rows: Iterable[dict]) -> Corpus:
    """Build a canonical corpus from `review_source_files` rows.

    Only rows whose extraction succeeded are used; the rest are reported by
    the caller rather than silently dropped, because a review that indexed 40
    of 50 PDFs and did not say so is not reproducible either.
    """
    records: list[Record] = []
    for row in rows:
        if row.get("extraction_status") != "extracted":
            continue
        record = record_from_extracted(
            row.get("extracted_text") or "",
            row.get("extracted_metadata") or {},
        )
        if not record.full_text and not record.abstract:
            continue
        records.append(record)
    return Corpus(records)


def partition_source_files(rows: Sequence[dict]) -> dict[str, list[dict]]:
    """Group rows by extraction status so the caller can report coverage."""
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row.get("extraction_status") or "unknown", []).append(row)
    return buckets


class LitRevClient:
    """Minimal client for the LitRev REST API."""

    def __init__(self, base_url: str, token: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        import httpx

        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                f"{self.base_url}{path}", headers=self._headers(), params=params or {}
            )
            response.raise_for_status()
            return response.json()

    def review(self, review_id: int) -> LitRevReview:
        return LitRevReview.from_payload(self._get(f"/api/reviews/{review_id}")["data"])

    def source_files(self, review_id: int) -> list[dict]:
        payload = self._get(f"/api/reviews/{review_id}/source-files")
        return payload.get("data", [])

    def corpus(self, review_id: int) -> Corpus:
        return corpus_from_source_files(self.source_files(review_id))
