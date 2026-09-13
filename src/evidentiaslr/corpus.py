"""Canonical corpus representation.

A reproducibility claim is only meaningful against a corpus that is itself
identified unambiguously. Two exports of the same Scopus query differ in
whitespace, unicode composition, author order and DOI casing while describing
the same literature. Canonicalisation collapses those differences, and the
corpus hash is then the identifier that goes into the evidence certificate.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_WHITESPACE = re.compile(r"\s+")
_DOI_PREFIX = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:)", re.IGNORECASE)


def normalise_text(value: str | None) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("\u00ad", "")  # soft hyphen
    return _WHITESPACE.sub(" ", value).strip()


def normalise_doi(value: str | None) -> str:
    if not value:
        return ""
    value = _DOI_PREFIX.sub("", normalise_text(value))
    return value.lower().rstrip(".")


@dataclass(frozen=True)
class Record:
    """One bibliographic record, in canonical form."""

    title: str
    abstract: str = ""
    doi: str = ""
    year: int | None = None
    authors: tuple[str, ...] = ()
    venue: str = ""
    full_text: str = ""
    sections: tuple[tuple[str, str], ...] = ()
    source: str = ""
    extra: dict = field(default_factory=dict, compare=False)

    @classmethod
    def build(cls, **kwargs) -> Record:
        authors = kwargs.pop("authors", ()) or ()
        sections = kwargs.pop("sections", ()) or ()
        if isinstance(sections, dict):
            sections = tuple(sections.items())
        return cls(
            title=normalise_text(kwargs.pop("title", "")),
            abstract=normalise_text(kwargs.pop("abstract", "")),
            doi=normalise_doi(kwargs.pop("doi", "")),
            year=kwargs.pop("year", None),
            authors=tuple(sorted(normalise_text(a) for a in authors if normalise_text(a))),
            venue=normalise_text(kwargs.pop("venue", "")),
            full_text=normalise_text(kwargs.pop("full_text", "")),
            sections=tuple(
                (normalise_text(name).lower(), normalise_text(body))
                for name, body in sections
                if normalise_text(body)
            ),
            source=normalise_text(kwargs.pop("source", "")),
            extra=kwargs.pop("extra", {}) or {},
        )

    def identity(self) -> dict:
        """The fields that define record equality for hashing purposes."""
        return {
            "title": self.title,
            "abstract": self.abstract,
            "doi": self.doi,
            "year": self.year,
            "authors": list(self.authors),
            "venue": self.venue,
            "full_text": self.full_text,
            "sections": [list(pair) for pair in self.sections],
        }

    @property
    def record_id(self) -> str:
        payload = json.dumps(self.identity(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    @property
    def dedup_key(self) -> str:
        """DOI when present, otherwise normalised title plus year."""
        if self.doi:
            return f"doi:{self.doi}"
        title_key = re.sub(r"[^a-z0-9]+", "", self.title.lower())
        return f"ty:{title_key}:{self.year or ''}"

    def screening_text(self) -> str:
        return normalise_text(f"{self.title}. {self.abstract}")


def _content_length(record: Record) -> int:
    """How much text a record carries, used to pick between duplicates."""
    return (
        len(record.abstract)
        + len(record.full_text)
        + sum(len(body) for _, body in record.sections)
    )


class Corpus:
    """An ordered, deduplicated, hashable collection of records."""

    def __init__(self, records: Iterable[Record]):
        grouped: dict[str, list[Record]] = {}
        for record in records:
            grouped.setdefault(record.dedup_key, []).append(record)

        self.duplicates: list[tuple[str, str]] = []
        kept: list[Record] = []
        for candidates in grouped.values():
            # Two exports of the same work can disagree — one carries the
            # abstract, the other does not. Keeping whichever arrived first
            # would make the corpus hash depend on file order, which is the
            # one thing it must not do. Prefer the richer record, breaking
            # ties on record_id so the choice is a function of content alone.
            winner, *rest = sorted(
                candidates, key=lambda r: (-_content_length(r), r.record_id)
            )
            kept.append(winner)
            self.duplicates.extend((winner.record_id, other.record_id) for other in rest)

        self.records: tuple[Record, ...] = tuple(sorted(kept, key=lambda r: r.record_id))

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    def __getitem__(self, item):
        return self.records[item]

    @property
    def corpus_hash(self) -> str:
        digest = hashlib.sha256()
        for record in self.records:
            digest.update(record.record_id.encode("ascii"))
        return digest.hexdigest()

    def stats(self) -> dict:
        """What kind of corpus this is, which determines the right chunker.

        A corpus of abstracts and a corpus of full texts need different
        chunking, and using the screening-stage chunker on full texts yields
        near-empty chunks. `recommended_chunker` encodes that choice so it
        does not have to be rediscovered.
        """
        with_abstract = sum(1 for r in self.records if r.abstract)
        with_full_text = sum(1 for r in self.records if r.full_text)
        with_sections = sum(1 for r in self.records if r.sections)

        if with_sections >= max(1, len(self.records) // 2):
            recommended = "tei-section"
        elif with_full_text >= max(1, len(self.records) // 2):
            recommended = "sentence"
        else:
            recommended = "record"

        return {
            "records": len(self.records),
            "with_abstract": with_abstract,
            "with_full_text": with_full_text,
            "with_sections": with_sections,
            "duplicates_removed": len(self.duplicates),
            "recommended_chunker": recommended,
        }

    def by_id(self, record_id: str) -> Record | None:
        for record in self.records:
            if record.record_id == record_id:
                return record
        return None

    def to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                payload = record.identity()
                payload["record_id"] = record.record_id
                payload["source"] = record.source
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Corpus:
        records = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(Record.build(**json.loads(line)))
        return cls(records)

    @classmethod
    def from_dicts(cls, rows: Sequence[dict]) -> Corpus:
        return cls(Record.build(**row) for row in rows)
