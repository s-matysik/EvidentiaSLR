"""Scopus and Web of Science CSV import.

Column names differ between Scopus, WoS and their various export profiles,
so the mapping is declarative and the loader reports what it could not find
rather than silently producing empty fields — a corpus that lost its abstracts
to a renamed column is worse than one that failed loudly.

Scopus writes the literal string `[No abstract available]` where an abstract
is missing. Treating that as text would put 24 meaningless characters into the
index and give every such record a spurious similarity to every other one, so
it is normalised to empty here.
"""

from __future__ import annotations

import csv
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..corpus import Corpus, Record
from ..exceptions import CorpusError

__all__ = ["ImportReport", "load_scopus_csv", "load_wos_csv", "records_from_rows"]

# Scopus increases the CSV field size well beyond the default 128 KB limit
# when references are included; without this a long reference list aborts the
# read with _csv.Error.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

_MISSING_ABSTRACT = {"[no abstract available]", "no abstract available", ""}

#: Fields whose absence means the mapping is wrong, not merely incomplete.
ESSENTIAL_FIELDS = ("title", "abstract", "doi", "year", "authors")

SCOPUS_COLUMNS = {
    "title": ("Title",),
    "abstract": ("Abstract",),
    "doi": ("DOI",),
    "year": ("Year",),
    "authors": ("Authors", "Author full names"),
    "venue": ("Source title", "Abbreviated Source Title"),
    "author_keywords": ("Author Keywords",),
    "index_keywords": ("Index Keywords",),
    "document_type": ("Document Type",),
    "cited_by": ("Cited by",),
    "eid": ("EID",),
}

WOS_COLUMNS = {
    "title": ("Article Title", "TI"),
    "abstract": ("Abstract", "AB"),
    "doi": ("DOI", "DI"),
    "year": ("Publication Year", "PY"),
    "authors": ("Authors", "AU", "Author Full Names"),
    "venue": ("Source Title", "SO"),
    "author_keywords": ("Author Keywords", "DE"),
    "index_keywords": ("Keywords Plus", "ID"),
    "document_type": ("Document Type", "DT"),
    "cited_by": ("Times Cited, All Databases", "TC"),
    "eid": ("UT (Unique WOS ID)", "UT"),
}


@dataclass
class ImportReport:
    """What the loader found, and what it had to drop."""

    path: str
    rows_read: int = 0
    records_built: int = 0
    duplicates_removed: int = 0
    missing_abstract: int = 0
    missing_doi: int = 0
    dropped_empty: int = 0
    unmapped_fields: list[str] = field(default_factory=list)
    document_types: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "rows_read": self.rows_read,
            "records_built": self.records_built,
            "duplicates_removed": self.duplicates_removed,
            "missing_abstract": self.missing_abstract,
            "missing_doi": self.missing_doi,
            "dropped_empty": self.dropped_empty,
            "unmapped_fields": self.unmapped_fields,
            "document_types": self.document_types,
        }

    def summary(self) -> str:
        lines = [
            f"{self.records_built} records from {self.rows_read} rows",
            f"  duplicates removed : {self.duplicates_removed}",
            f"  missing abstract   : {self.missing_abstract}",
            f"  missing DOI        : {self.missing_doi}",
            f"  dropped (no text)  : {self.dropped_empty}",
        ]
        if self.unmapped_fields:
            lines.append(f"  UNMAPPED COLUMNS   : {', '.join(self.unmapped_fields)}")
        return "\n".join(lines)


def _pick(row: dict, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _split_authors(value: str) -> list[str]:
    if not value:
        return []
    separator = ";" if ";" in value else ","
    return [part.strip() for part in value.split(separator) if part.strip()]


def _parse_year(value: str) -> int | None:
    digits = "".join(ch for ch in value[:4] if ch.isdigit())
    return int(digits) if len(digits) == 4 else None


def records_from_rows(
    rows: Iterable[dict],
    columns: dict[str, tuple[str, ...]],
    report: ImportReport,
    keep_types: tuple[str, ...] | None = None,
    require_abstract: bool = False,
) -> list[Record]:
    """Map raw CSV rows onto canonical records, filling the report as it goes."""
    records: list[Record] = []

    for row in rows:
        report.rows_read += 1

        raw_abstract = _pick(row, columns["abstract"]).strip()
        abstract = "" if raw_abstract.lower() in _MISSING_ABSTRACT else raw_abstract
        title = _pick(row, columns["title"]).strip()
        doi = _pick(row, columns["doi"])
        document_type = _pick(row, columns["document_type"]) or "unknown"

        report.document_types[document_type] = report.document_types.get(document_type, 0) + 1

        if not abstract:
            report.missing_abstract += 1
        if not doi:
            report.missing_doi += 1

        if keep_types and document_type not in keep_types:
            continue
        if require_abstract and not abstract:
            continue
        if not title and not abstract:
            report.dropped_empty += 1
            continue

        keywords = _split_authors(_pick(row, columns["author_keywords"]))
        records.append(
            Record.build(
                title=title,
                abstract=abstract,
                doi=doi,
                year=_parse_year(_pick(row, columns["year"])),
                authors=_split_authors(_pick(row, columns["authors"])),
                venue=_pick(row, columns["venue"]),
                source="scopus" if columns is SCOPUS_COLUMNS else "wos",
                extra={
                    "document_type": document_type,
                    "author_keywords": keywords,
                    "cited_by": _pick(row, columns["cited_by"]),
                    "external_id": _pick(row, columns["eid"]),
                },
            )
        )

    return records


def _load(
    path: str | Path,
    columns: dict[str, tuple[str, ...]],
    keep_types: tuple[str, ...] | None,
    require_abstract: bool,
    encoding: str,
) -> tuple[Corpus, ImportReport]:
    path = Path(path)
    report = ImportReport(path=str(path))

    with path.open(encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CorpusError(f"{path} has no header row")

        present = set(reader.fieldnames)
        report.unmapped_fields = sorted(
            field_name
            for field_name in ESSENTIAL_FIELDS
            if not present & set(columns[field_name])
        )
        header = list(reader.fieldnames)
        records = records_from_rows(
            reader, columns, report, keep_types=keep_types, require_abstract=require_abstract
        )

    if not records:
        missing = ", ".join(report.unmapped_fields) or "none"
        raise CorpusError(
            f"{path} produced no usable records. Unmapped essential columns: {missing}. "
            f"Header was: {', '.join(header)}"
        )

    corpus = Corpus(records)
    report.records_built = len(corpus)
    report.duplicates_removed = len(corpus.duplicates)
    return corpus, report


def load_scopus_csv(
    path: str | Path,
    keep_types: tuple[str, ...] | None = None,
    require_abstract: bool = False,
    encoding: str = "utf-8-sig",
) -> tuple[Corpus, ImportReport]:
    """Load a Scopus CSV export.

    `keep_types` filters on the Document Type column, e.g.
    `("Article", "Review", "Conference paper")` to drop book chapters and
    editorials. `require_abstract=True` drops records Scopus exported without
    one — worth doing before an embedding run, since a title-only record
    competes on far less text than the rest of the corpus.

    Default encoding is `utf-8-sig` because Scopus writes a BOM, which
    otherwise corrupts the first column name.
    """
    return _load(path, SCOPUS_COLUMNS, keep_types, require_abstract, encoding)


def load_wos_csv(
    path: str | Path,
    keep_types: tuple[str, ...] | None = None,
    require_abstract: bool = False,
    encoding: str = "utf-8-sig",
) -> tuple[Corpus, ImportReport]:
    """Load a Web of Science CSV or tab-delimited export."""
    return _load(path, WOS_COLUMNS, keep_types, require_abstract, encoding)
