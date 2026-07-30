"""RIS import.

RIS is the lowest common denominator: EndNote, Zotero, Mendeley, PubMed,
Embase and most publisher sites all emit it, which makes it the format a
review team is most likely to have when Scopus is not the source.

The parser is deliberately tolerant. Real RIS in the wild has inconsistent
line endings, repeated tags, continuation lines with no tag, and files that
omit the closing `ER  -`. All of those are handled rather than rejected,
because a review corpus that fails to load is a corpus that gets screened by
hand instead.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..corpus import Corpus, Record
from ..exceptions import CorpusError
from .scopus import ImportReport

__all__ = ["load_ris", "parse_ris"]

_TAG = re.compile(r"^([A-Z][A-Z0-9])\s{2}-\s?(.*)$")

# RIS types that are not primary literature and normally do not belong in a
# screening corpus.
_NON_ARTICLE = {"GEN", "ICOMM", "ECOMM", "SLIDE", "SOUND", "VIDEO"}


def _entries(text: str) -> list[dict[str, list[str]]]:
    entries: list[dict[str, list[str]]] = []
    current: dict[str, list[str]] = {}
    last_tag: str | None = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue

        match = _TAG.match(line)
        if match:
            tag, value = match.group(1), match.group(2).strip()
            if tag == "ER":
                if current:
                    entries.append(current)
                current, last_tag = {}, None
                continue
            if tag == "TY" and current:
                # A new record began without a closing ER; flush the previous.
                entries.append(current)
                current = {}
            current.setdefault(tag, []).append(value)
            last_tag = tag
        elif last_tag:
            # Continuation line: append to the value it belongs to.
            current[last_tag][-1] = f"{current[last_tag][-1]} {line.strip()}".strip()

    if current:
        entries.append(current)
    return entries


def _first(entry: dict[str, list[str]], *tags: str) -> str:
    for tag in tags:
        values = entry.get(tag)
        if values:
            return values[0]
    return ""


def _year(entry: dict[str, list[str]]) -> int | None:
    raw = _first(entry, "PY", "Y1", "DA")
    digits = "".join(ch for ch in raw[:4] if ch.isdigit())
    return int(digits) if len(digits) == 4 else None


def parse_ris(text: str, require_abstract: bool = False) -> tuple[list[Record], ImportReport]:
    """Parse RIS text into records plus an import report."""
    report = ImportReport(path="<string>")
    records: list[Record] = []

    for entry in _entries(text):
        report.rows_read += 1
        ris_type = _first(entry, "TY") or "unknown"
        report.document_types[ris_type] = report.document_types.get(ris_type, 0) + 1

        title = _first(entry, "TI", "T1", "BT")
        abstract = " ".join(entry.get("AB", []) or entry.get("N2", []))
        doi = _first(entry, "DO", "DI")

        if not abstract:
            report.missing_abstract += 1
        if not doi:
            report.missing_doi += 1

        if ris_type in _NON_ARTICLE:
            continue
        if require_abstract and not abstract:
            continue
        if not title and not abstract:
            report.dropped_empty += 1
            continue

        records.append(
            Record.build(
                title=title,
                abstract=abstract,
                doi=doi,
                year=_year(entry),
                authors=entry.get("AU", []) or entry.get("A1", []),
                venue=_first(entry, "JO", "JF", "T2", "J2"),
                source="ris",
                extra={
                    "document_type": ris_type,
                    "author_keywords": entry.get("KW", []),
                },
            )
        )

    return records, report


def load_ris(
    path: str | Path, require_abstract: bool = False, encoding: str = "utf-8-sig"
) -> tuple[Corpus, ImportReport]:
    """Load one RIS file."""
    path = Path(path)
    records, report = parse_ris(
        path.read_text(encoding=encoding, errors="replace"), require_abstract=require_abstract
    )
    report.path = str(path)

    if not records:
        raise CorpusError(f"{path} produced no usable records")

    corpus = Corpus(records)
    report.records_built = len(corpus)
    report.duplicates_removed = len(corpus.duplicates)
    return corpus, report
