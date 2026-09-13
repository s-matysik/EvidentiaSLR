"""BibTeX import.

A minimal, dependency-free parser covering what bibliographic exports
actually emit: `@type{key, field = {value},}` with braces or quotes, nested
braces in titles, and LaTeX escapes for accented characters.

It is not a general BibTeX implementation — no `@string` macros, no
concatenation, no cross-references. Those appear in hand-maintained .bib
files, not in database exports, and pretending to support them would be worse
than declining to.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..corpus import Corpus, Record
from ..exceptions import CorpusError
from .scopus import ImportReport

__all__ = ["load_bibtex", "parse_bibtex"]

_ENTRY_START = re.compile(r"@(\w+)\s*\{\s*([^,\s]*)\s*,", re.S)
_FIELD_NAME = re.compile(r"\s*(\w+)\s*=\s*")

_LATEX = {
    r"\&": "&", r"\%": "%", r"\_": "_", r"\$": "$", r"\#": "#",
    "``": '"', "''": '"', "--": "-", "~": " ",
}

_NON_ARTICLE = {"misc", "manual", "unpublished"}


def _scan_fields(body: str) -> dict[str, str]:
    """Extract `name = value` pairs, tracking brace depth.

    A regex cannot do this correctly: `title = {{EEG} recognition}` needs
    matched-brace counting, and a non-greedy `\\{.*?\\}` stops at the first
    inner closing brace, silently truncating the title to "EEG".
    """
    fields: dict[str, str] = {}
    position = 0

    while position < len(body):
        match = _FIELD_NAME.match(body, position)
        if not match:
            position += 1
            continue

        name = match.group(1).lower()
        cursor = match.end()
        if cursor >= len(body):
            break

        opener = body[cursor]
        if opener == "{":
            depth, start = 0, cursor
            while cursor < len(body):
                if body[cursor] == "{":
                    depth += 1
                elif body[cursor] == "}":
                    depth -= 1
                    if depth == 0:
                        cursor += 1
                        break
                cursor += 1
            raw = body[start:cursor]
        elif opener == '"':
            cursor += 1
            start = cursor
            while cursor < len(body) and body[cursor] != '"':
                cursor += 1
            raw = body[start:cursor]
            cursor += 1
        else:
            start = cursor
            while cursor < len(body) and body[cursor] not in ",\n":
                cursor += 1
            raw = body[start:cursor]

        fields[name] = _clean(raw)
        position = cursor + 1

    return fields


def _split_entries(text: str) -> list[tuple[str, str]]:
    """Split a .bib file into (entry_type, body) pairs by brace matching."""
    entries: list[tuple[str, str]] = []
    for match in _ENTRY_START.finditer(text):
        entry_type = match.group(1).lower()
        cursor = text.rindex("{", match.start(), match.end())
        depth = 0
        while cursor < len(text):
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
                if depth == 0:
                    break
            cursor += 1
        body = text[match.end() : cursor]
        entries.append((entry_type, body))
    return entries


def _clean(value: str) -> str:
    value = value.strip().strip(",").strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    elif value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    for latex, plain in _LATEX.items():
        value = value.replace(latex, plain)
    # Strip the brace groups BibTeX uses to protect capitalisation.
    value = re.sub(r"[{}]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_bibtex(text: str, require_abstract: bool = False) -> tuple[list[Record], ImportReport]:
    """Parse BibTeX text into records plus an import report."""
    report = ImportReport(path="<string>")
    records: list[Record] = []

    for entry_type, body in _split_entries(text):
        report.rows_read += 1
        report.document_types[entry_type] = report.document_types.get(entry_type, 0) + 1

        fields = _scan_fields(body)

        title = fields.get("title", "")
        abstract = fields.get("abstract", "")
        doi = fields.get("doi", "")

        if not abstract:
            report.missing_abstract += 1
        if not doi:
            report.missing_doi += 1

        if entry_type in _NON_ARTICLE:
            continue
        if require_abstract and not abstract:
            continue
        if not title and not abstract:
            report.dropped_empty += 1
            continue

        year_raw = fields.get("year", "")
        digits = "".join(ch for ch in year_raw[:4] if ch.isdigit())

        records.append(
            Record.build(
                title=title,
                abstract=abstract,
                doi=doi,
                year=int(digits) if len(digits) == 4 else None,
                authors=[a.strip() for a in fields.get("author", "").split(" and ") if a.strip()],
                venue=fields.get("journal") or fields.get("booktitle", ""),
                source="bibtex",
                extra={
                    "document_type": entry_type,
                    "author_keywords": [
                        k.strip() for k in re.split(r"[;,]", fields.get("keywords", "")) if k.strip()
                    ],
                },
            )
        )

    return records, report


def load_bibtex(
    path: str | Path, require_abstract: bool = False, encoding: str = "utf-8"
) -> tuple[Corpus, ImportReport]:
    """Load one .bib file."""
    path = Path(path)
    records, report = parse_bibtex(
        path.read_text(encoding=encoding, errors="replace"), require_abstract=require_abstract
    )
    report.path = str(path)

    if not records:
        raise CorpusError(f"{path} produced no usable records")

    corpus = Corpus(records)
    report.records_built = len(corpus)
    report.duplicates_removed = len(corpus.duplicates)
    return corpus, report
