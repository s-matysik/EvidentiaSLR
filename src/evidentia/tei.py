"""GROBID TEI parsing.

LitRev already runs GROBID behind its `extractor` service and stores the raw
TEI in `review_source_files.extracted_metadata->tei`. This module turns that
TEI into a canonical `Record` with IMRaD sections, which is exactly what
`TEISectionChunker` needs. No second extraction pass is required.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from .corpus import Record, normalise_text

_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def _text_of(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return normalise_text(" ".join(element.itertext()))


def parse_tei(tei: str, source: str = "grobid") -> Record:
    """Build a `Record` from a GROBID TEI document."""
    root = ET.fromstring(tei)

    title = _text_of(root.find(".//tei:titleStmt/tei:title", _NS))

    authors = []
    for author in root.findall(".//tei:sourceDesc//tei:author/tei:persName", _NS):
        forename = " ".join(_text_of(n) for n in author.findall("tei:forename", _NS))
        surname = _text_of(author.find("tei:surname", _NS))
        full = normalise_text(f"{forename} {surname}")
        if full:
            authors.append(full)

    doi = ""
    for idno in root.findall(".//tei:idno", _NS):
        if (idno.get("type") or "").upper() == "DOI":
            doi = _text_of(idno)
            break

    year = None
    date = root.find(".//tei:publicationStmt/tei:date", _NS)
    if date is None:
        date = root.find(".//tei:sourceDesc//tei:date", _NS)
    if date is not None:
        raw = date.get("when") or _text_of(date)
        digits = "".join(ch for ch in raw[:4] if ch.isdigit())
        if len(digits) == 4:
            year = int(digits)

    venue = _text_of(root.find(".//tei:monogr/tei:title", _NS))

    abstract_parts = [
        _text_of(p) for p in root.findall(".//tei:profileDesc//tei:abstract//tei:p", _NS)
    ]
    abstract = normalise_text(" ".join(abstract_parts))

    sections: list[tuple[str, str]] = []
    for div in root.findall(".//tei:body/tei:div", _NS):
        head = _text_of(div.find("tei:head", _NS)) or "other"
        body = normalise_text(" ".join(_text_of(p) for p in div.findall("tei:p", _NS)))
        if body:
            sections.append((head, body))

    full_text = normalise_text(" ".join(body for _, body in sections))

    return Record.build(
        title=title,
        abstract=abstract,
        doi=doi,
        year=year,
        authors=authors,
        venue=venue,
        full_text=full_text,
        sections=sections,
        source=source,
    )


def record_from_extracted(extracted_text: str, extracted_metadata: dict | None) -> Record:
    """Build a record from a LitRev `review_source_files` row.

    Prefers the stored TEI (structure preserved); falls back to the flattened
    text LitRev also keeps, which yields a record without sections.
    """
    metadata = extracted_metadata or {}
    tei = metadata.get("tei")
    if tei:
        try:
            return parse_tei(tei)
        except ET.ParseError:
            pass
    return Record.build(title="", full_text=extracted_text or "", source="litrev-text")
