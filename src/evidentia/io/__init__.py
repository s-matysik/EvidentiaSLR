"""Bibliographic and full-text import.

Two entry points, matching the two things a review team actually has:

    metadata  Scopus / WoS CSV, RIS, BibTeX  -> load_corpus()
    PDFs      a directory of files           -> corpus_from_pdf_dir()

and `merge_pdf_corpus()` to join them, so the DOIs and years come from the
database export while the full text and IMRaD sections come from GROBID.

Every loader returns a report alongside the corpus. How much was dropped and
why is part of the provenance record, not a diagnostic to be discarded.
"""

from __future__ import annotations

from pathlib import Path

from ..exceptions import UnknownBackendError
from .bibtex import load_bibtex, parse_bibtex
from .pdf import (
    GrobidClient,
    LocalPdfExtractor,
    PdfIngestReport,
    corpus_from_pdf_dir,
    corpus_from_tei_dir,
    extract_text_locally,
    merge_pdf_corpus,
)
from .ris import load_ris, parse_ris
from .scopus import ImportReport, load_scopus_csv, load_wos_csv

__all__ = [
    "GrobidClient",
    "ImportReport",
    "LocalPdfExtractor",
    "PdfIngestReport",
    "corpus_from_pdf_dir",
    "corpus_from_tei_dir",
    "extract_text_locally",
    "load_bibtex",
    "load_corpus",
    "load_ris",
    "load_scopus_csv",
    "load_wos_csv",
    "merge_pdf_corpus",
    "parse_bibtex",
    "parse_ris",
    "LOADERS",
]

LOADERS = {
    "scopus": load_scopus_csv,
    "wos": load_wos_csv,
    "ris": load_ris,
    "bibtex": load_bibtex,
}


def load_corpus(path: str | Path, fmt: str | None = None, **kwargs):
    """Load a metadata corpus, inferring the format from the extension.

    `.ris` and `.bib` are unambiguous. `.csv` defaults to Scopus, since that
    is the more common export and a wrong guess would report every essential
    column as unmapped — a loud failure rather than a silent one.
    """
    path = Path(path)
    if fmt is None:
        suffix = path.suffix.lower()
        fmt = {".ris": "ris", ".bib": "bibtex", ".bibtex": "bibtex", ".csv": "scopus",
               ".txt": "wos", ".tsv": "wos"}.get(suffix)
        if fmt is None:
            raise UnknownBackendError(
                f"cannot infer format from {path.suffix!r}; pass fmt= explicitly "
                f"(one of {sorted(LOADERS)})"
            )
    if fmt not in LOADERS:
        raise UnknownBackendError(f"unknown format {fmt!r}; available: {sorted(LOADERS)}")
    return LOADERS[fmt](path, **kwargs)
