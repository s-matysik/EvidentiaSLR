"""PDF ingestion.

The entry point for full texts. A PDF becomes a `Record` with IMRaD sections
via GROBID, which is the same engine LitRev already runs behind its
`extractor` service — so a corpus built here and a corpus built there are
interchangeable.

Two things make this reproducible rather than merely convenient:

TEI is cached by file content hash, not by filename. Re-running ingestion
never re-invokes GROBID on an unchanged file, so the corpus hash is stable
across runs even though GROBID itself is a large stateful Java service whose
output can shift between versions. The cached TEI is the artefact; keep it
alongside the corpus and the ingestion becomes replayable without GROBID at
all.

Files are processed in sorted order and the resulting corpus is id-sorted, so
a partial or interrupted run yields a prefix of the same corpus rather than an
arbitrary subset.

A local fallback (`pdftotext` or pypdf) exists for trying the pipeline without
standing up GROBID. It produces no section structure, so anything that depends
on `TEISectionChunker` or section filtering will silently degrade — the report
says so explicitly rather than leaving it to be discovered later.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..corpus import Corpus, Record
from ..exceptions import BackendUnavailableError, ConfigurationError, CorpusError

__all__ = [
    "GrobidClient",
    "PdfIngestReport",
    "corpus_from_pdf_dir",
    "extract_text_locally",
]

DEFAULT_GROBID_URL = "http://localhost:8070"


@dataclass
class PdfIngestReport:
    """Coverage record for a PDF ingestion run."""

    pdf_dir: str = ""
    files_found: int = 0
    processed: int = 0
    from_cache: int = 0
    failed: int = 0
    engine: str = ""
    with_sections: int = 0
    without_sections: int = 0
    errors: list[dict] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.processed / self.files_found if self.files_found else 0.0

    def as_dict(self) -> dict:
        return {
            "pdf_dir": self.pdf_dir,
            "files_found": self.files_found,
            "processed": self.processed,
            "from_cache": self.from_cache,
            "failed": self.failed,
            "engine": self.engine,
            "with_sections": self.with_sections,
            "without_sections": self.without_sections,
            "coverage": self.coverage,
            "errors": self.errors,
        }

    def summary(self) -> str:
        unit = "TEI files" if self.engine == "tei-cache" else "PDFs"
        lines = [
            f"{self.processed} of {self.files_found} {unit} ingested "
            f"({self.coverage:.0%}) via {self.engine}",
            f"  from cache        : {self.from_cache}",
            f"  failed            : {self.failed}",
            f"  with IMRaD sections: {self.with_sections}",
            f"  without sections   : {self.without_sections}",
        ]
        if self.without_sections and self.engine != "grobid":
            lines.append(
                "  NOTE: this engine recovers no section structure; "
                "section-filtered retrieval will not work on this corpus."
            )
        for error in self.errors[:5]:
            lines.append(f"  ! {error['file']}: {error['error']}")
        if len(self.errors) > 5:
            lines.append(f"  ... and {len(self.errors) - 5} more")
        return "\n".join(lines)


def _require_httpx():
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise BackendUnavailableError(
            "httpx is required for GROBID ingestion; install evidentia[fulltext]"
        ) from exc
    return httpx


def file_digest(path: Path) -> str:
    """SHA-256 of file contents, used as the TEI cache key."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class GrobidClient:
    """Minimal GROBID client.

    Defaults match LitRev's `extractor` service: `processFulltextDocument`
    with header and citation consolidation enabled, so TEI produced here and
    TEI produced there parse identically.
    """

    def __init__(
        self,
        url: str = DEFAULT_GROBID_URL,
        timeout: float = 300.0,
        consolidate_header: bool = True,
        consolidate_citations: bool = False,
        tei_cache: str | Path | None = ".evidentia-cache/tei",
    ):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.consolidate_header = consolidate_header
        self.consolidate_citations = consolidate_citations
        self.tei_cache = Path(tei_cache) if tei_cache else None
        if self.tei_cache:
            self.tei_cache.mkdir(parents=True, exist_ok=True)

    @property
    def engine(self) -> str:
        return "grobid"

    def alive(self) -> bool:
        """True when the service answers. Check before a long batch."""
        httpx = _require_httpx()
        try:
            with httpx.Client(timeout=10.0) as client:
                return client.get(f"{self.url}/api/isalive").status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def _cache_path(self, digest: str) -> Path | None:
        return self.tei_cache / f"{digest}.tei.xml" if self.tei_cache else None

    def process(self, path: Path) -> tuple[str, bool]:
        """Return (TEI, came_from_cache) for one PDF."""
        digest = file_digest(path)
        cached = self._cache_path(digest)
        if cached and cached.exists():
            return cached.read_text(encoding="utf-8"), True

        httpx = _require_httpx()
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.url}/api/processFulltextDocument",
                files={"input": (path.name, path.read_bytes(), "application/pdf")},
                data={
                    "consolidateHeader": "1" if self.consolidate_header else "0",
                    "consolidateCitations": "1" if self.consolidate_citations else "0",
                },
            )
        response.raise_for_status()
        tei = response.text

        if cached:
            cached.write_text(tei, encoding="utf-8")
        return tei, False


def extract_text_locally(path: Path) -> str:
    """Plain text from a PDF without GROBID. No section structure.

    Prefers poppler's `pdftotext` because its layout handling is better than
    pypdf's; falls back to pypdf where poppler is not installed.
    """
    if shutil.which("pdftotext"):
        result = subprocess.run(
            ["pdftotext", "-q", str(path), "-"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise BackendUnavailableError(
            "neither pdftotext nor pypdf is available for local extraction"
        ) from exc

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


class LocalPdfExtractor:
    """Fallback engine with the same interface as `GrobidClient`."""

    def __init__(self, tei_cache: str | Path | None = None):
        self.tei_cache = None  # no TEI to cache

    @property
    def engine(self) -> str:
        return "pdftotext" if shutil.which("pdftotext") else "pypdf"

    def alive(self) -> bool:
        return True

    def process(self, path: Path) -> tuple[str, bool]:
        return extract_text_locally(path), False


def _guess_title(text: str, max_words: int = 30) -> str:
    """First plausible title line of a plain-text extraction.

    Crude by necessity: without TEI there is no structure to consult. Takes
    the first non-trivial line that is not a running header and is short
    enough to be a title. Wrong sometimes; better than the filename, which is
    wrong almost always.
    """
    for line in text.splitlines():
        line = line.strip()
        words = line.split()
        if 3 <= len(words) <= max_words and not line.lower().startswith(
            ("abstract", "keywords", "doi", "http", "downloaded", "page ")
        ):
            return line
    return ""


def _pdf_files(pdf_dir: Path) -> list[Path]:
    return sorted(
        p for p in pdf_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".pdf" and not p.name.startswith(".")
    )


def corpus_from_pdf_dir(
    pdf_dir: str | Path,
    client: GrobidClient | LocalPdfExtractor | None = None,
    limit: int | None = None,
    require_sections: bool = False,
) -> tuple[Corpus, PdfIngestReport]:
    """Ingest every PDF in a directory into a canonical corpus.

    `require_sections=True` drops records from which no IMRaD structure could
    be recovered. Use it when the study depends on section-filtered retrieval,
    so a paper that GROBID could not segment does not quietly dilute the
    result; leave it off when you want maximum coverage.
    """
    from ..chunking import canonical_section
    from ..tei import parse_tei

    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        raise ConfigurationError(f"{pdf_dir} is not a directory")

    client = client or GrobidClient()
    report = PdfIngestReport(pdf_dir=str(pdf_dir), engine=client.engine)

    files = _pdf_files(pdf_dir)
    report.files_found = len(files)
    if not files:
        raise CorpusError(f"no PDF files found under {pdf_dir}")

    records: list[Record] = []
    for path in files[:limit] if limit else files:
        try:
            payload, cached = client.process(path)
        except Exception as exc:  # noqa: BLE001 - recorded per file, batch continues
            report.failed += 1
            report.errors.append({"file": path.name, "error": str(exc)})
            continue

        if cached:
            report.from_cache += 1

        if client.engine == "grobid":
            try:
                record = parse_tei(payload)
            except Exception as exc:  # noqa: BLE001
                report.failed += 1
                report.errors.append({"file": path.name, "error": f"TEI parse failed: {exc}"})
                continue
        else:
            record = Record.build(
                title=_guess_title(payload) or path.stem.replace("_", " "),
                full_text=payload,
                source=client.engine,
            )

        has_sections = any(
            canonical_section(name) in {"methods", "results", "introduction", "discussion"}
            for name, _ in record.sections
        )
        if has_sections:
            report.with_sections += 1
        else:
            report.without_sections += 1
            if require_sections:
                continue

        if not record.full_text and not record.abstract:
            report.failed += 1
            report.errors.append({"file": path.name, "error": "no text extracted"})
            continue

        records.append(record)
        report.processed += 1

    if not records:
        raise CorpusError(
            f"no usable records from {report.files_found} PDFs under {pdf_dir}; "
            f"{report.failed} failed"
        )

    return Corpus(records), report


def corpus_from_tei_dir(
    tei_dir: str | Path,
    require_sections: bool = False,
) -> tuple[Corpus, PdfIngestReport]:
    """Rebuild a corpus from cached TEI, without PDFs and without GROBID.

    The TEI cache is the durable artefact of an ingestion run: GROBID is a
    large stateful service whose output shifts between versions, so a corpus
    that can only be regenerated by re-running it is not really reproducible.
    Keeping the cache alongside the corpus makes the ingestion replayable
    years later, and recoverable now if the source files are lost.
    """
    from ..chunking import canonical_section
    from ..tei import parse_tei

    tei_dir = Path(tei_dir)
    if not tei_dir.is_dir():
        raise ConfigurationError(f"{tei_dir} is not a directory")

    files = sorted(
        p for p in tei_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".xml", ".tei"} or p.name.endswith(".tei.xml")
    )
    report = PdfIngestReport(pdf_dir=str(tei_dir), engine="tei-cache")
    report.files_found = len(files)
    if not files:
        raise CorpusError(f"no TEI files found under {tei_dir}")

    records: list[Record] = []
    for path in files:
        try:
            record = parse_tei(path.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            report.failed += 1
            report.errors.append({"file": path.name, "error": str(exc)})
            continue

        has_sections = any(
            canonical_section(name) in {"methods", "results", "introduction", "discussion"}
            for name, _ in record.sections
        )
        if has_sections:
            report.with_sections += 1
        else:
            report.without_sections += 1
            if require_sections:
                continue

        if not record.full_text and not record.abstract:
            report.failed += 1
            report.errors.append({"file": path.name, "error": "no text in TEI"})
            continue

        records.append(record)
        report.processed += 1
        report.from_cache += 1

    if not records:
        raise CorpusError(f"no usable records from {report.files_found} TEI files under {tei_dir}")

    return Corpus(records), report


def merge_pdf_corpus(
    metadata_corpus: Corpus,
    pdf_corpus: Corpus,
) -> tuple[Corpus, dict]:
    """Attach full texts to a bibliographic corpus, matching on DOI then title.

    Metadata from the database export wins over GROBID's, which recovers DOIs,
    years and venues less reliably than Scopus already did. GROBID supplies
    what the export cannot: full text and section structure.
    """
    import re

    def title_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    by_doi = {r.doi: r for r in pdf_corpus if r.doi}
    by_title = {title_key(r.title): r for r in pdf_corpus if r.title}

    merged: list[Record] = []
    matched = 0

    for record in metadata_corpus:
        source = by_doi.get(record.doi) if record.doi else None
        if source is None:
            source = by_title.get(title_key(record.title))

        if source is None:
            merged.append(record)
            continue

        matched += 1
        merged.append(
            Record.build(
                title=record.title or source.title,
                abstract=record.abstract or source.abstract,
                doi=record.doi or source.doi,
                year=record.year or source.year,
                authors=list(record.authors) or list(source.authors),
                venue=record.venue or source.venue,
                full_text=source.full_text,
                sections=source.sections,
                source=f"{record.source}+{source.source}",
                extra=record.extra,
            )
        )

    stats = {
        "metadata_records": len(metadata_corpus),
        "pdf_records": len(pdf_corpus),
        "matched": matched,
        "unmatched_pdfs": len(pdf_corpus) - matched,
        "match_rate": matched / len(metadata_corpus) if len(metadata_corpus) else 0.0,
    }
    return Corpus(merged), stats
