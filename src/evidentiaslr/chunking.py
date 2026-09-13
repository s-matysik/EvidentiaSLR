"""Deterministic chunking.

Semantic chunkers that call an embedding model or an LLM to place boundaries
are a hidden source of non-determinism: change the batch size and the split
points move. Every chunker here is a pure function of the input text and its
declared parameters, and each one publishes a `spec` string that goes into
the evidence certificate.

`TEISectionChunker` is the one that matters for literature review work. It
splits along IMRaD structure recovered from GROBID output, so a query can be
restricted to, say, method sections only.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Protocol

from .corpus import Corpus, Record, normalise_text
from .exceptions import ConfigurationError

#: Terminator, whitespace, then something that can start a sentence. The next
#: token may be a digit, quote or bracket, not only a capital: "...as follows.
#: 385 respondents..." is a boundary.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\u201c\"'])")

#: Abbreviations that end in a period without ending a sentence. Academic prose
#: is full of them, and without this "(Campbell et al., 2020). When used..."
#: splits after "al." Python forbids variable-width lookbehind, so the split is
#: done first and over-eager cuts are rejoined afterwards.
_ABBREVIATIONS = frozenset("""
    et al e.g i.e cf vs etc fig figs tab tabs eq eqs no nos vol pp ref refs
    approx resp dr prof mr mrs ms st ca ch sec min max jr sr inc ltd
""".split())

_LAST_TOKEN = re.compile(r"([A-Za-z.]+)\.$")


def _ends_with_abbreviation(fragment: str) -> bool:
    match = _LAST_TOKEN.search(fragment.strip())
    if not match:
        return False
    return match.group(1).lower().strip(".") in _ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Split into sentences, rejoining splits made after an abbreviation.

    Deliberately conservative: over-splitting a citation into two chunks is
    worse than under-splitting, because the second half loses the context that
    made the first half meaningful.
    """
    fragments = [part for part in _SENTENCE_END.split(text) if part.strip()]
    if not fragments:
        return []

    merged = [fragments[0]]
    for fragment in fragments[1:]:
        if _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {fragment}"
        else:
            merged.append(fragment)
    return [fragment.strip() for fragment in merged if fragment.strip()]


def canonical_section(name: str) -> str:
    """Canonical IMRaD label for a heading. See `evidentiaslr.sections`.

    Kept as a thin delegation so existing call sites do not change, while the
    classification rules live in one auditable place.
    """
    from .sections import classify_section

    return classify_section(name)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    record_id: str
    ordinal: int
    text: str
    section: str = "unknown"
    start: int = -1
    end: int = -1

    @property
    def located(self) -> bool:
        """Whether this chunk was placed in the record's canonical text."""
        return self.start >= 0 and self.end > self.start

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "record_id": self.record_id,
            "ordinal": self.ordinal,
            "section": self.section,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


def _chunk_id(record_id: str, spec: str, ordinal: int, text: str) -> str:
    payload = f"{record_id}|{spec}|{ordinal}|{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def body_text(record: Record) -> str:
    """The text every full-text chunker must operate on.

    Chunkers previously disagreed about this: the sentence and window
    chunkers read `full_text` and fell back to title-plus-abstract, while the
    section-aware ones read `sections`. On a corpus carrying sections but no
    flattened body — which is what GROBID ingestion produces — they were
    therefore chunking different strings, and any comparison between them
    measured that discrepancy rather than the chunking strategy.
    """
    if record.full_text:
        return record.full_text
    if record.sections:
        return " ".join(body for _, body in record.sections)
    return record.screening_text()


def canonical_source(record: Record) -> str:
    """The single text against which every chunker's output is located.

    Chunkers disagree about what a chunk is — a whole abstract, a sentence
    window, a method section — so their chunk ids are incomparable by
    construction. Character offsets into one shared string are not. This is
    what makes "did two chunkers retrieve the same evidence?" a well-posed
    question rather than a category error.
    """
    screening = record.screening_text()
    body = body_text(record)
    if body == screening:
        return screening
    return " ".join(part for part in (screening, body) if part)


def _locate(chunks: list[Chunk], record: Record) -> list[Chunk]:
    """Assign character spans by scanning forward through the source text.

    Forward scanning rather than a global search: a repeated sentence would
    otherwise collapse onto its first occurrence, and overlapping windows would
    all claim the same span. The cursor only moves forward, so successive
    chunks get successive positions.

    A chunk that cannot be located keeps (-1, -1) and span-based metrics fall
    back to record-level comparison for it. That happens when normalisation
    altered the text — worth degrading gracefully rather than failing, but the
    caller can count `located` to know how much of the comparison is exact.
    """
    source = canonical_source(record)
    if not source:
        return chunks

    located: list[Chunk] = []
    cursor = 0
    for chunk in chunks:
        position = source.find(chunk.text, cursor)
        if position < 0:
            # Overlapping windows legitimately step backwards; retry once
            # from the start before giving up.
            position = source.find(chunk.text)
        if position < 0:
            located.append(chunk)
            continue
        end = position + len(chunk.text)
        located.append(replace(chunk, start=position, end=end))
        cursor = position + max(1, len(chunk.text) // 2)
    return located


class Chunker(Protocol):
    @property
    def spec(self) -> str: ...

    def split(self, record: Record) -> list[Chunk]: ...


class _BaseChunker:
    @property
    def spec(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _emit(self, record: Record, pieces: Iterable[tuple[str, str]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for ordinal, (section, text) in enumerate(pieces):
            text = normalise_text(text)
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(record.record_id, self.spec, ordinal, text),
                    record_id=record.record_id,
                    ordinal=ordinal,
                    text=text,
                    section=section,
                )
            )
        return chunks

    def chunk_corpus(self, corpus: Corpus) -> list[Chunk]:
        chunks: list[Chunk] = []
        for record in corpus:
            chunks.extend(_locate(self.split(record), record))
        # Stable global order, independent of iteration order upstream.
        chunks.sort(key=lambda c: (c.record_id, c.ordinal))
        return chunks


class RecordChunker(_BaseChunker):
    """One chunk per record: title plus abstract. The screening-stage view."""

    @property
    def spec(self) -> str:
        return "record/v1"

    def split(self, record: Record) -> list[Chunk]:
        return self._emit(record, [("abstract", record.screening_text())])


class FixedWindowChunker(_BaseChunker):
    """Fixed word window with overlap. The baseline everyone uses."""

    def __init__(self, size: int = 200, overlap: int = 40):
        if overlap >= size:
            raise ConfigurationError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    @property
    def spec(self) -> str:
        return f"fixed/v1(size={self.size},overlap={self.overlap})"

    def _body(self, record: Record) -> str:
        return body_text(record)

    def split(self, record: Record) -> list[Chunk]:
        words = self._body(record).split()
        if not words:
            return []
        step = self.size - self.overlap
        pieces = []
        for start in range(0, len(words), step):
            window = words[start : start + self.size]
            if not window:
                break
            pieces.append(("unknown", " ".join(window)))
            if start + self.size >= len(words):
                break
        return self._emit(record, pieces)


class SentenceChunker(_BaseChunker):
    """Groups whole sentences up to a word budget. No model in the loop."""

    def __init__(self, max_words: int = 180):
        self.max_words = max_words

    @property
    def spec(self) -> str:
        return f"sentence/v1(max_words={self.max_words})"

    def split(self, record: Record) -> list[Chunk]:
        body = body_text(record)
        sentences = split_sentences(body)
        pieces, buffer, count = [], [], 0
        for sentence in sentences:
            words = len(sentence.split())
            if buffer and count + words > self.max_words:
                pieces.append(("unknown", " ".join(buffer)))
                buffer, count = [], 0
            buffer.append(sentence)
            count += words
        if buffer:
            pieces.append(("unknown", " ".join(buffer)))
        return self._emit(record, pieces)


class TEISectionChunker(_BaseChunker):
    """IMRaD-aware chunking over sections recovered from GROBID TEI.

    Falls back to sentence chunking when a record carries no section
    structure, so a mixed corpus (some full texts, some abstracts only)
    still produces a coherent index.
    """

    def __init__(self, max_words: int = 180, keep_sections: tuple[str, ...] | None = None):
        self.max_words = max_words
        self.keep_sections = tuple(sorted(keep_sections)) if keep_sections else None
        self._fallback = SentenceChunker(max_words=max_words)

    @property
    def spec(self) -> str:
        keep = ",".join(self.keep_sections) if self.keep_sections else "all"
        return f"tei-section/v1(max_words={self.max_words},keep={keep})"

    def split(self, record: Record) -> list[Chunk]:
        if not record.sections:
            fallback = self._fallback.split(record)
            return [
                Chunk(
                    chunk_id=_chunk_id(record.record_id, self.spec, chunk.ordinal, chunk.text),
                    record_id=chunk.record_id,
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    section="unknown",
                )
                for chunk in fallback
            ]

        from .sections import classify_record_sections

        pieces: list[tuple[str, str]] = []
        # The abstract is a section like any other and must obey the filter.
        # Emitting it unconditionally meant `keep_sections=("methods",)` still
        # returned abstracts, which the retriever's own filter then happened
        # to mask — a bug that only surfaced once both were tested apart.
        if record.abstract and (not self.keep_sections or "abstract" in self.keep_sections):
            pieces.append(("abstract", record.abstract))

        # Context-aware: a subsection inherits its parent's label, so
        # "2.1 Non-fungible tokens" under "2. Theoretical background" is not
        # dropped from a background filter for want of a keyword.
        labels = classify_record_sections(record.sections)
        for (_raw_name, body), match in zip(record.sections, labels, strict=True):
            section = match.label
            if self.keep_sections and section not in self.keep_sections:
                continue
            sentences = split_sentences(body)
            buffer, count = [], 0
            for sentence in sentences:
                words = len(sentence.split())
                if buffer and count + words > self.max_words:
                    pieces.append((section, " ".join(buffer)))
                    buffer, count = [], 0
                buffer.append(sentence)
                count += words
            if buffer:
                pieces.append((section, " ".join(buffer)))
        return self._emit(record, pieces)


def _extra(name: str):
    """Lazily resolve chunkers that live in `chunkers_extra`.

    Kept out of this module's import graph because `SemanticChunker` needs an
    embedder, and a chunker that cannot be constructed without one does not
    belong beside chunkers that are pure functions of text.
    """
    from . import chunkers_extra

    return getattr(chunkers_extra, name)


CHUNKERS = {
    # deterministic, pure functions of the text
    "record": RecordChunker,
    "fixed": FixedWindowChunker,
    "sentence": SentenceChunker,
    "tei-section": TEISectionChunker,
    "paragraph": lambda **kw: _extra("ParagraphChunker")(**kw),
    "recursive": lambda **kw: _extra("RecursiveCharacterChunker")(**kw),
    "sliding-sentence": lambda **kw: _extra("SlidingSentenceChunker")(**kw),
    # requires an embedder; boundaries depend on it
    "semantic": lambda **kw: _extra("SemanticChunker")(**kw),
}

#: Chunkers whose output depends on a model rather than on the text alone.
MODEL_DEPENDENT_CHUNKERS = frozenset({"semantic"})


def build_chunker(name: str, embedder=None, **params):
    """The one place a chunker is constructed from a name.

    Exists because the CLI, the HTTP service and the experiment driver each
    grew their own construction path, and only the driver remembered that a
    model-dependent chunker needs the pipeline embedder injected. The other
    two crashed with a bare TypeError — the CLI at the prompt, the service as
    an unhandled 500 — which is the kind of divergence a single factory ends.
    """
    from .exceptions import ConfigurationError, UnknownBackendError

    if name not in CHUNKERS:
        raise UnknownBackendError(
            f"unknown chunker {name!r}; available: {sorted(CHUNKERS)}"
        )
    if name in MODEL_DEPENDENT_CHUNKERS and "embedder" not in params:
        if embedder is None:
            raise ConfigurationError(
                f"chunker {name!r} places boundaries with an embedding model; "
                f"pass embedder= (the CLI and service inject the pipeline "
                f"embedder automatically)"
            )
        params["embedder"] = embedder
    return CHUNKERS[name](**params)
