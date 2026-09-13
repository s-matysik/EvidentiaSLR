"""Additional chunkers, including the non-deterministic kind.

`chunking.py` holds the chunkers that are pure functions of their input. This
module holds the rest: the baselines every RAG stack ships, and — the reason
it exists — a semantic chunker that places boundaries by calling an embedding
model.

That last one matters more than the index question. A semantic chunker asks a
model where the topic changes, so its boundaries move when the batch size,
the device or the model version moves. And it sits *upstream* of retrieval: a
shifted boundary changes the text of the evidence itself, not merely which
pre-cut piece gets selected. Whatever instability the index contributes, this
contributes it first and larger.

`SemanticChunker` is therefore instrumented rather than fixed. It accepts a
seed and a device so an experiment can vary them deliberately, and its `spec`
records both, so a certificate over a semantically chunked corpus states
plainly what it depended on.
"""

from __future__ import annotations

import re

import numpy as np

from .chunking import Chunk, _BaseChunker, body_text, split_sentences
from .corpus import Record, normalise_text
from .exceptions import ConfigurationError

__all__ = [
    "ParagraphChunker",
    "RecursiveCharacterChunker",
    "SlidingSentenceChunker",
    "SemanticChunker",
]

_PARAGRAPH = re.compile(r"\n\s*\n+")


class ParagraphChunker(_BaseChunker):
    """One chunk per paragraph, merging short ones up to a word budget.

    The structurally honest baseline for full texts: paragraph boundaries were
    put there by the author, so they carry more information than any fixed
    window. Deterministic.
    """

    def __init__(self, min_words: int = 30, max_words: int = 250):
        if min_words >= max_words:
            raise ConfigurationError("min_words must be below max_words")
        self.min_words = min_words
        self.max_words = max_words

    @property
    def spec(self) -> str:
        return f"paragraph/v1(min={self.min_words},max={self.max_words})"

    def _paragraphs(self, record: Record) -> list[tuple[str, str]]:
        if record.sections:
            from .chunking import canonical_section

            return [
                (canonical_section(name), part)
                for name, body in record.sections
                for part in _PARAGRAPH.split(body)
                if part.strip()
            ]
        body = body_text(record)
        return [("unknown", part) for part in _PARAGRAPH.split(body) if part.strip()]

    def split(self, record: Record) -> list[Chunk]:
        pieces: list[tuple[str, str]] = []
        buffer, count, section = [], 0, "unknown"

        for para_section, paragraph in self._paragraphs(record):
            words = len(paragraph.split())
            if buffer and (para_section != section or count + words > self.max_words):
                pieces.append((section, " ".join(buffer)))
                buffer, count = [], 0
            section = para_section
            buffer.append(paragraph)
            count += words
            if count >= self.min_words and count >= self.max_words:
                pieces.append((section, " ".join(buffer)))
                buffer, count = [], 0

        if buffer:
            pieces.append((section, " ".join(buffer)))
        return self._emit(record, pieces)


class RecursiveCharacterChunker(_BaseChunker):
    """Recursive split on a separator hierarchy, with character overlap.

    The de facto default across RAG frameworks, included so the comparison is
    against what people actually run rather than a straw man. Deterministic,
    but its boundaries ignore meaning entirely — which is the point of
    comparing it with the semantic and structural alternatives.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
    ):
        if overlap >= chunk_size:
            raise ConfigurationError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators

    @property
    def spec(self) -> str:
        return f"recursive/v1(size={self.chunk_size},overlap={self.overlap})"

    def _split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self.chunk_size or not separators:
            return [text]

        separator, rest = separators[0], separators[1:]
        parts = list(text) if separator == "" else text.split(separator)

        out, buffer = [], ""
        for part in parts:
            candidate = part if not buffer else f"{buffer}{separator}{part}"
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue
            if buffer:
                out.append(buffer)
            buffer = part if len(part) <= self.chunk_size else ""
            if not buffer:
                out.extend(self._split(part, rest))
        if buffer:
            out.append(buffer)
        return [piece for piece in out if piece.strip()]

    def split(self, record: Record) -> list[Chunk]:
        text = body_text(record)
        pieces = self._split(text, self.separators)

        if self.overlap and len(pieces) > 1:
            overlapped = [pieces[0]]
            for previous, current in zip(pieces[:-1], pieces[1:], strict=True):
                tail = previous[-self.overlap :]
                overlapped.append(normalise_text(f"{tail} {current}"))
            pieces = overlapped

        return self._emit(record, [("unknown", piece) for piece in pieces])


class SlidingSentenceChunker(_BaseChunker):
    """Fixed-size sentence windows with sentence-level stride.

    Sentence-window retrieval, the standard remedy for chunks that cut an
    argument in half. Deterministic; the parameter of interest is `stride`,
    which trades index size against boundary sensitivity.
    """

    def __init__(self, window: int = 3, stride: int = 1):
        if window < 1 or stride < 1:
            raise ConfigurationError("window and stride must be at least 1")
        if stride > window:
            raise ConfigurationError("stride above window would skip text")
        self.window = window
        self.stride = stride

    @property
    def spec(self) -> str:
        return f"sliding-sentence/v1(window={self.window},stride={self.stride})"

    def split(self, record: Record) -> list[Chunk]:
        body = body_text(record)
        sentences = split_sentences(body)
        if not sentences:
            return []

        pieces = []
        for start in range(0, len(sentences), self.stride):
            window = sentences[start : start + self.window]
            if not window:
                break
            pieces.append(("unknown", " ".join(window)))
            if start + self.window >= len(sentences):
                break
        return self._emit(record, pieces)


class SemanticChunker(_BaseChunker):
    """Embedding-based boundary placement — the non-deterministic one.

    Follows the percentile method used by the mainstream implementations:
    embed each sentence, measure the distance between consecutive sentence
    embeddings, and cut wherever that distance exceeds a percentile of the
    distribution.

    Three things make it unstable, and all three are exposed rather than
    hidden:

    - the embedder is injected, so its device and batch size are visible in
      the spec and can be varied by an experiment;
    - the percentile threshold is computed over *this record's* distances, so
      adding or removing a sentence moves every boundary in the document;
    - `jitter` optionally perturbs the embeddings by a seeded amount, which
      simulates the hardware-level divergence a real deployment sees across
      machines without needing two machines to observe it.

    With `jitter=0` and a deterministic embedder this chunker is reproducible,
    which is what makes it usable as a controlled variable: the instability
    can be switched on and off.
    """

    def __init__(
        self,
        embedder,
        percentile: float = 95.0,
        min_sentences: int = 2,
        max_words: int = 400,
        jitter: float = 0.0,
        seed: int = 42,
    ):
        if not 50.0 <= percentile <= 99.9:
            raise ConfigurationError("percentile should be between 50 and 99.9")
        self.embedder = embedder
        self.percentile = percentile
        self.min_sentences = min_sentences
        self.max_words = max_words
        self.jitter = jitter
        self.seed = seed

    @property
    def spec(self) -> str:
        return (
            f"semantic/v1(embedder={self.embedder.fingerprint},"
            f"pct={self.percentile},min={self.min_sentences},"
            f"max_words={self.max_words},jitter={self.jitter},seed={self.seed})"
        )

    def _boundaries(self, sentences: list[str]) -> set[int]:
        vectors = np.asarray(self.embedder.encode(sentences), dtype=np.float64)

        if self.jitter:
            from .determinism import rng

            generator = rng(self.seed, "semantic-jitter", len(sentences))
            vectors = vectors + generator.normal(scale=self.jitter, size=vectors.shape)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        unit = vectors / norms

        distances = 1.0 - np.sum(unit[:-1] * unit[1:], axis=1)
        if distances.size == 0:
            return set()

        threshold = float(np.percentile(distances, self.percentile))
        return {i + 1 for i, distance in enumerate(distances) if distance >= threshold}

    def split(self, record: Record) -> list[Chunk]:
        body = body_text(record)
        sentences = split_sentences(body)
        if not sentences:
            return []
        if len(sentences) <= self.min_sentences:
            return self._emit(record, [("unknown", " ".join(sentences))])

        cuts = self._boundaries(sentences)

        pieces, buffer, count = [], [], 0
        for position, sentence in enumerate(sentences):
            words = len(sentence.split())
            too_long = buffer and count + words > self.max_words
            at_boundary = position in cuts and len(buffer) >= self.min_sentences
            if too_long or at_boundary:
                pieces.append(("unknown", " ".join(buffer)))
                buffer, count = [], 0
            buffer.append(sentence)
            count += words

        if buffer:
            pieces.append(("unknown", " ".join(buffer)))
        return self._emit(record, pieces)
