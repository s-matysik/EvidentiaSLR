"""Retrieval pipeline.

`Retriever` ties a corpus, a chunker, an embedder and an index into one
object whose output is an `EvidenceSet` — an ordered list of chunks plus
everything needed to reconstruct how they were selected.

Two design decisions carry the reproducibility argument:

1. Reciprocal rank fusion for hybrid retrieval uses ranks, not scores, so
   the dense and lexical halves do not need commensurable scales, and the
   result is invariant to monotone rescaling of either.
2. Section filtering happens before fusion, not after, so restricting a
   query to method sections changes which chunks compete rather than merely
   trimming the winners.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .chunking import Chunk, Chunker, RecordChunker
from .corpus import Corpus
from .determinism import DEFAULT_SCORE_PRECISION, DeterminismConfig
from .embed import EmbeddingModel
from .exceptions import ConfigurationError, PipelineStateError
from .index import FlatIndex
from .index.base import BaseIndex
from .lexical import BM25


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    record_id: str
    section: str
    text: str
    score: float
    rank: int
    start: int = -1
    end: int = -1

    @property
    def located(self) -> bool:
        return self.start >= 0 and self.end > self.start


@dataclass
class EvidenceSet:
    query: str
    k: int
    chunks: list[RetrievedChunk]
    index_spec: str
    chunker_spec: str
    embedder_fingerprint: str
    corpus_hash: str
    mode: str = "dense"
    section_filter: tuple[str, ...] = ()
    extra: dict = field(default_factory=dict)

    @property
    def chunk_ids(self) -> list[str]:
        return [chunk.chunk_id for chunk in self.chunks]

    @property
    def record_ids(self) -> list[str]:
        seen, out = set(), []
        for chunk in self.chunks:
            if chunk.record_id not in seen:
                seen.add(chunk.record_id)
                out.append(chunk.record_id)
        return out

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "k": self.k,
            "mode": self.mode,
            "section_filter": list(self.section_filter),
            "corpus_hash": self.corpus_hash,
            "chunker": self.chunker_spec,
            "embedder": self.embedder_fingerprint,
            "index": self.index_spec,
            "chunks": [
                {
                    "rank": chunk.rank,
                    "chunk_id": chunk.chunk_id,
                    "record_id": chunk.record_id,
                    "section": chunk.section,
                    "start": chunk.start,
                    "end": chunk.end,
                    "score": chunk.score,
                }
                for chunk in self.chunks
            ],
        }


class Retriever:
    def __init__(
        self,
        corpus: Corpus,
        embedder: EmbeddingModel,
        chunker: Chunker | None = None,
        index: BaseIndex | None = None,
        determinism: DeterminismConfig | None = None,
        score_precision: int = DEFAULT_SCORE_PRECISION,
    ):
        self.corpus = corpus
        self.embedder = embedder
        self.chunker = chunker or RecordChunker()
        self.index = index or FlatIndex()
        self.determinism = determinism or DeterminismConfig()
        self.score_precision = score_precision

        self.chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self._bm25: BM25 | None = None
        self._vectors: np.ndarray | None = None

    def prepare(self, with_lexical: bool = False) -> Retriever:
        self.chunks = self.chunker.chunk_corpus(self.corpus)
        if not self.chunks:
            raise ConfigurationError("chunking produced no chunks")
        self._by_id = {chunk.chunk_id: chunk for chunk in self.chunks}

        texts = [chunk.text for chunk in self.chunks]
        ids = [chunk.chunk_id for chunk in self.chunks]
        self._vectors = self.embedder.encode(texts)
        self.index.build(self._vectors, ids)

        if with_lexical:
            self._bm25 = BM25(score_precision=self.score_precision)
            self._bm25.build(texts, ids)
        return self

    def _pool(self, section_filter: Sequence[str]) -> set[str] | None:
        if not section_filter:
            return None
        wanted = set(section_filter)
        return {chunk.chunk_id for chunk in self.chunks if chunk.section in wanted}

    def _to_evidence(
        self,
        query: str,
        pairs: list[tuple[str, float]],
        k: int,
        mode: str,
        section_filter: Sequence[str],
    ) -> EvidenceSet:
        chunks = []
        for rank, (chunk_id, score) in enumerate(pairs[:k], start=1):
            chunk = self._by_id[chunk_id]
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    record_id=chunk.record_id,
                    section=chunk.section,
                    text=chunk.text,
                    score=score,
                    rank=rank,
                    start=chunk.start,
                    end=chunk.end,
                )
            )
        return EvidenceSet(
            query=query,
            k=k,
            chunks=chunks,
            index_spec=self.index.spec,
            chunker_spec=self.chunker.spec,
            embedder_fingerprint=self.embedder.fingerprint,
            corpus_hash=self.corpus.corpus_hash,
            mode=mode,
            section_filter=tuple(sorted(section_filter)),
        )

    def retrieve(
        self,
        query: str,
        k: int = 20,
        mode: str = "dense",
        section_filter: Sequence[str] = (),
        rrf_k: int = 60,
        overfetch: int = 4,
    ) -> EvidenceSet:
        if self._vectors is None:
            raise PipelineStateError("call prepare() first")
        if mode not in {"dense", "lexical", "hybrid"}:
            raise ConfigurationError(f"unknown mode: {mode}")

        pool = self._pool(section_filter)
        # Over-fetch so that section filtering does not silently shorten the
        # evidence set; the filter must not be a post-hoc truncation.
        fetch = k * overfetch if pool is not None else k

        dense_pairs: list[tuple[str, float]] = []
        if mode in {"dense", "hybrid"}:
            query_vector = self.embedder.encode([query])[0]
            dense_pairs = self.index.search(query_vector, fetch)
            if pool is not None:
                dense_pairs = [pair for pair in dense_pairs if pair[0] in pool]

        lexical_pairs: list[tuple[str, float]] = []
        if mode in {"lexical", "hybrid"}:
            if self._bm25 is None:
                raise PipelineStateError("prepare(with_lexical=True) is required for this mode")
            lexical_pairs = self._bm25.search(query, fetch)
            if pool is not None:
                lexical_pairs = [pair for pair in lexical_pairs if pair[0] in pool]

        if mode == "dense":
            return self._to_evidence(query, dense_pairs, k, mode, section_filter)
        if mode == "lexical":
            return self._to_evidence(query, lexical_pairs, k, mode, section_filter)

        fused = reciprocal_rank_fusion(
            [pairs for pairs in (dense_pairs, lexical_pairs)],
            rrf_k=rrf_k,
            score_precision=self.score_precision,
        )
        return self._to_evidence(query, fused, k, mode, section_filter)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[str, float]]],
    rrf_k: int = 60,
    score_precision: int = DEFAULT_SCORE_PRECISION,
) -> list[tuple[str, float]]:
    """Fuse ranked lists by reciprocal rank, breaking ties on identifier."""
    totals: dict[str, float] = {}
    for ranking in rankings:
        for rank, (identifier, _score) in enumerate(ranking, start=1):
            totals[identifier] = totals.get(identifier, 0.0) + 1.0 / (rrf_k + rank)
    rounded = {identifier: round(value, score_precision) for identifier, value in totals.items()}
    return sorted(rounded.items(), key=lambda pair: (-pair[1], pair[0]))
