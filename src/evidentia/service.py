"""HTTP service.

Deliberately shaped like LitRev's existing `extractor` service: same
container conventions, same `/health` contract, same request style, so it
drops into the compose file next to it and a Laravel job can call it the way
`ExtractReviewSourceFileJob` already calls GROBID.

Every response carries the certificate, not just the results. A caller that
persists only the ranked list has thrown away the ability to verify it later.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .certificate import issue
from .chunking import CHUNKERS, build_chunker
from .corpus import Corpus, Record
from .determinism import DeterminismConfig, enforce
from .embed import HashEmbedder
from .exceptions import BackendUnavailableError, ConfigurationError, UnknownBackendError
from .index import build_index
from .litrev import corpus_from_source_files
from .retrieve import Retriever

app = FastAPI(title="Evidentia", version=__version__)

_DETERMINISM = enforce(DeterminismConfig())


class SourceFile(BaseModel):
    extraction_status: str = "extracted"
    extracted_text: str = ""
    extracted_metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveRequest(BaseModel):
    query: str
    source_files: list[SourceFile] = Field(default_factory=list)
    records: list[dict[str, Any]] = Field(default_factory=list)
    k: int = 20
    mode: Literal["dense", "lexical", "hybrid"] = "hybrid"
    chunker: str = "tei-section"
    chunker_params: dict[str, Any] = Field(default_factory=dict)
    index: str = "flat"
    index_params: dict[str, Any] = Field(default_factory=dict)
    section_filter: list[str] = Field(default_factory=list)
    embedder: Literal["hash", "sbert"] = "hash"
    embedder_params: dict[str, Any] = Field(default_factory=dict)


def _build_embedder(name: str, params: dict):
    if name == "hash":
        return HashEmbedder(**params)
    if name == "sbert":
        from .embed import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(**params)
    raise HTTPException(status_code=422, detail=f"unknown embedder: {name}")


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "version": __version__,
        "determinism": _DETERMINISM.as_dict(),
        "litrev_url": os.getenv("LITREV_URL", ""),
    }


@app.get("/backends")
async def backends() -> dict:
    available = {}
    for name in ("faiss", "sentence_transformers", "qdrant_client", "chromadb", "psycopg", "transformers"):
        try:
            __import__(name)
            available[name] = True
        except ImportError:
            available[name] = False
    return {"chunkers": sorted(CHUNKERS), "optional_dependencies": available}


@app.post("/retrieve")
async def retrieve(request: RetrieveRequest) -> dict:
    if request.source_files:
        corpus = corpus_from_source_files(
            [file.model_dump() for file in request.source_files]
        )
    elif request.records:
        corpus = Corpus([Record.build(**row) for row in request.records])
    else:
        raise HTTPException(status_code=422, detail="provide source_files or records")

    if len(corpus) == 0:
        raise HTTPException(status_code=422, detail="corpus is empty after canonicalisation")

    embedder = _build_embedder(request.embedder, request.embedder_params)

    chunker_params = dict(request.chunker_params)
    if request.chunker == "tei-section" and request.section_filter:
        chunker_params.setdefault("keep_sections", tuple(request.section_filter))
    try:
        chunker = build_chunker(request.chunker, embedder=embedder, **chunker_params)
    except (UnknownBackendError, ConfigurationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TypeError as exc:
        # Wrong chunker_params for this chunker: the caller's mistake, and it
        # must come back as 422, not surface as an unhandled server error.
        raise HTTPException(status_code=422, detail=f"chunker_params: {exc}") from exc

    try:
        index = build_index(request.index, **request.index_params)
    except (UnknownBackendError, BackendUnavailableError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    retriever = Retriever(corpus, embedder, chunker=chunker, index=index)
    retriever.prepare(with_lexical=request.mode in {"lexical", "hybrid"})

    evidence = retriever.retrieve(
        request.query,
        k=request.k,
        mode=request.mode,
        section_filter=tuple(request.section_filter),
    )
    certificate = issue(evidence, corpus_size=len(corpus), determinism=_DETERMINISM)

    return {
        "evidence": evidence.as_dict(),
        "certificate": certificate.as_dict(),
        "corpus": {
            "size": len(corpus),
            "hash": corpus.corpus_hash,
            "duplicates_removed": len(corpus.duplicates),
        },
    }
