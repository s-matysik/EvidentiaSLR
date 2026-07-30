"""Evidence certificates.

A certificate records everything needed to re-derive an evidence set, plus a
digest of the evidence set itself. A reviewer runs `evidentia verify` and gets
a binary answer: either the pipeline reproduces the reported evidence, or it
does not and the certificate says exactly which component drifted.

This is the artefact that makes a retrieval-augmented review auditable in the
same sense a PRISMA flow diagram makes a screening process auditable.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .determinism import DeterminismConfig

if TYPE_CHECKING:  # pragma: no cover
    from .retrieve import EvidenceSet

SCHEMA_VERSION = "evidentia-certificate/1"


def _environment() -> dict:
    versions = {"python": sys.version.split()[0]}
    for module in ("numpy", "faiss", "sentence_transformers", "torch", "transformers"):
        try:
            imported = __import__(module)
            versions[module] = getattr(imported, "__version__", "unknown")
        except Exception:
            continue
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "versions": versions,
    }


def evidence_digest(chunk_ids) -> str:
    """Order-sensitive digest: rank changes are detected, not only membership."""
    digest = hashlib.sha256()
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        digest.update(f"{rank}:{chunk_id}".encode("ascii"))
    return digest.hexdigest()


@dataclass
class Certificate:
    query: str
    k: int
    mode: str
    corpus_hash: str
    corpus_size: int
    chunker_spec: str
    embedder_fingerprint: str
    index_spec: str
    section_filter: list[str]
    determinism: dict
    evidence_digest: str
    chunk_ids: list[str]
    schema: str = SCHEMA_VERSION
    created_at: str = ""
    environment: dict = field(default_factory=_environment)
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        payload = {
            "schema": self.schema,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
            "query": self.query,
            "k": self.k,
            "mode": self.mode,
            "section_filter": self.section_filter,
            "corpus": {"hash": self.corpus_hash, "size": self.corpus_size},
            "pipeline": {
                "chunker": self.chunker_spec,
                "embedder": self.embedder_fingerprint,
                "index": self.index_spec,
            },
            "determinism": self.determinism,
            "evidence": {"digest": self.evidence_digest, "chunk_ids": self.chunk_ids},
            "environment": self.environment,
            "notes": self.notes,
        }
        return payload

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))


def issue(
    evidence: EvidenceSet,
    corpus_size: int,
    determinism: DeterminismConfig | None = None,
    notes: dict | None = None,
) -> Certificate:
    determinism = determinism or DeterminismConfig()
    chunk_ids = evidence.chunk_ids
    return Certificate(
        query=evidence.query,
        k=evidence.k,
        mode=evidence.mode,
        corpus_hash=evidence.corpus_hash,
        corpus_size=corpus_size,
        chunker_spec=evidence.chunker_spec,
        embedder_fingerprint=evidence.embedder_fingerprint,
        index_spec=evidence.index_spec,
        section_filter=list(evidence.section_filter),
        determinism=determinism.as_dict(),
        evidence_digest=evidence_digest(chunk_ids),
        chunk_ids=chunk_ids,
        created_at=datetime.now(timezone.utc).isoformat(),
        notes=notes or {},
    )


@dataclass
class VerificationResult:
    ok: bool
    mismatches: list[str]
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "mismatches": self.mismatches, "detail": self.detail}


def verify(certificate: dict, evidence: EvidenceSet, corpus_size: int | None = None) -> VerificationResult:
    """Compare a fresh run against a stored certificate, component by component."""
    mismatches: list[str] = []
    detail: dict = {}

    checks = [
        ("corpus.hash", certificate["corpus"]["hash"], evidence.corpus_hash),
        ("pipeline.chunker", certificate["pipeline"]["chunker"], evidence.chunker_spec),
        ("pipeline.embedder", certificate["pipeline"]["embedder"], evidence.embedder_fingerprint),
        ("pipeline.index", certificate["pipeline"]["index"], evidence.index_spec),
        ("query", certificate["query"], evidence.query),
        ("mode", certificate["mode"], evidence.mode),
        ("k", certificate["k"], evidence.k),
    ]
    if corpus_size is not None:
        checks.append(("corpus.size", certificate["corpus"]["size"], corpus_size))

    for name, expected, actual in checks:
        if expected != actual:
            mismatches.append(name)
            detail[name] = {"expected": expected, "actual": actual}

    actual_digest = evidence_digest(evidence.chunk_ids)
    if certificate["evidence"]["digest"] != actual_digest:
        mismatches.append("evidence.digest")
        expected_ids = certificate["evidence"]["chunk_ids"]
        actual_ids = evidence.chunk_ids
        detail["evidence.digest"] = {
            "expected": certificate["evidence"]["digest"],
            "actual": actual_digest,
            "missing": sorted(set(expected_ids) - set(actual_ids)),
            "spurious": sorted(set(actual_ids) - set(expected_ids)),
            "reordered_only": set(expected_ids) == set(actual_ids),
        }

    return VerificationResult(ok=not mismatches, mismatches=mismatches, detail=detail)
