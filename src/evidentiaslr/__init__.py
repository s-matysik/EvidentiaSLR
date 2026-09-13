"""EvidentiaSLR — reproducible retrieval for literature analysis.

Vector databases are treated as neutral plumbing in retrieval-augmented
literature work. They are not. Approximate indexes drop documents in a
seed-dependent way, quantisation is lossy, and the same encoder on two devices
disagrees in the low-order bits — so the same corpus and the same question can
yield a different evidence set, and therefore a different conclusion.

EvidentiaSLR makes that measurable (Evidence Set Fidelity, Evidence Set
Stability, Synthesis Divergence) and makes a given run verifiable (evidence
certificates).
"""

from __future__ import annotations

# Single source of truth is pyproject.toml; read it from installed metadata so
# the number cannot drift again (it already had, three ways: 0.1.0 here,
# 1.0.0 in CITATION.cff, 0.1.0 in pyproject).
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("evidentiaslr")
except Exception:  # pragma: no cover - source tree without installation
    __version__ = "1.0.0"

from .certificate import Certificate, issue, verify
from .chunking import (
    Chunk,
    FixedWindowChunker,
    RecordChunker,
    SentenceChunker,
    TEISectionChunker,
)
from .corpus import Corpus, Record
from .determinism import DeterminismConfig, enforce
from .embed import HashEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder
from .exceptions import (
    BackendUnavailableError,
    ConfigurationError,
    CorpusError,
    EvidentiaError,
    PipelineStateError,
    UnknownBackendError,
)
from .index import FlatIndex, RandomProjectionLSH, build_index
from .io import ImportReport, load_bibtex, load_corpus, load_ris, load_scopus_csv, load_wos_csv
from .metrics import (
    crossover_analysis,
    evidence_set_fidelity,
    evidence_set_stability,
    record_agreement,
    span_agreement,
    synthesis_divergence,
)
from .retrieve import EvidenceSet, Retriever

__all__ = [
    "__version__",
    "BackendUnavailableError",
    "Certificate",
    "Chunk",
    "ConfigurationError",
    "CorpusError",
    "Corpus",
    "DeterminismConfig",
    "EvidenceSet",
    "EvidentiaError",
    "FixedWindowChunker",
    "FlatIndex",
    "HashEmbedder",
    "ImportReport",
    "OpenAIEmbedder",
    "RandomProjectionLSH",
    "PipelineStateError",
    "Record",
    "RecordChunker",
    "Retriever",
    "SentenceChunker",
    "SentenceTransformerEmbedder",
    "TEISectionChunker",
    "UnknownBackendError",
    "build_index",
    "crossover_analysis",
    "record_agreement",
    "span_agreement",
    "enforce",
    "evidence_set_fidelity",
    "evidence_set_stability",
    "issue",
    "load_bibtex",
    "load_corpus",
    "load_ris",
    "load_scopus_csv",
    "load_wos_csv",
    "synthesis_divergence",
    "verify",
]
